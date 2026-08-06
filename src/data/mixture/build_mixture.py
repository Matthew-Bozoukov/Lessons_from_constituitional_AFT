# ABOUTME: Builds an N-source SFT mixture at per-source budgets, with an optional
# ABOUTME: constitution filter stage and HF push checkpoints after each pipeline stage.

"""Build a training mixture: sample sources, optionally filter, optionally add synthetic.

Two output modes, decided by the config (never per row — a mixture file is homogeneous):

* **Interchange (the default for new configs)** — rows are model-agnostic chat
  transcripts, `{"messages": [...], "source": ...}` with optional per-turn
  `reasoning_content`/`tool_calls` (see src/data/mixture/sources/). No chat template is
  applied at build time; train_lora renders with the training family's `ModelProfile`
  and the tokenizer here is used only to *count* tokens for budgets and length caps.
* **Legacy rendered** — any source declaring `reasoning: strip` or `format: rendered`
  switches the whole build to the historical pre-rendered `{"text": ...}` form, exactly
  as the pre-2026-08-06 builder produced it. Existing configs regenerate their artifacts
  unchanged; new configs should not use these kinds.

The optional stages, both config-driven (a config with neither behaves as a single pass):

* `filter:` — the spec-alignment judge (src/data/mixture/spec_filter.py) screens every
  NON-synthetic row against a constitution; sources marked `synthetic: true` join only
  after the filter (they are constitution-generated — judging them against it is
  circular). `filter.keep_examples` then downsamples the kept rows stratified by source.
* `hf:` — push checkpoints as they are produced (a dead run loses nothing):
  `mixture_unfiltered.jsonl` after the initial mix and `mixture_filtered.jsonl` +
  `verdicts.jsonl` + `filter_report.json` after filtering, both to `hf.base_repo`;
  the final `mixture.jsonl` (synthetic mixed in) to `hf.final_repo`.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.data.mixture.sources import SOURCES, clean_messages
from src.utils import git_sha, think_census, model_profile, timestamp, write_run_meta

# Rendering policy for LEGACY rendered mode (the repo-wide default 2026-08-04 until the
# interchange format replaced build-time rendering on 2026-08-06): the profile's
# render_kwargs (Qwen3.6: preserve_thinking=True) make the template emit a think block on
# EVERY assistant turn — reasoning_content where the row has it, the empty marker where it
# does not. The generation-boundary mask (src/train/masking.py) wholly masks empty markers
# and supervises real traces, so empty markers are safe by construction.
# Each source declares what its DATA carries via `reasoning:`:
#   native — rows carry reasoning_content (validated: every row keeps a real trace)
#   none   — rows have no reasoning at all (validated)
#   strip  — LEGACY: deliberate pre-policy no-think rendering; forces legacy mode.
_REASONING_KINDS = ("native", "none", "strip")

# The sentinel trick behind `strip`: the template only thinks on the FINAL assistant turn,
# so appending a throwaway user turn pushes the assistant off the end and takes the
# no-think branch; the throwaway turn is then cut.
_SENTINEL = "__MIXTURE_SENTINEL__"

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# Every budget (tokens or examples) is divided by this under --smoke, so a smoke run
# exercises the full wiring (rendering, streaming, validation, stats) in seconds.
_SMOKE_SCALE = 20

# Judge calls a --smoke run is allowed to spend (a fraction of a cent): enough to prove
# the checkpoint/parse/report wiring, never enough to look like a completed filter.
_SMOKE_JUDGE_LIMIT = 3


def _render_preserved(tok, messages: list[dict], render_kwargs: dict) -> str:
    """Render under the preserve-thinking policy: a think block on every assistant turn.

    Plain `apply_chat_template` with the family profile's render kwargs — no sentinel, no
    post-hoc surgery. Verified live (2026-08-04): with preserve_thinking=True Qwen3.6
    renders each turn's reasoning_content, and the empty marker where a turn has none.
    """
    assert messages[-1]["role"] == "assistant", "conversation must end with an assistant turn"
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                   **render_kwargs)
    n_turns = sum(1 for m in messages if m["role"] == "assistant")
    assert text.count("<think>") == n_turns, (
        f"expected a think block on every assistant turn ({n_turns}), got "
        f"{text.count('<think>')} — template drift from the verified preserve behaviour?")
    return text


def _render_without_think(tok, messages: list[dict]) -> str:
    """Render with no <think> block at all (`reasoning: strip` — legacy, non-default).

    Args:
        tok: The tokenizer.
        messages: Conversation ending in an assistant turn.

    Returns:
        The rendered text, truncated before the appended throwaway user turn.
    """
    assert messages[-1]["role"] == "assistant", "conversation must end with an assistant turn"
    padded = messages + [{"role": "user", "content": _SENTINEL}]
    text = tok.apply_chat_template(padded, tokenize=False, add_generation_prompt=False)
    text = text[: text.rindex("<|im_start|>user")]
    assert _SENTINEL not in text, "failed to strip the throwaway turn"
    assert "<think>" not in text, "strip rendering must contain no think block"
    return text


def _usable(msgs: list[dict]) -> bool:
    """Return True when a conversation is well-formed enough to render (legacy check)."""
    if len(msgs) < 2 or msgs[-1].get("role") != "assistant":
        return False
    if not all(isinstance(m.get("content"), str) and m["content"] for m in msgs):
        return False
    return all(m.get("role") in ("system", "user", "assistant") for m in msgs)


def _ntok(tok, text: str) -> int:
    """Token count of a rendered example."""
    return len(tok(text)["input_ids"])


def _budget(name: str, spec: dict, scale: int) -> tuple[str, int]:
    """Return one source's sampling budget as (kind, n), kind `tokens` | `examples`.

    Exactly one of `tokens:` / `examples:` must be declared — token budgets build
    token-share mixtures (the historical arms), example budgets build count-share
    mixtures (e.g. the Table-2 counts). Both divide by the smoke scale so `--smoke`
    exercises the same path.
    """
    declared = [k for k in ("tokens", "examples") if spec.get(k) is not None]
    if len(declared) != 1:
        raise ValueError(
            f"source {name!r}: declare exactly one of `tokens` or `examples`, "
            f"got {declared or 'neither'}")
    kind = declared[0]
    n = int(spec[kind])
    assert n > 0, f"source {name!r}: {kind} must be positive, got {n}"
    return kind, max(1, n // scale)


def _fill(rows: list[dict], budget: int, seed: int) -> list[dict]:
    """Greedily take shuffled rows (field: n_tokens) up to a token budget."""
    random.Random(seed).shuffle(rows)
    out, total = [], 0
    for r in rows:
        if total + r["n_tokens"] > budget:
            continue
        out.append(r)
        total += r["n_tokens"]
        if total >= budget * 0.995:
            break
    return out


def _fill_budget(rows: list[dict], budget: tuple[str, int], seed: int) -> list[dict]:
    """Sample loaded rows to a (kind, n) budget: token-greedy or exactly-n examples.

    An example budget is exact by contract — a source that cannot supply n rows fails
    loudly rather than silently shrinking its share of the mixture.
    """
    kind, n = budget
    if kind == "tokens":
        return _fill(rows, n, seed)
    assert len(rows) >= n, (
        f"asked for {n} examples but the source holds only {len(rows)}")
    random.Random(seed).shuffle(rows)
    return rows[:n]


def balanced_take(groups: dict[str, list[dict]], n: int, seed: int, name: str,
                  key: str) -> list[dict]:
    """Take exactly n rows split as evenly as possible across groups (absorbs the old
    balanced_subset.py: the mixture's token/example budgets do not preserve per-group
    counts, so balance is enforced at selection time).

    Quotas differ by at most one (the remainder goes to the first groups in sorted
    order, deterministically); a group that cannot fill its quota fails loudly —
    the balance is the experiment, never a silent shrink.
    """
    assert groups, f"{name}: balance_by={key!r} produced no groups"
    per, rem = divmod(n, len(groups))
    rng = random.Random(seed)
    picked: list[dict] = []
    for i, g in enumerate(sorted(groups)):
        quota = per + (1 if i < rem else 0)
        avail = groups[g]
        assert len(avail) >= quota, (
            f"{name}: {key}={g!r} has {len(avail)} rows, quota needs {quota}")
        rng.shuffle(avail)
        picked += avail[:quota]
    rng.shuffle(picked)
    assert len(picked) == n, len(picked)
    return picked


def stratified_subset(rows: list[dict], n: int, seed: int) -> tuple[list[dict], dict]:
    """Downsample rows to exactly n, holding each source's share (rounded).

    Rounding can land a few off n; the remainder is trimmed or topped up from the
    whole pool after shuffling, so the result is deterministic given the seed.

    Returns:
        (subset, quota_by_source).
    """
    assert len(rows) >= n, f"cannot take {n} of {len(rows)} rows"
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["source"], []).append(r)
    rng = random.Random(seed)
    quota = {s: round(len(g) * n / len(rows)) for s, g in sorted(by.items())}
    picked: list[dict] = []
    for s, g in sorted(by.items()):
        rng.shuffle(g)
        picked += g[:quota[s]]
    rng.shuffle(picked)
    if len(picked) > n:
        picked = picked[:n]
    elif len(picked) < n:
        chosen = {id(r) for r in picked}
        spare = [r for r in rows if id(r) not in chosen]
        rng.shuffle(spare)
        picked += spare[:n - len(picked)]
    assert len(picked) == n, len(picked)
    return picked, quota


# --------------------------------------------------------------------------------------
# Legacy rendered loaders — kept byte-for-byte compatible with the pre-interchange
# builder so every existing config regenerates its published artifact unchanged.
# --------------------------------------------------------------------------------------

def _take_messages(tok, path: Path, budget: tuple[str, int], seed: int, source: str,
                   render_kwargs: dict) -> list[dict]:
    """Sample a raw chat jsonl to a budget, reasoning preserved on every turn."""
    rows = []
    for line in path.open():
        text = _render_preserved(tok, json.loads(line)["messages"], render_kwargs)
        rows.append({"text": text, "source": source, "n_tokens": _ntok(tok, text)})
    assert rows, f"no rows in {path}"
    return _fill_budget(rows, budget, seed)


def _take_rendered(path: Path, budget: tuple[str, int], seed: int, source: str) -> list[dict]:
    """Sample an already-rendered jsonl (fields: text, n_tokens) to a budget.

    An optional per-row `supervise` field ("all" | "final") rides through into the
    mixture so the trainer can mask non-final assistant turns (the model-eval-model
    self-reflection records); rows without it train every assistant turn as before.
    """
    rows = [
        {"text": r["text"], "source": source, "n_tokens": r["n_tokens"],
         **({"supervise": r["supervise"]} if r.get("supervise") else {})}
        for r in map(json.loads, path.open())
    ]
    assert rows, f"no rows in {path}"
    return _fill_budget(rows, budget, seed)


def _take_hf(tok, repo: str, split: str, budget: tuple[str, int], seed: int, max_len: int,
             source: str, shuffle_buffer: int, renderer) -> list[dict]:
    """Stream an HF chat dataset and sample to a (kind, n) budget (legacy rendered).

    Rows are RENDERED FIRST and length-capped after, so the cap counts the think tokens
    the renderer added (the ordering defect PR #16 recorded in its always_think surgery
    cannot occur here).

    Raises:
        RuntimeError: An example budget the stream cannot fill — the mixture's shares
            are the experiment, so a short source is an error, never a silent shrink.
    """
    kind, want = budget
    ds = load_dataset(repo, split=split, streaming=True).shuffle(
        seed=seed, buffer_size=shuffle_buffer)
    out, total, skipped = [], 0, 0
    for row in ds:
        msgs = row.get("messages") or []
        if not _usable(msgs):
            skipped += 1
            continue
        try:
            text = renderer(msgs)
        except (AssertionError, ValueError):
            skipped += 1
            continue
        n = _ntok(tok, text)
        if n > max_len:
            skipped += 1
            continue
        if kind == "tokens":
            if total + n > want:
                continue
            out.append({"text": text, "source": source, "n_tokens": n})
            total += n
            if total >= want * 0.995:
                break
        else:
            out.append({"text": text, "source": source, "n_tokens": n})
            if len(out) == want:
                break
    print(f"  (skipped {skipped} {source} rows: wrong shape, unsupported role, or too long)")
    if kind == "examples" and len(out) < want:
        raise RuntimeError(
            f"source {source!r}: stream exhausted at {len(out)}/{want} examples "
            f"(after {skipped} skips) — the declared mixture share cannot be met")
    return out


def _load_source_legacy(tok, cfg, name: str, spec: dict, budget: tuple[str, int],
                        seed: int, render_kwargs: dict) -> tuple[list[dict], str]:
    """Load one source in legacy rendered mode and classify it for validation.

    Returns:
        (rows, kind) where kind is the spec's `reasoning:` declaration for HF sources
        (`native` | `none` | `strip`), `think` for local messages jsonl (traces
        preserved on every turn), or `rendered` (pre-rendered, validated upstream).
    """
    if "repo" in spec:
        if "think_marker" in spec:
            raise ValueError(
                f"source {name!r}: `think_marker` was replaced on 2026-08-04 by the "
                "required `reasoning:` declaration (native|none|strip); preserved "
                "rendering with empty markers is now the default for `reasoning: none`.")
        kind = spec.get("reasoning")
        if kind not in _REASONING_KINDS:
            raise ValueError(
                f"source {name!r}: HF sources must declare `reasoning: "
                f"{'|'.join(_REASONING_KINDS)}` — what the DATA carries is part of the "
                "scientific record (like `thinking:` in train configs) and is validated "
                "against the rendered rows, never guessed.")
        renderer = (lambda msgs: _render_without_think(tok, msgs)) if kind == "strip" \
            else (lambda msgs: _render_preserved(tok, msgs, render_kwargs))
        rows = _take_hf(tok, spec["repo"], spec.get("split", "train"), budget, seed,
                        int(cfg.max_seq_len), name,
                        int(spec.get("shuffle_buffer", cfg.get("shuffle_buffer", 1000))),
                        renderer)
        return rows, kind
    fmt = spec["format"]
    if fmt == "messages":
        return _take_messages(tok, Path(spec["path"]), budget, seed, name,
                              render_kwargs), "think"
    if fmt == "rendered":
        return _take_rendered(Path(spec["path"]), budget, seed, name), "rendered"
    raise ValueError(f"source {name!r}: unknown format {fmt!r} (messages|rendered)")


# --------------------------------------------------------------------------------------
# Interchange loaders — model-agnostic messages rows; the tokenizer only counts.
# --------------------------------------------------------------------------------------

def _take_interchange(tok, cfg, name: str, spec: dict, budget: tuple[str, int],
                      seed: int, render_kwargs: dict) -> tuple[list[dict], str]:
    """Load one source as interchange messages rows and validate its reasoning kind.

    The spec names rows via a registry adapter (`source:`), a raw HF chat repo
    (`repo:`), or a local jsonl (`path:`); adapters supply repo/config/normaliser
    defaults the spec can override. Budgets and the `max_seq_len` cap are counted on
    the PRESERVED render of each row (what training will render), with the config's
    tokenizer — the counts are model-relative, the stored data is not.

    Returns:
        (rows, kind): rows carry `messages`, `source`, `n_tokens` (+`supervise` when a
        local row declares it); kind is the validated `reasoning:` declaration.
    """
    # A spec names its adapter explicitly (`source:`), or implicitly by its own key when
    # it declares no other origin — so `no_robots: {examples: N, reasoning: none}` just
    # works. Raw `repo:`/`path:` specs need no adapter at all.
    adapter_name = spec.get("source") or (
        name if not ("repo" in spec or "path" in spec) else None)
    adapter = None
    if adapter_name is not None:
        if adapter_name not in SOURCES:
            raise ValueError(f"source {name!r}: unknown adapter {adapter_name!r} "
                             f"(known: {', '.join(sorted(SOURCES))})")
        adapter = SOURCES[adapter_name]
    kind = spec.get("reasoning")
    if kind not in ("native", "none"):
        raise ValueError(
            f"source {name!r}: interchange sources must declare `reasoning: native|none`"
            " — what the DATA carries is part of the scientific record. (`strip` is a "
            "legacy RENDERED-mode kind: it forces the whole config to the pre-2026-08-06"
            " build-time rendering; nothink arms now choose their render at train time.)")
    to_messages = adapter.to_messages if adapter else \
        (lambda row: clean_messages(row.get("messages")))
    if spec.get("balance_by") and "path" not in spec:
        raise ValueError(
            f"source {name!r}: balance_by requires a local `path:` source — a stream "
            "cannot be grouped without loading the whole pool")

    def payload(raw: dict) -> dict | None:
        msgs = to_messages(raw)
        if msgs is None:
            return None
        # return_dict + explicit ["input_ids"]: with tokenize=True this transformers
        # version hands back a BatchEncoding either way, and len() of THAT is its key
        # count (2), not the token count — caught live 2026-08-06 (2 "tokens" per row).
        n = len(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                        return_dict=True, **render_kwargs)["input_ids"])
        if n > int(cfg.max_seq_len):
            return None
        out = {"messages": msgs, "source": name, "n_tokens": n}
        # supervise rides top-level or under metadata (synthdoc stage-5 exports put it
        # there); losing it would silently train non-target turns (supervise: final).
        supervise = raw.get("supervise") or (raw.get("metadata") or {}).get("supervise")
        if supervise:
            out["supervise"] = supervise
        return out

    if "path" in spec:
        bkey = spec.get("balance_by")
        rows, groups = [], {}
        for line in Path(spec["path"]).open():
            raw = json.loads(line)
            p = payload(raw)
            if p is None:
                continue
            rows.append(p)
            if bkey:
                g = raw.get(bkey) or (raw.get("metadata") or {}).get(bkey)
                assert g is not None, \
                    f"source {name!r}: row missing balance_by field {bkey!r}"
                groups.setdefault(str(g), []).append(p)
        assert rows, f"no usable rows in {spec['path']}"
        if bkey:
            b_kind, want = budget
            assert b_kind == "examples", \
                f"source {name!r}: balance_by needs an `examples:` budget (got {b_kind})"
            return balanced_take(groups, want, seed, name, str(bkey)), kind
        return _fill_budget(rows, budget, seed), kind

    repo = spec.get("repo") or (adapter.repo if adapter else None)
    if not repo:
        raise ValueError(f"source {name!r}: needs `source:` (registry), `repo:` or `path:`")
    hf_config = spec.get("config") or (adapter.hf_config if adapter else None)
    split = spec.get("split") or (adapter.split if adapter else "train")
    args = [repo] + ([hf_config] if hf_config else [])
    ds = load_dataset(*args, split=split, streaming=True).shuffle(
        seed=seed,
        buffer_size=int(spec.get("shuffle_buffer", cfg.get("shuffle_buffer", 1000))))
    b_kind, want = budget
    out, total, skipped = [], 0, 0
    for raw in ds:
        p = payload(raw)
        if p is None:
            skipped += 1
            continue
        if b_kind == "tokens":
            if total + p["n_tokens"] > want:
                continue
            out.append(p)
            total += p["n_tokens"]
            if total >= want * 0.995:
                break
        else:
            out.append(p)
            if len(out) == want:
                break
    print(f"  (skipped {skipped} {name} rows: wrong shape, unsupported role, or too long)")
    if b_kind == "examples" and len(out) < want:
        raise RuntimeError(
            f"source {name!r}: stream exhausted at {len(out)}/{want} examples "
            f"(after {skipped} skips) — the declared mixture share cannot be met")
    return out, kind


def _validate_interchange(name: str, kind: str, rows: list[dict]) -> None:
    """Enforce a source's `reasoning:` declaration on its sampled messages rows."""
    def real_traces(r):
        return sum(1 for m in r["messages"]
                   if str(m.get("reasoning_content") or "").strip())
    if kind == "native":
        traceless = sum(1 for r in rows if real_traces(r) == 0)
        assert traceless == 0, (
            f"{name}: reasoning: native, but {traceless} rows carry no real "
            "reasoning_content — the trace would silently render as an empty marker")
    else:  # none
        with_traces = sum(real_traces(r) for r in rows)
        assert with_traces == 0, (
            f"{name}: reasoning: none, but {with_traces} turns carry reasoning_content "
            "— the declaration mislabels this source")


# --------------------------------------------------------------------------------------
# Stats, cards, pushes
# --------------------------------------------------------------------------------------

def _source_stats(rows: list[dict]) -> dict[str, dict]:
    """Per-source composition of the built mixture, with BOTH share definitions.

    A mixture's split is declared in one unit but reads differently in the other
    (model-eval-model docs run ~3.4× longer than replay rows, so 20% of examples is
    ~46% of tokens): recording `share_pct_examples` and `share_pct_tokens` side by side
    keeps the design share and its token-weight consequence both explicit in the stats.
    """
    grand_tok = sum(r["n_tokens"] for r in rows)
    by_source: dict[str, dict] = {}
    for r in rows:
        b = by_source.setdefault(r["source"], {"examples": 0, "tokens": 0})
        b["examples"] += 1
        b["tokens"] += r["n_tokens"]
    for b in by_source.values():
        b["share_pct_examples"] = round(100 * b["examples"] / len(rows), 2)
        b["share_pct_tokens"] = round(100 * b["tokens"] / grand_tok, 2)
    return by_source


def _write_rows(path: Path, rows: list[dict]) -> None:
    """Write mixture rows (either form), keeping only the interchange/artifact fields."""
    payload_key = "messages" if "messages" in rows[0] else "text"
    with path.open("w") as f:
        for r in rows:
            rec = {payload_key: r[payload_key], "source": r["source"]}
            if r.get("supervise"):
                rec["supervise"] = r["supervise"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    assert sum(1 for _ in path.open()) == len(rows), f"{path} is truncated"


def _origin_url() -> str:
    """This repo's origin URL, best-effort (provenance only)."""
    try:
        return subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            stderr=subprocess.DEVNULL).decode().strip() or "this repository"
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "this repository"


def _card_fields(cfg, config_path: str, stage_desc: str, files_desc: str,
                 interchange: bool, filter_cfg, report: dict | None) -> dict:
    """Assemble the CLAUDE.md-required dataset-card fields for one push checkpoint."""
    judge = f"filter judge: {filter_cfg.model}" if filter_cfg is not None else "none"
    constitution = (f"{filter_cfg.constitution} (full text given to the filter judge)"
                    if filter_cfg is not None else str(cfg.hf.get("constitution", "none")))
    schema = (
        "jsonl rows {messages: [{role, content, reasoning_content?, tool_calls?}], "
        "source, supervise?} — model-agnostic interchange; rendered with the training "
        "family's chat template at train time (src/utils.py ModelProfile)"
        if interchange else
        "jsonl rows {text, source, supervise?} — pre-rendered with the config tokenizer's"
        " chat template (legacy mode)")
    gen = {"seed": int(cfg.seed), "max_seq_len": int(cfg.max_seq_len),
           "budget_tokenizer": str(cfg.tokenizer)}
    if filter_cfg is not None:
        gen["judge"] = {"model": str(filter_cfg.model), "temperature": 0.0,
                        "max_tokens": 900, "reasoning_effort": "low"}
    if report:
        gen["filter_result"] = {k: report[k] for k in
                                ("samples_in", "samples_kept", "reject_rate_pct")}
    return {
        "experiment": f"{cfg.hf.experiment} — {stage_desc}",
        "title": str(cfg.hf.experiment),
        "date_generated": timestamp()[:8],
        "constitution": constitution,
        "source_repo": f"{_origin_url()} @ {git_sha()}",
        "models": judge,
        "generation_config": json.dumps(gen),
        "schema": f"{files_desc}. {schema}",
        "provenance": ("uv run scripts/data/mixture/build_mixture.py "
                       f"--config {config_path}"),
    }


def _push(paths: list[Path], repo: str, fields: dict, private: bool, smoke: bool) -> None:
    """Push one checkpoint's files, or explain why not (smoke never pushes)."""
    if smoke:
        print(f">>> smoke: NOT pushing {[p.name for p in paths]} -> {repo}")
        return
    from src.hf_publish import push_files
    url = push_files(paths, repo, fields, private=private)
    print(f">>> pushed {[p.name for p in paths]} -> {url}")


# --------------------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------------------

def _load_all(tok, cfg, specs: dict, scale: int, seed: int, render_kwargs: dict,
              legacy: bool) -> tuple[list[dict], dict[str, str]]:
    """Load every source in `specs` under the config's output mode."""
    rows: list[dict] = []
    kinds: dict[str, str] = {}
    for name, spec in specs.items():
        budget = _budget(name, spec, scale)
        loader = _load_source_legacy if legacy else _take_interchange
        got, kinds[name] = loader(tok, cfg, name, spec, budget, seed, render_kwargs)
        print(f"  {name:<24} {len(got):>5} docs  {sum(r['n_tokens'] for r in got):>9,} tok "
              f"(budget {budget[1]:,} {budget[0]}, {kinds[name]})")
        rows += got
    return rows, kinds


def _validate_written(out_path: Path, rows: list[dict], kinds: dict[str, str],
                      legacy: bool) -> None:
    """Validate what actually landed on disk, not just the in-memory rows.

    Legacy mode uses the shared per-turn census (src/utils.py) — the same yardstick
    train_lora's mask gate applies; interchange mode checks reasoning_content fields
    directly. Rendered sources are exempt: validated at render time upstream.
    """
    written = [json.loads(line) for line in out_path.open()]
    assert len(written) == len(rows), "mixture file is truncated"
    for name, kind in kinds.items():
        got = [r for r in written if r["source"] == name]
        if not got:  # a filter stage may legitimately empty a small source
            print(f"{name}: no rows remain in {out_path.name}")
            continue
        if not legacy:
            _validate_interchange(name, kind, got)
            n_traces = sum(1 for r in got for m in r["messages"]
                           if str(m.get("reasoning_content") or "").strip())
            print(f"{name}: {kind} — {n_traces} reasoning turns over {len(got)} rows")
            continue
        texts = [r["text"] for r in got]
        census = think_census(texts)
        print(f"{name}: {kind} — {census['real']} real / {census['empty']} empty / "
              f"{census['absent']} absent think blocks over {census['turns']} turns")
        if kind in ("think", "native"):
            assert census["absent"] == 0, \
                f"{name}: every assistant turn must carry a think block (preserve-thinking)"
            n_traceless = sum(1 for t in texts
                              if not any(b.strip() for b in _THINK_BLOCK.findall(t)))
            assert n_traceless == 0, \
                f"{name}: {n_traceless} rows carry no real reasoning trace at all"
        elif kind == "none":
            assert census["absent"] == 0 and census["real"] == 0, \
                f"{name}: reasoning:none rows must carry exactly the empty marker per turn"
        elif kind == "strip":
            assert census["turns"] > 0 and census["real"] + census["empty"] == 0, \
                f"{name}: strip rows must contain no think block"


def main(config: str, smoke: bool = False) -> None:
    """Build and write the training mixture, with optional filter and push stages.

    Args:
        config: OmegaConf YAML. `sources` maps name -> spec:
            * `source:` a registry adapter name (src/data/mixture/sources/) — repo,
              config and normaliser come from the adapter; `split`/`config`/`repo`
              override it; or `repo:` a raw HF chat dataset; or `path:` a local jsonl
              (interchange rows or whatever the adapter's normaliser reads).
            * exactly one of `tokens:` (greedy token-share fill) or `examples:` (exact
              row count — short sources fail loudly).
            * `reasoning: native|none` — what the DATA carries, validated. (`strip`
              and `format: rendered` force the whole config into legacy rendered mode,
              reproducing pre-2026-08-06 artifacts.)
            * `synthetic: true` — the source joins AFTER the filter stage.
            * `balance_by: <field>` — local-path sources only: take the `examples:`
              budget split evenly across that field's values (top-level or under
              `metadata`), e.g. `trait_id` to trait-balance the difficult-advice share
              (absorbs the old balanced_subset.py). Quotas fail loudly when short.
        Optional `filter:` block — constitution, model, workers?, max_chars?,
            keep_examples? (stratified downsample of the kept rows).
        Optional `hf:` block — experiment, base_repo?, final_repo?, private?
            (checkpoint pushes; see the module docstring).
        smoke: Divide every budget by 20, cap judge calls, never push.
    """
    cfg = OmegaConf.load(config)
    assert "tulu3_repo" not in cfg, (
        "tulu3_repo/tulu3_tokens were folded into `sources`: add an entry like "
        "`tulu3: {repo: allenai/tulu-3-sft-mixture, tokens: N, shuffle_buffer: 10000}`")
    scale = _SMOKE_SCALE if smoke else 1
    seed = int(cfg.seed)
    sources: dict[str, dict] = OmegaConf.to_container(cfg.sources, resolve=True)
    filter_cfg = cfg.get("filter")
    hf_cfg = cfg.get("hf")

    legacy = any(s.get("reasoning") == "strip" or s.get("format") == "rendered"
                 for s in sources.values())
    if legacy:
        print(">>> LEGACY rendered mode (a source declares strip/rendered): rows are "
              "pre-rendered text, reproducing pre-2026-08-06 artifacts. New configs "
              "should use interchange sources (reasoning: native|none).")

    base_specs = {k: v for k, v in sources.items() if not v.get("synthetic")}
    synth_specs = {k: v for k, v in sources.items() if v.get("synthetic")}
    if synth_specs and filter_cfg is None:
        raise ValueError(
            "`synthetic: true` orders a source AFTER the filter stage, but this config "
            "has no `filter:` block — drop the flags for a single-pass mixture, or add "
            "the filter.")
    if hf_cfg is not None:
        assert "experiment" in hf_cfg, "hf: block needs `experiment:` for the dataset card"
        if filter_cfg is not None:
            assert "base_repo" in hf_cfg, "hf: block needs base_repo for filter checkpoints"

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
    render_kwargs = model_profile(str(cfg.tokenizer)).render_kwargs

    out_dir = Path(cfg.output_dir) / (f"smoke_{timestamp()}" if smoke else timestamp())
    out_dir.mkdir(parents=True, exist_ok=True)
    private = bool(hf_cfg.get("private", True)) if hf_cfg is not None else True

    # --- stage 1: the base mixture ----------------------------------------------------
    rows, kinds = _load_all(tok, cfg, base_specs, scale, seed, render_kwargs, legacy)
    random.Random(seed).shuffle(rows)
    report = None

    if filter_cfg is not None:
        base_path = out_dir / "mixture_unfiltered.jsonl"
        _write_rows(base_path, rows)
        _validate_written(base_path, rows, kinds, legacy)
        base_stats = {"total": {"examples": len(rows),
                                "tokens": sum(r["n_tokens"] for r in rows)},
                      "by_source": _source_stats(rows)}
        (out_dir / "mixture_stats_unfiltered.json").write_text(
            json.dumps(base_stats, indent=2))
        print(f">>> stage 1: wrote {base_path} "
              f"({base_stats['total']['examples']:,} examples)")
        if hf_cfg is not None:
            _push([base_path, out_dir / "mixture_stats_unfiltered.json"],
                  str(hf_cfg.base_repo),
                  _card_fields(cfg, config, "unfiltered initial mix",
                               "mixture_unfiltered.jsonl + stats", not legacy,
                               filter_cfg, None),
                  private, smoke)

        # --- stage 2: the spec filter -------------------------------------------------
        from src.data.mixture.spec_filter import run_filter
        from src.data.synthdoc.constitution import full_text
        from src.endpoints.openrouter import OpenRouterClient
        client = OpenRouterClient(api_key=os.environ.get("OPENROUTER_FILTER_KEY"))
        keep, report = run_filter(
            rows,
            constitution_text=full_text(str(filter_cfg.constitution)),
            model=str(filter_cfg.model), dest=out_dir,
            workers=int(filter_cfg.get("workers", 24)),
            max_chars=int(filter_cfg.get("max_chars", 12000)),
            limit=_SMOKE_JUDGE_LIMIT if smoke else None,
            client=client)
        rows = [r for r, k in zip(rows, keep) if k]
        filtered_path = out_dir / "mixture_filtered.jsonl"
        _write_rows(filtered_path, rows)
        print(f">>> stage 2: kept {report['samples_kept']:,}/{report['samples_in']:,} "
              f"({report['reject_rate_pct']}% rejected) -> {filtered_path}")
        if hf_cfg is not None:
            _push([filtered_path, out_dir / "verdicts.jsonl",
                   out_dir / "filter_report.json"],
                  str(hf_cfg.base_repo),
                  _card_fields(cfg, config, "spec-filtered, with per-sample verdicts",
                               "mixture_filtered.jsonl + verdicts.jsonl + "
                               "filter_report.json", not legacy, filter_cfg, report),
                  private, smoke)

        keep_n = filter_cfg.get("keep_examples")
        if keep_n is not None:
            keep_n = max(1, int(keep_n) // scale)
            rows, quota = stratified_subset(rows, keep_n, seed)
            print(f">>> stratified downsample to {keep_n:,} rows (quota: {quota})")

    # --- stage 3: synthetic sources join, final artifact ------------------------------
    if synth_specs:
        synth_rows, synth_kinds = _load_all(tok, cfg, synth_specs, scale, seed,
                                            render_kwargs, legacy)
        rows += synth_rows
        kinds |= synth_kinds
        random.Random(seed).shuffle(rows)

    out_path = out_dir / "mixture.jsonl"
    _write_rows(out_path, rows)
    _validate_written(out_path, rows, kinds, legacy)
    stats = {"total": {"examples": len(rows), "tokens": sum(r["n_tokens"] for r in rows)},
             "by_source": _source_stats(rows), "mixture_path": str(out_path),
             "filter": report}
    (out_dir / "mixture_stats.json").write_text(json.dumps(stats, indent=2))
    write_run_meta(out_dir, OmegaConf.to_container(cfg, resolve=True),
                   extra={"command": " ".join(sys.argv), "smoke": smoke, "stats": stats})

    # Loud sanity output: the actual rows the model will train on.
    payload_key = "text" if legacy else "messages"
    for wanted, header in (("think", "reasoning preserved on every turn"),
                           ("native", "real reasoning_content on assistant turns"),
                           ("none", "no reasoning carried"),
                           ("strip", "NO <think> at all (legacy pre-policy render)")):
        name = next((n for n, k in kinds.items() if k == wanted), None)
        row = next((r for r in rows if r["source"] == name), None) if name else None
        if row:
            print("\n" + "=" * 72)
            print(f"FIRST {name} EXAMPLE ({header}):")
            print("=" * 72)
            print(json.dumps(row[payload_key], ensure_ascii=False, indent=2)[:1200]
                  if payload_key == "messages" else row[payload_key][:1200])

    if hf_cfg is not None and hf_cfg.get("final_repo"):
        _push([out_path, out_dir / "mixture_stats.json"], str(hf_cfg.final_repo),
              _card_fields(cfg, config, "final training mixture"
                           + (" (synthetic sources mixed in)" if synth_specs else ""),
                           "mixture.jsonl + mixture_stats.json", not legacy,
                           filter_cfg, report),
              private, smoke)

    print("\n" + json.dumps(stats["total"], indent=2))
    print(f">>> wrote {out_path}")

    # Breaking out of a streaming dataset mid-shard leaves HF's parquet reader threads to
    # crash during interpreter finalization. The artifact above is written and verified, so
    # exit before finalization rather than surfacing a spurious fatal error.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def cli() -> None:
    """Console entry (`uv run build_mixture --config ...`, [project.scripts])."""
    import fire

    fire.Fire(main)
