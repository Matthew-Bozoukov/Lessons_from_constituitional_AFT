# ABOUTME: Builds an N-source SFT mixture at per-source budgets, with an optional
# ABOUTME: constitution filter stage and HF push checkpoints after each pipeline stage.

"""Build a training mixture: sample sources, optionally filter, optionally add synthetic.

Rows are model-agnostic chat transcripts — `{"messages": [...], "source": ...}` with
optional per-turn `reasoning_content`/`tool_calls` (see src/data/mixture/sources/). No
chat template is applied at build time; train_lora renders with the training family's
`ModelProfile`, and the tokenizer here only *counts* tokens for budgets and length caps.
(The legacy pre-rendered `{"text"}` mode — `reasoning: strip` / `format: rendered` — was
removed 2026-08-07: its published artifacts live on HF, and regenerating one byte-for-byte
means checking out a pre-removal commit. Git history is the archive.)

The optional stages, both config-driven (a config with neither behaves as a single pass):

* `filter:` — the spec-alignment judge (src/data/mixture/spec_filter.py) screens every
  NON-synthetic row against a constitution; sources marked `synthetic: true` join only
  after the filter (they are constitution-generated — judging them against it is
  circular). `filter.keep_examples` then downsamples the kept rows stratified by source.
* `hf:` — push checkpoints as they are produced (a dead run loses nothing):
  `mixture_unfiltered.jsonl` after the initial mix and `mixture_filtered.jsonl` +
  `verdicts.jsonl` + `filter_report.json` after filtering, both to `hf.base_repo`;
  the final `mixture.jsonl` (synthetic mixed in) to the same repo. Both stages share
  ONE repo — `<date>-<styles>-<pct>[-<variant>]-mix`, built from this config's stem and
  its declared `variant:` (src/naming.py) — the way a synth run's stages share one repo.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import OmegaConf
from transformers import AutoTokenizer

# HERE, not only in the filter branch. The credentials this module needs are the HF
# token's, and every `hf:` push needs it — but until 2026-08-19 .env was loaded only as a
# side effect of importing OpenRouterClient, which happens inside the `filter:` stage. A
# filter-less config therefore built its mixture, ran to the final push and died on a bare
# 401 from create_repo with the artifact already on disk.
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from src.data.mixture.sources import SOURCES, clean_messages  # noqa: E402
from src.model_profile import model_profile  # noqa: E402
from src.naming import check_style, mix_name  # noqa: E402
from src.utils import git_sha, origin_url, timestamp, write_run_meta  # noqa: E402

# Each source declares what its DATA carries via `reasoning:` — part of the scientific
# record, validated on the sampled rows, never guessed:
#   native — rows carry reasoning_content (validated: every row keeps a real trace)
#   none   — rows have no reasoning at all (validated)
# Think-tag syntax (markers, prefills) is a TRAIN-time concern: the profile's
# render_kwargs put a think block on every assistant turn and the generation-boundary
# mask (src/train/masking.py) keeps empty markers out of the loss.

# Every budget (tokens or examples) is divided by this under --smoke, so a smoke run
# exercises the full wiring (rendering, streaming, validation, stats) in seconds.
_SMOKE_SCALE = 20

# Judge calls a --smoke run is allowed to spend (a fraction of a cent): enough to prove
# the checkpoint/parse/report wiring, never enough to look like a completed filter.
_SMOKE_JUDGE_LIMIT = 3


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
# Interchange loaders — model-agnostic messages rows; the tokenizer only counts.
# --------------------------------------------------------------------------------------

def _take_interchange(tok, cfg, name: str, spec: dict, budget: tuple[str, int],
                      seed: int, render_kwargs: dict) -> tuple[list[dict], str]:
    """Load one source as interchange messages rows and validate its reasoning kind.

    The spec names rows via a registry adapter (`source:`), a synth-contract HF repo
    (`dataset:` [+ `revision:`] — load_dataset of the repo's default config, the
    canonical synthetic intake), a raw HF chat repo (`repo:`, streamed), one exact
    file of a pre-contract repo (`repo:` + `file:`, legacy), or a local jsonl
    (`path:`); adapters supply repo/config/normaliser defaults the spec can override. Budgets and the `max_seq_len` cap are counted on
    the PRESERVED render of each row (what training will render), with the config's
    tokenizer — the counts are model-relative, the stored data is not.

    Returns:
        (rows, kind): rows carry `messages`, `source`, `n_tokens` (+`supervise` when a
        local row declares it); kind is the validated `reasoning:` declaration.
    """
    # A spec names its adapter explicitly (`source:`), or implicitly by its own key when
    # it declares no other origin — so `no_robots: {examples: N, reasoning: none}` just
    # works. Raw `dataset:`/`repo:`/`path:` specs need no adapter at all.
    adapter_name = spec.get("source") or (
        name if not ("repo" in spec or "path" in spec or "dataset" in spec) else None)
    adapter = None
    if adapter_name is not None:
        if adapter_name not in SOURCES:
            raise ValueError(f"source {name!r}: unknown adapter {adapter_name!r} "
                             f"(known: {', '.join(sorted(SOURCES))})")
        adapter = SOURCES[adapter_name]
    if "format" in spec:
        raise ValueError(
            f"source {name!r}: `format: messages|rendered` was the legacy pre-rendered "
            "mode, removed 2026-08-07 — use `path:` (+ `source:` adapter) with "
            "`reasoning: native|none`; pre-removal artifacts live on HF and regenerate "
            "from a pre-removal checkout.")
    kind = spec.get("reasoning")
    if kind not in ("native", "none"):
        raise ValueError(
            f"source {name!r}: sources must declare `reasoning: native|none` — what the "
            "DATA carries is part of the scientific record. (`strip` was the legacy "
            "no-think build-time rendering, removed 2026-08-07; nothink arms choose "
            "their render at train time.)")
    to_messages = adapter.to_messages if adapter else \
        (lambda row: clean_messages(row.get("messages")))
    pool = None
    if "dataset" in spec:
        # THE canonical synth intake: `dataset: org/repo` loads the repo's DEFAULT
        # config — dataset.jsonl under the synth->mixture contract — via
        # load_dataset, pinned to the exact sha and fully materialised (balancing
        # needs the whole pool). load_dataset's schema inference None-fills optional
        # fields (e.g. reasoning_content on turns without a trace); clean_messages
        # drops falsy fields, so rows come out identical to reading the jsonl.
        from src.huggingface import hf_api, hf_token

        info = hf_api().repo_info(spec["dataset"], repo_type="dataset",
                                  revision=spec.get("revision"))
        print(f"{name}: {spec['dataset']}@{info.sha[:12]} (default config)")
        pool = load_dataset(spec["dataset"], revision=info.sha, split="train",
                            token=hf_token())
    elif "file" in spec:
        # LEGACY intake for pre-contract repos with no dataset.jsonl/default config
        # (e.g. stage_7_sft.jsonl mirrors): one exact file, sha-pinned via the shared
        # resolver, then read as a local path. New synth repos use `dataset:`.
        assert "repo" in spec, f"source {name!r}: `file:` needs `repo:`"
        from src.huggingface import resolve_dataset

        local, ref = resolve_dataset(spec["repo"], spec["file"], spec.get("revision"))
        print(f"{name}: {ref['repo']}@{ref['revision'][:12]} ({ref['file']} — legacy "
              "file intake; new-layout repos use `dataset:`)")
        spec = {**spec, "path": local}
    if spec.get("balance_by") and pool is None and "path" not in spec:
        raise ValueError(
            f"source {name!r}: balance_by needs the whole pool in hand (`dataset:` or "
            "`path:`) — a streamed `repo:` cannot be grouped")

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
        # supervise rides top-level or under metadata (synth stage-5 exports put it
        # there); losing it would silently train non-target turns (supervise: final).
        supervise = raw.get("supervise") or (raw.get("metadata") or {}).get("supervise")
        if supervise:
            out["supervise"] = supervise
        return out

    if pool is not None or "path" in spec:
        if pool is None:
            pool = (json.loads(line) for line in
                    Path(spec["path"]).open(encoding="utf-8"))
        bkey = spec.get("balance_by")
        rows, groups = [], {}
        for raw in pool:
            p = payload(raw)
            if p is None:
                continue
            rows.append(p)
            if bkey:
                g = raw.get(bkey) or (raw.get("metadata") or {}).get(bkey)
                assert g is not None, \
                    f"source {name!r}: row missing balance_by field {bkey!r}"
                groups.setdefault(str(g), []).append(p)
        assert rows, f"no usable rows in {spec.get('dataset') or spec['path']}"
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

def synthetic_pct(rows: list[dict], synthetic_sources: set[str]) -> int:
    """Percentage of a built mixture's rows that came from a synthetic source.

    Rounded to a whole number because that is what a name can carry and what anyone says
    out loud ("the 20% arm"); the exact counts stay in `mixture_stats.json`. Counted on
    EXAMPLES, not tokens — the same unit the mixture's split is declared in.
    """
    if not rows:
        return 0
    return round(100 * sum(r["source"] in synthetic_sources for r in rows) / len(rows))


def _base_sources(base_config: str) -> dict[str, dict]:
    """The non-synthetic blend a mixture inherits, from the base config it names."""
    base = OmegaConf.to_container(OmegaConf.load(base_config), resolve=True)
    assert not any(s.get("synthetic") for s in base["sources"].values()), (
        f"{base_config} is used as a BASE blend but declares synthetic sources. The base "
        "is the non-synthetic composition every arm shares; a mixture with a synthetic "
        "share cannot define it.")
    return base["sources"]


def blend(base: dict[str, dict], synthetic: dict[str, dict], synthetic_pct: int,
          total_examples: int) -> dict[str, dict]:
    """Scale a fixed non-synthetic blend around a synthetic share (pure; unit-tested).

    THE mechanism that makes an arm ladder a dose-response curve. Earlier arms replaced
    the replay portion with a single source, so `da-10` and `da-40` differed in their
    replay composition as well as their synthetic share and no arm was a clean control for
    the next. Here the base blend's PROPORTIONS are fixed and only its total shrinks: a
    source that is 27.79% of the base is 27.79% x (100 - pct)% of every mixture built from
    it.

    Args:
        base: The base blend's per-source specs, budgeted in `examples`.
        synthetic: The synthetic sources, which share the synthetic budget between them.
            Their declared budgets set the RATIO between them, not the totals.
        synthetic_pct: Percentage of rows that must be synthetic — the number in the name.
        total_examples: Rows in the finished mixture.

    Returns:
        The same specs with `examples` rewritten to the scaled counts, synthetic sources
        marked `synthetic: true` so they join after the filter stage.
    """
    assert 0 <= synthetic_pct <= 100, f"synthetic_pct out of range: {synthetic_pct}"
    assert bool(synthetic) == (synthetic_pct > 0), (
        f"a {synthetic_pct}% synthetic share and {len(synthetic)} synthetic source(s) do "
        "not agree — 0% means no synthetic sources, and any share needs at least one.")

    def share(specs: dict[str, dict], budget: int) -> dict[str, dict]:
        weights = {n: float(s.get("examples") or s.get("tokens") or 0) for n, s in specs.items()}
        total_w = sum(weights.values())
        assert total_w > 0 or not specs, "every source needs a budget to weight it by"
        return {n: {**s, "examples": round(budget * weights[n] / total_w)}
                for n, s in specs.items()}

    synth_budget = round(total_examples * synthetic_pct / 100)
    out = share(base, total_examples - synth_budget)
    out.update({n: {**s, "synthetic": True}
                for n, s in share(synthetic, synth_budget).items()})
    return out


def declared_synthetic_pct(sources: dict) -> int:
    """The synthetic share the config DESIGNS, from its per-source budgets.

    A mixture has to be named before its rows exist — the checkpoint pushes need a repo to
    land in — so the name comes from the design, and `synthetic_pct` over the built rows
    checks it. Budgets are declared in tokens or examples depending on the source, and
    both are used here for the same reason the name is approximate: this is the intent,
    and the built stats are the record.
    """
    def budget(spec: dict) -> float:
        return float(spec.get("tokens") or spec.get("examples") or 0)

    total = sum(budget(s) for s in sources.values())
    synth = sum(budget(s) for s in sources.values() if s.get("synthetic"))
    return round(100 * synth / total) if total else 0


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
    """Write mixture rows, keeping only the interchange/artifact fields."""
    # Explicit utf-8 on every jsonl hop: rows are written with ensure_ascii=False, and
    # a Windows-driven build otherwise reads and writes them as cp1252 (a local `path:`
    # source dies on the first non-latin-1 byte).
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            rec = {"messages": r["messages"], "source": r["source"]}
            if r.get("supervise"):
                rec["supervise"] = r["supervise"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    assert sum(1 for _ in path.open(encoding="utf-8")) == len(rows), f"{path} truncated"


def _card_fields(cfg, config_path: str, stage_desc: str, files_desc: str,
                 filter_cfg, report: dict | None) -> dict:
    """Assemble the CLAUDE.md-required dataset-card fields for one push checkpoint."""
    judge = f"filter judge: {filter_cfg.model}" if filter_cfg is not None else "none"
    constitution = (f"{filter_cfg.constitution} (full text given to the filter judge)"
                    if filter_cfg is not None else str(cfg.hf.get("constitution", "none")))
    schema = (
        "jsonl rows {messages: [{role, content, reasoning_content?, tool_calls?}], "
        "source, supervise?} — model-agnostic interchange; rendered with the training "
        "family's chat template at train time (src/model_profile.py ModelProfile)")
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
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": judge,
        "generation_config": json.dumps(gen),
        "schema": f"{files_desc}. {schema}",
        "provenance": ("uv run scripts/data/mixture/build_mixture.py "
                       f"--config {config_path}"),
    }


def _front_matter(cfg, config_path: str, filter_cfg, stage: str, data_file: str) -> dict:
    """Card front-matter for one push checkpoint: default config + discovery tags.

    The default `configs:` entry names the rows file, so `load_dataset(repo)` and the
    dashboard's byte-range reader both find it without guessing among the sidecars
    (stats, verdicts, filter report) pushed beside it. The tags are the Hub-indexed
    `training_data_tags`; `stage:` separates the base repo's unfiltered/filtered
    checkpoints from the final mixture.
    """
    from src.huggingface import training_data_tags

    constitution = (filter_cfg.constitution if filter_cfg is not None
                    else cfg.hf.get("constitution", "none"))
    return {
        "configs": [{"config_name": "default", "data_files": data_file, "default": True}],
        "tags": training_data_tags("mixture", Path(config_path).stem, str(constitution),
                                   extra=[f"stage:{stage}"]),
    }


def _push(paths: list[Path], repo: str, fields: dict, private: bool, smoke: bool,
          front_matter: dict) -> None:
    """Push one checkpoint's files, or explain why not (smoke never pushes)."""
    from src.huggingface import hf_repo_id

    repo = hf_repo_id(repo)  # the config names the repo, .env's HF_ORG the org
    if smoke:
        print(f">>> smoke: NOT pushing {[p.name for p in paths]} -> {repo}")
        return
    from src.huggingface import push_files
    url = push_files(paths, repo, fields, private=private, front_matter=front_matter)
    print(f">>> pushed {[p.name for p in paths]} -> {url}")


# --------------------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------------------

def _load_all(tok, cfg, specs: dict, scale: int, seed: int,
              render_kwargs: dict) -> tuple[list[dict], dict[str, str]]:
    """Load every source in `specs` as interchange rows."""
    rows: list[dict] = []
    kinds: dict[str, str] = {}
    for name, spec in specs.items():
        budget = _budget(name, spec, scale)
        got, kinds[name] = _take_interchange(tok, cfg, name, spec, budget, seed,
                                             render_kwargs)
        print(f"  {name:<24} {len(got):>5} docs  {sum(r['n_tokens'] for r in got):>9,} tok "
              f"(budget {budget[1]:,} {budget[0]}, {kinds[name]})")
        rows += got
    return rows, kinds


def _validate_written(out_path: Path, rows: list[dict], kinds: dict[str, str]) -> None:
    """Validate what actually landed on disk, not just the in-memory rows."""
    written = [json.loads(line) for line in out_path.open(encoding="utf-8")]
    assert len(written) == len(rows), "mixture file is truncated"
    for name, kind in kinds.items():
        got = [r for r in written if r["source"] == name]
        if not got:  # a filter stage may legitimately empty a small source
            print(f"{name}: no rows remain in {out_path.name}")
            continue
        _validate_interchange(name, kind, got)
        n_traces = sum(1 for r in got for m in r["messages"]
                       if str(m.get("reasoning_content") or "").strip())
        print(f"{name}: {kind} — {n_traces} reasoning turns over {len(got)} rows")


def main(config: str, smoke: bool = False) -> None:
    """Build and write the training mixture, with optional filter and push stages.

    Args:
        config: OmegaConf YAML. `sources` maps name -> spec:
            * `source:` a registry adapter name (src/data/mixture/sources/) — repo,
              config and normaliser come from the adapter; `split`/`config`/`repo`
              override it; or `dataset:` an HF synth-contract repo [+ `revision:`]
              (THE canonical synthetic intake — load_dataset of its default config);
              or `repo:` a raw HF chat dataset (streamed); or `repo:` + `file:` one
              pinned file of a pre-contract repo (legacy); or `path:` a local jsonl.
            * exactly one of `tokens:` (greedy token-share fill) or `examples:` (exact
              row count — short sources fail loudly).
            * `reasoning: native|none` — what the DATA carries, validated. (`strip` /
              `format: rendered` were the legacy pre-rendered mode, removed 2026-08-07;
              those artifacts live on HF and regenerate from a pre-removal checkout.)
            * `synthetic: true` — the source joins AFTER the filter stage.
            * `balance_by: <field>` — local-path sources only: take the `examples:`
              budget split evenly across that field's values (top-level or under
              `metadata`), e.g. `trait_id` to trait-balance the difficult-advice share
              (absorbs the old balanced_subset.py). Quotas fail loudly when short.
        Optional `filter:` block — constitution, model, workers?, max_chars?,
            keep_examples? (stratified downsample of the kept rows).
        Optional `hf:` block — experiment, constitution?, private? (the REPO is
            built from the config stem, never declared)
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
    if cfg.get("base"):
        sources = blend(_base_sources(str(cfg.base)), sources,
                        int(cfg.synthetic_pct), int(cfg.total_examples))
    filter_cfg = cfg.get("filter")
    hf_cfg = cfg.get("hf")

    base_specs = {k: v for k, v in sources.items() if not v.get("synthetic")}
    synth_specs = {k: v for k, v in sources.items() if v.get("synthetic")}
    if synth_specs and filter_cfg is None:
        raise ValueError(
            "`synthetic: true` orders a source AFTER the filter stage, but this config "
            "has no `filter:` block — drop the flags for a single-pass mixture, or add "
            "the filter.")
    if hf_cfg is not None:
        assert "experiment" in hf_cfg, "hf: block needs `experiment:` for the dataset card"
    # THE mixture's name (src/naming.py): this config's stem — its styles and any variant,
    # the parts a human chose — with the synthetic share spliced BETWEEN them, and today's
    # date in front. `da` + `reason-only` at 7% is `<date>-da-7-reason-only-mix`. The
    # variant is declared rather than inferred precisely because the share lands in the
    # middle, so the stem alone cannot say where the styles end.
    variant = str(cfg.get("variant") or "")
    stem = Path(config).stem
    if variant:
        assert stem.endswith(f"-{variant}"), (
            f"{stem}.yaml declares `variant: {variant}` but its stem does not end in it; "
            f"the stem is `<styles>-{variant}`.")
        stem = stem[: -len(variant) - 1]
    style = stem if stem == "0" else check_style(
        stem, what="styles (mixture config stem)")

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
    render_kwargs = model_profile(str(cfg.tokenizer)).render_kwargs

    out_dir = Path(cfg.output_dir) / (f"smoke_{timestamp()}" if smoke else timestamp())
    out_dir.mkdir(parents=True, exist_ok=True)
    private = bool(hf_cfg.get("private", True)) if hf_cfg is not None else True

    # The synthetic share is part of the mixture's NAME, and the name has to exist before
    # the first checkpoint push — so it comes from the share the config designs, and the
    # share the built rows actually carry is asserted against it at stage 3. A mixture
    # cannot be published under a percentage its own rows disagree with.
    declared_pct = declared_synthetic_pct(sources)
    repo = mix_name(style if style != "0" else "", declared_pct, variant)

    # --- stage 1: the base mixture ----------------------------------------------------
    rows, kinds = _load_all(tok, cfg, base_specs, scale, seed, render_kwargs)
    random.Random(seed).shuffle(rows)
    report = None

    if filter_cfg is not None:
        base_path = out_dir / "mixture_unfiltered.jsonl"
        _write_rows(base_path, rows)
        _validate_written(base_path, rows, kinds)
        base_stats = {"total": {"examples": len(rows),
                                "tokens": sum(r["n_tokens"] for r in rows)},
                      "by_source": _source_stats(rows)}
        (out_dir / "mixture_stats_unfiltered.json").write_text(
            json.dumps(base_stats, indent=2))
        print(f">>> stage 1: wrote {base_path} "
              f"({base_stats['total']['examples']:,} examples)")
        if hf_cfg is not None:
            _push([base_path, out_dir / "mixture_stats_unfiltered.json"], repo,
                  _card_fields(cfg, config, "unfiltered initial mix",
                               "mixture_unfiltered.jsonl + stats", filter_cfg, None),
                  private, smoke,
                  _front_matter(cfg, config, filter_cfg, "unfiltered", base_path.name))

        # --- stage 2: the spec filter -------------------------------------------------
        from src.data.mixture.spec_filter import run_filter
        from src.data.synth.constitution import full_text
        from src.infra.endpoints.openrouter import OpenRouterClient
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
                   out_dir / "filter_report.json"], repo,
                  _card_fields(cfg, config, "spec-filtered, with per-sample verdicts",
                               "mixture_filtered.jsonl + verdicts.jsonl + "
                               "filter_report.json", filter_cfg, report),
                  private, smoke,
                  _front_matter(cfg, config, filter_cfg, "filtered", filtered_path.name))

        keep_n = filter_cfg.get("keep_examples")
        if keep_n is not None:
            keep_n = max(1, int(keep_n) // scale)
            rows, quota = stratified_subset(rows, keep_n, seed)
            print(f">>> stratified downsample to {keep_n:,} rows (quota: {quota})")

    # --- stage 3: synthetic sources join, final artifact ------------------------------
    if synth_specs:
        synth_rows, synth_kinds = _load_all(tok, cfg, synth_specs, scale, seed,
                                            render_kwargs)
        rows += synth_rows
        kinds |= synth_kinds
        random.Random(seed).shuffle(rows)

    # The name says 20%; the rows had better be 20%. Rounding is the only slack allowed,
    # because everything trained on this mixture inherits the number from its name.
    built_pct = synthetic_pct(rows, {n for n in synth_specs})
    assert abs(built_pct - declared_pct) <= 1, (
        f"this mixture is named for a {declared_pct}% synthetic share but its rows are "
        f"{built_pct}% ({sum(r['source'] in synth_specs for r in rows):,} of {len(rows):,}). "
        "The name would be wrong, and every arm trained on it would inherit the wrong "
        "number. Fix the source budgets, or the config stem.")

    out_path = out_dir / "mixture.jsonl"
    _write_rows(out_path, rows)
    _validate_written(out_path, rows, kinds)
    stats = {"total": {"examples": len(rows), "tokens": sum(r["n_tokens"] for r in rows)},
             "synthetic_pct": built_pct, "by_source": _source_stats(rows),
             "mixture_path": str(out_path), "filter": report}
    (out_dir / "mixture_stats.json").write_text(json.dumps(stats, indent=2))
    write_run_meta(out_dir, OmegaConf.to_container(cfg, resolve=True),
                   extra={"command": " ".join(sys.argv), "smoke": smoke, "stats": stats})

    # Loud sanity output: the actual rows the model will train on.
    for wanted, header in (("native", "real reasoning_content on assistant turns"),
                           ("none", "no reasoning carried")):
        name = next((n for n, k in kinds.items() if k == wanted), None)
        row = next((r for r in rows if r["source"] == name), None) if name else None
        if row:
            print("\n" + "=" * 72)
            print(f"FIRST {name} EXAMPLE ({header}):")
            print("=" * 72)
            print(json.dumps(row["messages"], ensure_ascii=False, indent=2)[:1200])

    if hf_cfg is not None:
        _push([out_path, out_dir / "mixture_stats.json"], repo,
              _card_fields(cfg, config, "final training mixture"
                           + (" (synthetic sources mixed in)" if synth_specs else ""),
                           "mixture.jsonl + mixture_stats.json", filter_cfg, report),
              private, smoke, _front_matter(cfg, config, filter_cfg, "final", out_path.name))

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
