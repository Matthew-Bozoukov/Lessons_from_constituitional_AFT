# ABOUTME: The specgen stages: pin source -> extract claim inventory (once, shared) ->
# ABOUTME: per arm/seed cluster to N principles -> write units -> assemble the spec.

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import threading
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import prompts  # noqa: E402
from openrouter import map_threaded  # noqa: E402
from utils import extract_json, git_sha, read_jsonl, timestamp  # noqa: E402

PKG = Path(__file__).parent
PRIORITY_ORDER = ["safe", "ethical", "compliant", "helpful"]
STOPWORDS = {"being", "broadly", "the", "a", "an", "of", "and", "to", "on", "for",
             "claude", "claude's", "why", "what", "how"}

# All generation runs through headless Claude Code subagents (`claude -p`, subscription
# auth) rather than any API provider. A minimal replacement system prompt keeps the
# per-call overhead small and the output plain text.
SYSTEM = ("You are a careful technical writer. Follow the format instructions exactly "
          "and reply with the requested output only.")


class Spend:
    """Thread-safe tally of the nominal cost the claude CLI reports per call."""

    def __init__(self) -> None:
        self.usd, self.calls = 0.0, 0
        self._lock = threading.Lock()

    def add(self, usd: float) -> None:
        with self._lock:
            self.usd += usd
            self.calls += 1


def _ask(prompt: str, model: str, spend: Spend, timeout: int = 900) -> str:
    """One headless Claude Code subagent call with tools disabled.

    Args:
        prompt: The full task prompt (conversations are flattened into it).
        model: A claude CLI model alias, e.g. "fable" or "opus".
        spend: Tally for the CLI-reported nominal cost.
        timeout: Seconds before the subprocess is killed.

    Returns:
        The subagent's text reply.
    """
    for attempt in (1, 2):  # one bounded retry for transient CLI/API failures
        res = subprocess.run(
            ["claude", "-p", "--model", model, "--system-prompt", SYSTEM,
             "--disallowedTools", "*", "--output-format", "json"],
            input=prompt, capture_output=True, text=True, timeout=timeout)
        if res.returncode == 0:
            data = json.loads(res.stdout)
            if not data.get("is_error"):
                spend.add(float(data.get("total_cost_usd") or 0))
                return data["result"].strip()
        err = (f"claude -p ({model}) failed (rc={res.returncode}): "
               f"stdout={res.stdout[:300]!r} stderr={res.stderr[:300]!r}")
        print(f"    attempt {attempt}: {err}")
    raise RuntimeError(err)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@lru_cache(maxsize=2)
def _counter(tokenizer_name: str):
    """Return a text -> token-count callable for a HF tokenizer."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    return lambda text: len(tok.encode(text))


def out_dir(cfg: dict, smoke: bool = False) -> Path:
    """The run's output root (smoke runs never pollute the real one)."""
    root = Path(cfg["output_dir"])
    return root / "smoke" if smoke else root


def _push(cfg: dict, files: list[Path], prefix: str) -> None:
    """Mirror files into the HF dataset repo that records spec evolution.

    Every generate run pushes under a fresh timestamped prefix, so successive
    refinement runs accumulate as an inspectable history rather than overwriting.
    No-op when hf_repo is unset.
    """
    repo = cfg.get("hf_repo")
    if not repo:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", exist_ok=True,
                    private=bool(cfg.get("hf_private", False)))
    for f in files:
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f"{prefix}/{f.name}",
                        repo_id=repo, repo_type="dataset",
                        commit_message=f"{prefix}/{f.name}")
    print(f"    -> hf:{repo}/{prefix}")


def _card(cfg: dict) -> str:
    """Dataset card for the HF evolution repo (repo policy: required fields)."""
    return f"""# specgen — constitution-granularity distills (evolution record)

| field | value |
|---|---|
| experiment | Distill the published Claude constitution into specs at 3 granularities (4/12/24 principles); every draft and revision is recorded here |
| date_generated | see per-file commit dates; run prefixes are UTC timestamps |
| constitution | Anthropic, *Claude's Constitution* ({cfg["source_url"]}), CC0 1.0 |
| source_repo | teaching_claude_why_replication @ {git_sha()} |
| models | {json.dumps(cfg["models"])} |
| generation_config | headless Claude Code CLI defaults; see meta.json beside each spec (models, prompt hashes)  |
| schema | `claims/<ts>/inventory.jsonl` extracted claims; `<arm>/seed<k>/<ts>/constitution.draft.md` pre-revision assembly, `constitution.md` final, + clusters.json/meta.json |
| provenance | `uv run scratch/specgen/cli.py generate --config scratch/specgen/specgen.yaml` |
"""


def _unit_ratio(unit: str, count) -> float:
    """Explanation share of one unit: (Why + not-apply tokens) / unit tokens."""
    part = unit.split("*Why:*", 1)
    rest = part[1] if len(part) == 2 else ""
    na = rest.split("*When this does NOT apply:*", 1)
    why = re.split(r"^- ", na[0], flags=re.M)[0]
    expl = count(why) + count(na[1] if len(na) == 2 else "")
    return expl / max(count(unit), 1)


def _assemble(units: list[str], preamble: str, closing: str) -> str:
    """Number the units and join preamble -> units -> closing."""
    numbered = [u.replace("## ", f"## {i + 1}. ", 1) for i, u in enumerate(units)]
    return preamble + "\n---\n\n" + "\n\n".join(numbered) + "\n\n---\n\n" + closing


# --- source ------------------------------------------------------------------------


def pin(cfg: dict, file: str) -> Path:
    """Pin the source constitution: copy it verbatim and write a hash lock.

    The source is fetched manually (it is a JS-rendered page; save it as markdown) and
    pinned from a local file so the lock records exactly what every stage reads.

    Args:
        cfg: Run config.
        file: Path to the manually saved constitution markdown.

    Returns:
        Path to the pinned copy.
    """
    text = Path(file).read_text()
    dst = out_dir(cfg) / "source"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "constitution.md").write_text(text)
    lock = {"sha256": _sha256(text), "source_url": cfg["source_url"],
            "pinned_from": str(file), "pinned_at": timestamp(),
            "note": "Anthropic's published Claude constitution, CC0 1.0."}
    (dst / "source.lock.json").write_text(json.dumps(lock, indent=2))
    n = len(sections(text))
    print(f">>> pinned {len(text)} chars, {n} sections, sha {lock['sha256'][:12]}")
    return dst / "constitution.md"


def sections(text: str) -> list[dict]:
    """Split the source on its own H2/H3 boundaries into [{section_id, title, text}].

    H3 subsections become their own chunks (titled "H2 — H3") so extraction calls see
    evenly sized text; the source's H2 sections vary by an order of magnitude.
    """
    parts = re.split(r"^(#{2,3}) +(.+)$", text, flags=re.M)
    # parts = [pre, level1, title1, body1, level2, title2, body2, ...]
    out, h2 = [], ""
    for level, title, body in zip(parts[1::3], parts[2::3], parts[3::3]):
        title, body = title.strip(), body.strip()
        if level == "##":
            h2 = title
        else:
            title = f"{h2} — {title}"
        if not body:
            continue
        words = [w for w in re.findall(r"[a-z']+", title.lower()) if w not in STOPWORDS]
        slug = next((w[:3] for w in reversed(words)), f"s{len(out):02d}")
        while any(s["section_id"] == slug for s in out):
            slug += "x"
        out.append({"section_id": slug, "title": title, "text": body})
    return out


# --- stage 1: extraction (run once, shared by all arms) ----------------------------


def extract(cfg: dict, smoke: bool = False) -> Path:
    """Extract atomic normative claims per source section into claims/inventory.jsonl.

    Per-section calls, never one pass over the whole document: at ~10x compression a
    single pass attends unevenly and over-weights early sections, which would break
    the coverage invariant for every arm.
    """
    root = out_dir(cfg, smoke)
    source = (out_dir(cfg) / "source" / "constitution.md").read_text()
    secs = sections(source)
    if smoke:
        secs = secs[:2]
    spend = Spend()
    model = cfg["models"]["extract"]

    def one(i: int) -> list[dict]:
        s = secs[i]
        reply = _ask(prompts.EXTRACT.format(title=s["title"], text=s["text"]),
                     model, spend)
        claims = extract_json(reply)
        for j, c in enumerate(claims):
            assert c["modality"] in ("never", "always", "prefer", "weigh"), c
            c.update(claim_id=f"{s['section_id']}-{j:03d}",
                     section_id=s["section_id"], section_title=s["title"])
        return claims

    inventory = [c for chunk in map_threaded(one, len(secs), int(cfg["workers"]),
                                             "extract") for c in chunk]
    dst = root / "claims"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "inventory.jsonl").write_text(
        "\n".join(json.dumps(c) for c in inventory) + "\n")
    lock = {"sha256": _sha256((dst / "inventory.jsonl").read_text()),
            "source_sha256": _sha256(source), "model": model,
            "n_claims": len(inventory), "git_sha": git_sha(), "at": timestamp()}
    (dst / "inventory.lock.json").write_text(json.dumps(lock, indent=2))
    print(f">>> {len(inventory)} claims from {len(secs)} sections | "
          f"{spend.calls} calls (~${spend.usd:.2f} nominal)")
    if not smoke and not 150 <= len(inventory) <= 400:
        print(f"!!! claim count outside the expected 150-400 band — inspect before generating")
    if not smoke:
        _push(cfg, [dst / "inventory.jsonl", dst / "inventory.lock.json"],
              f"claims/{lock['at']}")
    return dst / "inventory.jsonl"


# --- stage 2: cluster to resolution N ----------------------------------------------


def cluster(inventory: list[dict], n: int, cfg: dict, spend: Spend, seed: int) -> list[dict]:
    """Partition the inventory into exactly n clusters; exact ID accounting enforced.

    Retries with error feedback; raises after cfg[cluster_retries] failures rather
    than silently accepting an unbalanced partition.
    """
    model = cfg["models"]["cluster"]
    lines = "\n".join(f"{c['claim_id']} | {c['modality']} | {c['claim']}"
                      for c in inventory)
    prompt = prompts.CLUSTER.format(n_principles=n, claims=lines)
    want = {c["claim_id"] for c in inventory}
    for attempt in range(int(cfg["cluster_retries"]) + 1):
        reply = _ask(prompt, model, spend)
        clusters = extract_json(reply)["clusters"]
        got = [i for c in clusters for i in c["claim_ids"]]
        orphans, dupes = want - set(got), len(got) - len(set(got))
        bad_prio = [c["parent_priority"] for c in clusters
                    if c["parent_priority"] not in PRIORITY_ORDER]
        if len(clusters) == n and not orphans and not dupes and not bad_prio:
            return clusters
        problem = (f"{len(clusters)} clusters (need {n}), {len(orphans)} orphaned, "
                   f"{dupes} duplicated, bad priorities {bad_prio}")
        print(f"    cluster attempt {attempt + 1}: {problem}")
        prompt += (f"\n\nYour previous attempt:\n\n{reply}\n\nInvalid partition: "
                   f"{problem}. Missing ids: {sorted(orphans)[:20]}. "
                   f"Return the corrected full JSON.")
    raise RuntimeError(f"clustering failed after retries: {problem}")


# --- stage 3+4: write units and assemble -------------------------------------------


def write_spec(inventory: list[dict], clusters: list[dict], arm: str, cfg: dict,
               spend: Spend, seed: int, smoke: bool = False) -> tuple[str, str]:
    """Write every principle unit (one isolated call per cluster) and assemble the doc.

    Each unit call sees only its own cluster's claims and anchors — even attention,
    and the per-unit token budget is enforceable. Units landing outside the band get
    one revision round with the measured count fed back.

    Returns:
        (draft_doc, final_doc): the assembly of first-pass units and of post-revision
        units, so the doc's evolution is on the record. Identical when nothing revised.
    """
    a = cfg["arms"][arm]
    model = cfg["models"]["write"]
    count = _counter(cfg["tokenizer"])
    preamble, closing = (PKG / "preamble.md").read_text(), (PKG / "closing.md").read_text()
    overhead = count(preamble) + count(closing)
    budget = max(int(cfg["unit_floor_tokens"]),
                 round((a["target_tokens"] - overhead) / a["n_principles"]))
    by_id = {c["claim_id"]: c for c in inventory}
    ordered = sorted(clusters, key=lambda c: PRIORITY_ORDER.index(c["parent_priority"]))

    def one(i: int) -> tuple[str, str]:
        c = ordered[i]
        claims = "\n".join(
            f"- ({x['claim_id']}, {x['modality']}) {x['claim']}  [anchor: \"{x['anchor']}\"]"
            for x in (by_id[j] for j in c["claim_ids"]))
        prompt = prompts.WRITE_UNIT.format(
            title=c["working_title"], claims=claims, token_budget=budget,
            cue_block=prompts.cue_block(int(a["cues"])))
        draft = unit = _ask(prompt, model, spend)
        measured, ratio = count(unit), _unit_ratio(unit, count)
        low, high = cfg["unit_band"]
        rlow = float(cfg["explanation_ratio_band"][0])
        if not smoke and ((not low * budget <= measured <= high * budget)
                          or ratio < rlow):
            unit = _ask(prompt + f"\n\nYour previous unit:\n\n{unit}\n\n"
                        + prompts.REVISE.format(measured=measured, token_budget=budget,
                                                expl_pct=round(100 * ratio)),
                        model, spend)
        assert unit.startswith("## ") and "*Why:*" in unit \
            and "*When this does NOT apply:*" in unit, f"malformed unit:\n{unit[:300]}"
        return draft, unit

    units = map_threaded(one, len(ordered), int(cfg["workers"]), f"write:{arm}/s{seed}")
    return (_assemble([d for d, _ in units], preamble, closing),
            _assemble([u for _, u in units], preamble, closing))


def generate(cfg: dict, arm: str | None = None, seeds: int | None = None,
             smoke: bool = False) -> None:
    """Generate specs: for each arm and seed, cluster -> write -> assemble -> meta."""
    root = out_dir(cfg, smoke)
    inventory = read_jsonl(root / "claims" / "inventory.jsonl")
    inv_lock = json.loads((root / "claims" / "inventory.lock.json").read_text())
    spend = Spend()
    arms = [arm] if arm else list(cfg["arms"])
    seed_list = range(seeds if seeds is not None else int(cfg["seeds"]))
    if smoke:
        arms, seed_list = arms[:1], [0]
    count = _counter(cfg["tokenizer"])
    run_ts = timestamp()
    if not smoke and cfg.get("hf_repo"):
        card = root / "README.hf.md"
        card.write_text(_card(cfg))
        _push(cfg, [card], ".")

    for a in arms:
        n = min(int(cfg["arms"][a]["n_principles"]), len(inventory)) if smoke \
            else int(cfg["arms"][a]["n_principles"])
        for seed in seed_list:
            dst = root / a / f"seed{seed}"
            if (dst / "constitution.md").exists():
                print(f">>> {a}/seed{seed}: exists, skipping")
                continue
            dst.mkdir(parents=True, exist_ok=True)
            # Clustering is cached: a crash in the write stage must not cost a recluster.
            if (dst / "clusters.json").exists():
                clusters = json.loads((dst / "clusters.json").read_text())["clusters"]
                print(f">>> {a}/seed{seed}: reusing cached clusters.json")
            else:
                clusters = cluster(inventory, n, cfg, spend, seed)
                (dst / "clusters.json").write_text(json.dumps(
                    {"arm": a, "n_principles": n, "seed": seed,
                     "inventory_sha256": inv_lock["sha256"], "clusters": clusters},
                    indent=2))
            draft, doc = write_spec(inventory, clusters, a, cfg, spend, seed, smoke)
            (dst / "constitution.draft.md").write_text(draft)
            (dst / "constitution.md").write_text(doc)
            (dst / "meta.json").write_text(json.dumps(
                {"arm": a, "n_principles": n, "seed": seed,
                 "models": cfg["models"], "source_sha256": inv_lock["source_sha256"],
                 "inventory_sha256": inv_lock["sha256"],
                 "prompt_sha256": prompts.hashes(), "git_sha": git_sha(),
                 "timestamp": timestamp(), "token_count": count(doc),
                 "hf_prefix": f"{a}/seed{seed}/{run_ts}"}, indent=2))
            if not smoke:
                _push(cfg, [dst / "constitution.draft.md", dst / "constitution.md",
                            dst / "clusters.json", dst / "meta.json"],
                      f"{a}/seed{seed}/{run_ts}")
            print(f">>> {a}/seed{seed}: {count(doc)} tokens, {n} principles")
            if spend.usd > float(cfg["budget_usd"]):
                raise RuntimeError(
                    f"budget_usd={cfg['budget_usd']} exceeded (~${spend.usd:.2f} "
                    f"nominal, {spend.calls} calls); completed outputs are kept")
    print(f">>> generate done | {spend.calls} calls (~${spend.usd:.2f} nominal)")
