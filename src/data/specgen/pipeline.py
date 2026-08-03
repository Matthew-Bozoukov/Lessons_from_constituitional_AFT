# ABOUTME: The specgen stages: pin source -> extract claim inventory (once, shared) ->
# ABOUTME: per arm/seed cluster to N principles -> write units -> assemble the spec.

from __future__ import annotations

import hashlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openrouter import OpenRouterClient, map_threaded  # noqa: E402
from utils import extract_json, git_sha, read_jsonl, timestamp  # noqa: E402

from . import prompts  # noqa: E402
from ..synthdoc.stages import Usage  # noqa: E402

PKG = Path(__file__).parent
PRIORITY_ORDER = ["safe", "ethical", "compliant", "helpful"]
STOPWORDS = {"being", "broadly", "the", "a", "an", "of", "and", "to", "on", "for",
             "claude", "claude's", "why", "what", "how"}


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
    """Split the source on its own H2 boundaries into [{section_id, title, text}]."""
    parts = re.split(r"^## +(.+)$", text, flags=re.M)
    # parts = [before-first-heading, title1, body1, title2, body2, ...]
    out = []
    for title, body in zip(parts[1::2], parts[2::2]):
        words = [w for w in re.findall(r"[a-z']+", title.lower()) if w not in STOPWORDS]
        slug = (words[0][:3] if words else f"s{len(out):02d}")
        while any(s["section_id"] == slug for s in out):
            slug += "x"
        out.append({"section_id": slug, "title": title.strip(), "text": body.strip()})
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
    client, usage = OpenRouterClient(), Usage()
    m = cfg["models"]["extract"]

    def one(i: int) -> list[dict]:
        s = secs[i]
        res = client.chat(m["model"], [{"role": "user", "content": prompts.EXTRACT.format(
            title=s["title"], text=s["text"])}],
            temperature=m["temperature"], max_tokens=m["max_tokens"])
        usage.add(m["model"], res, "extract")
        claims = extract_json(res.content)
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
            "source_sha256": _sha256(source), "model": m["model"],
            "n_claims": len(inventory), "git_sha": git_sha(), "at": timestamp()}
    (dst / "inventory.lock.json").write_text(json.dumps(lock, indent=2))
    print(f">>> {len(inventory)} claims from {len(secs)} sections | ${usage.usd:.2f}")
    if not smoke and not 150 <= len(inventory) <= 400:
        print(f"!!! claim count outside the expected 150-400 band — inspect before generating")
    return dst / "inventory.jsonl"


# --- stage 2: cluster to resolution N ----------------------------------------------


def cluster(inventory: list[dict], n: int, cfg: dict, usage: Usage, seed: int) -> list[dict]:
    """Partition the inventory into exactly n clusters; exact ID accounting enforced.

    Retries with error feedback; raises after cfg[cluster_retries] failures rather
    than silently accepting an unbalanced partition.
    """
    client = OpenRouterClient()
    m = cfg["models"]["cluster"]
    lines = "\n".join(f"{c['claim_id']} | {c['modality']} | {c['claim']}"
                      for c in inventory)
    messages = [{"role": "user",
                 "content": prompts.CLUSTER.format(n_principles=n, claims=lines)}]
    want = {c["claim_id"] for c in inventory}
    for attempt in range(int(cfg["cluster_retries"]) + 1):
        res = client.chat(m["model"], messages, temperature=m["temperature"],
                          max_tokens=m["max_tokens"], seed=seed)
        usage.add(m["model"], res, "cluster")
        clusters = extract_json(res.content)["clusters"]
        got = [i for c in clusters for i in c["claim_ids"]]
        orphans, dupes = want - set(got), len(got) - len(set(got))
        bad_prio = [c["parent_priority"] for c in clusters
                    if c["parent_priority"] not in PRIORITY_ORDER]
        if len(clusters) == n and not orphans and not dupes and not bad_prio:
            return clusters
        problem = (f"{len(clusters)} clusters (need {n}), {len(orphans)} orphaned, "
                   f"{dupes} duplicated, bad priorities {bad_prio}")
        print(f"    cluster attempt {attempt + 1}: {problem}")
        messages += [{"role": "assistant", "content": res.content},
                     {"role": "user", "content":
                      f"Invalid partition: {problem}. Missing ids: "
                      f"{sorted(orphans)[:20]}. Return the corrected full JSON."}]
    raise RuntimeError(f"clustering failed after retries: {problem}")


# --- stage 3+4: write units and assemble -------------------------------------------


def write_spec(inventory: list[dict], clusters: list[dict], arm: str, cfg: dict,
               usage: Usage, seed: int, smoke: bool = False) -> str:
    """Write every principle unit (one isolated call per cluster) and assemble the doc.

    Each unit call sees only its own cluster's claims and anchors — even attention,
    and the per-unit token budget is enforceable. Units landing outside the band get
    one revision round with the measured count fed back.
    """
    a = cfg["arms"][arm]
    client = OpenRouterClient()
    m = cfg["models"]["write"]
    count = _counter(cfg["tokenizer"])
    preamble, closing = (PKG / "preamble.md").read_text(), (PKG / "closing.md").read_text()
    overhead = count(preamble) + count(closing)
    budget = max(int(cfg["unit_floor_tokens"]),
                 round((a["target_tokens"] - overhead) / a["n_principles"]))
    by_id = {c["claim_id"]: c for c in inventory}
    ordered = sorted(clusters, key=lambda c: PRIORITY_ORDER.index(c["parent_priority"]))

    def one(i: int) -> str:
        c = ordered[i]
        claims = "\n".join(
            f"- ({x['claim_id']}, {x['modality']}) {x['claim']}  [anchor: \"{x['anchor']}\"]"
            for x in (by_id[j] for j in c["claim_ids"]))
        messages = [{"role": "user", "content": prompts.WRITE_UNIT.format(
            title=c["working_title"], claims=claims, token_budget=budget,
            cue_block=prompts.cue_block(int(a["cues"])))}]
        res = client.chat(m["model"], messages, temperature=m["temperature"],
                          max_tokens=m["max_tokens"], seed=seed)
        usage.add(m["model"], res, "write")
        unit = res.content.strip()
        measured = count(unit)
        low, high = cfg["unit_band"]
        if not smoke and not (low * budget <= measured <= high * budget):
            messages += [{"role": "assistant", "content": unit},
                         {"role": "user", "content": prompts.REVISE.format(
                             measured=measured, token_budget=budget)}]
            res = client.chat(m["model"], messages, temperature=m["temperature"],
                              max_tokens=m["max_tokens"], seed=seed)
            usage.add(m["model"], res, "write")
            unit = res.content.strip()
        assert unit.startswith("## ") and "*Why:*" in unit \
            and "*When this does NOT apply:*" in unit, f"malformed unit:\n{unit[:300]}"
        return unit

    units = map_threaded(one, len(ordered), int(cfg["workers"]), f"write:{arm}/s{seed}")
    numbered = [u.replace("## ", f"## {i + 1}. ", 1) for i, u in enumerate(units)]
    return preamble + "\n---\n\n" + "\n\n".join(numbered) + "\n\n---\n\n" + closing


def generate(cfg: dict, arm: str | None = None, seeds: int | None = None,
             smoke: bool = False) -> None:
    """Generate specs: for each arm and seed, cluster -> write -> assemble -> meta."""
    root = out_dir(cfg, smoke)
    inventory = read_jsonl(root / "claims" / "inventory.jsonl")
    inv_lock = json.loads((root / "claims" / "inventory.lock.json").read_text())
    usage = Usage()
    arms = [arm] if arm else list(cfg["arms"])
    seed_list = range(seeds if seeds is not None else int(cfg["seeds"]))
    if smoke:
        arms, seed_list = arms[:1], [0]
    count = _counter(cfg["tokenizer"])

    for a in arms:
        n = min(int(cfg["arms"][a]["n_principles"]), len(inventory)) if smoke \
            else int(cfg["arms"][a]["n_principles"])
        for seed in seed_list:
            dst = root / a / f"seed{seed}"
            if (dst / "constitution.md").exists():
                print(f">>> {a}/seed{seed}: exists, skipping")
                continue
            clusters = cluster(inventory, n, cfg, usage, seed)
            doc = write_spec(inventory, clusters, a, cfg, usage, seed, smoke)
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "constitution.md").write_text(doc)
            (dst / "clusters.json").write_text(json.dumps(
                {"arm": a, "n_principles": n, "seed": seed,
                 "inventory_sha256": inv_lock["sha256"], "clusters": clusters}, indent=2))
            (dst / "meta.json").write_text(json.dumps(
                {"arm": a, "n_principles": n, "seed": seed,
                 "models": cfg["models"], "source_sha256": inv_lock["source_sha256"],
                 "inventory_sha256": inv_lock["sha256"],
                 "prompt_sha256": prompts.hashes(), "git_sha": git_sha(),
                 "timestamp": timestamp(), "token_count": count(doc)}, indent=2))
            print(f">>> {a}/seed{seed}: {count(doc)} tokens, {n} principles")
            if usage.usd > float(cfg["budget_usd"]):
                raise RuntimeError(f"budget_usd={cfg['budget_usd']} exceeded "
                                   f"(${usage.usd:.2f}); completed outputs are kept")
    print(f">>> generate done | ${usage.usd:.2f}")
