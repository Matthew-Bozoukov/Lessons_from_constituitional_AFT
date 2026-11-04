# ABOUTME: Builds a side-by-side reader for the latest low-stakes DA corpus against the latest DA corpus.
# ABOUTME: Run: uv run --no-sync python scratch/lowstakes_vs_da_viewer_20261102.py (writes output/2026-11-02_lowstakes_vs_da_viewer.html)
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

LOW = ("dougalldeepmind/2026-10-04-da-low-synth", "c8a61e2661be8d564b27bfb513ba3f31eec27897")
DA = ("dougalldeepmind/2026-10-30-da-synth", "8fadea6299915605d0b270feddfd85d91344e58f")
PER_TRAIT = 12
SEED = 0
TEMPLATE = Path(__file__).with_name("lowstakes_vs_da_viewer_20261102.html")
OUT = Path("output/2026-11-02_lowstakes_vs_da_viewer.html")
META_KEYS = ("scenario_id", "trait_id", "domain", "sector", "ai", "situation", "shortcut")


def load(repo, rev):
    path = hf_hub_download(repo, "dataset.jsonl", repo_type="dataset", revision=rev)
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def doc(row):
    system, user, assistant = row["messages"]
    return {
        "meta": {k: row["metadata"][k] for k in META_KEYS},
        "system": system["content"],
        "user": user["content"],
        "reasoning": assistant.get("reasoning_content") or "",
        "response": assistant["content"],
    }


def lengths(rows):
    out = {}
    for name, pick in [
        ("user", lambda m: m[1]["content"]),
        ("reasoning", lambda m: m[2].get("reasoning_content") or ""),
        ("response", lambda m: m[2]["content"]),
    ]:
        words = [len(pick(r["messages"]).split()) for r in rows]
        out[name] = round(statistics.median(words))
    return out


def corpus_info(repo, rev, rows):
    return {"repo": repo, "rev": rev, "rows": len(rows), "median_words": lengths(rows)}


def pairs(low, da):
    # Pair on the shared scenario id: same trait, and the seeded deal usually gives the same sector and AI line.
    da_by_id = {r["metadata"]["scenario_id"]: r for r in da}
    by_trait = defaultdict(list)
    for r in low:
        other = da_by_id.get(r["metadata"]["scenario_id"])
        if other is None:
            continue
        same_deal = all(r["metadata"][k] == other["metadata"][k] for k in ("sector", "ai"))
        by_trait[r["metadata"]["trait_id"]].append((same_deal, r, other))
    rng = random.Random(SEED)
    chosen = []
    for trait in sorted(by_trait):
        cands = by_trait[trait]
        rng.shuffle(cands)
        cands.sort(key=lambda c: not c[0])
        picked = cands[:PER_TRAIT]
        picked.sort(key=lambda c: c[1]["metadata"]["scenario_id"])
        chosen += [{"same_deal": s, "low": doc(a), "da": doc(b)} for s, a, b in picked]
    return chosen


def main():
    low, da = load(*LOW), load(*DA)
    traits = sorted({(r["metadata"]["trait_id"], r["metadata"]["trait_name"]) for r in low})
    data = {
        "low": corpus_info(*LOW, low),
        "da": corpus_info(*DA, da),
        "traits": [{"id": t, "name": n} for t, n in traits],
        "pairs": pairs(low, da),
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"{OUT}: {len(data['pairs'])} pairs, {OUT.stat().st_size / 1e6:.2f} MB")
    print(data["low"]["median_words"], data["da"]["median_words"])


if __name__ == "__main__":
    main()
