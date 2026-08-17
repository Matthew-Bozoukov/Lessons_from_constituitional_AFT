# ABOUTME: Sweep embedding_dedup's cosine_min over the difficult-advice scenario corpus to
# ABOUTME: find a threshold that actually catches near-clones, since 0.90 provably drops zero.

"""Why this exists.

`embedding_dedup` ships `cosine_min: 0.90`. The registry comment records that the
difficult-advice corpus produced 0 pairs at every threshold from 0.90 up -- its worst
pair sits at 0.886. That same corpus has 46.9% of its mass in ten domains and contains
acknowledged near-clones (the grad-student / funding-cliff / data-shortcut cluster).

So the gate was calibrated against a corpus assumed healthy that we now know is not, and
it is a no-op on exactly the data it was measured on. This sweeps the threshold to find
where the clones actually live, and reports what a drop at each threshold would cost.

Run: uv run python scratch/measure_dedup_threshold.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synth.check_corpus import _components  # noqa: E402
from src.data.synth.embeddings import DEFAULT_MODEL, embed  # noqa: E402

# Read-only: the corpus lives in the main checkout, this worktree has no output/.
SCENARIOS = Path("/Users/kunwar/projects/lessons_from_constitutional_aft/output/"
                 "corpus_browse/difficult_advice/stage_2_scenarios.jsonl")
THRESHOLDS = [0.95, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.75, 0.70]


def load() -> list[dict]:
    return [json.loads(line) for line in SCENARIOS.open(encoding="utf-8") if line.strip()]


def main() -> None:
    import numpy as np

    records = load()
    situations = [str(r.get("situation") or "") for r in records]
    domains = [str(r.get("domain") or "").strip().lower() for r in records]
    n = len(records)
    mean_words = sum(len(s.split()) for s in situations) / max(n, 1)
    print(f"{n} scenarios, mean {mean_words:.0f} words in `situation`, "
          f"embedder {DEFAULT_MODEL}\n")

    X = embed(situations)
    G = X @ X.T
    np.fill_diagonal(G, -1.0)

    nn = G.max(axis=1)
    qs = [50, 90, 95, 99, 100]
    print("nearest-neighbour cosine distribution")
    for q in qs:
        print(f"  p{q:<3} {np.percentile(nn, q):.4f}")
    print(f"  mean {nn.mean():.4f}\n")

    print(f"{'thresh':>7} {'pairs':>7} {'clusters':>9} {'would_drop':>11} "
          f"{'drop %':>8} {'max_cluster':>12} {'same-domain %':>14}")
    for t in THRESHOLDS:
        a_idx, b_idx = np.where(np.triu(G >= t, k=1))
        pairs = list(zip(a_idx.tolist(), b_idx.tolist()))
        clusters = _components(n, pairs)
        drop = sorted(i for g in clusters for i in g[1:])
        same = (sum(1 for a, b in pairs if domains[a] == domains[b]) / len(pairs)
                if pairs else 0.0)
        print(f"{t:>7.2f} {len(pairs):>7} {len(clusters):>9} {len(drop):>11} "
              f"{len(drop) / n:>7.1%} {max((len(g) for g in clusters), default=0):>12} "
              f"{same:>13.0%}")

    # The concentration the dedup is supposed to relieve, for contrast.
    top = Counter(d for d in domains if d).most_common(10)
    print(f"\ntop-10 domains = {sum(c for _, c in top) / n:.1%} of the corpus")
    for name, count in top:
        print(f"  {count:>4}  {name}")

    # What the clones actually look like at the threshold that first catches them.
    for t in (0.86, 0.82):
        a_idx, b_idx = np.where(np.triu(G >= t, k=1))
        if not len(a_idx):
            continue
        order = np.argsort(-G[a_idx, b_idx])[:3]
        print(f"\n--- closest pairs at cosine >= {t} ---")
        for k in order:
            a, b = int(a_idx[k]), int(b_idx[k])
            print(f"\n[{G[a, b]:.3f}] {records[a]['scenario_id']} ({domains[a]}) "
                  f"<-> {records[b]['scenario_id']} ({domains[b]})")
            print(f"  A: {situations[a][:240]}")
            print(f"  B: {situations[b][:240]}")


if __name__ == "__main__":
    main()
