# ABOUTME: Label-free diversity of the RENDERED user turns, pilot vs baseline, size-matched.
# ABOUTME: Answers "is the corpus actually diverse" without trusting a metadata field.

"""Why not just count domains.

`metadata.domain` is written at stage 2 and never updated, while stage 4 replaces or
reframes most prompts -- so a domain count measures labels, not content (see
scratch/refine_drift.py). This measures the text itself.

Cosine is length-dependent (0.37 mean pairwise at 68 words, 0.59 at 203, 0.76 at 1044),
so the two corpora are compared at the same N and their mean word counts are printed:
if those diverge materially the cosine comparison is not clean and the run says so.

Lower mean-pairwise and lower mean-nearest-neighbour both mean MORE diverse.

Usage: uv run python scratch/text_diversity.py <pilot_run_dir>
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

BASELINE = Path("/Users/kunwar/projects/lessons_from_constitutional_aft/output/"
                "corpus_browse/difficult_advice/stage_4_refined_prompts.jsonl")
DRAWS = 40


def load(path: Path, field: str = "user") -> list[str]:
    return [str(json.loads(x).get(field) or "")
            for x in path.open(encoding="utf-8") if x.strip()]


def stats(texts: list[str]) -> tuple[float, float]:
    """(mean pairwise cosine, mean nearest-neighbour cosine)."""
    import numpy as np

    from src.data.synth.embeddings import embed

    X = embed(texts)
    G = X @ X.T
    np.fill_diagonal(G, -1.0)
    n = len(texts)
    mean_pair = float(G[G > -1].sum() / max(n * (n - 1), 1))
    return mean_pair, float(G.max(axis=1).mean())


def main() -> None:
    run_dir = Path(sys.argv[1])
    pilot = [t for t in load(run_dir / "stage_5_refined_prompts.jsonl") if t]
    base_all = [t for t in load(BASELINE) if t]
    n = len(pilot)

    pw_pilot, nn_pilot = stats(pilot)
    pilot_words = sum(len(t.split()) for t in pilot) / n

    rng = random.Random(0)
    pws, nns, words = [], [], []
    for _ in range(DRAWS):
        s = rng.sample(base_all, n)
        pw, nn = stats(s)
        pws.append(pw)
        nns.append(nn)
        words.append(sum(len(t.split()) for t in s) / n)
    pws.sort()
    nns.sort()
    lo, hi = int(0.05 * DRAWS), int(0.95 * DRAWS) - 1

    print(f"\nRendered user turns, label-free. n={n} each, {DRAWS} baseline draws.\n"
          + "=" * 68)
    print(f"\nmean words   pilot {pilot_words:6.0f}   baseline {sum(words) / DRAWS:6.0f}")
    if abs(pilot_words - sum(words) / DRAWS) / max(pilot_words, 1) > 0.20:
        print("  !! lengths differ by >20%; cosine is length-dependent, read with care")

    for name, val, dist in (("mean pairwise cosine", pw_pilot, pws),
                            ("mean nearest-neighbour", nn_pilot, nns)):
        med = dist[len(dist) // 2]
        out = "OUTSIDE" if (val < dist[lo] or val > dist[hi]) else "inside"
        arrow = "more diverse" if val < med else "less diverse"
        print(f"\n{name}")
        print(f"  pilot     {val:.4f}   ({arrow})")
        print(f"  baseline  {med:.4f}   p5 {dist[lo]:.4f}  p95 {dist[hi]:.4f}"
              f"   -> pilot is {out} the baseline range")


if __name__ == "__main__":
    main()
