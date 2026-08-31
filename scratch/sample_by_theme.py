# ABOUTME: Draws a theme-balanced subsample from the k=9 scenario clustering.
# ABOUTME: Equal per theme, with the indivisible remainder given to the largest themes.
"""Take N scenarios spread evenly across the nine themes.

179 does not divide by 9, so the split is 8 themes of 20 and 1 of 19. The short theme is the
SMALLEST one, chosen by cluster size rather than at random, so the shortfall lands where it
costs the least proportionally and the choice is reproducible.

Within a theme the draw is uniform at the given seed. Selection is by scenario_id, so the
manifest is stable even if the corpus file is re-ordered.

Run: uv run python scratch/sample_by_theme.py --n 179
"""
import collections
import json
from pathlib import Path

import fire

from src.utils import read_jsonl, timestamp, write_run_meta

SRC = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl"
THEMES = "output/scenario_themes/20260827_184339"
OUT_ROOT = Path("output/theme_balanced_sample")


def main(n: int = 179, seed: int = 0, themes_dir: str = THEMES) -> None:
    """Write a theme-balanced sample of n scenarios.

    Args:
        n: Total scenarios to draw.
        seed: Sampling seed.
        themes_dir: Directory holding assignments.jsonl from the k=9 clustering.
    """
    import random

    assign = list(read_jsonl(Path(themes_dir) / "assignments.jsonl"))
    rows = {r["metadata"]["scenario_id"]: r for r in read_jsonl(SRC)}
    by_theme = collections.defaultdict(list)
    for a in assign:
        by_theme[a["theme"]].append(a)
    k = len(by_theme)
    assert k, "no themes found"

    base, extra = divmod(n, k)
    # Largest themes take the remainder, so the one short theme is the smallest.
    order = sorted(by_theme, key=lambda t: -len(by_theme[t]))
    quota = {t: base + (1 if i < extra else 0) for i, t in enumerate(order)}
    print(f"n={n} over {k} themes -> {base} each, +1 for the {extra} largest\n")

    rnd = random.Random(seed)
    picked = []
    for t in order:
        pool = sorted(by_theme[t], key=lambda a: a["scenario_id"])
        assert len(pool) >= quota[t], f"{t}: {len(pool)} available < {quota[t]} requested"
        take = rnd.sample(pool, quota[t])
        picked += take
        print(f"  {t:<34} pool {len(pool):>4}  taken {quota[t]:>3}")

    picked.sort(key=lambda a: (a["theme"], a["scenario_id"]))
    out_dir = OUT_ROOT / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "sample.jsonl").open("w") as f:
        for a in picked:
            r = rows[a["scenario_id"]]
            f.write(json.dumps({**r, "theme": a["theme"], "cluster": a["cluster"]},
                               ensure_ascii=False) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(
        {"n": len(picked), "seed": seed, "themes_dir": themes_dir,
         "per_theme": {t: quota[t] for t in order},
         "scenario_ids": [a["scenario_id"] for a in picked]}, indent=2))

    tr = collections.Counter(a["trait"] for a in picked)
    L = [f"# Theme-balanced sample — {len(picked)} scenarios", "",
         f"Drawn from the k=9 clustering (`{themes_dir}`), seed {seed}.", "",
         "| theme | pool | taken |", "|---|--:|--:|"]
    for t in order:
        L.append(f"| {t} | {len(by_theme[t])} | {quota[t]} |")
    L += ["", "## Trait spread of the sample (not balanced on — reported for transparency)",
          "", "| trait | n |", "|---|--:|"]
    for t, c in sorted(tr.items()):
        L.append(f"| {t} | {c} |")
    (out_dir / "sample.md").write_text("\n".join(L) + "\n")
    write_run_meta(out_dir, {"src": SRC, "themes_dir": themes_dir, "n": n, "seed": seed},
                   {"n_picked": len(picked), "per_theme": quota})
    print(f"\n  trait spread: {dict(sorted(tr.items()))}")
    print(f"\n>>> {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
