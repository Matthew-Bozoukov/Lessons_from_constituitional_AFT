# ABOUTME: Applies arena-hard's style-controlled win rate to an LMSYS pair run, which
# ABOUTME: otherwise reports only a raw preference rate with length/markdown baked in.

"""Style-control an LMSYS chat SxS result.

`configs/eval/lmsys.yaml` has no style control: its win rate is a raw preference count, so
a model that simply writes longer or adds more headers scores better for that alone. This
reuses arena-hard's validated machinery rather than inventing a second one — the same
`style_features` extractor and the same Bradley-Terry regression over four features
(length, header/list/bold density asymmetry), so the two evals' controlled numbers mean
the same thing.

One structural difference to keep in mind: arena-hard judges every prompt TWICE with the
positions swapped, and averages. LMSYS judges each prompt once with the position randomly
assigned (`target_is_a`). Position bias therefore averages out across prompts here rather
than within them, which is weaker — it adds variance rather than removing bias, and there
is no swap-consistency number to report.

    uv run python scratch/lmsys_style_control.py --run_dir output/lmsys/pair_...
"""

from __future__ import annotations

import json
from pathlib import Path

import fire

from src.eval.capabilities.arena_hard import arena_hard_stats as stats
from src.eval.capabilities.arena_hard.arena_hard_metrics import style_features

_WINNER_TO_SCORE = {"target": 1.0, "reference": 0.0, "tie": 0.5}


def main(run_dir: str, rounds: int = 2000, alpha: float = 0.05, seed: int = 0) -> None:
    """Recompute an LMSYS pair result with and without style control.

    Args:
        run_dir: An LMSYS pair output directory (contains one subdir per arm).
        rounds: Bootstrap resamples.
        alpha: Two-sided level.
        seed: RNG seed.
    """
    judged = next(Path(run_dir).rglob("judged.jsonl"))
    rows = [json.loads(line) for line in judged.open()]
    print(f">>> {judged}\n>>> {len(rows)} judged prompts")

    battles, model_meta, base_meta = [], {}, {}
    skipped = 0
    for r in rows:
        winner = str(r.get("winner", "")).lower()
        if winner not in _WINNER_TO_SCORE:
            skipped += 1          # judge failure: no verdict to score
            continue
        uid = str(r["id"])
        battles.append({"uid": uid, "category": "lmsys", "model": "target",
                        "game": 0, "score": _WINNER_TO_SCORE[winner],
                        "decisive": winner != "tie"})
        # Features come from the user-visible answer only — the <think> trace is not what
        # the judge compared, and including it would make "length" mean trace length.
        model_meta[uid] = style_features(r.get("target_answer") or "")
        base_meta[uid] = style_features(r.get("reference_answer") or "")

    print(f">>> scored {len(battles)}, skipped {skipped} (no parsable verdict)")
    res = stats.evaluate_arm(battles, model_meta, base_meta, threshold=0.45,
                             rounds=rounds, alpha=alpha, seed=seed)
    u, c = res["uncontrolled"], res["controlled"]
    print(f"\n  win/tie/loss           {res['split']}")
    print(f"  uncontrolled           {u['mean']*100:5.1f}%  "
          f"[{u['ci_lower']*100:.1f}, {u['ci_upper']*100:.1f}]")
    print(f"  STYLE-CONTROLLED       {c['mean']*100:5.1f}%  "
          f"[{c['ci_lower']*100:.1f}, {c['ci_upper']*100:.1f}]")
    print(f"  style gap              {res['style_gap_pp']:+.1f}pp  "
          f"(uncontrolled - controlled; positive = the lead came from style)")

    def mean(meta, key):
        vals = [m[key] if isinstance(m[key], (int, float)) else sum(m[key].values())
                for m in meta.values()]
        return sum(vals) / len(vals)

    print("\n  mean style features    target   reference")
    for key in ("token_len", "header_count", "list_count", "bold_count"):
        print(f"    {key:<16} {mean(model_meta, key):8.1f} {mean(base_meta, key):10.1f}")

    out = Path(run_dir) / "style_controlled.json"
    out.write_text(json.dumps({
        "n_scored": len(battles), "n_skipped": skipped,
        "split": res["split"], "uncontrolled": u, "controlled": c,
        "style_gap_pp": res["style_gap_pp"],
        "mean_features": {
            "target": {k: mean(model_meta, k) for k in
                       ("token_len", "header_count", "list_count", "bold_count")},
            "reference": {k: mean(base_meta, k) for k in
                          ("token_len", "header_count", "list_count", "bold_count")},
        },
    }, indent=2))
    print(f"\n>>> {out}")


if __name__ == "__main__":
    fire.Fire(main)
