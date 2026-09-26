# ABOUTME: Judge the episodes the run left unscored — the ones that hit the turn or token cap —
# ABOUTME: so a rate can be read with them included instead of silently dropped.
"""Grade a run's INCOMPLETE episodes as they stand.

    uv run python scratch/delegated_harm/grade_incomplete.py --run-dir <run> \\
        [--judge-model google/gemini-3-flash-preview]

Both the runner and rescore.py judge only `status == "complete"`: an episode that ran to the
90-turn cap or blew the 16k output budget is recorded and then excluded from every rate. That
is the right default — a truncated transcript has no settled outcome — but it also means a
checkpoint that rambles quietly loses episodes, and the exclusions are not evenly spread
(2026-09-26: nosynth 81/108 valid, multiparty-human 105/108).

This judges those transcripts AS THEY ARE and writes the result beside the official one,
never over it: `results/incomplete_graded.json` holds the per-episode verdicts plus rates
computed three ways — official (valid only), incomplete only, and pooled. A truncated
episode's label is weaker evidence than a finished one's, so the three stay separate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.misalignment.delegated_harm.judging import Judge
from src.eval.misalignment.delegated_harm.runner import judge_episode
from src.eval.misalignment.delegated_harm.source import prepare


def read(path: Path):
    """Parse one JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def rates(records: list[dict]) -> dict:
    """Mean of each metric over the records that carry metrics."""
    scored = [r for r in records if r.get("metrics")]
    if not scored:
        return {}
    return {k: round(sum(r["metrics"][k] for r in scored) / len(scored), 4)
            for k in scored[0]["metrics"]}


def main() -> None:
    """Judge every unscored episode of the run and write the three-way comparison."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--judge-model", default="google/gemini-3-flash-preview")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.run_dir)
    cfg = OmegaConf.create(read(out / "metadata/protocol.json")["protocol"])
    cfg.judge.model = args.judge_model
    cfg.judge.budget_usd = None          # these are a handful of episodes, not a sweep
    cfg.judge.workers = args.workers
    scenarios, _, notes, _ = prepare(cfg)
    descriptions = read(Path(cfg.source.path) / "descriptions.json")
    judge = Judge(cfg.judge, out / "metadata" / "incomplete_judge")

    official = [read(p) for p in sorted((out / "results/episodes").glob("*.json"))]
    unscored = [r for r in official if not r.get("metrics")]
    print(f">>> {len(official)} episodes recorded, {len(unscored)} unscored "
          f"({', '.join(sorted({r.get('status', '?') for r in unscored}))})")

    graded = []
    for rec in unscored:
        trace_path = out / "rollouts" / f"{rec['id']}.json"
        if not trace_path.exists():
            print(f"    {rec['id']}: no transcript on disk; skipped")
            continue
        trace = read(trace_path)
        sc = scenarios[rec["scenario"]]
        try:
            judged = judge_episode(sc, trace, notes[sc.key]["clear"], judge, descriptions)
        except Exception as exc:  # noqa: BLE001 — a judge that refuses this transcript is
            # reported, not counted: an unjudgeable truncation is still an exclusion.
            print(f"    {rec['id']} ({rec.get('status')}): judge {type(exc).__name__}")
            graded.append({**rec, "judgment_error": str(exc)[:200]})
            continue
        print(f"    {rec['id']} ({rec.get('status')}): graded "
              f"harmful={judged['metrics'].get('harmful_action')} "
              f"refusal={judged['metrics'].get('explicit_refusal')}")
        graded.append({**rec, **judged})

    scored_official = [r for r in official if r.get("metrics")]
    payload = {
        "judge_model": args.judge_model,
        "counts": {"recorded": len(official), "official_valid": len(scored_official),
                   "unscored": len(unscored), "newly_graded": sum(1 for r in graded if r.get("metrics"))},
        "statuses": {s: sum(1 for r in unscored if r.get("status") == s)
                     for s in sorted({r.get("status", "?") for r in unscored})},
        "rates_official_valid_only": rates(scored_official),
        "rates_incomplete_only": rates(graded),
        "rates_pooled": rates(scored_official + graded),
        "episodes": [{k: v for k, v in r.items() if k in ("id", "status", "metrics", "judgment_error")}
                     for r in graded],
    }
    dest = out / "results" / "incomplete_graded.json"
    dest.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f">>> wrote {dest}")


if __name__ == "__main__":
    main()
