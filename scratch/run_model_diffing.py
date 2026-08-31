# ABOUTME: Runs N model-diffing investigations against a served pair and reports the
# ABOUTME: false-positive rate, which is the whole point of the A-vs-A control.

"""Drive the diffing agent over several seeds.

Run: uv run python scratch/run_model_diffing.py <pod-id> <model_a> <model_b> [seeds]

For the control, pass the same served name twice: every `difference_found: true` is then a
false positive by construction, and the agent's calibration is exactly its rate of them.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.audits.model_diffing.agent import run_investigation  # noqa: E402
from src.eval.audits.model_diffing.prompts import SEEDS  # noqa: E402
from src.eval.audits.model_diffing.tools import PairedSampler  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

load_dotenv("src/eval/audits/surf/third_party/SURF/.env", override=True)

POD = sys.argv[1]
MODEL_A = sys.argv[2]
MODEL_B = sys.argv[3]
N_SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 10
AUDITOR = os.environ.get("DIFF_AUDITOR", "z-ai/glm-5.2")
BASE_URL = f"https://{POD}-8000.proxy.runpod.net/v1"

IS_CONTROL = MODEL_A == MODEL_B
OUT = Path(f"output/model_diffing/{'control' if IS_CONTROL else 'diff'}_{timestamp()}")


def main():
    """Run the investigations concurrently and summarise calibration."""
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS[:N_SEEDS]
    print(f"auditor={AUDITOR}  A={MODEL_A}  B={MODEL_B}  seeds={len(seeds)}")
    print(f"{'A-vs-A CONTROL: every finding is a FALSE POSITIVE' if IS_CONTROL else 'diff run'}")

    def one(i):
        """Run investigation i with its own sampler, checkpointing the moment it files.

        Written per investigation, not at the end: a single end-of-run write means a
        timeout or crash discards every completed investigation, which is exactly what
        happened on the first control attempt (47 minutes, zero output).
        """
        sampler = PairedSampler(BASE_URL, MODEL_A, MODEL_B)
        try:
            r = run_investigation(AUDITOR, sampler, seeds[i])
            (OUT / f"seed_{i:02d}.json").write_text(
                json.dumps(r, indent=2, ensure_ascii=False))
            rep = r["report"]
            print(f"  [{i + 1}/{len(seeds)}] turns={r['turns']:2d} "
                  f"found={rep.get('difference_found')} "
                  f"conf={rep.get('confidence')} :: {str(rep.get('headline'))[:90]}",
                  flush=True)
            return r
        except Exception as e:  # noqa: BLE001 - one dead seed must not lose the rest
            print(f"  [{i + 1}] FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
            return {"seed": seeds[i], "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=5) as ex:
        runs = list(ex.map(one, range(len(seeds))))

    # Transcripts are large; keep them, but separately from the summary.
    (OUT / "runs.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False))

    ok = [r for r in runs if "report" in r]
    pos = [r for r in ok if r["report"].get("difference_found")]
    high = [r for r in pos if r["report"].get("confidence") == "high"]
    capped = [r for r in ok if r["report"].get("hit_turn_cap")]
    summary = {
        "git_sha": git_sha(), "auditor": AUDITOR, "model_a": MODEL_A, "model_b": MODEL_B,
        "is_control": IS_CONTROL, "n_seeds": len(seeds), "completed": len(ok),
        "errors": len(runs) - len(ok), "hit_turn_cap": len(capped),
        "difference_found": len(pos), "of_which_high_confidence": len(high),
        "false_positive_rate": round(len(pos) / len(ok), 3) if (IS_CONTROL and ok) else None,
        "mean_turns": round(sum(r["turns"] for r in ok) / len(ok), 1) if ok else None,
        "target_calls_total": sum(r.get("target_calls", 0) for r in ok),
        "headlines": [r["report"].get("headline") for r in ok],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 72}")
    print(f"completed {len(ok)}/{len(seeds)}   mean turns {summary['mean_turns']}   "
          f"target calls {summary['target_calls_total']}")
    if IS_CONTROL:
        print(f"FALSE POSITIVES: {len(pos)}/{len(ok)} = "
              f"{(len(pos) / len(ok) * 100 if ok else 0):.0f}%   "
              f"(high-confidence: {len(high)})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
