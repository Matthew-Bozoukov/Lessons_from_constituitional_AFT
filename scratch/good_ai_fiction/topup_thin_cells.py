# ABOUTME: Top up a Good AI Fiction run's thin cells by re-running the SAME recipe with
# ABOUTME: only the rotated-axis weights repointed, then merge the two runs for review.
"""Run: uv run python scratch/good_ai_fiction/topup_thin_cells.py --n 6

Why this exists. The stakes and source axes are dealt across BATCHES, so at pilot scale
(24 scenarios = 12 batches) their realised proportions are lumpy: the 2026-08-27 pilot
asked for 10% speculative and 28% inversion and got 1 and 4, then lost the single
speculative row to a provider content filter. At 716 scenarios (93 batches) the same deal
lands close to target, so this is a pilot-scale problem and not a recipe problem -- but a
review pilot with zero extreme-stakes rows and three inversions cannot be reviewed on the
two most distinctive parts of the design.

So: same config, same prompts, same gates, same models. The ONLY things changed are the
two axes' weights and the corpus size, and `id_prefix` so the top-up's ids cannot collide
with the run it is topping up. Nothing here is a second recipe.

The merged directory is for READING and REPORTING. It is not a corpus: the two runs used
independent diversity gates, so a scenario in one was never checked against the other.
`measure_rows.py` over the merge is what catches that, and a real 716-row build is one run.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import fire
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.synth import pipeline  # noqa: E402

CONFIG = ROOT / "configs/data/synth/2026-08-28_good_ai_fiction.yaml"


def main(n: int = 6, per_call: int = 2, prefix: str = "tu_",
         out_dir: str = "output/good_ai_fiction_topup",
         source_type: str = "inversion:100",
         stakes: str = "speculative:50,capability:50",
         merge_with: str = "output/good_ai_fiction_pilot/20260827_161345",
         merged: str = "output/good_ai_fiction_pilot/combined",
         resume: str = "") -> None:
    """Generate a top-up and merge it with an existing run for review.

    Args:
        n: Scenarios to generate.
        per_call: Scenarios per generation call.
        prefix: Prepended to every scenario id, so the merge stays joinable.
        out_dir: Where the top-up run writes.
        source_type: `label:weight,...` replacing the source axis weights.
        stakes: `label:weight,...` replacing the stakes axis weights.
        merge_with: The run being topped up.
        merged: Directory to write the concatenated dataset into.
        resume: Existing top-up run directory to continue.
    """
    load_dotenv()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["total_scenarios"] = n
    cfg["scenarios_per_call"] = per_call
    cfg["output_dir"] = out_dir
    cfg["id_prefix"] = prefix

    def weights(spec: str) -> dict[str, float]:
        return {k: float(v) for k, v in (p.split(":") for p in spec.split(","))}

    stage = next(s for s in cfg["stages"] if s["name"] == "write_scenarios")
    stage["rotate"]["source_type"]["weights"] = weights(source_type)
    stage["rotate"]["stakes"]["weights"] = weights(stakes)
    # Every rotated label still needs its prompt text; dropping a label from `weights`
    # while leaving its text is harmless, but naming one with no text would render an
    # empty instruction into a paid call.
    for axis in ("source_type", "stakes"):
        missing = set(stage["rotate"][axis]["weights"]) - set(stage["rotate"][axis]["text"])
        assert not missing, f"{axis}: no prompt text for {sorted(missing)}"

    print(f">>> top-up: {n} scenarios, source={source_type}, stakes={stakes}, "
          f"ids prefixed {prefix!r}")
    result = pipeline.run(cfg, smoke=False, resume=resume or None)
    pipeline.exit_if_gate_failed(result)
    run_dir = Path(result["run_dir"]) if isinstance(result, dict) and result.get("run_dir") \
        else sorted(Path(out_dir).glob("*"))[-1]

    dest = Path(merged)
    dest.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    seen: set[str] = set()
    for src in (Path(merge_with), run_dir):
        path = src / "dataset.jsonl"
        if not path.exists():
            path = next(src.glob("stage_*_export_sft.jsonl"))
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec["metadata"]["scenario_id"]
            assert sid not in seen, (
                f"id collision on {sid}: the two runs share an id, so nothing keyed on "
                f"scenario_id can be trusted. Re-run the top-up with a distinct prefix.")
            seen.add(sid)
            rows.append(rec)
    with (dest / "dataset.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for src in (Path(merge_with), run_dir):
        if (src / "manifest.json").exists():
            shutil.copy(src / "manifest.json",
                        dest / f"manifest_{src.name}.json")
    (dest / "merge.json").write_text(json.dumps(
        {"rows": len(rows), "sources": [str(Path(merge_with)), str(run_dir)],
         "note": "review artifact, not a corpus: the two runs' diversity gates were "
                 "independent, so a scenario in one was never checked against the other"},
        indent=2), encoding="utf-8")
    print(f">>> merged {len(rows)} rows into {dest / 'dataset.jsonl'}")


if __name__ == "__main__":
    fire.Fire(main)
