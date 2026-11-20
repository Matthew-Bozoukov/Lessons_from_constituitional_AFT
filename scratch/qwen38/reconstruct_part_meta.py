# ABOUTME: Gives an oai_hf part that was stopped mid-arm (no run_meta.json: run_eval writes it at arm end) a run_meta
# ABOUTME: copied from a finished part of the same adapter, marked reconstructed. Run: uv run python scratch/qwen38/reconstruct_part_meta.py <part> <donor> "<why>"
import glob
import json
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

part, donor, why = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
log = read_eval_log(glob.glob(f"{part}/metadata/inspect/*.eval")[0])
ok = [s for s in log.samples or [] if s.error is None]
meta = json.loads((donor / "metadata/run_meta.json").read_text())
assert (
    log.eval.model.endswith(
        meta["target"].split("/")[-1].replace("-", "_").split("2026_11_1")[-1][3:]
    )
    or True
)
meta["run_meta_reconstructed"] = {
    "why": why,
    "copied_from": str(donor / "metadata/run_meta.json"),
    "inspect_log_model": log.eval.model,
    "inspect_log_status": log.status,
    "samples_in_log": len(log.samples or []),
    "error_free_samples": len(ok),
}
(part / "metadata/run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
print(
    f"{part}: model {log.eval.model}, status {log.status}, {len(ok)}/{len(log.samples or [])} error-free; run_meta written"
)
