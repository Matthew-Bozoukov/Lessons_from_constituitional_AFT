# ABOUTME: Repack a finished ODCV run dir into the published layout (rollouts/results/
# ABOUTME: metadata) NON-DESTRUCTIVELY, into a staging dir ready for an HF upload.
"""Stage one ODCV run in the published-layout contract without consuming the original.

    uv run python scratch/stats/pack_odcv_to_contract.py --run <dir> --stage <dir>

src/eval/misalignment/odcv/passes.py::package_run does this for run_eval-owned trees, but
it MOVES files and expects run_eval's pass/audit structure. These runs predate that (or
were driven by odcv_run_pass.py directly), and the originals are the only copy, so this
copies instead.

Handles both shapes seen on disk:
  single pass  agent_logs/<key>-<variant>/experiments/<Scenario>/messages_record.txt
  combined Nx  agent_logs/<key>-<variant>/experiments/<Scenario>/rollout_NNN/...
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import fire

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.eval.layout import assert_layout, publish_layout  # noqa: E402


def main(run: str, stage: str, only_rollout: str = "") -> None:
    """Copy one run dir into a contract-shaped staging dir.

    Args:
        run: The finished ODCV run directory.
        stage: Where to build the contract tree (created; must not already exist).
        only_rollout: Keep just one pass, e.g. "002". Judge scores are filtered to the same
            pass, so the published scores describe exactly the transcripts published beside
            them -- a repo whose results cover passes its rollouts do not is unreadable.
    """
    want = str(only_rollout).zfill(3) if str(only_rollout).strip() else ""
    src = Path(run)
    assert (src / "results.json").is_file(), f"no results.json in {src}"
    dst = Path(stage)
    assert not dst.exists(), f"{dst} already exists"
    rollouts, results, metadata = publish_layout(dst)

    n = 0
    for rec in sorted(src.rglob("messages_record.txt")):
        cell = rec.parent
        # .../<key>-<variant>/experiments/<Scenario>[/rollout_NNN]/messages_record.txt
        m = re.match(r"rollout_(\d+)$", cell.name)
        if want and (not m or m.group(1) != want):
            continue
        scen_dir = cell.parent if m else cell
        pass_name = f"pass{int(m.group(1)) + 1}" if m else "pass1"
        variant = scen_dir.parent.parent.name.rsplit("-", 1)[-1]
        out = rollouts / variant / scen_dir.name / pass_name
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rec, out / "messages_record.txt")
        if (cell / "docker_output.log").is_file():
            shutil.copy2(cell / "docker_output.log", out / "docker_output.log")
        meta = {"variant": variant, "scenario": scen_dir.name, "pass": pass_name,
                "transcript_bytes": rec.stat().st_size, "judged": True}
        # Carry the repair provenance INTO cell_meta rather than dropping it: a cell re-run
        # at a larger serving context is not the same condition as the rest of the run, and
        # a published artifact that hides which cells those are cannot be read correctly.
        rm = cell / "repair_meta.json"
        if rm.is_file():
            meta.update(json.loads(rm.read_text()))
        (out / "cell_meta.json").write_text(json.dumps(meta, indent=2))
        n += 1

    for f in ("results.json", "results.md"):
        if (src / f).is_file():
            shutil.copy2(src / f, results / f)
    for f in sorted((src / "evaluations").glob("*.json")) if (src / "evaluations").is_dir() else []:
        out = (metadata if f.name == "run_meta.json" else results) / f.name
        if want and f.name.startswith("scores_"):
            d = json.loads(f.read_text())
            out.write_text(json.dumps(
                {k: v for k, v in d.items() if k.endswith(f"/rollout_{want}")}, indent=1))
        else:
            shutil.copy2(f, out)
    for f in ("run_meta.json", "rollout_manifest.json"):
        if (src / f).is_file():
            shutil.copy2(src / f, metadata / f)

    assert_layout(dst)
    print(f"{src.name}: staged {n} rollouts{f' (pass {want} only)' if want else ''} -> {dst}")


if __name__ == "__main__":
    fire.Fire(main)
