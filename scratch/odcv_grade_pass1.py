# ABOUTME: Judge a single ODCV pass when the run was stopped before its remaining passes,
# ABOUTME: reusing the runner's own combine -> judge -> package tail so nothing diverges.

"""Grade one already-complete ODCV pass.

`src.eval.misalignment.odcv.runner.run()` judges only after ALL passes finish, so killing a
2-pass run part-way through pass 2 leaves pass 1 complete on disk and ungraded. This replays
exactly the tail of run() against that one pass — same combine_passes, same odcv_judge, same
package_run — so the result is byte-for-byte what a `passes=1` run would have produced.

    uv run python scratch/odcv_grade_pass1.py --out_dir <run_eval out_dir> --pass_dir <pass>
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.eval.misalignment.odcv import odcv_judge
from src.eval.misalignment.odcv.passes import audit_pass, combine_passes, package_run
from src.utils import timestamp


def main(out_dir: str, pass_dir: str, judge_workers: int = 8,
         allow_dirty: bool = False) -> None:
    """Combine, judge and package one ODCV pass.

    Args:
        out_dir: The run_eval out_dir holding odcv_config.yaml (judging keys off that path).
        pass_dir: The single complete pass directory to grade.
        judge_workers: Concurrent judge calls.
        allow_dirty: Judge a pass that did NOT audit clean. Off by default because the
            runner drops a dirty pass for a reason: the cells that produced no transcript
            are not a random sample (they are the slow, long-trajectory ones), so the
            surviving cell set is smaller AND differently composed than a sibling arm's.
            Pass it only when you intend to report the reduced cell count explicitly.
    """
    load_dotenv()
    out, p = Path(out_dir), Path(pass_dir)
    cfg_path = out / "odcv_config.yaml"
    assert cfg_path.is_file(), f"no odcv_config.yaml under {out}"
    cfg = OmegaConf.load(cfg_path)

    audit = audit_pass(p)
    print(f">>> auditing {p.name}: clean={audit['clean']} "
          f"missing={audit.get('missing_cells')} statuses={audit.get('statuses')}")
    if not audit["clean"]:
        assert allow_dirty, (
            f"pass {p} did not audit clean; refusing to judge an incomplete pass "
            f"(missing {audit.get('missing_cells')}). Pass --allow_dirty to judge it "
            "anyway, and report the reduced cell count.")
        print(f"!!! judging a DIRTY pass: {audit.get('missing_cells')} cell(s) have no "
              "transcript and are absent from every number below")
    # audit_pass returns only what it can see in the pass dir; the runner's _run_pass adds
    # these two before handing the record on, and package_run reads `path`.
    audit["kept"] = True
    audit["path"] = str(p)
    audit.setdefault("pass_dir", p.name)

    combined = Path(cfg.output_root) / str(cfg.model_key) / f"combined1x_{timestamp()}"
    manifest = combine_passes([p], combined, str(cfg.model_key),
                              OmegaConf.to_container(cfg, resolve=True))
    print(f">>> combined {manifest['n_transcripts']} transcripts -> {combined}")

    odcv_judge.main(rollout_dir=str(combined), config=str(cfg_path),
                    max_workers=int(judge_workers), smoke=False)

    results = json.loads((combined / "results.json").read_text())
    package_run(out, str(cfg.model_key), [audit], combined)
    results["passes"] = {"requested": 1, "kept": 1, "dropped": 0,
                         "n_transcripts": manifest["n_transcripts"], "audits": [audit]}
    (out / "results_pass1.json").write_text(json.dumps(results, indent=2))
    print(f">>> wrote {out / 'results_pass1.json'}")
    for k in ("misalignment_rate", "mandated", "incentivized", "n_cells", "severity"):
        if k in results:
            print(f"    {k}: {results[k]}")


if __name__ == "__main__":
    fire.Fire(main)
