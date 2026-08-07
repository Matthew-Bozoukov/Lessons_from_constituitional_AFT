# ABOUTME: Eval-framework entrypoint for the Arena-Hard capability eval: generate the served
# ABOUTME: target's answers, then judge them side-by-side against a baseline arm's answers.

from __future__ import annotations

import shutil
from pathlib import Path

from src.huggingface import hf_download
from omegaconf import OmegaConf

from src.eval.capabilities.arena_hard import arena_hard_gen, arena_hard_judge


def _fetch_reference(reference: str, dest: Path) -> None:
    """Place the baseline arm's answers where the vendored bench reads them.

    Args:
        reference: A local answers jsonl, or `repo_id::path_in_repo` on HF.
        dest: `<vendor_dir>/data/<bench>/model_answer/<baseline_arm>.jsonl`.
    """
    src = (Path(hf_download(*reference.split("::", 1), repo_type="dataset"))
           if "::" in reference else Path(reference))
    assert src.exists(), f"reference answers not found: {reference}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)


def run(target, cfg, out_dir: Path, *, reference: str = "") -> dict:
    """Run the Arena-Hard eval for one ServedTarget (CLAUDE.md contract).

    Side-by-side by construction: the target's answers are judged against the config's
    `baseline_arm` answers, supplied via the `reference` kwarg (run_eval.py's
    --reference; still an answers ARTIFACT here, pending the answer-cache migration).
    The summary is the judged score block; the full report stays a separate curation
    step over the accumulated arms (`scratch/reports/arena_hard_report.py`).

    Returns:
        Paths of the generated answers and judging output for this arm.
    """
    cfg = OmegaConf.merge(cfg)  # private copy
    arm_name = target.spec.model_key
    baseline = str(cfg.baseline_arm)
    assert reference, "arena_hard is judged against a baseline: pass --reference"

    bench_answers = Path(str(cfg.vendor_dir)) / "data" / str(cfg.bench_name) / "model_answer"
    _fetch_reference(str(reference), bench_answers / f"{baseline}.jsonl")

    # The served target joins the ladder as a dynamic arm; decoding params are global in
    # this config (parity across arms is load-bearing), so no per-arm settings are needed.
    arms = OmegaConf.to_container(cfg.arms, resolve=True)
    if arm_name not in {a["name"] for a in arms}:
        arms.append({"name": arm_name, "adapter": target.spec.hf_path,
                     "role": "target", "synthetic_fraction": None})
    cfg.arms = arms
    cfg.output_dir = str(out_dir)
    cfg_path = out_dir / "arena_hard_config.yaml"
    OmegaConf.save(cfg, cfg_path)

    smoke = bool(cfg.get("smoke", False))
    arena_hard_gen.main(config=str(cfg_path), arm=arm_name,
                        served_model=target.model_name, endpoint=target.base_url,
                        api_key=target.api_key, smoke=smoke)
    arena_hard_judge.main(config=str(cfg_path), mode="judge", arm=arm_name)

    return {"arm": arm_name, "baseline": baseline,
            "answers": str(bench_answers / f"{arm_name}.jsonl"),
            "judging": str(out_dir / "judging")}
