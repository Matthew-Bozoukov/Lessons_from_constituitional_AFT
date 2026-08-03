# ABOUTME: Eval-framework entrypoint for the constitution-internalization proxy eval:
# ABOUTME: routes the served target into the self-contained pipeline, returns its summary.

from __future__ import annotations

from pathlib import Path


def run(target, cfg, out_dir: Path) -> dict:
    """Run the internalization eval against a ServedTarget (CLAUDE.md contract).

    Args:
        target: ServedTarget from src/endpoints/vllm_server.py.
        cfg: configs/eval/internalization.yaml + CLI overrides. `internal_config` names the
            self-contained config under control/configs/; `set` passes dotted overrides
            through to it; `max_items` caps items for a quick pass.
        out_dir: Per-target run directory owned by run_eval.py.

    Returns:
        Summary dict: run dir, item counts and the pipeline's own summary text.
    """
    from src.eval.misalignment.internalization.cli import _load
    from src.eval.misalignment.internalization.pipeline.run import run_eval

    internal = _load(
        str(cfg.get("internal_config", "base.yaml")),
        cfg.get("set"),
        bool(cfg.get("smoke", False)),
        **{
            "target.provider": "vllm",
            "target.model": target.model_name,
            "target.base_url": target.base_url,
            "output_dir": str(out_dir),
        },
    )
    result = run_eval(internal, max_items=int(cfg.get("max_items", 0)))
    return {"run_dir": str(result.run_dir), "summary": result.summary()}
