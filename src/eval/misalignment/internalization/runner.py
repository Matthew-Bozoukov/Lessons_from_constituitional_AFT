# ABOUTME: Eval-framework entrypoint for the constitution-internalization proxy eval: routes
# ABOUTME: the served target into the self-contained pipeline, then repacks to the layout contract.

from __future__ import annotations

import json
import shutil
from pathlib import Path


def run(target, cfg, out_dir: Path) -> dict:
    """Run the internalization eval against a ServedTarget (CLAUDE.md contract).

    The pipeline writes its own `runs/<run_id>/` working tree; run() repacks it into the
    published layout (src/eval/layout.py). Completions are joined with their itemset
    prompts on the way into rollouts/ — the pipeline's completions.jsonl carries only
    item_id + text, and a rollout must be self-contained (CLAUDE.md).

    Args:
        target: ServedTarget from src/infra/endpoints/vllm.py.
        cfg: configs/eval/internalization.yaml + CLI overrides. `internal_config` names the
            self-contained config under control/configs/; `set` passes dotted overrides
            through to it; `max_items` caps items for a quick pass.
        out_dir: Per-target run directory owned by run_eval.py.

    Returns:
        Summary dict: run dir, item counts and the pipeline's own summary text.
    """
    from src.eval.layout import publish_layout
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
    summary_text = result.summary()  # before the repack moves the files it may read

    rollouts, results, metadata = publish_layout(out_dir)
    run_dir = Path(result.run_dir)
    manifest = json.loads((run_dir / "run_meta.json").read_text())

    # Join prompts in: items.jsonl lives in the itemset dir OUTSIDE out_dir, referenced
    # only by itemset_id — without the join the published rollouts are unreadable alone.
    items_file = (Path(str(internal["itemset"]["dir"])) / str(manifest["itemset_id"])
                  / "items.jsonl")
    items = {row["item_id"]: row
             for row in map(json.loads, items_file.read_text().splitlines()) if row}
    with (rollouts / "completions.jsonl").open("w", encoding="utf-8") as fh:
        for line in (run_dir / "completions.jsonl").read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            fh.write(json.dumps({"prompt": items[row["item_id"]]["prompt"], **row},
                                ensure_ascii=False) + "\n")

    (run_dir / "results.jsonl").rename(results / "results.jsonl")
    if (run_dir / "results.parquet").is_file():
        (run_dir / "results.parquet").rename(results / "results.parquet")
    # The pipeline's own manifest; the bare run_meta.json name is the framework's.
    (run_dir / "run_meta.json").rename(metadata / "pipeline_manifest.json")
    shutil.rmtree(out_dir / "runs")

    return {"run_dir": str(out_dir), "summary": summary_text}
