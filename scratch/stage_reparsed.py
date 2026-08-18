# ABOUTME: Stage the re-parsed sycophancy summaries into the output/<eval>/<arm>/<ts>/ layout
# ABOUTME: the plot script reads. Run: uv run python scratch/stage_reparsed.py

"""The re-parse writes one JSON per arm; the plotter expects run directories.

Trace statistics are carried over from the pod's own results.json (fetched from the Hub
alongside the rollouts) because the re-parse recomputes the SCORES, not the generation
telemetry — think-character means and truncation rates are properties of the run and do not
change when the parser does.
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from scratch.reparse_sycophancy import REPOS

KEYS = {
    "CR": "qwen3_6-27b-lora-t2-9284-courtroom716-r64-dynbatch",
    "PC": "qwen3_6-27b-lora-t2-9284-peercritique716-r64-dynbatch",
    "DA": "qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch",
    "T2": "qwen3_6-27b-lora-table2-only-9284-r64",
    "base": "Qwen3_6-27B",
}


def main(src: str = "output/sycophancy_reparsed", dest: str = "output/sycophancy") -> str:
    written = []
    for arm, key in KEYS.items():
        summary = json.loads((Path(src) / f"{arm}.json").read_text())
        pod = json.loads(Path(hf_hub_download(REPOS[arm], "results.json",
                                              repo_type="dataset")).read_text())
        for field in ("trace_turn1", "trace_turn2", "target", "mode", "dataset"):
            if field in pod:
                summary[field] = pod[field]
        out = Path(dest) / key / "20260818_reparsed"
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(summary, indent=2))
        written.append(str(out))
    return "staged:\n" + "\n".join(written)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
