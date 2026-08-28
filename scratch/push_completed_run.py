# ABOUTME: Push a completed eval run dir to HF when run_eval's own epilogue push failed,
# ABOUTME: rebuilding the card fields from the run's metadata/run_meta.json.

"""Recover the push for a run whose epilogue died after the results were written.

run_eval pushes as its last step; a 403 on the org (or any other push-side failure) leaves a
complete, contract-shaped run dir on disk with nothing on the Hub. This re-does exactly that
step — same repo-id convention, same required card fields, same discovery tags — reading what
the epilogue would have known from metadata/run_meta.json instead of the live ServedTarget.

    uv run python scratch/push_completed_run.py --run_dir <dir> --name strongreject --org matboz
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.huggingface import push_run_dir


def main(run_dir: str, name: str, org: str, private: bool = False) -> None:
    """Push one completed run dir, mirroring run_eval's epilogue.

    Args:
        run_dir: The eval run directory (must already hold rollouts/ results/ metadata/).
        name: The registered eval name, e.g. strongreject.
        org: HF namespace to publish under.
        private: Publish privately; runs are public by default so the dashboard can read them.
    """
    load_dotenv()
    d = Path(run_dir)
    meta = json.loads((d / "metadata" / "run_meta.json").read_text())
    summary = json.loads((d / "results" / "results.json").read_text())
    # write_run_meta flattens its `extra` kwargs onto the top level, so target/base_model/
    # mode/command sit beside git_sha rather than under an "extra" key.
    cfg = meta.get("config", {})
    target = meta.get("target") or summary["target"]
    mode = meta.get("mode") or summary.get("mode", "")
    model_key = d.parent.name

    repo_id = (f"{org}/{date.today().isoformat()}-{name.replace('_', '-')}-"
               f"{model_key.replace('_', '-')}")
    fields = {
        "experiment": f"{name} eval of {target} (mode={mode})",
        "date_generated": date.today().isoformat(),
        "constitution": str(cfg.get("constitution", "none")),
        "source_repo": f"teaching_claude_why_replication @ {meta['git_sha']}",
        "models": f"target={target} base={meta.get('base_model', '')}",
        "generation_config": json.dumps(cfg.get("generation", {})),
        "schema": ("rollouts/: self-contained transcripts; results/: results.json + judge/eval "
                   "outputs; metadata/: run_meta.json + config + provenance"),
        "provenance": meta["command"],
    }
    url = push_run_dir(d, repo_id, fields, private=private,
                       front_matter={"tags": ["eval-run", f"eval:{name}",
                                              f"model:{model_key}", f"mode:{mode}"]})
    print(f">>> pushed {url}")


if __name__ == "__main__":
    fire.Fire(main)
