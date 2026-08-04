# ABOUTME: Eval-framework entrypoint for the agentic-misalignment honeypots: drives the
# ABOUTME: vendored harness against a served target, judges via OpenRouter, aggregates rates.

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from src.eval.misalignment.agentic_misalignment import aggregate_eval, build_rollouts
from src.utils import timestamp

_HARNESS = Path(__file__).parents[1] / "third_party" / "agentic-misalignment"


def _harness_config(cfg, model_id: str, expid: str) -> dict:
    """Rewrite the eval config for one served target (pure; unit-tested offline)."""
    d = OmegaConf.to_container(cfg, resolve=True)
    d["experiment_id"] = expid
    g = d["global"]
    old_models = g.get("models") or []
    g["models"] = [model_id]
    conc = g.get("concurrency", {})
    if "models" in conc:
        # Per-model concurrency was keyed by the old served name; re-key for this target.
        values = [conc["models"][m] for m in old_models if m in conc["models"]]
        conc["models"] = {model_id: values[0] if values else 32}
    return d


def _step(argv: list[str], env: dict) -> None:
    subprocess.run(argv, cwd=_HARNESS, env=env, check=True)


def run(target, cfg, out_dir: Path) -> dict:
    """Run the honeypot suite against a ServedTarget (CLAUDE.md contract).

    The vendored harness stays untouched: it is driven as subprocesses with the served
    endpoint injected via VLLM_BASE_URL (its patched vllm/ provider reads that; thinking
    mode is already pinned into the server's chat template, so no harness-side flag).

    Returns:
        The misalignment summary (per-condition rates + overall).
    """
    expid = f"{target.spec.model_key}_{timestamp()}"
    model_id = f"vllm/{target.model_name}"
    harness_cfg = out_dir / "harness_config.yaml"
    harness_cfg.write_text(yaml.safe_dump(_harness_config(cfg, model_id, expid)))

    env_file = _HARNESS / ".env"
    if not env_file.exists():
        env_file.symlink_to(Path(".env").resolve())
    env = os.environ | {"VLLM_BASE_URL": target.base_url, "VLLM_API_KEY": "EMPTY"}

    judge = str(cfg.get("classifier_model", "anthropic/claude-sonnet-4.5"))
    _step([sys.executable, "scripts/generate_prompts.py", "--config", str(harness_cfg.resolve())], env)
    _step([sys.executable, "scripts/run_experiments.py", "--config", str(harness_cfg.resolve()),
           "--no-classification"], env)
    _step([sys.executable, "scripts/classify_results.py", "--results-dir", f"results/{expid}",
           "--classifier-model", judge], env)

    results_dir = _HARNESS / "results" / expid
    summary_path = out_dir / "misalignment_summary.json"
    aggregate_eval.main(results_dir=str(results_dir), label=expid, out=str(summary_path))

    # "Logs" means ROLLOUTS: stitch prompts + responses into self-contained transcripts,
    # then move the whole result tree under out_dir so nothing lives only in the harness.
    build_rollouts.main(results_dir=str(results_dir), out=str(out_dir / "rollouts"))
    shutil.move(str(results_dir), str(out_dir / "results"))

    return json.loads(summary_path.read_text())
