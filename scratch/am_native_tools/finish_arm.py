# ABOUTME: Finish an AM run whose generation completed but which stopped before judging (the 2026-11-23
# ABOUTME: native-tools DA+tools arm): judge, validate, aggregate, build rollouts, publish -- no GPU.
# Run: uv run python scratch/am_native_tools/finish_arm.py <out_dir> [--no-push]
"""The same steps, in the same order, as the tail of src/eval/misalignment/agentic_misalignment/runner.py
run() and src/eval/run_eval.py's per-target epilogue, applied to a run dir whose generation finished.
The run dir's own pre-run run_meta.json (written by run_eval at launch) supplies the config, target,
revisions and name facets, so the published name and card are what run_eval would have produced. What
differs is recorded: `finished_by` (this script, its commit, and why) in run_meta and the summary."""

import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from src.eval.layout import publish_layout, run_tags
from src.eval.misalignment.agentic_misalignment import aggregate_eval, build_rollouts, runner
from src.eval.run_eval import _card_fields, _publish
from src.infra.endpoints.openrouter import provider_pin
from src.utils import git_sha

out_dir = Path(sys.argv[1]).resolve()
push = "--no-push" not in sys.argv
rollouts_dir, results_out, metadata_dir = publish_layout(out_dir)
results_dir = results_out / "harness"
meta_path = out_dir / "run_meta.json"
meta = json.loads(meta_path.read_text())
cfg = OmegaConf.create(meta["config"])
native = cfg.get("protocol") == runner.NATIVE_TOOLS
samples = int(cfg["global"].samples_per_condition)
model_id = f"vllm/{meta['model_key']}"
harness = yaml.safe_load((metadata_dir / "harness_config.yaml").read_text())
condition_manifest = json.loads((metadata_dir / "condition_manifest.json").read_text())
protocol = json.loads((metadata_dir / "scoring_protocol.json").read_text())

# Generation must be complete before anything is judged (runner.run's own gate).
runner.validate_results(results_dir, model_id, samples, native_tools=native)

finished_by = {
    "script": "scratch/am_native_tools/finish_arm.py", "git_sha": git_sha(),
    "why": ("generation finished on the pod (run_eval launch at run_meta.git_sha) but the run stopped "
            "before judging on one empty no-call reply; judged + published here, on the laptop, after "
            "VENDORED PATCH 22 made an empty no-call reply a scored 'no action' trial (option 1, 2026-11-24)"),
}
# The serving stack: this arm was served by the same vLLM process (one invocation, LoRA swap) as the
# da-15 arm, whose published run_meta recorded it after its own run.
sibling = json.load(open(hf_hub_download(
    "dougalldeepmind/2026-11-23-am-native-tools-qwen36-0-da-15", "metadata/run_meta.json",
    repo_type="dataset")))
meta["serving_env"] = sibling.get("serving_env")
meta["serving_env_from"] = "dougalldeepmind/2026-11-23-am-native-tools-qwen36-0-da-15 (same server process)"
meta["finished_by"] = finished_by
meta_path.write_text(json.dumps(meta, indent=2))

load_dotenv()
judge = str(cfg.get("classifier_model"))
env = os.environ | {"PYTHONUTF8": "1", "OPENROUTER_PROVIDER_PIN": json.dumps(provider_pin(judge))}
runner._step([sys.executable, "scripts/classify_results.py", "--results-dir", str(results_dir),
              "--classifier-model", judge], env)
runner.validate_results(results_dir, model_id, samples, classified=True, native_tools=native)

summary_path = results_out / "misalignment_summary.json"
aggregate_eval.main(results_dir=str(results_dir), label=harness["experiment_id"], out=str(summary_path))
build_rollouts.main(results_dir=str(results_dir), out=str(rollouts_dir))
(rollouts_dir / "run_meta.json").rename(metadata_dir / "rollout_build_meta.json")
summary = json.loads(summary_path.read_text())
summary["condition_protocol"] = condition_manifest["protocol"]
summary["conditions"] = condition_manifest["conditions"]
summary["classifier_protocol"] = runner._CLASSIFIER_PROTOCOL
if native:
    summary["tool_use"] = runner.tool_use_summary(results_dir)
summary["scoring_protocol"] = protocol
summary["finished_by"] = finished_by
summary_path.write_text(json.dumps(summary, indent=2))

summary = {"target": meta["target"], "mode": meta["mode"], **summary}
name = "agentic_misalignment"
url = _publish(
    out_dir, name=name, model_key=meta["model_key"], mode=meta["mode"], target=meta["target"],
    summary=summary, push=push, variant=meta["variant"],
    card=_card_fields(
        name, cfg, meta["command"] + "  # finished by scratch/am_native_tools/finish_arm.py",
        experiment=f"{name} ({meta['variant']}) eval of {meta['target']} (mode={meta['mode']})",
        models=json.dumps({"target": meta["target"], "target_revision": meta["target_revision"],
                           "base": meta["base_model"], "base_revision": meta["base_model_revision"]}),
        source_revision=meta["git_sha"]),
    tags=run_tags(name, meta["model_key"], meta["mode"], variant=meta["variant"]))
print("overall", summary["overall"], "\ntool_use overall", summary.get("tool_use", {}).get("overall"))
print("url", url)
