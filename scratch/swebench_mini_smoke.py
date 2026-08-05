#!/usr/bin/env python3
# ABOUTME: Wiring smoke for the SWE-bench baseline — 2 instances via OpenRouter, then real
# ABOUTME: grading. Proves the plumbing end to end without a GPU. NOT a baseline number.

"""Does the whole path work, before we rent anything?

This exercises every moving part except our own vLLM endpoint: the pinned agent environment,
the untouched official config plus overlay, the instance filter, docker containers, preds.json,
the pinned grading harness, and the metrics. It substitutes an OpenRouter model for the served
target, so it needs no GPU and no serving stack.

What it deliberately does NOT test: tool-call parsing on our own vLLM server, thinking-mode
pinning, and the 65536-context path. Those need the box.

The step limit is cut to keep the smoke cheap. That alone makes any resolved rate here
meaningless as a score — this run answers "does it work", never "how good is the model".

    uv run scratch/swebench_mini_smoke.py
    uv run scratch/swebench_mini_smoke.py --model openrouter/openai/gpt-4.1-mini --n 2
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.capabilities.swebench_mini import agent, grade as grading, images, metrics, subset  # noqa: E402
from src.eval.docker import docker_preflight  # noqa: E402
from src.utils import timestamp  # noqa: E402

DATASET = "princeton-nlp/SWE-Bench_Verified"


def main(model: str = "openrouter/google/gemini-3-flash-preview", n: int = 2,
         step_limit: int = 25, workers: int = 2, seed: int = 0,
         out_root: str = "output/swebench_mini_smoke") -> None:
    """Run the 2-instance wiring smoke, rollouts through grading.

    Args:
        model: A litellm model id. OpenRouter needs OPENROUTER_API_KEY in .env.
        n: Instances, taken from the SAME stratified order the real run uses.
        step_limit: Cut well below upstream's 250 to bound the spend. Why this makes the
            result a wiring check rather than a score — see the module docstring.
        workers: Parallel instances (each holds one container).
        seed: Subset seed; matches the eval config's default so the smoke instances are the
            first two of the real 10% draw.
        out_root: Output directory root.
    """
    load_dotenv()
    docker_preflight()
    if "openrouter/" in model and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("no OPENROUTER_API_KEY in .env — needed for the smoke's model calls")

    out_dir = Path(out_root) / timestamp()
    (out_dir / "rollouts").mkdir(parents=True, exist_ok=True)

    instances, revision = subset.load_instances(DATASET, "test")
    chosen = subset.select(instances, seed, n=n)
    selection = subset.summarize_selection(chosen, len(instances), seed, DATASET, revision)
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2))
    print(f">>> smoke on {selection['n_selected']} instances: {selection['instance_ids']}")
    images.pull_all(chosen, workers=workers)

    # Same shape as the real overlay (network off, official config untouched underneath), plus
    # the reduced step limit. Written here rather than in agent.py because it is a smoke-only
    # deviation and must not leak into the baseline's code path.
    overlay = out_dir / "smoke_overlay.yaml"
    OmegaConf.save(OmegaConf.create({
        "agent": {"step_limit": step_limit},
        "environment": {"run_args": ["--rm", "--network", "none"]},
    }), overlay)

    official_config = agent.official_config_path()
    print(f">>> scaffold {agent.agent_version()} | config sha256 "
          f"{agent.config_sha256(official_config)[:16]}… | model {model}")

    code = agent.run_rollouts(
        agent.rollout_command(dataset=DATASET, split="test",
                              filter_regex=subset.id_filter_regex(chosen), workers=workers,
                              model_name=model, rollouts_dir=out_dir / "rollouts",
                              overlay=overlay, official_config=official_config),
        agent.rollout_env(registry=agent.write_cost_registry(out_dir, model),
                          global_config_dir=out_dir / "mini_global_config"),
        out_dir / "rollouts.log")
    print(f">>> rollouts exited {code}")

    preds_path = out_dir / "rollouts" / "preds.json"
    rollout = metrics.rollout_summary(metrics.load_preds(preds_path),
                                      selection["instance_ids"], out_dir / "rollouts")
    print(json.dumps(rollout, indent=2))
    if not rollout["n_with_patch"]:
        print("!!! no non-empty patches — grading anyway so the harness path is still "
              f"exercised; read {out_dir / 'rollouts.log'} for why")

    report = grading.grade(preds_path=preds_path, selected_ids=selection["instance_ids"],
                           dataset=DATASET, revision=revision,
                           run_id=f"smoke_{selection['subset_hash']}",
                           grade_dir=out_dir / "grading", max_workers=workers)
    scores = metrics.resolution_summary(report, selection["instance_ids"])
    (out_dir / "results.json").write_text(json.dumps(
        {"selection": selection, **rollout, **scores, "harness": report["_harness"],
         "smoke": True, "step_limit": step_limit, "model": model}, indent=2))

    print("\n=== SMOKE (wiring check, NOT a baseline number) ===")
    print(json.dumps(scores, indent=2))
    print(f"\nharness {report['_harness']['version']} | artifacts in {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
