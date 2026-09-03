#!/usr/bin/env python3
# ABOUTME: Verify a machine can grade SWE-bench at all, using gold patches and no model —
# ABOUTME: run this on every fresh grading host BEFORE trusting a single pass@1.

"""Is this host's grading environment real?

Submits each instance's own reference patch through the pinned official harness. A correct
setup resolves 100% of them. Anything less means the environment is broken — wrong images,
docker misconfigured, tests not executing — and every model score produced there would be
wrong in the same direction, with nothing in the output to say so.

Costs a few minutes of CPU and no API credit. Instances come from the same stratified order
the real runs use, so the images it pulls are ones a real run will need anyway.

    uv run scripts/eval/swebench_mini_check_env.py            # 2 instances
    uv run scripts/eval/swebench_mini_check_env.py --n 5
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
from omegaconf import OmegaConf

from src.eval.capabilities.swebench_mini import grade as grading
from src.eval.capabilities.swebench_mini import images, subset
from src.eval.docker import docker_preflight
from src.utils import timestamp


def main(n: int = 2, config: str = "configs/eval/swebench_mini.yaml",
         max_workers: int = 2, out_root: str = "output/swebench_mini_env_check") -> None:
    """Run the gold-patch environment check.

    Args:
        n: How many instances to verify. 2 is enough to catch a broken environment; more
            costs only time and disk.
        config: Eval config — supplies the dataset, split and image namespace.
        max_workers: Parallel instances.
        out_root: Where the report and logs land.
    """
    docker_preflight()
    cfg = OmegaConf.load(config)
    out_dir = Path(out_root) / timestamp()

    instances, revision = subset.load_instances(str(cfg.dataset), str(cfg.split))
    chosen = subset.select(instances, int(cfg.subset.seed), n=n)
    ids = sorted(row["instance_id"] for row in chosen)
    print(f">>> gold check on {len(ids)} instances: {ids}")
    print(f">>> dataset {cfg.dataset}@{revision[:12]}")

    # Same reason as the real run: upstream's container-start timeout cannot cover a cold pull.
    images.pull_all(chosen, workers=max_workers)

    result = grading.verify_environment(
        dataset=str(cfg.dataset), split=str(cfg.split), instance_ids=ids, out_dir=out_dir,
        max_workers=max_workers, cache_level=str(cfg.grading.cache_level),
        namespace=str(cfg.grading.namespace))
    result |= {"dataset_revision": revision}
    (out_dir / "env_check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    if not result["passed"]:
        raise SystemExit(
            f"\nGOLD CHECK FAILED: {result['n_resolved']}/{result['n_requested']} resolved.\n"
            f"  Unresolved with the REFERENCE patch: {result['unresolved_gold']}\n"
            "  This host cannot grade correctly — do not trust any pass@1 produced here.\n"
            f"  Read {out_dir / 'gold_check.log'} and the per-instance logs beside it.")
    print(f"\nGOLD CHECK PASSED — {result['n_resolved']}/{result['n_requested']} resolved. "
          "This host grades correctly.")


if __name__ == "__main__":
    fire.Fire(main)
