# ABOUTME: Runs the standard ODCV lifecycle with one counted pilot cell before the remaining cells.
# ABOUTME: Disables shared-daemon pruning and releases the owned GPU before local judging.
import json
from pathlib import Path

from omegaconf import OmegaConf

from src.eval import run_eval
from src.eval.misalignment.odcv import odcv_rollout, runner
from src.eval.misalignment.odcv.odcv import VARIANTS, scenario_names


def run(target, cfg, out_dir):
    original_main = odcv_rollout.main
    original_prune = runner._prune_networks
    original_combine = runner.combine_passes
    first = True

    def staged_main(config, smoke=False, resume="", **overrides):
        nonlocal first
        if resume or not first:
            return original_main(config=config, smoke=smoke, resume=resume, **overrides)
        first = False
        full = OmegaConf.load(config)
        bench = Path(full.bench_dir)
        excluded = set(full.get("exclude_scenarios", []))
        cells = [f"{v}/{s}" for v in VARIANTS for s in scenario_names(bench, v)
                 if f"{v}/{s}" not in excluded]
        assert cells
        pilot = OmegaConf.merge(full, {"concurrency": 1,
                                      "exclude_scenarios": sorted(excluded | set(cells[1:]))})
        pilot_path = out_dir / "pilot_config.yaml"
        OmegaConf.save(pilot, pilot_path)
        print(f">>> Counted pilot cell: {cells[0]}; remaining {len(cells)-1}", flush=True)
        pass_dir = original_main(config=str(pilot_path), smoke=False)
        records = list(pass_dir.rglob("messages_record.txt"))
        assert len(records) == 1 and "role: assistant" in records[0].read_text(encoding="utf-8"), (
            "Pilot produced no assistant transcript; refusing broad dispatch")
        manifest = json.loads((pass_dir / "rollout_manifest.json").read_text(encoding="utf-8"))
        (out_dir / "pilot_receipt.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        # Resume caches this exact pilot; it is one of the requested cells, never an extra pass.
        return original_main(config=config, smoke=False, resume=str(pass_dir), **overrides)

    def combine_and_release(*args, **kwargs):
        manifest = original_combine(*args, **kwargs)
        # All model outputs are now local and copied into the combined transcript tree.
        target.release()
        return manifest

    odcv_rollout.main = staged_main
    runner._prune_networks = lambda: None
    runner.combine_passes = combine_and_release
    try:
        return runner.run(target, cfg, out_dir)
    finally:
        odcv_rollout.main = original_main
        runner._prune_networks = original_prune
        runner.combine_passes = original_combine


if __name__ == "__main__":
    run_eval.main(runner=run)
