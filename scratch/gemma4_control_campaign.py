# ABOUTME: Launch the pinned Gemma control smoke or full run through the canonical trainer.
# ABOUTME: Smoke includes the longest rows and tool trajectories; full launch requires its saved finite-loss evidence.
from __future__ import annotations
import json
import math
import os
import subprocess
from pathlib import Path
import fire
from omegaconf import OmegaConf


def main(phase: str, config: str = "scratch/gemma4_control_launch.yaml"):
    cfg = OmegaConf.load(config)
    root = Path("output/gemma4-control")
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               TOKENIZERS_PARALLELISM="false")
    command = ["uv", "run", "--frozen", "torchrun", "--nproc_per_node=2",
               "scripts/train/train_lora.py", "--config", cfg.recipe, *cfg.overrides]
    if phase == "smoke":
        audit = json.loads((root / "audit.json").read_text())
        assert audit["verified"] == audit["rows"] == 1832
        rows = audit["evidence"]
        chosen = [r["row"] for r in sorted(rows, key=lambda r: r["tokens"], reverse=True)[:2]]
        chosen += [r["row"] for r in rows if r["tool"] and r["row"] not in chosen][:6]
        chosen += [r["row"] for r in rows if r["row"] not in chosen][:16-len(chosen)]
        command += ["--smoke", "output_dir=output/gemma4-control/smoke", "push=false",
                    "smoke_indices=" + json.dumps(chosen, separators=(",", ":"))]
    elif phase == "train":
        assert (root / "smoke_passed.json").exists(), "Run and verify the smoke first"
    else:
        raise ValueError("phase must be smoke or train")
    (root / f"{phase}_command.json").write_text(json.dumps(command, indent=2))
    print("Launching", command, flush=True)
    completed = subprocess.run(command, env=env)
    (root / f"{phase}_exit.json").write_text(json.dumps({"returncode": completed.returncode}))
    completed.check_returncode()
    if phase == "smoke":
        metas = list((root / "smoke").glob("*/run_meta.json"))
        assert len(metas) == 1, "Smoke output must be fresh"
        meta = json.loads(metas[0].read_text())
        logged = [r for r in meta["log_history"] if "loss" in r]
        assert len(logged) == 2 and all(math.isfinite(r["loss"]) and r["loss"] > 0 for r in logged)
        assert all(math.isfinite(r["grad_norm"]) and r["grad_norm"] > 0 for r in logged)
        assert meta["world_size"] == 2 and meta["n_examples"] == 16
        from safetensors import safe_open
        with safe_open(metas[0].parent / "adapter/adapter_model.safetensors", framework="pt") as f:
            b = [k for k in f.keys() if "lora_B" in k]
            assert b and all(f.get_tensor(k).isfinite().all().item() for k in f.keys())
            assert any(f.get_tensor(k).abs().max().item() > 0 for k in b), "No learned adapter update"
        (root / "smoke_passed.json").write_text(json.dumps({"meta": str(metas[0]), "steps": logged}, indent=2))
        print("SMOKE_PASSED", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
