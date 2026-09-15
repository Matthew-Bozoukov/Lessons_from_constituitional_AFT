# ABOUTME: Prepare and launch one adapter's missing-only recovery on an owned RunPod GPU.
# ABOUTME: Run: uv run scratch/delegated_harm/recover_missing.py prepare|launch control|da --root <path>.
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

from scratch.delegated_harm.recovery import AuthorValidator, read, recover_saved_author
from src.eval.misalignment.delegated_harm.source import prepare, save
from src.infra import runpod
from src.infra.endpoints.vllm import resolve_target, native_context_window
from scratch.delegated_harm.launch import MODELS, account


def prepare_arm(arm, root):
    originals = {
        "control": "output/delegated_harm/2026-09-11_11-53-22_control/runs/2026-09-11_qwen3_6_27b_lora_9284_numina_control_716_r64_125934",
        "da": "output/delegated_harm/2026-09-11_11-41-32_da/runs/2026-09-11_qwen36_lora_table2_9284_difficult_advice_chunk_only_702_rank_64_dynbatch_125021"}
    source = Path(originals[arm]).resolve()
    cfg = OmegaConf.load("configs/eval/delegated_harm.yaml")
    scenarios, _, notes, _ = prepare(cfg)
    cfg.author.allow_near_copy = True
    cfg.author.max_tokens = 32768
    cfg.author.attempts = 4
    cfg.judge.budget_usd = None
    cfg.judge.author_max_tokens = 8192
    cfg.serving.context_window = 131072
    assert native_context_window("Qwen/Qwen3.6-27B") >= cfg.serving.context_window
    spec = resolve_target(MODELS[arm])
    assert spec.revision == cfg.expected_revisions[MODELS[arm]]
    assert spec.base_revision == cfg.expected_base_revision
    rows = [read(f) for f in (source / "results/episodes").glob("*.json")]
    missing = sorted(r["id"] for r in rows if not r.get("metrics"))
    assert len(missing) == {"control": 134, "da": 69}[arm]
    root.mkdir(parents=True, exist_ok=True)
    validator = AuthorValidator(cfg.judge, root / "author_validation")
    def one(f):
        record = read(f)
        result = recover_saved_author(record, scenarios[record["scenario"]],
                                      notes[record["scenario"]]["clear"], validator)
        print(f"{arm} author {f.stem}: {'ready' if result['accepted'] else 'needs same-adapter repair'}", flush=True)
        return f.stem, result
    with ThreadPoolExecutor(max_workers=4) as pool:
        authors = dict(pool.map(one, sorted((source / "metadata/authors").glob("*.json"))))
    save(root / "prepared_authors.json", authors)
    prior_sources = read("output/delegated_harm/2026-09-11-delegated-harm-comparison/2026-09-11-delegated-harm-comparison_results.json")["sources"]
    cfg.recovery = {"source_run": source.as_posix(), "source_hf": prior_sources[arm],
                    "missing_ids": missing, "prepared_authors": (root / "prepared_authors.json").resolve().as_posix(),
                    "output_limit_tokens": 32768, "turn_limit": 120,
                    "description": "One missing-only recovery attempt. Preserve every original completed observation; allow near-copy authored requests; expand context to 131072; count tokens before every turn. Only prior output/turn cutoffs receive higher corresponding limits."}
    cfg.output_root = (root / "runs").resolve().as_posix()
    OmegaConf.save(cfg, root / "config.yaml")
    save(root / "preparation.json", {"arm": arm, "planned": len(missing),
         "requests_ready": sum(r["accepted"] for r in authors.values()),
         "requests_need_gpu_repair": [k for k, r in authors.items() if not r["accepted"]]})
    print(f"{arm}: preparation finished, {len(missing)} missing episodes", flush=True)


def launch(arm, root):
    cfg = OmegaConf.load(root / "config.yaml")
    assert read(root / "preparation.json")["arm"] == arm
    statefile = root / "controller.json"
    if statefile.exists():
        raise RuntimeError("Recovery launch already has a controller; inspect it instead of duplicating rental")
    snapshot = account()
    if float(snapshot["myself"]["clientBalance"]) < 30:
        raise RuntimeError("Insufficient shared RunPod balance for bounded recovery")
    gpu = "NVIDIA H100 80GB HBM3"
    quote = next(float(g["securePrice"]) for g in snapshot["gpuTypes"] if g["id"] == gpu)
    hours = 3.5
    if quote * hours > 20:
        raise RuntimeError("Unexpected GPU price; recovery GPU allocation is $20 per adapter")
    pair = runpod.default_keypair()
    assert pair
    state = {"arm": arm, "model": MODELS[arm], "pid": os.getpid(), "status": "provisioning",
             "recovery": True, "max_hours": hours, "quoted_gpu_hourly_usd": quote,
             "planned_missing": len(cfg.recovery.missing_ids), "api_spending_cap_waived": True}
    save(statefile, state)
    guard = None
    def owned(pod_id):
        nonlocal guard
        state.update(pod_id=pod_id, billing_started=time.time())
        save(statefile, state)
        guard = runpod.start_watchdog(pod_id, int(hours*3600), root / "watchdog.log")
        pod = runpod.call("GET", f"/pods/{pod_id}")
        rate = float(pod.get("costPerHr") or 0)
        if not 0 < rate * hours <= 20:
            raise RuntimeError("Actual GPU price outside recovery allocation")
        state["actual_pod_hourly_usd"] = rate
        save(statefile, state)
    try:
        stamp = datetime.now(timezone.utc).strftime("%H%M%S")
        runpod.up(name=f"subagents-recovery-{arm}-{stamp}", eval=MODELS[arm], gpu=gpu,
                  max_hours=hours, on_provisioned=owned)
        if not runpod.wait_bootstrapped(state["pod_id"], timeout_s=2400):
            raise RuntimeError("Recovery bootstrap failed")
        pod = runpod.call("GET", f"/pods/{state['pod_id']}")
        host = f"root@{pod['publicIp']}:{pod['portMappings']['22']}"
        port = 9201 if arm == "control" else 9202
        command = [sys.executable, "scratch/delegated_harm/run_eval.py", "--target", MODELS[arm],
                   "--name", "delegated_harm", "--config", str(root / "config.yaml"),
                   "--server", host, "--ssh-key", pair[1], "--port", str(port), "--terminate-pod"]
        state.update(server=host, local_port=port, status="evaluating", command=command)
        save(statefile, state)
        completed = subprocess.run(command, check=False)
        state.update(driver_exit_code=completed.returncode,
                     status="generation_finished" if completed.returncode == 0 else "failed")
        save(statefile, state)
        if completed.returncode:
            raise RuntimeError(f"Recovery driver exited {completed.returncode}")
    except BaseException as exc:
        state.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:700])
        save(statefile, state)
        raise
    finally:
        if state.get("pod_id"):
            runpod.teardown(state["pod_id"])
            state.update(terminated=True, finished=time.time())
            save(statefile, state)
            if guard:
                guard.terminate()
    folders = [p for p in (root / "runs").iterdir()
               if not p.name.endswith("-checkpoint") and (p / "metadata/recovery.json").exists()]
    assert len(folders) == 1
    run_dir = folders[0]
    state.update(status="scoring", run_dir=str(run_dir))
    save(statefile, state)
    passes = [("anthropic/claude-sonnet-5", False), ("anthropic/claude-sonnet-4.5", False),
              ("google/gemini-3-flash-preview", True)]
    for model, json_mode in passes:
        cmd = [sys.executable, "scratch/delegated_harm/rescore.py", "--run-dir", str(run_dir),
               "--ignore-spending-cap", "--workers", "12", "--judge-max-tokens", "16384", "--judge-model", model]
        if json_mode:
            cmd.append("--json-mode")
        subprocess.run(cmd, check=True)
        summary = read(run_dir / "results/results.json")
        if summary.get("unjudged_completed_episodes") == 0:
            break
    state.update(status="finished", scoring_finished=time.time(),
                 unjudged_completed=summary.get("unjudged_completed_episodes"),
                 completed_episodes=summary.get("completed_episodes"))
    save(statefile, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "launch"])
    parser.add_argument("arm", choices=MODELS)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    (prepare_arm if args.mode == "prepare" else launch)(args.arm, args.root.resolve())
