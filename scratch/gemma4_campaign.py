# ABOUTME: Owned one-H200 Gemma ODCV campaign using the standard serving/eval lifecycle.
# ABOUTME: Performs preflight, live tool qualification, durable status, finite guards and teardown.
import argparse
import ctypes
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from dotenv import load_dotenv
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument("--env-file", required=True)
parser.add_argument("--root", required=True)
parser.add_argument("--preflight-only", action="store_true")
args = parser.parse_args()
load_dotenv(args.env_file, override=True)
os.environ["HF_ORG"] = "dougalldeepmind"
os.environ["PYTHONPATH"] = str(ROOT)
os.environ["PYTHONUTF8"] = "1"
from src.infra import runpod
from src.infra.huggingface import hf_api, hf_repo_id
from src.infra.endpoints.vllm import assert_local_port_free, SshExec
from src.eval.run_eval import main as eval_main, _preflight, _credentials_preflight, _run_repo
from src.eval.misalignment.odcv.runner import run as odcv_run
from src.eval.misalignment.odcv.odcv import scenario_names
from src.eval.misalignment.odcv import odcv_rollout
from src.model_profile import model_key
from openai import OpenAI

OUT = Path(args.root).resolve()
OUT.mkdir(parents=True, exist_ok=True)
CONFIG = ROOT / "configs/eval/odcv/gemma.yaml"
cfg = OmegaConf.load(CONFIG)
MODEL = "google/gemma-4-31B-it"
PORT = 8097
state = dict(pid=os.getpid(), stage="preflight", source_root=str(ROOT),
             config=str(CONFIG), model=MODEL, revision=cfg.target_revision,
             start_utc=datetime.now(timezone.utc).isoformat(), port=PORT,
             emergency_gpu_hours=4, judge_cap_usd=12)
guard = None
stop_event = threading.Event()

def save(**updates):
    state.update(updates, updated_utc=datetime.now(timezone.utc).isoformat())
    temp = OUT / "status.tmp"
    temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temp.replace(OUT / "status.json")
    print(json.dumps(updates), flush=True)

def keepalive():
    while not stop_event.wait(20):
        try:
            subprocess.run(["docker", "ps", "-q"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=15)
        except Exception:
            pass

def provisioned(pod_id):
    global guard
    save(stage="bootstrapping", pod_id=pod_id, provisioned_utc=datetime.now(timezone.utc).isoformat())
    guard = runpod.start_watchdog(pod_id, 4 * 3600, OUT / "watchdog.log")
    save(watchdog_pid=guard.pid)

def qualified_run(target, config, out_dir):
    save(stage="loading_model", run_dir=str(out_dir))
    endpoint = target.base_url
    remote = SshExec(state["host"], port=PORT, identity=state["identity"])
    (OUT / "gpu-start.txt").write_text(remote._ssh("nvidia-smi"), encoding="utf-8")
    save(stage="live_probe")
    client = OpenAI(base_url=endpoint, api_key="unused", timeout=300, max_retries=0)
    tool = {"type": "function", "function": {"name": "add", "description": "Add two integers",
            "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                           "required": ["a", "b"]}}}
    messages = [{"role": "system", "content": "Use the provided calculator tool when asked. After its result, give the numeric answer."},
                {"role": "user", "content": "Use add exactly once to add 17 and 25."}]
    first = client.chat.completions.create(model=target.model_name, messages=messages, tools=[tool],
        temperature=config.temperature, max_tokens=4096, extra_body=OmegaConf.to_container(config.sampling))
    m = first.choices[0].message
    assert m.tool_calls and len(m.tool_calls) == 1, "Live probe has no single parsed tool call"
    call = m.tool_calls[0]
    assert call.function.name == "add" and json.loads(call.function.arguments) == {"a": 17, "b": 25}
    assert getattr(m, "reasoning", None) or getattr(m, "reasoning_content", None), "Live probe has no reasoning"
    messages += [m.model_dump(exclude_none=True), {"role": "tool", "tool_call_id": call.id, "content": "42"}]
    second = client.chat.completions.create(model=target.model_name, messages=messages, tools=[tool],
        temperature=config.temperature, max_tokens=4096, extra_body=OmegaConf.to_container(config.sampling))
    assert "42" in (second.choices[0].message.content or ""), "Tool result round trip failed"
    meta = out_dir / "metadata"
    meta.mkdir(exist_ok=True)
    (meta / "live_probe.json").write_text(json.dumps({"first": first.model_dump(), "second": second.model_dump(),
        "sampling": OmegaConf.to_container(config.sampling), "temperature": config.temperature}, indent=2), encoding="utf-8")
    save(stage="rollouts", live_probe="passed")
    result = odcv_run(target, config, out_dir)
    save(stage="publishing", transcripts=result["passes"]["n_transcripts"], judged=result["n_judged"])
    return result

def main():
    _preflight("odcv", None, cfg)
    _credentials_preflight(push=True)
    assert_local_port_free("0.0.0.0", PORT)
    assert sum(len(scenario_names(Path(cfg.bench_dir), v)) for v in ["mandated", "incentivized"]) == 80
    # Build a representative workspace and check the longest actual Windows path.
    bench = Path(cfg.bench_dir).resolve()
    longest = max(scenario_names(bench, "mandated"), key=len)
    ws = OUT / "path-check" / longest
    assert ws.resolve().is_relative_to(OUT)
    odcv_rollout._build_workspace(bench, "mandated", longest, ws)
    assert all(p.is_file() for p in [ws / "docker-compose.yml", ws / "mission_executor/agent_main.py"])
    repo = hf_repo_id(_run_repo("odcv", model_key(MODEL), ""))
    assert not hf_api().repo_exists(repo, repo_type="dataset"), "Destination exists; refuse overwrite"
    quote = runpod.gpu_price("NVIDIA H200")
    account = runpod.graphql("query { myself { clientBalance currentSpendPerHr } }")["myself"]
    assert quote and quote <= 6 and account["clientBalance"] >= 40
    runpod.validate_scheduled_provision()
    keys = runpod.default_keypair()
    assert keys, "No local SSH keypair"
    save(stage="preflight_passed", quote_gpu_usd_hour=quote, account_before=account, hf_repo=repo,
         source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
         identity=keys[1], prior_pods=[p["id"] for p in runpod.active_pods()])
    if args.preflight_only:
        return
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    threading.Thread(target=keepalive, daemon=True).start()
    expires = datetime.now(timezone.utc) + timedelta(hours=4)
    pod = runpod.provision_eval_pod(MODEL, name=f"{os.environ['USER_PREFIX']}-odcv-gemma431b-20261009",
        gpu="NVIDIA H200", count=1, disk_gb=200, eval="odcv",
        revisions={MODEL: cfg.target_revision}, terminate_at=expires.strftime("%Y-%m-%dT%H:%M:%SZ"),
        pubkey_path=keys[0], identity=keys[1], cuda_versions="13.0,13.1,13.2",
        env={"LASR_POD_OWNER": runpod.POD_OWNER, "LASR_POD_DEADLINE": str(expires.timestamp())},
        on_provisioned=provisioned)
    save(host=pod.host, provider_deadline_utc=expires.isoformat())
    assert pod.reachable, "SSH did not become ready"
    assert runpod.wait_bootstrapped(pod.id, timeout_s=2400, poll_s=20), "Bootstrap deadline exceeded"
    save(stage="boot_ready")
    eval_main(["--name", "odcv", "--config", str(CONFIG), "--server", pod.host,
        "--server-bind", "0.0.0.0", "--port", str(PORT), "--ssh-key", keys[1], "--terminate-pod",
        f"output_root={OUT.as_posix()}/runs", f"judge_budget.ledger={OUT.as_posix()}/judge-ledger.json",
        "judge_budget.cap_usd=12", "judge_budget.max_tokens=8192",
        "--target", MODEL], runner=qualified_run)
    info = hf_api().repo_info(repo, repo_type="dataset")
    assert state["transcripts"] == state["judged"] == 80, "Incomplete coverage; inspect saved run"
    from huggingface_hub import hf_hub_download
    import hashlib
    published = Path(hf_hub_download(repo, "results/results.json", repo_type="dataset", revision=info.sha))
    local = Path(state["run_dir"]) / "results/results.json"
    assert hashlib.sha256(published.read_bytes()).digest() == hashlib.sha256(local.read_bytes()).digest()
    files = hf_api().list_repo_files(repo, repo_type="dataset", revision=info.sha)
    assert len([f for f in files if f.startswith("rollouts/") and f.endswith("messages_record.txt")]) == 80
    save(stage="complete", hf_revision=info.sha)

try:
    main()
except BaseException as exc:
    save(stage="failed", error_type=type(exc).__name__, error=str(exc)[:1200])
    traceback.print_exc()
    raise
finally:
    if state.get("pod_id"):
        try:
            runpod.teardown(state["pod_id"])
            save(gpu_terminated=True)
            if guard:
                guard.terminate()
        except Exception:
            save(gpu_terminated=False)
            traceback.print_exc()
    stop_event.set()
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
