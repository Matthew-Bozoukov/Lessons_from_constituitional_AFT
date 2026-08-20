# ABOUTME: The on-box ODCV supervisor: runs its passes, health-checks each one, pushes it to
# ABOUTME: HF immediately, and heartbeats so the pod watchdog knows the run is still alive.

"""Drive one docker host's share of an ODCV run, unattended.

    uv run python scratch/odcv_box_run.py --config <cfg> --passes 2 --hf_repo <org/name> \
        --box_id a1 [--pod <runpod-id>]

This is what runs under nohup on a rented box after the laptop goes away. Everything it
does is shaped by one fact: THE BOX IS NOT DURABLE. vast instances get their IP remapped,
get stuck `loading` while billing, and occasionally just vanish (all three happened in
docs/swebench_run_postmortem.md). So:

  - A pass is pushed to Hugging Face THE MOMENT it finishes, not at the end of the run.
    GOTCHAS is explicit that artifacts must come off continuously; a box that dies then
    costs at most the one pass still in flight, instead of everything it ever produced.
  - The heartbeat is refreshed on a background thread, not between passes. A pass takes
    tens of minutes, so between-pass touching cannot distinguish "working" from "wedged"
    and the pod watchdog would either kill a healthy run or never fire.
  - Every pass is health-checked for the EMPTY-TRANSCRIPT signature before being trusted.
    A scenario that finishes `ok` while writing no messages_record.txt is the failure that
    cost a full run on 2026-08-18: the summary looks clean and the cells are simply gone.
    Here it is counted, written into the status file, and pushed with the pass.

The status file is the thing to read when you come back: it is a single JSON with per-pass
counts, the empty-transcript alarm, and the endpoint health at each pass boundary.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import fire
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# load_dotenv HERE. src/huggingface.py resolves the token from the environment and
# does NOT read .env itself, so a supervisor started without it runs the whole pass
# and then fails its push with a bare 401 from create_repo -- which is exactly what
# happened on 2026-08-20, silently disabling the crash-safety this script exists for.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

STATE = Path("/root/odcv")


def _heartbeat_loop(path: Path, stop: threading.Event, period_s: float = 60.0) -> None:
    """Refresh the watchdog's liveness file until told to stop."""
    while not stop.is_set():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        stop.wait(period_s)


def _endpoint_ok(base_url: str, want_model: str) -> dict:
    """Is the tunnel up and is the LoRA this config asks for actually served?"""
    import requests
    url = base_url.replace("host.docker.internal", "127.0.0.1").rstrip("/") + "/models"
    try:
        names = [m["id"] for m in requests.get(url, timeout=30).json()["data"]]
        return {"reachable": True, "models": names, "has_target": want_model in names}
    except Exception as e:  # noqa: BLE001 - a dead endpoint is data, not a crash
        return {"reachable": False, "error": f"{type(e).__name__}: {e}"}


def _audit_pass(run_dir: Path) -> dict:
    """Count what a finished pass actually produced, including the silent failure.

    `ok` in the summary is not evidence: the 2026-08-18 run had scenarios report ok while
    writing no transcript at all, which is indistinguishable from success unless the
    transcript file is checked directly.
    """
    logs = list(run_dir.rglob("messages_record.txt"))
    nonempty = [p for p in logs if p.stat().st_size > 0]
    statuses: dict[str, int] = {}
    n_scenarios = None
    cost = None
    manifest = run_dir / "rollout_manifest.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            n_scenarios = data.get("n_scenarios")
            cost = data.get("rollout_cost_usd")
            for r in data.get("results", []):
                st = str(r.get("status", "unknown"))
                statuses[st] = statuses.get(st, 0) + 1
        except Exception:  # noqa: BLE001
            statuses = {"unparseable_manifest": 1}
    else:
        # No manifest means the driver died before writing it, so the pass is incomplete
        # however many transcripts happen to be on disk.
        statuses = {"NO_MANIFEST": 1}
    expected = n_scenarios or 70
    return {
        "run_dir": run_dir.name,
        "n_scenarios": n_scenarios,
        "transcripts_written": len(logs),
        "transcripts_nonempty": len(nonempty),
        "empty_transcripts": len(logs) - len(nonempty),
        "statuses": statuses,
        "rollout_cost_usd": cost,
        # THE alarm. `ok`/`cached` in the manifest is not evidence a cell produced anything:
        # on 2026-08-18 scenarios reported ok while writing no transcript, which is
        # invisible in the status counts and silently removed ~21% of every pass. Only a
        # non-empty messages_record.txt proves a cell ran.
        "ALARM_missing_cells": max(0, expected - len(nonempty)),
    }


def _push(run_dir: Path, hf_repo: str, box_id: str, audit: dict, cfg_path: str) -> str:
    """Put one finished pass on the Hub, where the box dying cannot take it."""
    from src.huggingface import hf_api
    from src.utils import git_sha, origin_url
    api = hf_api()
    api.create_repo(hf_repo, repo_type="dataset", private=False, exist_ok=True)
    prefix = f"passes/{box_id}/{run_dir.name}"
    (run_dir / "pass_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (run_dir / "pass_provenance.json").write_text(json.dumps({
        "config": cfg_path, "box_id": box_id, "source_repo": origin_url(),
        "git_sha": git_sha(),
    }, indent=2), encoding="utf-8")
    api.upload_folder(folder_path=str(run_dir), path_in_repo=prefix,
                      repo_id=hf_repo, repo_type="dataset")
    return f"{hf_repo}/{prefix}"


def main(config: str, passes: int = 2, hf_repo: str = "", box_id: str = "box",
         state_dir: str = str(STATE)) -> None:
    """Run this box's passes, auditing and publishing each as it lands."""
    cfg = OmegaConf.load(config)
    sd = Path(state_dir)
    sd.mkdir(parents=True, exist_ok=True)
    status_path = sd / "status.json"
    hb = sd / "heartbeat"

    stop = threading.Event()
    threading.Thread(target=_heartbeat_loop, args=(hb, stop), daemon=True).start()

    out_root = Path(str(cfg.output_root)) / str(cfg.model_key)
    state = {"box_id": box_id, "config": config, "passes_requested": passes,
             "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "passes": [], "done": False}

    def save() -> None:
        status_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    save()
    try:
        for i in range(1, passes + 1):
            health = _endpoint_ok(str(cfg.base_url), str(cfg.model))
            state.setdefault("endpoint_checks", []).append({"before_pass": i, **health})
            save()
            if not health.get("has_target"):
                # Running a pass against a dead or wrong endpoint burns an hour and
                # produces 70 useless cells. Stop and leave the box up to be inspected.
                state["fatal"] = f"endpoint not serving {cfg.model!r} before pass {i}"
                save()
                print(f"!!! {state['fatal']}", flush=True)
                break

            before = {p.name for p in out_root.glob("*") if p.is_dir()}
            t0 = time.time()
            print(f">>> pass {i}/{passes} starting", flush=True)
            r = subprocess.run(
                ["uv", "run", "python", "scratch/odcv_rollout_cli.py", "--config", config],
                cwd=str(ROOT), capture_output=True, text=True)
            (sd / f"pass_{box_id}_{i}.log").write_text(
                (r.stdout or "") + "\n--- stderr ---\n" + (r.stderr or ""), encoding="utf-8")

            new = [p for p in out_root.glob("*") if p.is_dir() and p.name not in before]
            entry = {"pass": i, "returncode": r.returncode,
                     "minutes": round((time.time() - t0) / 60, 1)}
            if new:
                run_dir = sorted(new)[-1]
                entry.update(_audit_pass(run_dir))
                if hf_repo:
                    try:
                        entry["published"] = _push(run_dir, hf_repo, box_id,
                                                   entry, config)
                    except Exception as e:  # noqa: BLE001 - a failed push must not stop the run
                        entry["publish_error"] = f"{type(e).__name__}: {e}"
            else:
                entry["ALARM_no_run_dir"] = True
            state["passes"].append(entry)
            save()
            print(f">>> pass {i} done: {json.dumps(entry)[:300]}", flush=True)
        state["done"] = True
    finally:
        state["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()
        stop.set()
        # The marker the watchdog and the operator both key on. Written last, and written
        # even on an exception, so "no DONE file" unambiguously means "still running or
        # died hard" rather than "maybe finished".
        (sd / "DONE").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(">>> ALL PASSES COMPLETE", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
