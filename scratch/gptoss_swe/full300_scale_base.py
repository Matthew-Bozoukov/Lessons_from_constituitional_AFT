# ABOUTME: Add Base-only consumers while preserving the original owner's live children and results.
# ABOUTME: Run this separately from the frozen runtime; pause/resume only its coordinator PID.
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil
from omegaconf import OmegaConf

from src.eval.capabilities.swebench_mini.fleet_state import State, atomic, read


PREFIX = "base-scale80-"


def identity(pid):
    p = psutil.Process(pid)
    return dict(pid=pid, birth=p.create_time(), cmd=p.cmdline())


def alive(receipt):
    try:
        p = psutil.Process(receipt["pid"])
        return (
            p.create_time() == receipt["birth"] and p.status() != psutil.STATUS_ZOMBIE
        )
    except psutil.NoSuchProcess:
        return False


def capacity(originals, supplemental, total):
    return max(0, total - sum(alive(p) for p in originals) - supplemental)


def extra_running(data):
    return [
        iid
        for iid, t in data["tasks"].items()
        if t["status"] == "running" and t["attempts"][-1]["worker"].startswith(PREFIX)
    ]


def resume(root, handoff, reason):
    assert alive(handoff["original_owner"])
    assert not extra_running(read(root / "base/swe/metadata/state.json"))
    os.kill(handoff["original_owner"]["pid"], signal.SIGCONT)
    atomic(root / "base-scale80-resumed.json", dict(at=time.time(), reason=reason))


def guard(root):
    path = root / "base-scale80-handoff.json"
    while not path.exists():
        if (root / "base-scale80-admission-failed.json").exists():
            return
        time.sleep(2)
    h = read(path)
    while not (root / "base-scale80-resumed.json").exists():
        if not alive(h["supplemental_owner"]):
            data = read(root / "base/swe/metadata/state.json")
            if extra_running(data):
                atomic(
                    root / "base-scale80-guard-held.json",
                    dict(
                        at=time.time(),
                        reason="Supplemental owner exited with unfinalized attempts; preserve endpoint and diagnose.",
                    ),
                )
            else:
                resume(
                    root,
                    h,
                    "independent guard: supplemental owner exited, no active supplemental attempts",
                )
            return
        time.sleep(5)


def run(root, total):
    from dotenv import load_dotenv
    from src.eval.capabilities.swebench_mini.fleet_worker import consume

    load_dotenv("/srv/lasr/credentials.env")
    assert total == 80
    claim = root / "base-scale80-owner.json"
    with claim.open("x") as stream:
        json.dump(
            identity(os.getpid()) | dict(at=time.time(), total_slots=total), stream
        )
    original = identity(read(root / "owner-started.json")["pid"])
    assert "scratch.gptoss_swe.full300" in original["cmd"]
    assert "--authorize-inference" in original["cmd"]
    assert subprocess.check_output(
        [
            "systemctl",
            "show",
            "lasr-gptoss-full300.service",
            "--property=MainPID",
            "--value",
        ],
        text=True,
    ).strip() == str(original["pid"])
    for arm in ("control", "da15"):
        assert read(root / arm / "status.json")["phase"] == "inference_finished"
        assert all(
            t["status"] == "valid"
            for t in read(root / arm / "swe/metadata/state.json")["tasks"].values()
        )
        assert all(
            r["state"] == "completed"
            for r in read(root / arm / "budget.json")["requests"].values()
        )
    cfg = OmegaConf.load(root / "base/config.yaml")
    assert (
        cfg.tinker.context_window == 65000 and cfg.worker.max_response_tokens == 16384
    )
    assert (
        cfg.worker.tool_concurrency == 32 and cfg.worker.min_available_memory_gib == 32
    )
    assert cfg.worker.max_task_tokens == 262144 and cfg.worker.step_limit == 500
    assert cfg.workers == 27
    state = State(root / "base/swe")
    snapshot_dir = root / "base-scale80-before"
    snapshot_dir.mkdir()
    hashes = {}
    for arm in ("base", "control", "da15"):
        for name in (
            "config.yaml",
            "budget.json",
            "status.json",
            "swe/metadata/state.json",
        ):
            p = root / arm / name
            content = p.read_bytes()
            destination = snapshot_dir / arm / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            hashes[str(p.relative_to(root))] = hashlib.sha256(content).hexdigest()
    original_children = []
    # Holding the existing state lock prevents pausing its owner inside a state mutation.
    # Do not use State.edit here: no state rewrite is needed or permitted for this handoff.
    from src.eval.capabilities.swebench_mini.fleet_state import lock

    with lock(state.root / ".state.lock"):
        data = read(state.path)
        assert not data.get("halt")
        assert all(
            t["status"] in ("pending", "running", "valid")
            for t in data["tasks"].values()
        )
        for iid, task in data["tasks"].items():
            if task["status"] != "running":
                continue
            attempt = task["attempts"][-1]
            out = state.root / "rollouts" / iid / attempt["id"]
            process = read(out / "process.json")
            ident = identity(process["pid"])
            assert str(out / "request.json") in ident["cmd"]
            assert psutil.Process(process["pid"]).ppid() == original["pid"]
            original_children.append(
                ident | dict(instance_id=iid, attempt_id=attempt["id"])
            )
        pending = [
            iid for iid in cfg.instance_ids if data["tasks"][iid]["status"] == "pending"
        ]
        assert pending and len(original_children) <= 27
        assert all(not data["tasks"][iid]["attempts"] for iid in pending)
        h = dict(
            at=time.time(),
            original_owner=original,
            supplemental_owner=identity(os.getpid()),
            original_children=original_children,
            pending=pending,
            total_slots=total,
            snapshot_sha256=hashes,
            valid_before={
                iid: t for iid, t in data["tasks"].items() if t["status"] == "valid"
            },
            authorization="User requested reassigning unused arm slots to Base without data loss.",
        )
        atomic(root / "base-scale80-handoff.json", h)
        subprocess.Popen(
            [sys.executable, __file__, "guard", "--root", str(root)],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.kill(original["pid"], signal.SIGSTOP)
    # SIGSTOP delivery to a multithreaded owner is asynchronous.
    for _ in range(100):
        if psutil.Process(original["pid"]).status() == psutil.STATUS_STOPPED:
            break
        time.sleep(0.02)
    else:
        raise RuntimeError("Original coordinator did not confirm stopped state")
    worker = OmegaConf.merge(
        cfg.worker,
        dict(
            root=str(state.root),
            serving=dict(context_window=cfg.tinker.context_window),
            endpoint_api_key_env="TINKER_API_KEY",
            fleet_owner_root=str(root),
            protocol_version=cfg.protocol,
            sampling=cfg.sampling,
        ),
    )
    # A separate admission pool keeps the original in-flight requests unchanged.
    # The scheduler below caps combined live task processes at the qualified 80.
    admission = dict(
        directory=str(root / "base/scale80-token-slots"),
        budget_tokens=65000 * total,
        expires=time.time() + 7 * 86400,
        fairness_seconds=30,
    )

    def task(index, iid):
        own_cfg = OmegaConf.create(OmegaConf.to_container(worker, resolve=True))
        consume(
            "http://127.0.0.1:18705/v1",
            "hosted_vllm/openai/gpt-oss-120b",
            own_cfg,
            PREFIX + str(index),
            [iid],
            time.time() + 7 * 86400,
            admission=admission,
        )

    queue = list(enumerate(pending))
    futures = set()
    errors = []
    with ThreadPoolExecutor(max_workers=total) as pool:
        while queue or futures:
            assert alive(original), (
                "Original coordinator identity lost; stop new admission"
            )
            live = read(state.path)
            if live.get("halt"):
                queue.clear()
            slots = capacity(original_children, len(futures), total)
            for _ in range(min(slots, len(queue))):
                index, iid = queue.pop(0)
                futures.add(pool.submit(task, index, iid))
            atomic(
                root / "base-scale80-status.json",
                dict(
                    at=time.time(),
                    phase="inference",
                    original_processes_running=sum(alive(p) for p in original_children),
                    original_results_awaiting_owner=sum(
                        not alive(p) for p in original_children
                    ),
                    supplemental_slots=len(futures),
                    unstarted=len(queue),
                    total_slots=total,
                ),
            )
            done, futures = wait(futures, timeout=2, return_when=FIRST_COMPLETED)
            for f in done:
                try:
                    f.result()
                except Exception as exc:
                    errors.append(repr(exc))
                    queue.clear()
    final = read(state.path)
    assert not extra_running(final)
    for iid, task_before in h["valid_before"].items():
        assert final["tasks"][iid] == task_before, "Prior outcome changed"
    assert not errors, errors
    assert all(final["tasks"][iid]["status"] == "valid" for iid in pending), (
        "Supplemental coverage held"
    )
    atomic(
        root / "base-scale80-complete.json",
        dict(
            at=time.time(),
            tasks=len(pending),
            preserved_valid=len(h["valid_before"]),
            original_attempts_retained=len(original_children),
        ),
    )
    resume(
        root,
        h,
        "Supplemental queue finished; original owner commits its preserved children and grades",
    )


def selftest():
    import tempfile
    from unittest.mock import patch

    with patch(__name__ + ".alive", side_effect=lambda p: p["live"]):
        assert capacity([{"live": True}] * 27, 53, 80) == 0
        assert capacity([{"live": False}] * 27, 53, 80) == 27
        assert capacity([], 80, 80) == 0
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "metadata").mkdir()
        data = dict(
            deadline=None,
            halt=None,
            tasks={str(i): dict(status="pending", attempts=[]) for i in range(100)},
        )
        data["retained"] = None
        data["tasks"]["valid"] = dict(
            status="valid", attempts=[dict(id="old", valid=True)]
        )
        data["tasks"]["running"] = dict(
            status="running", attempts=[dict(id="live", worker="old")]
        )
        atomic(root / "metadata/state.json", data)

        def claim(_):
            return State(root).claim("test", list(data["tasks"]), 3, 6)

        with ThreadPoolExecutor(max_workers=80) as pool:
            results = list(pool.map(claim, range(160)))
        leased = [r[0] for r in results if r]
        assert len(leased) == len(set(leased)) == 100
        after = read(root / "metadata/state.json")
        assert after["tasks"]["valid"] == data["tasks"]["valid"]
        assert after["tasks"]["running"] == data["tasks"]["running"]
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    p = identity(child.pid)
    assert alive(p)
    child.terminate()
    child.wait()
    assert not alive(p)
    print(
        "PASS: capacity ramp, 80-way exclusive claims, retained outcomes, PID lifecycle",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "guard", "selftest"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--total", type=int, default=80)
    args = parser.parse_args()
    if args.action == "selftest":
        selftest()
    elif args.action == "guard":
        guard(args.root)
    else:
        run(args.root, args.total)
