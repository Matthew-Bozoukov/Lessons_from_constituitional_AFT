# ABOUTME: Drain and replace only the Base bridge without restarting any evaluation task or coordinator.
# ABOUTME: Preserve ledger identity and unknown requests; own replacement cleanup and exact routing rollback.
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import psutil

ROOT = Path("/srv/lasr/runs/gptoss-full300-65000-20261013")
REPO = Path("/srv/lasr/repo")
HERE = Path(__file__).resolve().parent
GATE = Path("/run/lasr-gptoss-compact-handoff")
BRIDGE_SERVICE = "lasr-gptoss-base-compact-bridge"
GATE_SERVICE = "lasr-gptoss-base-request-gate"
RULE = [
    "OUTPUT",
    "-p",
    "tcp",
    "-d",
    "127.0.0.1",
    "--dport",
    "18705",
    "-m",
    "owner",
    "--uid-owner",
    "0",
    "-m",
    "comment",
    "--comment",
    "lasr-gptoss-compact-20261014",
    "-j",
    "REDIRECT",
    "--to-ports",
    "18704",
]


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    p = Path(p)
    temp = p.with_suffix(".tmp")
    temp.write_text(json.dumps(v, indent=2))
    os.replace(temp, p)


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def identity(pid):
    p = psutil.Process(pid)
    return dict(pid=pid, birth=p.create_time(), cmd=p.cmdline())


def alive(v):
    try:
        p = psutil.Process(v["pid"])
        return (
            abs(p.create_time() - v["birth"]) < 0.01
            and p.status() != psutil.STATUS_ZOMBIE
        )
    except psutil.NoSuchProcess:
        return False


def mode(port=None, fallback=False):
    write(
        GATE / "control.json",
        dict(
            port=port,
            fallback_port=18705 if fallback else None,
            lease_until=time.time() + 45,
        ),
    )


def remove_rule():
    if (
        subprocess.run(
            ["iptables", "-t", "nat", "-C", *RULE], capture_output=True
        ).returncode
        == 0
    ):
        run("iptables", "-t", "nat", "-D", *RULE)


def pending():
    return [
        k
        for k, v in read(ROOT / "base/budget.json")["requests"].items()
        if v["state"] != "completed"
    ]


def connections(owner):
    if not alive(owner):
        return []
    return [
        c
        for c in psutil.Process(owner["pid"]).net_connections(kind="tcp")
        if c.laddr.port == 18705 and c.status != psutil.CONN_LISTEN
    ]


def probe(path, body=None):
    # Same credentials and native endpoint; UID exclusion bypasses the queue only
    # for local identity/tokenization probes. Never sends a sampling request.
    timeout = 120 if body is not None else 10
    payload = dict(
        path=path, body=body, key=os.environ["TINKER_API_KEY"], timeout=timeout
    )
    code = """import sys,json,urllib.request
d=json.load(sys.stdin)
r=urllib.request.Request('http://127.0.0.1:18705'+d['path'],data=None if d['body'] is None else json.dumps(d['body']).encode(),headers={'Authorization':'Bearer '+d['key'],'Content-Type':'application/json'})
with urllib.request.urlopen(r,timeout=d['timeout']) as f: print(f.read().decode())
"""

    def nobody():
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)

    result = subprocess.run(
        ["/usr/bin/python3", "-c", code],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=timeout + 5,
        preexec_fn=nobody,
    )
    if result.returncode:
        raise RuntimeError("Local bridge probe failed")
    return json.loads(result.stdout)


def start_bridge(original=False):
    service = BRIDGE_SERVICE + ("-rollback" if original else "")
    args = [
        "systemd-run",
        "--unit=" + service,
        "--property=WorkingDirectory=" + str(REPO),
        "--property=Restart=no",
        "--property=TimeoutStopSec=180",
        "--property=StandardOutput=append:" + str(ROOT / "compact-bridge.log"),
        "--property=StandardError=append:" + str(ROOT / "compact-bridge.err"),
        sys.executable,
        "-u",
        str(HERE / "compact_budget_bridge.py"),
        str(ROOT),
    ]
    if original:
        args.append("--original")
    run(*args)
    expected = read(ROOT / "base/identity.json")
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            actual = probe("/v1/models")
            assert actual == expected, "Replacement model identity changed"
            tokens = probe(
                "/tokenize",
                dict(
                    model="openai/gpt-oss-120b",
                    messages=[dict(role="user", content="transport probe")],
                    tools=[],
                ),
            )
            assert tokens["count"] > 0
            pid = int(
                run("systemctl", "show", service, "-p", "MainPID", "--value").strip()
            )
            return identity(pid) | dict(service=service)
        except (RuntimeError, subprocess.TimeoutExpired):
            time.sleep(1)
    raise RuntimeError("Replacement bridge failed readiness; do not release queue")


def drain_gate():
    for _ in range(180):
        v = read(GATE / "state/status.json")
        assert not v["errors"], v["errors"]
        if v["active"] == 0:
            return v
        time.sleep(1)
    raise RuntimeError("Gate connections remain; keep gate running and inspect")


def cleanup_monitor():
    owner = read(ROOT / "compact-bridge-handoff.json")["new_bridge"]
    while True:
        tasks = read(ROOT / "base/swe/metadata/state.json")["tasks"]
        if all(t["status"] == "valid" for t in tasks.values()) and not pending():
            if alive(owner):
                run("systemctl", "stop", owner["service"])
            assert not alive(owner)
            write(
                ROOT / "compact-bridge-cleanup.json",
                dict(at=time.time(), bridge=owner, stopped=True),
            )
            return
        if not alive(owner):
            write(
                ROOT / "compact-bridge-unexpected-exit.json",
                dict(at=time.time(), pending=pending()),
            )
            raise RuntimeError(
                "Replacement bridge exited before inference completion; no blind restart"
            )
        time.sleep(2)


def launch_cleanup():
    run(
        "systemd-run",
        "--unit=lasr-gptoss-compact-cleanup",
        "--property=WorkingDirectory=" + str(REPO),
        "--property=Restart=no",
        "--property=StandardOutput=append:" + str(ROOT / "compact-cleanup.log"),
        "--property=StandardError=append:" + str(ROOT / "compact-cleanup.err"),
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "cleanup",
    )


def guard():
    while True:
        receipt = read(ROOT / "compact-bridge-handoff.json")
        if receipt["phase"] in ("complete", "rolled_back"):
            return
        if not alive(receipt["controller"]):
            if receipt["phase"] in ("admitted", "draining") and alive(
                receipt["old_bridge"]
            ):
                if (GATE / "control.json").exists():
                    mode(18705)
                remove_rule()
                if (GATE / "state/status.json").exists():
                    drain_gate()
                subprocess.run(["systemctl", "stop", GATE_SERVICE], capture_output=True)
                write(
                    ROOT / "compact-bridge-guard.json",
                    dict(at=time.time(), action="original_restored_before_cutover"),
                )
            else:
                # A controller failure across listener replacement is ambiguous.
                # Keep queued requests, never replay unknown paid work.
                write(
                    ROOT / "compact-bridge-guard.json",
                    dict(
                        at=time.time(),
                        action="held_for_inspection",
                        phase=receipt["phase"],
                    ),
                )
            return
        time.sleep(2)


def main():
    from dotenv import load_dotenv

    load_dotenv("/srv/lasr/credentials.env")
    assert not (ROOT / "compact-bridge-handoff.json").exists(), (
        "Exclusive admission already recorded"
    )
    old = identity(read(ROOT / "base/shim-owner.json")["pid"])
    assert any("src.infra.endpoints.tinker_server" == v for v in old["cmd"])
    handoff = read(ROOT / "base-scale80-handoff.json")
    assert alive(handoff["original_owner"]) and alive(handoff["supplemental_owner"])
    assert (
        psutil.Process(handoff["original_owner"]["pid"]).status()
        == psutil.STATUS_STOPPED
    )
    for name, digest in [
        (
            "tinker_budget.py",
            "0457ec43f3f634397470b571832a16c6c005bf8f1c517a2be2717ab4d5064c04",
        ),
        (
            "tinker_server.py",
            "32b89e0ca039c2be484274646b73e2c62fef97a8d14fe699edac95f4366296f1",
        ),
    ]:
        assert (
            hashlib.sha256(
                (REPO / "src/infra/endpoints" / name).read_bytes()
            ).hexdigest()
            == digest
        )
    assert read(HERE / "qualification.json")["routing"]["exactly_once"]
    original_identity = probe("/v1/models")
    assert original_identity == read(ROOT / "base/identity.json")
    original_tokens = probe(
        "/tokenize",
        dict(
            model="openai/gpt-oss-120b",
            messages=[dict(role="user", content="transport probe")],
            tools=[],
        ),
    )
    before = read(ROOT / "base/swe/metadata/state.json")
    original_valid = {
        k: v for k, v in before["tasks"].items() if v["status"] == "valid"
    }
    write(
        ROOT / "compact-bridge-before.json",
        dict(at=time.time(), old_bridge=old, valid=original_valid),
    )
    receipt = dict(
        at=time.time(),
        phase="admitted",
        controller=identity(os.getpid()),
        old_bridge=old,
        authorization="User: do that; implement tested accounting optimization and drain handoff",
        model_changed=False,
        task_restarts=0,
    )
    write(ROOT / "compact-bridge-handoff.json", receipt)
    run(
        "systemd-run",
        "--unit=lasr-gptoss-compact-guard",
        "--property=WorkingDirectory=" + str(REPO),
        "--property=Restart=no",
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "guard",
    )
    gate_started = old_stopped = released = False
    try:
        GATE.mkdir(mode=0o755)
        (GATE / "state").mkdir()
        os.chown(GATE / "state", 65534, 65534)
        (GATE / "gate.py").write_bytes((HERE / "bridge_request_gate.py").read_bytes())
        mode(fallback=True)
        run(
            "systemd-run",
            "--unit=" + GATE_SERVICE,
            "--property=User=nobody",
            "--property=Group=nogroup",
            "--property=Restart=no",
            "/usr/bin/python3",
            str(GATE / "gate.py"),
            "18704",
            str(GATE / "control.json"),
            str(GATE / "state/status.json"),
        )
        gate_started = True
        for _ in range(100):
            if (GATE / "state/status.json").exists():
                break
            time.sleep(0.1)
        assert (GATE / "state/status.json").exists()
        run("iptables", "-t", "nat", "-I", *RULE)
        receipt.update(phase="draining", drain_at=time.time())
        write(ROOT / "compact-bridge-handoff.json", receipt)
        quiet = 0
        for _ in range(300):
            mode(fallback=True)
            assert alive(old) and alive(handoff["supplemental_owner"])
            empty = not connections(old) and not pending()
            quiet = quiet + 1 if empty else 0
            if quiet >= 3:
                break
            time.sleep(1)
        else:
            raise RuntimeError(
                "Drain deadline exceeded; revert queue to original bridge"
            )
        mode()  # Disable fail-open before taking away original listener.
        ledger = (ROOT / "base/budget.json").read_bytes()
        assert not [
            v
            for v in json.loads(ledger)["requests"].values()
            if v["state"] != "completed"
        ]
        (ROOT / "compact-bridge-drained-budget.json").write_bytes(ledger)
        receipt.update(
            phase="replacing",
            drained_at=time.time(),
            ledger_sha256=hashlib.sha256(ledger).hexdigest(),
        )
        write(ROOT / "compact-bridge-handoff.json", receipt)
        os.kill(old["pid"], signal.SIGTERM)
        for _ in range(100):
            if not alive(old):
                break
            time.sleep(0.1)
        assert not alive(old), (
            "Old bridge did not exit; never start concurrent ledger owner"
        )
        old_stopped = True
        new = start_bridge()
        assert read(ROOT / "base/budget.json") == json.loads(ledger), (
            "Accounting changed during unsampled startup"
        )
        assert (
            probe(
                "/tokenize",
                dict(
                    model="openai/gpt-oss-120b",
                    messages=[dict(role="user", content="transport probe")],
                    tools=[],
                ),
            )
            == original_tokens
        )
        current = read(ROOT / "base/swe/metadata/state.json")["tasks"]
        assert all(current[k] == v for k, v in original_valid.items())
        receipt.update(
            phase="releasing",
            new_bridge=new,
            ready_at=time.time(),
            preserved_valid=len(original_valid),
            preserved_requests=len(json.loads(ledger)["requests"]),
        )
        write(ROOT / "compact-bridge-handoff.json", receipt)
        # Start durable cleanup owner BEFORE any queued generation is released.
        launch_cleanup()
        mode(18705)
        released = True
        remove_rule()
        gate_status = drain_gate()
        run("systemctl", "stop", GATE_SERVICE)
        gate_started = False
        receipt.update(
            phase="complete",
            completed_at=time.time(),
            gate=gate_status,
            routing_removed=True,
        )
        write(ROOT / "compact-bridge-handoff.json", receipt)
    except BaseException as exc:
        receipt.update(error=repr(exc), failure_at=time.time(), released=released)
        if not released:
            if old_stopped or not alive(old):
                # No queued sample was dispatched. Safely restore original implementation.
                subprocess.run(
                    ["systemctl", "stop", BRIDGE_SERVICE], capture_output=True
                )
                restored = start_bridge(original=True)
                receipt.update(new_bridge=restored, rollback_original_code=True)
                write(ROOT / "compact-bridge-handoff.json", receipt)
                launch_cleanup()
            if gate_started:
                mode(18705)
                remove_rule()
                drain_gate()
                run("systemctl", "stop", GATE_SERVICE)
            receipt["phase"] = "rolled_back"
        else:
            receipt["phase"] = "held_after_release"
        write(ROOT / "compact-bridge-handoff.json", receipt)
        raise


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        cleanup_monitor()
    elif len(sys.argv) > 1 and sys.argv[1] == "guard":
        guard()
    else:
        main()
