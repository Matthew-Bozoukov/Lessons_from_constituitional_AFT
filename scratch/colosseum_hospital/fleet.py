# ABOUTME: Run a fleet of Hospital pods from one plan file: rent each with a watchdog, bootstrap, launch its
# ABOUTME: queue, then pull every finished pod's runs and env logs and terminate it as soon as they are safe.
"""One process drives many colosseum_hospital pods; each is torn down the moment its work is pulled.

    uv run python scratch/colosseum_hospital/fleet.py run    --plan <plan.yaml>
    uv run python scratch/colosseum_hospital/fleet.py status --plan <plan.yaml>

`run` rents every planned pod in parallel (src.infra.runpod.up, --eval shape, HF env pushed),
starts a detached watchdog for each (src.infra.runpod watchdog) bound to ONE keeper process
and a per-pod lifetime cap, so nothing bills on if this driver or the session dies (kill the
keeper to stop the whole fleet), waits for the boot log's READY, bootstraps
(pod_bootstrap.sh at the pushed HEAD), launches the pod's queue (run_hospital_queue.sh), and
then polls. A pod whose queue log says QUEUE_DONE is pulled (runs to
<out_root>/<group>/<pod>/, env snapshots to <env_root>/<date>_<group>/<pod>/, logs to
<out_root>/logs/<pod>/), counted against its expected episodes, and terminated at once. A pod
that fails to boot, bootstrap or launch is terminated too. The process exits (FLEET_DONE)
when no pod of the plan is left billing. State is one JSON file in out_root, so a re-run
resumes where the last one stopped and never rents a second pod under a name already on
the account.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.infra.runpod import active_pods, terminate, up

HERE = Path(__file__).resolve().parent
LOCK = threading.RLock()
SSH_OPTS = [
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "ConnectTimeout=20",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "LogLevel=ERROR",
]
TERMINAL = {
    "down",
    "rent_failed",
    "boot_failed",
    "bootstrap_failed",
    "launch_failed",
    "bringup_failed",
    "lost",
    "terminate_failed",
}
RATE = 3.49  # $/h of the profile's H100 SECURE card, for the running cost estimate
LOG_PATH: Path | None = None


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if LOG_PATH:
        with LOCK, open(LOG_PATH, "a") as f:
            f.write(line + "\n")


def alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True
    return True


def seed_count(job: str) -> int:
    rng = job.split(":")[1]
    if "," in rng:
        return len(rng.split(","))
    lo, hi = rng.split("-")
    return int(hi) - int(lo) + 1


def runpod_balance() -> tuple[float, float]:
    load_dotenv(".env")
    r = requests.post(
        "https://api.runpod.io/graphql",
        json={"query": "query { myself { clientBalance currentSpendPerHr } }"},
        headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"},
        timeout=30,
    )
    me = r.json()["data"]["myself"]
    return float(me["clientBalance"]), float(me.get("currentSpendPerHr") or 0.0)


class Fleet:
    def __init__(self, plan_path: str):
        global LOG_PATH
        self.plan_path = Path(plan_path)
        self.plan = OmegaConf.to_container(OmegaConf.load(plan_path), resolve=True)
        self.out = Path(self.plan["out_root"])
        (self.out / "logs").mkdir(parents=True, exist_ok=True)
        LOG_PATH = self.out / f"fleet_{self.plan_path.stem}.log"
        self.state_path = self.out / f"fleet_state_{self.plan_path.stem}.json"
        self.state = (
            json.loads(self.state_path.read_text())
            if self.state_path.exists()
            else {"pods": {}}
        )
        self.pods = {p["name"]: p for p in self.plan["pods"]}
        for name in self.pods:
            self.state["pods"].setdefault(name, {"status": "planned"})
        self.save()

    # ── state ────────────────────────────────────────────────────────────────
    def save(self) -> None:
        with LOCK:
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True))
            tmp.replace(self.state_path)

    def st(self, name: str) -> dict:
        return self.state["pods"][name]

    def set(self, name: str, **kw) -> None:
        with LOCK:
            self.state["pods"][name].update(kw)
            self.save()

    # ── plumbing ─────────────────────────────────────────────────────────────
    def ssh(self, host: str, remote: str, timeout: int = 120) -> tuple[int, str]:
        user_ip, port = host.rsplit(":", 1)
        try:
            r = subprocess.run(
                ["ssh", "-p", port, *SSH_OPTS, user_ip, remote],
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return 124, "ssh timed out"
        return r.returncode, r.stdout + r.stderr

    def listed(self) -> dict[str, dict]:
        return {p.get("name"): p for p in active_pods()}

    def keeper(self) -> int:
        with LOCK:
            pid = self.state.get("keeper_pid")
            if pid and alive(pid):
                return int(pid)
            p = subprocess.Popen(
                ["sleep", str(int(self.plan.get("keeper_s", 43200)))],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.state["keeper_pid"] = p.pid
            self.save()
            log(f"keeper pid {p.pid}: kill it to terminate every pod of this fleet")
            return p.pid

    def watchdog_alive(self, pod_id: str) -> bool:
        r = subprocess.run(
            ["pgrep", "-f", f"src.infra.runpod watchdog {pod_id}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return r.returncode == 0 and bool(r.stdout.strip())

    def watchdog(self, name: str, pod_id: str) -> None:
        if self.watchdog_alive(pod_id):
            return
        # A watchdog re-created after its first one died with a killed driver gets only what
        # is left of the pod's cap, counted from when the pod was rented.
        elapsed = time.time() - float(self.st(name).get("rented_at") or time.time())
        cap_s = max(600, int(float(self.pods[name]["cap_h"]) * 3600 - elapsed))
        logp = self.out / "logs" / f"{name}_watchdog.log"
        # The detached watchdog keeps writing to this log after the driver exits.
        f = open(logp, "a")  # noqa: SIM115
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "src.infra.runpod",
                "watchdog",
                pod_id,
                str(self.keeper()),
                str(cap_s),
                str(logp),
            ],
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=os.getcwd(),
        )
        self.set(name, watchdog_for=pod_id, cap_s=cap_s)

    def push_env(self, name: str) -> bool:
        from src.infra.endpoints.vllm import POD_WORKDIR, SshExec

        host = self.st(name)["host"]
        for _ in range(30):
            if self.ssh(host, "true", timeout=40)[0] == 0:
                break
            time.sleep(30)
        else:
            return False
        SshExec(host, port=int(self.plan["port"]), workdir=POD_WORKDIR).push_hf_env(
            Path(".env")
        )
        self.set(name, env_pushed=True)
        return True

    def kill(self, name: str, status: str, reason: str) -> None:
        s = self.st(name)
        gone = terminate(s["id"]) if s.get("id") else True
        hours = (time.time() - s["rented_at"]) / 3600 if s.get("rented_at") else 0.0
        self.set(
            name,
            status=status if gone else "terminate_failed",
            reason=reason,
            down_at=time.time(),
            hours=round(hours, 2),
            cost=round(hours * RATE, 2),
        )
        log(
            f"{name}: {status} ({reason}); terminated={gone}; {hours:.2f} h ~ ${hours * RATE:.0f}"
        )

    # ── bring-up ─────────────────────────────────────────────────────────────
    def adopt(self, name: str, pod_id: str) -> bool:
        from src.infra.runpod import _ssh_endpoint

        self.set(
            name, id=pod_id, rented_at=self.st(name).get("rented_at") or time.time()
        )
        self.watchdog(name, pod_id)
        try:
            ip, port = _ssh_endpoint(pod_id)
        except Exception as e:  # noqa: BLE001 - a pod we cannot reach is terminated, not kept
            self.kill(name, "rent_failed", f"adopted but no ssh endpoint: {e}")
            return False
        self.set(name, host=f"root@{ip}:{port}", status="rented")
        return self.push_env(name)

    def rent(self, name: str) -> bool:
        pod = self.pods[name]
        if self.st(name).get("id"):
            return True
        existing = self.listed().get(name)
        if existing:
            log(f"{name}: already on the account as {existing['id']}; adopting it")
            return self.adopt(name, existing["id"])
        targets = list(dict.fromkeys([pod["target"], self.plan["peer"]]))
        for attempt in range(1, int(self.plan.get("rent_attempts", 3)) + 1):
            try:
                out = up(
                    name=name,
                    # `--eval` is the eval NAME — it picks the inference card from the
                    # model's profile — and the checkpoints moved to `--target`. No
                    # clone_repo: pod_bootstrap.sh puts this repo and the patched
                    # Colosseum on the box itself, at the pushed commit.
                    eval="colosseum_hospital",
                    target=targets,
                    push_env=True,
                    gpu=self.plan.get("gpu"),
                    cloud=self.plan.get("cloud", "SECURE"),
                )
            except Exception as e:  # noqa: BLE001
                log(
                    f"{name}: rent attempt {attempt} raised {type(e).__name__}: {str(e)[:300]}"
                )
                existing = self.listed().get(name)
                if existing:  # provisioned, then something after it failed: never lose track of it
                    return self.adopt(name, existing["id"])
                time.sleep(60 * attempt)
                continue
            pod_id = re.search(r"^pod:\s+(\S+)", out, re.MULTILINE).group(1)
            host = re.search(r"^host:\s+(\S+)", out, re.MULTILINE).group(1)
            ready = bool(re.search(r"^ssh:\s+ready", out, re.MULTILINE))
            self.set(
                name,
                id=pod_id,
                host=host,
                status="rented",
                rented_at=time.time(),
                env_pushed=ready,
            )
            self.watchdog(name, pod_id)
            log(f"{name}: rented {pod_id} at {host}")
            return ready or self.push_env(name)
        self.set(name, status="rent_failed", reason="no pod after retries")
        log(f"{name}: rent_failed")
        return False

    def wait_ready(self, name: str, timeout_s: int = 3600) -> bool:
        s = self.st(name)
        if s.get("ready_at"):
            return True
        t0, shown = time.time(), ""
        while time.time() - t0 < timeout_s:
            try:
                r = requests.get(
                    f"https://{s['id']}-8080.proxy.runpod.net/boot.log", timeout=20
                )
                if r.ok:
                    if "READY" in r.text:
                        self.set(name, ready_at=time.time())
                        log(
                            f"{name}: READY after {(time.time() - s['rented_at']) / 60:.0f} min"
                        )
                        return True
                    tail = (r.text.strip().splitlines() or [""])[-1][:160]
                    if tail != shown and re.search(r"FAIL|Error|error:", tail):
                        log(f"{name}: boot log: {tail}")
                    shown = tail
            except requests.RequestException:
                pass
            time.sleep(30)
        return False

    def bootstrap(self, name: str) -> bool:
        s = self.st(name)
        if s.get("bootstrapped_at"):
            return True
        if not s.get("env_pushed") and not self.push_env(name):
            return False
        for attempt in (1, 2):
            logp = self.out / "logs" / f"{name}_bootstrap_{attempt}.log"
            with open(logp, "w") as f:
                try:
                    rc = subprocess.run(
                        [
                            "bash",
                            str(HERE / "pod_bootstrap.sh"),
                            s["host"],
                            self.plan["branch"],
                            self.state["sha"],
                        ],
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        timeout=3600,
                        check=False,
                    ).returncode
                except subprocess.TimeoutExpired:
                    rc = 124
            text = logp.read_text()
            if rc == 0 and "BOOTSTRAP_DONE" in text and "[FAIL]" not in text:
                self.set(name, bootstrapped_at=time.time(), status="bootstrapped")
                log(f"{name}: bootstrapped")
                return True
            log(f"{name}: bootstrap attempt {attempt} rc={rc}: {text.strip()[-300:]}")
        return False

    def launch(self, name: str) -> bool:
        pod, s = self.pods[name], self.st(name)
        if s.get("launched_at"):
            return True
        # Overrides first: the queue appends EXTRA after its own `condition=... seeds=...`, and
        # argparse takes one run of positionals, so an override after `--config` is rejected.
        extra = " ".join([*pod.get("extra", []), "--config", pod["config"]])
        remote = (
            f"cd /root/work && EXTRA='{extra}' bash scratch/colosseum_hospital/run_hospital_queue.sh "
            f"{self.plan['port']} {pod['target']} -- {' '.join(pod['jobs'])}"
        )
        rc, out = self.ssh(s["host"], remote, timeout=120)
        if rc == 0 and "queued" in out:
            self.set(name, status="launched", launched_at=time.time())
            log(f"{name}: launched {' '.join(pod['jobs'])} with {extra}")
            return True
        log(f"{name}: launch rc={rc}: {out.strip()[-300:]}")
        return False

    def bring_up(self, name: str) -> None:
        try:
            if not self.rent(name):
                return
            if not self.wait_ready(name):
                return self.kill(
                    name, "boot_failed", "no READY in the boot log within 60 min"
                )
            if not self.bootstrap(name):
                return self.kill(
                    name,
                    "bootstrap_failed",
                    "bootstrap did not finish (logs/<pod>_bootstrap_*.log)",
                )
            if not (self.launch(name) or self.launch(name)):
                return self.kill(name, "launch_failed", "the queue did not start")
        except Exception as e:  # noqa: BLE001 - a failed bring-up must not leave a pod billing
            log(f"{name}: bring-up raised {type(e).__name__}: {e}")
            if self.st(name).get("id") and self.st(name)["status"] not in TERMINAL:
                self.kill(name, "bringup_failed", f"{type(e).__name__}: {str(e)[:200]}")

    # ── watch, pull, tear down ───────────────────────────────────────────────
    def count_local(self, name: str) -> int:
        d = self.out / self.pods[name]["group"] / name
        return sum(
            1 for p in d.rglob("blackboards.json") if "fixes_smoke" not in p.parts
        )

    def pull(self, name: str) -> bool:
        pod, s = self.pods[name], self.st(name)
        user_ip, port = s["host"].rsplit(":", 1)
        rsh = f"ssh -p {port} " + " ".join(SSH_OPTS)
        dest = self.out / pod["group"] / name
        env = Path(self.plan["env_root"]) / f"{self.plan['date']}_{pod['group']}" / name
        logs = self.out / "logs" / name
        for d in (dest, env, logs):
            d.mkdir(parents=True, exist_ok=True)
        cmds = [
            [
                "rsync",
                "-az",
                "--timeout=300",
                "--exclude",
                "server_*",
                "--exclude",
                "pooled",
                "--exclude",
                "fixes_smoke",
                "--exclude",
                "colosseum_env_logs",
                "--exclude",
                "blackboard_*.txt",
                "--exclude",
                "agent_trajectories.json",
                "--exclude",
                "*.png",
                "-e",
                rsh,
                f"{user_ip}:/root/work/output/colosseum_hospital/",
                f"{dest}/",
            ],
            [
                "rsync",
                "-az",
                "--timeout=300",
                "--include",
                "*/",
                "--include",
                "data_iteration_*.json",
                "--exclude",
                "*",
                "--prune-empty-dirs",
                "-e",
                rsh,
                f"{user_ip}:/root/colosseum/logs/",
                f"{env}/",
            ],
            [
                "rsync",
                "-az",
                "--timeout=120",
                "-e",
                rsh,
                f"{user_ip}:/root/work/output/logs/",
                f"{logs}/",
            ],
        ]
        ok = True
        for c in cmds:
            try:
                r = subprocess.run(
                    c, capture_output=True, text=True, timeout=3600, check=False
                )
                if r.returncode != 0:
                    ok = False
                    log(f"{name}: rsync rc={r.returncode}: {r.stderr.strip()[-300:]}")
            except subprocess.TimeoutExpired:
                ok = False
                log(f"{name}: rsync timed out")
        return ok

    def finish(self, name: str) -> None:
        pod, s = self.pods[name], self.st(name)
        expected = sum(seed_count(j) for j in pod["jobs"])
        if not self.pull(name):
            tries = s.get("pull_tries", 0) + 1
            self.set(name, pull_tries=tries)
            if tries < 3:
                return  # try again next poll; the pod stays up until its work is safe
        n = self.count_local(name)
        bad = [e for e in s.get("exits", []) if e[0] != "0"]
        verdict = (
            "complete"
            if n >= expected and not bad
            else f"incomplete {n}/{expected} exits={bad}"
        )
        self.set(name, pulled=n, expected=expected)
        self.kill(name, "down", verdict)

    def check(self, name: str) -> None:
        s = self.st(name)
        port = self.plan["port"]
        rc, out = self.ssh(
            s["host"],
            f"tail -n 40 /root/work/output/logs/queue_{port}.log 2>/dev/null; "
            "echo FINISHED=$(find /root/work/output/colosseum_hospital -path '*fixes_smoke*' -prune "
            "-o -name blackboards.json -print 2>/dev/null | wc -l)",
            timeout=90,
        )
        if rc != 0:
            misses = s.get("misses", 0) + 1
            self.set(name, misses=misses)
            if misses >= 5 and s.get("id") not in {p.get("id") for p in active_pods()}:
                self.set(
                    name,
                    status="lost",
                    reason="pod no longer listed (watchdog cap, credit or console)",
                    down_at=time.time(),
                )
                log(f"{name}: LOST")
            return
        m = re.search(r"FINISHED=(\d+)", out)
        exits = re.findall(r"EXIT (\d+) (\S+) (\S+)", out)
        self.set(
            name,
            finished=int(m.group(1)) if m else None,
            exits=exits,
            misses=0,
            checked_at=time.time(),
        )
        if "QUEUE_DONE" in out:
            log(
                f"{name}: queue done ({self.st(name)['finished']} episodes, exits {exits}); pulling"
            )
            self.finish(name)

    def write_status(self) -> str:
        rows, total = [], 0.0
        for name, pod in self.pods.items():
            s = self.st(name)
            exp = sum(seed_count(j) for j in pod["jobs"])
            hrs = (
                ((s.get("down_at") or time.time()) - s["rented_at"]) / 3600
                if s.get("rented_at")
                else 0.0
            )
            total += hrs * RATE
            rows.append(
                f"| {name} | {s['status']} | {s.get('finished', '-')}/{exp} | {hrs:.2f} | {s.get('reason', '')} |"
            )
        bal = self.state.get("balance")
        text = (
            f"updated {time.strftime('%H:%M:%S')}; est. spend ${total:.0f}"
            + (f"; runpod balance ${bal[1]:.0f}" if bal else "")
            + "\n\n| pod | status | episodes | hours | note |\n|---|---|---|---|---|\n"
            + "\n".join(rows)
            + "\n"
        )
        (self.out / f"fleet_status_{self.plan_path.stem}.md").write_text(text)
        return text

    def resolve_sha(self) -> str:
        branch = self.plan["branch"]
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        subprocess.run(["git", "fetch", "--quiet", "origin", branch], check=True)
        remote = subprocess.check_output(
            ["git", "rev-parse", f"origin/{branch}"], text=True
        ).strip()
        assert head == remote, (
            f"HEAD {head[:8]} is not origin/{branch} {remote[:8]}: push first, pods clone origin"
        )
        return head

    def run(self, interval: int) -> None:
        needs_boot = [
            n
            for n in self.pods
            if self.st(n)["status"] not in TERMINAL
            and not self.st(n).get("bootstrapped_at")
        ]
        if needs_boot or not self.state.get("sha"):
            self.state["sha"] = self.resolve_sha()
        self.save()
        self.keeper()
        for (
            n
        ) in self.pods:  # one watchdog per live pod, re-created if its first one died
            s = self.st(n)
            if s.get("id") and s["status"] not in TERMINAL:
                self.watchdog(n, s["id"])
        log(
            f"fleet {self.plan_path.stem}: {len(self.pods)} pods at {self.state['sha'][:8]}"
        )
        todo = [
            n
            for n in self.pods
            if self.st(n)["status"] not in TERMINAL
            and not self.st(n).get("launched_at")
        ]
        pool = ThreadPoolExecutor(max_workers=max(1, len(todo)))
        futures = []
        for n in todo:
            futures.append(pool.submit(self.bring_up, n))
            time.sleep(float(self.plan.get("stagger_s", 5)))
        last_balance = 0.0
        while True:
            for n in self.pods:
                if self.st(n)["status"] == "launched":
                    try:
                        self.check(n)
                    except Exception as e:  # noqa: BLE001 - keep watching the others
                        log(f"{n}: check raised {type(e).__name__}: {e}")
            if time.time() - last_balance > 600:
                try:
                    b, rate = runpod_balance()
                    self.state["balance"] = [time.time(), b, rate]
                    self.save()
                    log(f"runpod balance ${b:.2f}, account spend ${rate:.2f}/h")
                    if b < float(self.plan.get("low_balance", 25)):
                        log("LOW BALANCE: top up RunPod or pods stop")
                except Exception as e:  # noqa: BLE001
                    log(f"balance check failed: {type(e).__name__}")
                last_balance = time.time()
            self.write_status()
            busy = [n for n in self.pods if self.st(n)["status"] not in TERMINAL]
            if not busy and all(f.done() for f in futures):
                log("FLEET_DONE")
                return
            time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["run", "status"])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--interval", type=int, default=90)
    a = ap.parse_args()
    fleet = Fleet(a.plan)
    if a.cmd == "run":
        fleet.run(a.interval)
    else:
        print(fleet.write_status())


if __name__ == "__main__":
    main()
