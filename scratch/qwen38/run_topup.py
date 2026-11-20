# ABOUTME: Unattended oai_hf Step 3 top-up for the Qwen3.8 arms, meant to run on a Docker VM under nohup: runs the
# ABOUTME: missing rollouts per arm against a RunPod vLLM server, merges each arm to n=192 and pushes, then tears the pod down.
# Run: nohup uv run python scratch/qwen38/run_topup.py --server root@IP:PORT --pod POD_ID --need nosynth=128,da=128,datools=128 </dev/null >topup.log 2>&1 &
import argparse
import glob
import os
import subprocess
import time

ARMS = {  # arm -> (adapter, its n=64 run dir from 2026-11-18/19, local run-dir stem)
    "nosynth": (
        "dougalldeepmind/2026-11-18-qwen38-0-plain",
        "output/oai_hf/2026-11-18_qwen38_0_plain_185529",
        "qwen38_0_plain",
    ),
    "da": (
        "dougalldeepmind/2026-11-19-qwen38-0-da-15",
        "output/oai_hf/2026-11-19_qwen38_0_da_15_131422",
        "qwen38_0_da_15",
    ),
    "datools": (
        "dougalldeepmind/2026-11-18-qwen38-0-da-tools-15",
        "output/oai_hf/2026-11-18_qwen38_0_da_tools_15_194726",
        "qwen38_0_da_tools_15",
    ),
}
TOTAL = 192


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd):
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--pod", required=True)
    ap.add_argument(
        "--need", required=True, help="arm=n,... rollouts still to run per arm"
    )
    ap.add_argument(
        "--extra_parts",
        default="",
        help="arm=dir;dir,... partial part dirs already made elsewhere",
    )
    ap.add_argument("--port", default="8096")
    a = ap.parse_args()
    need = {k: int(v) for k, v in (x.split("=") for x in a.need.split(","))}
    extra = {
        k: v.split(";")
        for k, v in (x.split("=", 1) for x in a.extra_parts.split(",") if x)
    }
    start = time.time()
    ran = []
    for i, (arm, n) in enumerate(need.items()):
        if n <= 0:
            continue
        ran.append(arm)
        adapter = ARMS[arm][0]
        # A finished run's tunnel can hold its local port for a while (docs/GOTCHAS.md 2026-11-03):
        # every arm gets its own port, or the next arm dies at startup (it did on 2026-11-19).
        port = str(int(a.port) + i)
        rc = run(
            [
                "uv",
                "run",
                "evals",
                "--name",
                "oai_hf",
                "--target",
                adapter,
                "--server",
                a.server,
                "--port",
                port,
                "--no-push",
                "only_steps=[step3]",
                f"n={n}",
                "inspect.max_samples=24",
                "inspect.max_sandboxes=30",
                "inspect.max_connections=24",
                "serving.concurrency=24",
            ]
        )
        log(f"{arm}: evals exited {rc}")
    ok = True
    stamp = time.strftime("%Y-%m-%d_%H%M%S", time.gmtime(start))
    for arm in ran:  # only the arms that ran here; the others were merged already
        adapter, old, stem = ARMS[arm]
        new = [
            d
            for d in sorted(glob.glob(f"output/oai_hf/*_{stem}_*"))
            if os.path.getmtime(d) >= start - 60 and d != old
        ]
        parts = [old] + extra.get(arm, []) + new
        rc = run(
            [
                "uv",
                "run",
                "python",
                "scratch/oai_hf_repro/merge_and_push.py",
                f"output/oai_hf_repro/{stamp}_merged{TOTAL}_{arm}",
                *parts,
                "--n",
                str(TOTAL),
                "--push",
            ]
        )
        ok &= rc == 0
        log(f"{arm}: merge+push exited {rc} from parts {parts}")
    rc = run(["uv", "run", "runpod", "down", "--pod", a.pod])
    log(f"runpod down exited {rc}; all pushes ok: {ok}")
    log("TOPUP DONE")


if __name__ == "__main__":
    main()
