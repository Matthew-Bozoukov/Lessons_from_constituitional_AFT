# ABOUTME: Freezes only the three owned single-GPU attempts for the requested dual-GPU correction.
# ABOUTME: Run from this isolated worktree with uv run python scratch/da_supervision/archive/freeze_single.py.
import json, shlex
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from src.infra.endpoints.vllm import SshExec


def main():
    root = Path("output/da_supervision/2026-09-16/runs")

    def freeze(name):
        p = root / name
        s = json.loads((p / "status.json").read_text())
        r = SshExec(s["host"], port=8000, workdir="/root/work")
        cmd = "p=/root/work/output/da-supervision/driver.pid; if test -f $p; then g=$(cat $p); case $g in ''|*[!0-9]*) exit 1;; esac; kill -STOP -- -$g; fi"
        r._ssh(cmd, timeout=30)
        probe = "import json; from pathlib import Path; p=Path('/root/work/output'); print(json.dumps({'logs':{str(f):f.read_text(errors='replace') for f in (p/'da-supervision').glob('*.log')},'files':[str(f) for f in (p/'train').rglob('*') if f.is_file()]}))"
        receipt = json.loads(r._ssh("python3 -c " + shlex.quote(probe), timeout=60))
        (p / "switch_receipt.json").write_text(
            json.dumps({"status": s, "remote": receipt}, indent=2)
        )
        print(
            json.dumps(
                {
                    "name": name,
                    "pod": s["owned_pod"],
                    "files": receipt["files"],
                    "train_tail": receipt["logs"].get(
                        "/root/work/output/da-supervision/train.log", ""
                    )[-600:],
                }
            ),
            flush=True,
        )

    with ThreadPoolExecutor(3) as ex:
        list(ex.map(freeze, ["answer", "cot-attempt2", "empty-attempt2"]))


if __name__ == "__main__":
    main()
