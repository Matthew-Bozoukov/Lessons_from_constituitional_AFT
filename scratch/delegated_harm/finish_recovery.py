# ABOUTME: Wait for two owned recovery jobs and render figures after their scoring completes.
# ABOUTME: Run: uv run scratch/delegated_harm/finish_recovery.py --control-root <path> --da-root <path>.
import argparse
import subprocess
import sys
import time
from pathlib import Path

from src.eval.misalignment.delegated_harm.source import save
from scratch.delegated_harm.recovery import read
from src.naming import artifact_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--da-root", type=Path, required=True)
    args = parser.parse_args()
    roots = {"control": args.control_root, "da": args.da_root}
    deadline = time.time() + 5 * 3600
    while time.time() < deadline:
        states = {k: read(p / "controller.json") for k, p in roots.items()}
        if any(s["status"] == "failed" for s in states.values()):
            raise RuntimeError("A recovery controller failed; inspect owned logs before continuing")
        if all(s["status"] == "finished" for s in states.values()):
            break
        time.sleep(30)
    else:
        raise TimeoutError("Recovery completion wait exceeded five hours")
    assert all(s["terminated"] and s["unjudged_completed"] == 0 for s in states.values())
    sources = {k: read(Path(s["run_dir"]) / "metadata/recovery.json")["source_hf"]["repo"]
               for k, s in states.items()}
    subject = "delegated-harm-recovery-comparison"
    subprocess.run([sys.executable, "scratch/delegated_harm/compare.py", "--control", sources["control"],
                    "--da", sources["da"], "--artifact-subject", subject], check=True)
    stem = artifact_name(subject)
    report = Path("output/delegated_harm") / stem / f"{stem}_results.json"
    assert all(read(report)["recovery"].values())
    subprocess.run([sys.executable, "scratch/delegated_harm/plot_by_world.py", "--comparison", str(report),
                    "--artifact-subject", "delegated-harm-recovery-complied-silent-by-world"], check=True)
    save(report.parent / "recovery_completion.json", {"controllers": states, "comparison": str(report.resolve())})
    print("Recovery scoring and charts complete: " + str(report.resolve()), flush=True)


if __name__ == "__main__":
    main()
