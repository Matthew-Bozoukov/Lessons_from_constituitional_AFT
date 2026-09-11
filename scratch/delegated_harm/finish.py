# ABOUTME: Wait for corrected HF publications, publish owned-cost audits, and draw the comparison.
# ABOUTME: Run: uv run scratch/delegated_harm/finish.py --control-run <path> --da-run <path>.
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

from src.infra.huggingface import hf_api, hf_download, hf_repo_id
from src.eval.misalignment.delegated_harm.source import save


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(run, controller):
    launches = [read(p) for p in Path("output/delegated_harm").glob("*/controller.json")]
    launches = [c for c in launches if c["arm"] == controller["arm"] and c.get("billing_started")]
    assert all(c.get("terminated") for c in launches), "An owned GPU is still billing"
    gpu_records = [{k: c[k] for k in ("pod_id", "billing_started", "finished", "actual_pod_hourly_usd", "terminated")}
                   for c in launches]
    gpu = sum((c["finished"]-c["billing_started"])/3600*c["actual_pod_hourly_usd"] for c in launches)
    allowance = sum((c["finished"]-c["billing_started"])/3600*.10 for c in launches)
    ledgers = [read(run/p) for p in ("metadata/judge_ledger.json", "metadata/rescoring/metadata/judge_ledger.json")]
    reported = sum(c["usd"] for ledger in ledgers for c in ledger["calls"] if c["status"] == "response")
    reserved = sum(c["usd"] for ledger in ledgers for c in ledger["calls"] if c["status"] == "reserved")
    total = gpu + allowance + reported + reserved
    waived = read(run/"metadata/rescoring/protocol.json").get("spending_cap_waived_by_user", False)
    return {"adapter": controller["model"], "budget_usd": controller["budget_usd"],
            "owned_gpu_launches": gpu_records, "estimated_gpu_usd": gpu,
            "gpu_extra_allowance_usd": allowance, "api_response_accounted_usd": reported,
            "api_unresolved_reservations_usd": reserved, "conservative_accounted_usd": total,
            "within_original_budget": total <= controller["budget_usd"],
            "spending_cap_waived_by_user": waived,
            "note": "GPU costs are rate times owned billing duration, not the shared account balance change. "
                    "Unresolved API reservations are upper bounds, not confirmed charges. "
                    "Includes failed startup and saved-transcript scoring checks. "
                    "No unrelated pods or account balances are included."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-run", required=True)
    parser.add_argument("--da-run", required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load("configs/eval/delegated_harm.yaml").analysis
    runs = {"control": Path(args.control_run), "da": Path(args.da_run)}
    controllers = {name: run.parents[1]/"controller.json" for name, run in runs.items()}
    repos = {name: hf_repo_id(read(path)["eval_name"]) for name, path in controllers.items()}
    status_path = runs["control"].parents[1]/"comparison_completion.json"
    deadline = time.monotonic()+float(cfg.wait_timeout_seconds)
    try:
        while True:
            ready = []
            for name, run in runs.items():
                if not read(controllers[name]).get("terminated"):
                    continue
                try:
                    info = hf_api().dataset_info(repos[name])
                    summary = read(Path(hf_download(repos[name], "results/results.json", repo_type="dataset", revision=info.sha)))
                    if summary.get("score_version") == "evidence-actions-v3" and summary["recorded"] == summary["scheduled"]:
                        ready.append(name)
                except Exception as exc:
                    print(f"{name}: publication check {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            if len(ready) == 2:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Corrected publications did not both complete; saved rollouts remain available")
            print(f"Waiting for corrected publications; ready: {ready}", flush=True)
            time.sleep(float(cfg.poll_seconds))
        for name, run in runs.items():
            record = audit(run, read(controllers[name]))
            path = run/"metadata/cost_audit.json"
            save(path, record)
            hf_api().upload_file(path_or_fileobj=str(path), path_in_repo="metadata/cost_audit.json",
                repo_id=repos[name], repo_type="dataset", commit_message="Record owned GPU cleanup and bounded evaluation costs")
            print(f"{name}: conservative accounted cost ${record['conservative_accounted_usd']:.2f}; "
                  f"unresolved API reservations ${record['api_unresolved_reservations_usd']:.2f}", flush=True)
            assert record["within_original_budget"] or record["spending_cap_waived_by_user"], "Recorded cost exceeds the adapter budget"
        subprocess.run([sys.executable, "scratch/delegated_harm/compare.py", "--control", repos["control"],
                        "--da", repos["da"]], check=True)
        save(status_path, {"status": "complete", "finished": time.time(), "repositories": repos})
        print("Corrected scores, cost audits and comparison are complete", flush=True)
    except BaseException as exc:
        save(status_path, {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:500]})
        raise


if __name__ == "__main__":
    main()
