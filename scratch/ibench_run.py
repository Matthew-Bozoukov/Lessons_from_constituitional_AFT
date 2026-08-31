# ABOUTME: Run ImpossibleBench LiveCodeBench (conflicting split, minimal scaffold) against a local
# ABOUTME: vLLM endpoint and print the cheating rate. Runs ON the box in the ibench venv.
# Usage: /root/ibench/bin/python ibench_run.py <served_model_name> [limit] [split]

import os
import sys

# Import the livecodebench task module DIRECTLY (bypassing the package __init__, which pulls in
# the heavy swebench/inspect_evals deps we don't need for LiveCodeBench).
sys.path.insert(0, "/root/impossiblebench/src/impossiblebench")
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:8000/v1")
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

import livecodebench_tasks  # noqa: E402
from inspect_ai import eval as inspect_eval  # noqa: E402


def main() -> None:
    model = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 125
    split = sys.argv[3] if len(sys.argv) > 3 else "conflicting"
    task = livecodebench_tasks.impossible_livecodebench(
        split=split, agent_type="minimal", sandbox="local", limit=limit)
    logs = inspect_eval(
        task, model=f"openai/{model}", max_connections=16,
        log_dir=f"/root/ibench_logs/{model}_{split}", temperature=0.0)
    log = logs[0]
    print(f"\n=== RESULT: {model} | split={split} | limit={limit} ===")
    print(f"status: {log.status}")
    if log.results:
        print(f"samples: {log.results.completed_samples}/{log.results.total_samples}")
        for score in log.results.scores:
            for mname, m in score.metrics.items():
                # On the conflicting split, accuracy == pass-rate on impossible tests == CHEATING RATE.
                print(f"  {score.name}.{mname} = {m.value}")
    else:
        print("NO RESULTS (check errors)")


if __name__ == "__main__":
    main()
