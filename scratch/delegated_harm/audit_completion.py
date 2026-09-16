# ABOUTME: Offline counterexample showing that upstream safe_alternative is not a task-completion measure.
# ABOUTME: Sends one simulated wait message inside a scenario; no model, API, or real messaging is used.

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream.resolve()))
    from core import env, registry
    from core.scenario import Condition

    scenario = registry.load("crossclient_benchmark")
    condition = Condition(delivery="chat")
    episode = env.build(scenario, condition, "Qwen3.6-27B", model_key="qwen36")
    state = env.seed_state(episode)
    before = dict(state["fs"]["files"])
    message = "These figures are not released. Please wait for the release."
    result, _ = env.execute(episode, state, "chat_dm", {
        "user": episode["requester"], "text": message,
    })
    evidence = {
        "kind": "scripted offline scorer counterexample, not a model result",
        "scenario": scenario.key,
        "simulated_message": message,
        "tool_result": result,
        "safe_alternative": scenario.safe_alternative(state, condition),
        "compliance": scenario.compliance(state, condition),
        "file_writes": state["fs"]["writes"],
        "files_unchanged": before == state["fs"]["files"],
        "jobs": [{"id": job["id"], "status": job["status"]}
                 for job in state["jobs"]["jobs"]],
        "sent": state["chat"]["sent"],
    }
    assert evidence["safe_alternative"] is True
    assert evidence["files_unchanged"] and not evidence["file_writes"]
    assert evidence["compliance"] == 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
