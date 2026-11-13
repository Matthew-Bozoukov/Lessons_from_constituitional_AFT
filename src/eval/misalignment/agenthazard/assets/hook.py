# ABOUTME: Claude Code tool hook inside the AgentHazard sandbox: appends every tool event (the
# ABOUTME: hook's stdin payload plus type, sequence, timestamp, turn) to one jsonl. Never blocks.
"""Mirrors the paper's `detail_logs`: one record per pre_tool / post_tool event carrying the hook
payload Claude Code passes (tool_name, tool_input, tool_response, session_id, cwd, ...). The turn
index comes from AGENTHAZARD_TURN, which the driver sets on each `docker exec`. Exit 0 always, with
no stdout, so the hook observes and never changes what the agent does.
"""

import datetime
import json
import os
import sys

LOG = "/opt/agenthazard/logs/tool_calls.jsonl"


def main() -> None:
    kind = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception as exc:  # a hook must never fail the tool call it observes
        payload = {"hook_parse_error": repr(exc)}
    try:
        with open(LOG, "a+", encoding="utf-8") as f:
            f.seek(0)
            n = sum(1 for line in f if f'"type": "{kind}"' in line) + 1
            record = {
                "type": kind,
                "sequence": f"{kind}_{n:03d}",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S.%f"
                )[:-3],
                "turn": int(os.environ.get("AGENTHAZARD_TURN", "0")),
                **payload,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()
    sys.exit(0)
