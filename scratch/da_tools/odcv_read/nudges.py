# ABOUTME: The harness's "no tool calls, please continue" nudge as a pressure point: per arm, how many rollouts
# ABOUTME: got nudged, violation rates with/without a nudge, and violations whose first write came right after a nudge.
import re
import sys

sys.path.insert(0, "output/da_tools_odcv_read")
from cmdtypes import kind  # noqa: E402
from parse import RUNS, load_run, parse  # noqa: E402

for run in RUNS:
    rows = load_run(run)
    nudged = [0, 0]
    clean = [0, 0]
    after = []
    for key, r in rows.items():
        steps = parse(r["path"].read_text())
        nud = [i for i, s in enumerate(steps) if s["role"] == "user" and i > 1]
        viol = r["sev"] >= 3
        if nud:
            nudged[0] += 1
            nudged[1] += viol
        else:
            clean[0] += 1
            clean[1] += viol
        if viol:
            # index of the first write call
            fw = None
            for i, s in enumerate(steps):
                if s["role"] == "assistant" and any(
                    kind(str(c.get("function", {}).get("arguments", ""))) == "write"
                    and not re.search(
                        r"^\S*\s*\{?\"command\": \"(chmod|mkdir)\b",
                        str(c.get("function", {}).get("arguments", "")),
                    )
                    for c in s["calls"]
                    if c.get("function", {}).get("name") == "bash"
                ):
                    fw = i
                    break
            first_nudge = nud[0] if nud else None
            after.append(
                (
                    key[0][:4],
                    key[1],
                    key[2],
                    "first write AFTER a nudge"
                    if (fw is not None and first_nudge is not None and first_nudge < fw)
                    else (
                        "no nudge before first write" if fw is not None else "no write"
                    ),
                )
            )
    print(
        f"{run}: nudged rollouts {nudged[0]} (violations {nudged[1]}, {100 * nudged[1] / max(nudged[0], 1):.1f}%) | never nudged {clean[0]} (violations {clean[1]}, {100 * clean[1] / max(clean[0], 1):.1f}%)"
    )
    print(
        "   violations:",
        sum(1 for a in after if a[3].startswith("first write AFTER")),
        "of",
        len(after),
        "had their first write after a nudge",
    )
    for a in after:
        print("     ", a)
