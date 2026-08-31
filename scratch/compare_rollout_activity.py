# ABOUTME: Compare what the agent actually DID across ODCV arms — tool calls, steps, task
# ABOUTME: completion — so a low misalignment rate can be told apart from doing nothing.
"""Is a low misalignment rate alignment, or inactivity?

An agent that refuses everything cannot be misaligned. This counts, per rollout, how many
tool calls it issued, how many steps it took, whether it reached task_complete, and how long
its output was — on the cells the arms share, so a terse arm is visible as terse rather than
as safe.

Run: uv run python scratch/compare_rollout_activity.py --arms <key>=<dir>,<key>=<dir>
"""
import re
import statistics
from pathlib import Path

import fire

STEP = re.compile(r"^== Step \d+ ==$", re.M)
ROLE = re.compile(r"^role: (\w+)", re.M)
CMD = re.compile(r'"command":\s*"((?:[^"\\]|\\.)*)"')
COMPLETE = re.compile(r"task_complete", re.I)
REASON = re.compile(r"^reason: (.*?)(?=^(?:content|call|role):|\Z)", re.M | re.S)


def stats_for(run_dir: Path) -> dict:
    """Per-cell activity stats for one combined run directory."""
    out = {}
    for rec in sorted(run_dir.glob("agent_logs/*/experiments/*/**/messages_record.txt")):
        variant = next((p.rsplit("-", 1)[-1] for p in rec.parts
                        if p.endswith(("-mandated", "-incentivized"))), None)
        scen = rec.parent.name if not rec.parent.name.startswith("rollout") \
            else rec.parent.parent.name
        text = rec.read_text()
        blocks = STEP.split(text)
        asst = [b for b in blocks if (m := ROLE.search(b)) and m.group(1) == "assistant"]
        words = sum(len(r.group(1).split()) for b in asst for r in [REASON.search(b)] if r)
        out.setdefault(f"{variant}/{scen}", []).append({
            "steps": len(asst),
            "tool_calls": len(CMD.findall(text)),
            "completed": bool(COMPLETE.search(text)),
            "reason_words": words,
            "chars": len(text),
        })
    return out


def main(arms: str) -> None:
    """Print activity stats per arm on the cells they share.

    Args:
        arms: Comma-separated `label=path` pairs naming combined run directories.
    """
    pairs = [a.split("=", 1) for a in arms.split(",")]
    data = {label: stats_for(Path(p)) for label, p in pairs}
    shared = set.intersection(*(set(d) for d in data.values()))
    print(f"cells shared by all {len(data)} arms: {len(shared)}\n")
    print(f"{'arm':26s} {'tool calls':>11} {'steps':>8} {'completed':>11} "
          f"{'reason w':>10} {'rollouts':>9}")
    for label, d in data.items():
        rs = [r for k in shared for r in d[k]]
        tc = statistics.mean(r["tool_calls"] for r in rs)
        st = statistics.mean(r["steps"] for r in rs)
        cp = 100 * sum(r["completed"] for r in rs) / len(rs)
        rw = statistics.mean(r["reason_words"] for r in rs)
        print(f"{label:26s} {tc:>11.1f} {st:>8.1f} {cp:>10.0f}% {rw:>10.0f} {len(rs):>9}")
    print("\nzero-tool-call rollouts (the agent never acted):")
    for label, d in data.items():
        rs = [r for k in shared for r in d[k]]
        z = sum(1 for r in rs if r["tool_calls"] == 0)
        print(f"  {label:26s} {z:>4}/{len(rs)} ({100*z/len(rs):.0f}%)")


if __name__ == "__main__":
    fire.Fire(main)
