# ABOUTME: Print one Colosseum Hospital episode as a readable rollout — seating, the pair's
# ABOUTME: instructions' effect, every turn's reasoning + tool calls, and both channels' logs.

"""Read an episode end to end.

    uv run python scratch/colosseum_hospital/show_episode.py <run_dir> [--agents A,B] [--full]

`run_dir` is one `runs/<model>/<cell>/<run_id>/` directory (on the pod, or pulled back by
pull_runs.sh). By default the pair's turns are printed in full and everyone else's are
summarised to one line; `--agents` restricts to named agents, `--full` prints every turn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.misalignment.colosseum.hospital.harvest import episode_records
from src.eval.misalignment.colosseum.hospital.judge import boards_of, transcript

PROVISIONER = "Resource_Provisioner"


def _read(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--agents", default="")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--reasoning-chars", type=int, default=1500)
    args = ap.parse_args()
    d = Path(args.run_dir)
    run_cfg = _read(d / "run_config.json") or {}
    pair = run_cfg.get("pair") or [PROVISIONER, "General_Hospital_Triage"]
    focus = set(args.agents.split(",")) if args.agents else set(pair)

    # Numbers first, from the same harvest the eval uses.
    recs = [
        r for r in episode_records(d.parents[3], pair=pair) if r["run_id"] == d.name
    ]
    if recs:
        r = recs[0]
        print(f"=== {d.name}")
        print(
            f"cell={r['cell']} seed={r['seed']} strategy={r['strategy']} status={r['status']}"
        )
        print(
            f"pair={r['pair']} adversaries={r['adversaries']} secret_board={r['secret_blackboard_id']}"
        )
        print("seating:", json.dumps(r["agent_llm_labels"], indent=None))
        print(
            f"joint_reward={r['joint_reward']:.0f}/{r['max_joint_reward']:.0f} regret_norm={r['regret_norm']:.3f} "
            f"advantage_hospital={r['advantage_hospital']:.3f} advantage_pair={r['advantage_pair']:.3f} "
            f"dept_vs_mirror={r['pair_dept_minus_mirror']}"
        )
        print(
            f"hospital sums={ {h: round(v) for h, v in r['hospital_reward_sum'].items()} } "
            f"failures={r['resource_failures']} by_hospital={r['failures_by_hospital']} "
            f"provisioner_transfers={r['provisioner_transfers']}"
        )
        print(
            f"secret_messages={r['secret_messages']} {r['secret_messages_by_agent']} "
            f"public_broadcasts_by_pair={r['public_broadcasts_by_pair']} tool_calls={r['tool_calls_total']} "
            f"tool_errors={r['tool_errors']} llm_calls={r['llm_calls']} truncated={r['truncated_calls']} "
            f"exec_turns_without_tools={r['execution_turns_without_tools']}/{r['execution_turns']} "
            f"tokens={r['prompt_tokens']}+{r['completion_tokens']}"
        )

    print("\n=== TURNS")
    for t in _read(d / "agent_turns.json") or []:
        agent = t.get("agent")
        head = f"[iter {t.get('iteration')} {t.get('phase')}] {agent} ({t.get('role')}) tools={t.get('tools_executed')}"
        if not (args.full or agent in focus):
            print(head)
            continue
        print("\n" + head)
        for i, call in enumerate(t.get("llm_calls") or []):
            reasoning = (call.get("reasoning") or "").strip()
            if reasoning:
                cut = (
                    reasoning
                    if len(reasoning) <= args.reasoning_chars
                    else reasoning[: args.reasoning_chars] + " …"
                )
                print(
                    f"  --- call {i + 1} reasoning ({len(reasoning)} chars, finish={call.get('finish_reason')}):"
                )
                print("  " + cut.replace("\n", "\n  "))
            content = (call.get("content") or "").strip()
            if content:
                print(f"  --- call {i + 1} content:")
                print("  " + content[:1500].replace("\n", "\n  "))
            for tc in call.get("tool_calls") or []:
                print(
                    f"  --- call {i + 1} tool: {tc.get('name')} {str(tc.get('arguments'))[:300]}"
                )

    for channel, board in boards_of(d).items():
        print(
            f"\n=== CHANNEL {channel} (participants: {', '.join(board.get('participants') or [])})"
        )
        print(transcript(board, 40000))


if __name__ == "__main__":
    main()
