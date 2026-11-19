# ABOUTME: Dump one episode exactly as Agent B (our reviewer) saw it: its system prompt, every message,
# ABOUTME: its tool calls and tool results, its thinking and final output, then the decision.
# Run: uv run python scratch/iai_code_review/rollout.py <run dir> <row index from summarize.load>
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log, resolve_sample_attachments

sys.path.insert(0, str(Path(__file__).parent))
from summarize import load  # noqa: E402


def text_of(m) -> str:
    if isinstance(m.content, str):
        return m.content
    parts = []
    for c in m.content:
        if c.type == "reasoning":
            parts.append(f"<think>\n{c.reasoning}\n</think>")
        elif c.type == "text":
            parts.append(c.text)
    return "\n".join(parts)


def main() -> None:
    run, idx = Path(sys.argv[1]), int(sys.argv[2])
    r = load(run)[idx]
    s = [
        x
        for x in read_eval_log(r["file"]).samples
        if x.metadata["scenario_name"] == r["scenario"] and x.epoch == r["epoch"]
    ][0]
    s = resolve_sample_attachments(s)
    b_calls = [e for e in s.events if e.event == "model" and "vllm" in str(e.model)]
    print(
        f"# condition={r['condition']} scenario={r['scenario']} epoch={r['epoch']} "
        f"decision={r['decision']} reviewer_calls={len(b_calls)}\n"
    )
    last = b_calls[-1]
    print("TOOLS OFFERED TO THE REVIEWER:", ", ".join(t.name for t in last.tools), "\n")
    for m in last.input:
        print(f"\n===== {m.role.upper()} =====")
        print(text_of(m))
        for tc in getattr(m, "tool_calls", None) or []:
            print(f"[tool call] {tc.function}({tc.arguments})")
    print("\n===== ASSISTANT (final reviewer output) =====")
    msg = last.output.choices[0].message
    print(text_of(msg))
    for tc in msg.tool_calls or []:
        print(f"[tool call] {tc.function}({tc.arguments})")
    print("\n===== SUBMITTED DECISION =====")
    print(s.metadata.get("submitted_decision"))


if __name__ == "__main__":
    main()
