#!/usr/bin/env python3
"""Render an extracted transcript JSON into compact readable markdown."""
import json
import sys
import textwrap

TOOL_TRIM = int(sys.argv[2]) if len(sys.argv) > 2 else 900
REASON_TRIM = int(sys.argv[3]) if len(sys.argv) > 3 else 1600


def trim(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + f" ...[+{len(s)-n} chars]"


def render(path):
    d = json.load(open(path, encoding="utf-8"))
    out = []
    out.append(f"# {d['uid']}  (target tool calls: {d['target_tool_calls']})")
    sc = d["scores"]
    out.append("SCORES: " + ", ".join(f"{k}={sc[k]}" for k in
               ["concerning", "disappointing", "needs_attention",
                "concerning_with_no_mitigating_factors", "cooperation_with_human_misuse",
                "unprompted_deception_toward_user", "unfaithful_thinking",
                "scenario_realism", "auditor_failure", "input_hallucination",
                "broken_tool_use", "eval_awareness"] if k in sc))
    out.append("\n## JUDGE SUMMARY\n" + str(d.get("judge_summary") or "(none)"))
    out.append("\n## JUDGE EXPLANATION\n" + (d.get("judge_explanation") or ""))
    out.append("\n## TRANSCRIPT (target turns, chronological)")
    for t in d["turns"]:
        out.append(f"\n--- TURN {t['turn']} (ctx={t['context_len']}"
                   + (", ROLLBACK" if t["rollback_detected"] else "") + ") ---")
        for m in t["new_input"]:
            body = trim(m["content"], TOOL_TRIM if m["role"] == "tool" else 2500)
            out.append(f"[{m['role'].upper()}] {body}")
        o = t["target_output"]
        out.append("[TARGET] " + trim(o["content"], REASON_TRIM))
        if o["tool_calls"]:
            for fn, args in zip(o["tool_calls"], o["tool_args"]):
                out.append(f"  >> TARGET TOOL CALL: {fn}({trim(json.dumps(args, ensure_ascii=False), 400)})")
        else:
            out.append("  >> (no tool call this turn)")
    return "\n".join(out)


if __name__ == "__main__":
    print(render(sys.argv[1]))
