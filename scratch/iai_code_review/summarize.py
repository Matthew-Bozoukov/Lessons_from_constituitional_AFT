# ABOUTME: Summarise a run_code_review.py run dir: per condition, did Agent B use its tools and decide,
# ABOUTME: approval rate (Wilson 95% CI), Agent A API cost per episode; writes a dated row-chart PDF.
# Run: uv run python scratch/iai_code_review/summarize.py output/iai_code_review/<run> [--transcript N]
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from collections import Counter
from pathlib import Path

from inspect_ai.log import read_eval_log

# OpenRouter x-ai/grok-4.20, endpoint `xai`, fetched 2026-11-19 from /api/v1/models/x-ai/grok-4.20/endpoints
PRICE = {"input": 1.25e-6, "output": 2.5e-6, "cache_read": 0.2e-6}
CONDITIONS = ["baseline", "persuasion", "deception"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def api_cost(usage) -> float:
    cache = usage.input_tokens_cache_read or 0
    return (
        usage.input_tokens * PRICE["input"]
        + cache * PRICE["cache_read"]
        + usage.output_tokens * PRICE["output"]
    )


def load(run: Path) -> list[dict]:
    rows = []
    for f in sorted(run.glob("*/*.eval")):
        log = read_eval_log(str(f))
        for s in log.samples or []:
            md = s.metadata or {}
            env = md.get("environment_state") or {}
            b_actions = [
                a["action"]
                for a in env.get("audit_log", [])
                if a.get("actor") == "code_reviewer"
            ]
            dec = (md.get("submitted_decision") or {}).get("decision")
            usage = s.model_usage or {}
            a_model = md.get("model_a") or log.eval.task_args.get("model_a")
            b_model = md.get("model_b") or log.eval.task_args.get("model_b")
            rows.append(
                {
                    "file": str(f),
                    "condition": log.eval.task_args["capability"],
                    "scenario": md.get("scenario_name"),
                    "epoch": s.epoch,
                    "error": s.error is not None,
                    "decision": dec,
                    "approved": dec == "APPROVED",
                    "turns": md.get("conversation_turns"),
                    "b_actions": b_actions,
                    "b_tool_calls": len(b_actions),
                    "a_cost": api_cost(usage[a_model])
                    if a_model in usage
                    else float("nan"),
                    "b_tokens": usage[b_model].model_dump(exclude_none=True)
                    if b_model in usage
                    else None,
                    "seconds": s.total_time,
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--transcript", type=int, default=None, help="print row N in full")
    args = ap.parse_args()
    run = Path(args.run)
    rows = load(run)
    meta = json.loads((run / "run_meta.json").read_text())

    summary = {}
    for c in CONDITIONS:
        rs = [r for r in rows if r["condition"] == c]
        if not rs:
            continue
        ok = [r for r in rs if not r["error"]]
        k = sum(r["approved"] for r in ok)
        lo, hi = wilson(k, len(ok))
        summary[c] = {
            "episodes": len(rs),
            "errored": len(rs) - len(ok),
            "decided": sum(r["decision"] is not None for r in ok),
            "used_a_tool_besides_submit": sum(
                any(a != "REVIEW_DECISION_SUBMITTED" for a in r["b_actions"]) for r in ok
            ),
            "approved": k,
            "approval_rate": k / len(ok) if ok else float("nan"),
            "ci95": [lo, hi],
            "by_scenario": dict(
                Counter(f"{r['scenario']}: {r['decision']}" for r in ok)
            ),
            "b_action_counts": dict(Counter(a for r in ok for a in r["b_actions"])),
            "mean_agent_a_usd": sum(r["a_cost"] for r in ok) / len(ok)
            if ok
            else float("nan"),
            "mean_episode_seconds": sum(r["seconds"] or 0 for r in ok) / len(ok)
            if ok
            else float("nan"),
        }
    print(json.dumps(summary, indent=2, default=str))
    (run / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    if args.transcript is not None:
        r = rows[args.transcript]
        log = read_eval_log(r["file"])
        s = [
            x
            for x in log.samples
            if x.metadata.get("scenario_name") == r["scenario"]
            and x.epoch == r["epoch"]
        ][0]
        print(
            json.dumps(
                {
                    k: r[k]
                    for k in ("condition", "scenario", "epoch", "decision", "b_actions")
                },
                indent=2,
            )
        )
        for m in s.messages:
            print(f"\n===== {m.role.upper()} =====")
            print(m.text if hasattr(m, "text") else m.content)
            for tc in getattr(m, "tool_calls", None) or []:
                print(f"[tool call] {tc.function}({json.dumps(tc.arguments)[:600]})")

    import matplotlib.pyplot as plt

    from src.utils import save_figure_pdf

    conds = [c for c in CONDITIONS if c in summary]
    fig, ax = plt.subplots(figsize=(6.5, 0.7 + 0.6 * len(conds)))
    for i, c in enumerate(conds):
        s = summary[c]
        y = len(conds) - 1 - i
        ax.plot(
            [100 * s["ci95"][0], 100 * s["ci95"][1]],
            [y, y],
            color="#8a8a8a",
            lw=3,
            solid_capstyle="round",
        )
        ax.plot(100 * s["approval_rate"], y, "o", color="#5a5a5a", ms=8)
        ax.text(
            101,
            y,
            f"{100 * s['approval_rate']:.0f}%  ({s['approved']}/{s['episodes'] - s['errored']})",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(list(reversed(conds)))
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(
        "Agent B (our reviewer) approves the vulnerable code, % of episodes (95% CI)"
    )
    tgt = meta.get("target", meta.get("model_b"))
    ax.set_title(
        f"Inter-Agent Influence code review, reviewer = {tgt.split('/')[-1]}\n"
        f"author = {meta['model_a'].split('/')[-1]}, {meta['epochs']} epochs x 2 scenarios",
        fontsize=9,
    )
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    date = dt.date.today().isoformat()
    out = Path("output/iai_code_review") / f"{date}_iai_code_review_approval_plain.pdf"
    save_figure_pdf(
        fig,
        out,
        title="Inter-Agent Influence code review sanity check",
        artifacts=[tgt],
        script="scratch/iai_code_review/summarize.py",
    )
    print(f">>> figure {out}")


if __name__ == "__main__":
    main()
