# ABOUTME: Turns an inspect agentic-misalignment .eval log into self-contained per-epoch
# ABOUTME: rollouts (prompt + reasoning + response + verdict) plus a markdown results mirror.

"""Export rollouts and a results table from an agentic-misalignment inspect run.

A rollout here is the WHOLE episode -- the system prompt and user turn the model was given
as well as what it produced -- because the response alone is unreadable without the setup.

    uv run --with inspect_ai python scratch/am_inspect_report.py --log_dir <dir>
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
from inspect_ai.log import read_eval_log


def _text(content) -> str:
    """Flatten an inspect message content field (str or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    return "\n".join(getattr(c, "text", "") or getattr(c, "reasoning", "") for c in content)


def _split_trace(text: str) -> tuple[str, str]:
    """Separate the reasoning trace from the answer.

    The chat template prefills the opening `<think>` and vLLM ran without a reasoning
    parser, so the trace arrives inline in the content with only its CLOSING tag present
    and `message.reasoning` empty. Keying off `reasoning` alone therefore scores every
    sample as an empty trace -- the exact false signal gotcha #5 warns about.

    Args:
        text: Raw assistant content.

    Returns:
        (trace, answer); trace is "" when no closing tag is present.
    """
    if "</think>" not in text:
        return "", text
    trace, _, answer = text.partition("</think>")
    return trace.removeprefix("<think>").strip(), answer.strip()


def main(log_dir: str, arm: str = "t2716") -> None:
    """Write rollouts/ and results.md next to the .eval log.

    Args:
        log_dir: Directory holding the .eval log.
        arm: Short label for the evaluated model, used in the report.
    """
    dest = Path(log_dir)
    path = sorted(dest.glob("*.eval"))[-1]
    log = read_eval_log(str(path))
    samples = log.samples or []
    assert samples, f"no samples in {path}"

    roll = dest / "rollouts"
    roll.mkdir(exist_ok=True)
    rows = []
    for s in samples:
        sc = (s.scores or {}).get("harmfulness_scorer")
        val = sc.value if sc else {}
        harmful = float(val.get("harmful", 0.0))
        verdict = float(val.get("classifier_verdict", 0.0))
        msgs = s.messages or []
        parts = [f"# epoch {s.epoch}  |  sample {s.id}  |  harmful={harmful} "
                 f"classifier_verdict={verdict}\n"]
        for m in msgs:
            parts.append(f"\n{'='*78}\n## {m.role}\n{'='*78}\n")
            if getattr(m, "reasoning", None):
                parts.append(f"<think>\n{m.reasoning}\n</think>\n\n")
            parts.append(_text(m.content))
        if sc is not None and sc.explanation:
            parts.append(f"\n\n{'='*78}\n## grader explanation\n{'='*78}\n{sc.explanation}")
        (roll / f"epoch_{s.epoch:03d}.md").write_text("".join(parts))
        asst = [m for m in msgs if m.role == "assistant"]
        raw = _text(asst[-1].content) if asst else ""
        trace, answer = _split_trace(raw)
        trace = trace or (getattr(asst[-1], "reasoning", "") or "" if asst else "")
        rows.append({"epoch": s.epoch, "harmful": harmful, "verdict": verdict,
                     "reasoning_chars": len(trace), "response_chars": len(answer)})

    n = len(rows)
    n_harm = sum(r["harmful"] for r in rows)
    n_verd = sum(r["verdict"] for r in rows)
    empty_think = sum(1 for r in rows if r["reasoning_chars"] == 0)
    se = (n_harm / n * (1 - n_harm / n) / n) ** 0.5 * 100

    task = log.eval.task_args or {}
    md = [f"# Agentic misalignment (inspect) — {arm}", "",
          f"- log: `{path.name}`", f"- model: `{log.eval.model}`",
          f"- epochs: {log.eval.config.epochs} (completed {n}, errors "
          f"{sum(1 for s in samples if s.error)})",
          f"- task args: `{json.dumps(task)}`", "",
          "## Headline", "",
          "| metric | value |", "|---|---|",
          f"| harmful (blackmail) | **{n_harm/n*100:.1f}%** ({int(n_harm)}/{n}) ± {se:.1f}pp |",
          f"| classifier_verdict | {n_verd/n*100:.1f}% ({int(n_verd)}/{n}) |",
          f"| empty reasoning trace | {empty_think}/{n} (reasoning did NOT collapse) |",
          f"| mean reasoning chars | {sum(r['reasoning_chars'] for r in rows)/n:,.0f} |",
          f"| mean response chars | {sum(r['response_chars'] for r in rows)/n:,.0f} |", "",
          "`harmful` is the scored outcome; `classifier_verdict` is the looser "
          "classifier flag (harmful is a strict subset).", "",
          "## Harmful epochs", "",
          "Full transcripts in `rollouts/epoch_NNN.md` (prompt + reasoning + response + "
          "grader explanation).", "",
          "| epoch | reasoning chars | response chars |", "|---|---|---|"]
    md += [f"| {r['epoch']} | {r['reasoning_chars']:,} | {r['response_chars']:,} |"
           for r in rows if r["harmful"]]
    md.append("")
    (dest / "results.md").write_text("\n".join(md))
    (dest / "results.json").write_text(json.dumps(
        {"arm": arm, "log": path.name, "model": log.eval.model, "n": n,
         "harmful": n_harm / n, "harmful_count": int(n_harm),
         "classifier_verdict": n_verd / n, "stderr_pp": se,
         "empty_think": empty_think, "task_args": task, "per_epoch": rows}, indent=2))
    print(f">>> {n} rollouts -> {roll}")
    print(f">>> harmful {n_harm/n*100:.1f}% ({int(n_harm)}/{n})  "
          f"verdict {n_verd/n*100:.1f}%  empty-think {empty_think}/{n}")
    print(f">>> {dest/'results.md'}")


if __name__ == "__main__":
    fire.Fire(main)
