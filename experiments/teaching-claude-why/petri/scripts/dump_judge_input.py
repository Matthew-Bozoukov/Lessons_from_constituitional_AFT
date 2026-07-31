# ABOUTME: Dumps exactly what the judge receives - prompt plus the answer-tool rubric schema.
# ABOUTME: For reviewing or handing to someone else to critique the measurement.
"""Write a single readable document of the judge's complete input.

Worth having as a script rather than an ad-hoc command: the rubrics ARE the
measurement in this experiment. The judge never reads the constitution or the
seed - it scores against these field descriptions alone - so if a principle is
encoded wrongly here, every arm is mis-scored identically and nothing looks
broken. This makes that surface reviewable.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util as _u
import json
from pathlib import Path

from inspect_petri._judge.dimensions import judge_dimensions
from inspect_petri._judge.judge import _alignment_answer_type

_spec = _u.spec_from_file_location("rj", str(Path(__file__).with_name("rejudge.py")))
_rj = _u.module_from_spec(_spec)
_spec.loader.exec_module(_rj)  # type: ignore[union-attr]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-glob", default="logs/v2-e1/base/*.eval")
    ap.add_argument("--sample-id", default="P2d-agentic-clear-alert")
    ap.add_argument("--dimensions", default="configs/judge_dimensions.yaml")
    ap.add_argument("--out", default="output/judge_input_full.md")
    a = ap.parse_args()

    from inspect_ai.log import read_eval_log

    files = sorted(glob.glob(a.log_glob))
    if not files:
        raise SystemExit(f"no logs matched {a.log_glob}")
    log = read_eval_log(files[-1], resolve_attachments=True)
    samples = log.samples or []
    sample = next((s for s in samples if str(s.id) == a.sample_id), samples[0])

    prompt = _rj.JUDGE_PROMPT.format(transcript=_rj.render_transcript(sample))
    dims = judge_dimensions(a.dimensions)
    schema = _alignment_answer_type(dims).model_json_schema()
    props = schema["properties"]

    L: list[str] = []
    L.append("# Exactly what the judge receives")
    L.append("")
    L.append("Two things are sent per transcript: **(A)** the prompt below, and **(B)** an")
    L.append("`answer()` tool the judge is forced to call, whose field descriptions carry the")
    L.append("scoring rubrics.")
    L.append("")
    L.append("The judge sees **no constitution document, no seed brief, and no arm label** —")
    L.append("it cannot tell `base` from `dose-40-60`. The constitution reaches it only")
    L.append("through the rubric text in Part B.")
    L.append("")
    L.append(f"Example transcript: `{sample.id}` from the `base` arm.")
    L.append("")
    L.append("---")
    L.append("")
    L.append(f"## PART A — the prompt ({len(prompt):,} characters)")
    L.append("")
    L.append("```text")
    L.append(prompt)
    L.append("```")
    L.append("")
    L.append("---")
    L.append("")
    L.append(f"## PART B — the `answer()` tool ({len(props)} required fields, "
             f"{len(json.dumps(schema)):,} chars of schema)")
    L.append("")
    L.append("Every scoring dimension is an integer 1–10 with the range enforced by the")
    L.append("schema, so the judge cannot return anything else.")
    L.append("")
    for name, spec in props.items():
        if spec.get("type") == "integer":
            L.append(f"### `{name}` — integer {spec.get('minimum')}–{spec.get('maximum')}")
        else:
            L.append(f"### `{name}` — {spec.get('type', 'string')}")
        L.append("")
        L.append("```text")
        L.append((spec.get("description") or "(no description)").strip())
        L.append("```")
        L.append("")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
