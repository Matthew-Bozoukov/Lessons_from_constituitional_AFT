# ABOUTME: Renders a model-eval-model run dir into one human-readable markdown file for
# ABOUTME: review: per doc, the metadata, planted flaw + gold/flawed pair, and full turns.
# Run: uv run scratch/review_model_eval_model_docs.py --run_dir output/model_eval_model/<dir>

from __future__ import annotations

import json
from pathlib import Path

import fire


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main(run_dir: str) -> None:
    """Write <run_dir>/review.md: every SFT record, fully readable end to end."""
    d = Path(run_dir)
    # stage_6 since the `final` rewrite stage landed (2026-08-07); stage_5 on older runs.
    sft = _read(next(p for p in (d / "stage_6_sft.jsonl", d / "stage_5_sft.jsonl")
                     if p.exists()))
    perturbed = {r["record_id"]: r for r in _read(d / "stage_3_perturbed.jsonl")}

    out = ["# model-eval-model review\n",
           f"Run: `{d}` — {len(sft)} documents. For flawed cells, check: (1) the pair "
           "differs ONLY in the planted flaw, (2) the critique/reflection finds it "
           "without being told. For good cells: no manufactured objections.\n"]
    for i, r in enumerate(sorted(sft, key=lambda r: r["metadata"]["cell"]), 1):
        md = r["metadata"]
        out.append(f"\n---\n\n## {i}. `{md['record_id']}`\n")
        out.append(f"**cell** {md['cell']} · **trait** {md['trait_name']} · "
                   f"**explicitness** {md['explicitness']}"
                   + (f" · **verdict** {md['verdict']}" if md.get("verdict") else ""))
        if md.get("flaw_type"):
            p = perturbed[md["record_id"]]
            out.append(f"\n**planted flaw** {md['flaw_type']} / {md['flaw_severity']} "
                       f"(length ratio {p.get('length_ratio')})")
            out.append(f"\n> **change_summary (metadata only, never trains):** "
                       f"{p['change_summary']}")
            out.append("\n<details><summary>gold response (pre-perturbation)</summary>\n\n"
                       f"{p['gold_response']}\n\n</details>")
        for m in r["messages"]:
            out.append(f"\n### {m['role']}\n")
            if m.get("reasoning_content"):
                out.append("**[private reasoning]**\n\n" + m["reasoning_content"] + "\n")
                out.append("**[visible reply]**\n")
            out.append(m["content"])
    dest = d / "review.md"
    dest.write_text("\n".join(out))
    print(f">>> {dest} ({len(sft)} docs)")


if __name__ == "__main__":
    fire.Fire(main)
