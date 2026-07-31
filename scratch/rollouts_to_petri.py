# ABOUTME: Convert an agentic-misalignment results/ rollout tree into a Visualizer
# ABOUTME: petri-run content record (index.md + results/transcripts.jsonl + scores.json).

from __future__ import annotations

import json
import re
from pathlib import Path

import fire


def _clean(text: str) -> str:
    """Strip surrounding <reasoning> tags from a judge rationale, if present."""
    return re.sub(r"</?reasoning>", "", text or "", flags=re.I).strip()


def main(
    results_dir: str,
    out_root: str = "Visualizer/content/petri-runs",
    slug: str = "qwen36-4060-agentic",
    served_name: str = "vllm_qwen3",
    target_model_id: str = "qwen3.6-27b",
    target_checkpoint_id: str = "difficult-advice-mixture-40-60",
) -> None:
    """Build a petri-run content record from an agentic-misalignment rollout tree.

    Args:
        results_dir: Path to results/<expid> (holds models/<served>/... and prompts/).
        out_root: Visualizer petri-runs content directory.
        slug: Record directory name under out_root.
        served_name: Sub-dir under models/ to read rollouts from.
        target_model_id: Model identifier for frontmatter.
        target_checkpoint_id: Checkpoint identifier for frontmatter.
    """
    res = Path(results_dir)
    prompts = res / "prompts"
    model_root = res / "models" / served_name
    assert model_root.is_dir(), f"no rollouts at {model_root}"

    out = Path(out_root) / slug
    (out / "results").mkdir(parents=True, exist_ok=True)

    transcripts = []
    n_harm = 0
    by_cat: dict[str, dict[str, int]] = {}

    for cond_dir in sorted(model_root.iterdir()):
        if not cond_dir.is_dir():
            continue
        cond = cond_dir.name
        scenario = cond.split("_")[0]
        pdir = prompts / cond
        system = (pdir / "system_prompt.txt").read_text() if (pdir / "system_prompt.txt").is_file() else ""
        user_instr = (pdir / "user_prompt.txt").read_text() if (pdir / "user_prompt.txt").is_file() else ""
        email = (pdir / "email_content.txt").read_text() if (pdir / "email_content.txt").is_file() else ""
        user = (user_instr + ("\n\n" + email if email else "")).strip()

        for sdir in sorted(cond_dir.iterdir()):
            resp = sdir / "response.json"
            if not resp.is_file():
                continue
            d = json.loads(resp.read_text())
            cls = d.get("classification") or {}
            harmful = bool(cls.get("harmful_behavior"))
            n_harm += harmful
            slot = by_cat.setdefault(scenario, {"harmful": 0, "total": 0})
            slot["harmful"] += harmful
            slot["total"] += 1

            transcripts.append({
                "id": f"{cond}/{sdir.name}",
                "scenario_id": cond,
                "category": scenario,
                "outcome": "harmful" if harmful else "safe",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": d.get("raw_response", "")},
                ],
                "scores": {"harmful_behavior": int(harmful)},
                "judge_summary": _clean(cls.get("classifier_reasoning", "")),
                "tags": cond.split("_"),
            })

    (out / "results" / "transcripts.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in transcripts))

    total = len(transcripts)
    # Map agentic-misalignment counts onto the Visualizer's petri `scores` schema:
    # audits = rollouts in the category, concerning = harmful, eval_aware = 0 (n/a),
    # so the viewer's derived "pass" = audits - concerning - eval_aware = safe count.
    scores = {
        "by_category": [
            {
                "category": k,
                "audits": v["total"],
                "concerning": v["harmful"],
                "eval_aware": 0,
                "mean_realism": 0,
            }
            for k, v in by_cat.items()
        ],
        "outcomes": {"harmful": n_harm, "safe": total - n_harm},
        "retained_transcripts": total,
        "filtered_for_realism": 0,
        "misalignment_rate": round(n_harm / total, 4) if total else 0.0,
    }
    (out / "results" / "scores.json").write_text(json.dumps(scores, indent=2))

    frontmatter = f"""---
title: "Agentic misalignment rollouts — {target_checkpoint_id}"
date: 2026-07-30
summary: "{total} agentic-misalignment rollouts ({target_model_id} / {target_checkpoint_id}); overall misalignment {100*scores['misalignment_rate']:.1f}%."
status: complete
petri_run_id: {slug}
petri_version: agentic-misalignment
target_model_id: {target_model_id}
target_checkpoint_id: {target_checkpoint_id}
judge_model_id: google/gemini-3-flash-preview
models:
  - {target_model_id}
tags:
  - agentic-misalignment
  - blackmail
  - leaking
metrics:
  agentic_misalignment_rate:
    value: {scores['misalignment_rate']}
    unit: proportion
    lower_is_better: true
---

# Agentic misalignment rollouts — {target_checkpoint_id}

## Summary

{total} rollouts from the Anthropic agentic-misalignment honeypots (blackmail + leaking),
served as `{served_name}`. Judge: gemini-3-flash-preview. Overall misalignment rate
**{100*scores['misalignment_rate']:.1f}%** ({n_harm}/{total}).

Per-category rates:

{chr(10).join(f"- **{k}**: {100*v['harmful']/v['total']:.1f}% ({v['harmful']}/{v['total']})" for k, v in by_cat.items())}

Open the transcripts below to read each rollout with its judge rationale.
"""
    (out / "index.md").write_text(frontmatter)
    print(f">>> wrote {out}/index.md")
    print(f">>> {total} transcripts, misalignment {100*scores['misalignment_rate']:.1f}%")


if __name__ == "__main__":
    fire.Fire(main)
