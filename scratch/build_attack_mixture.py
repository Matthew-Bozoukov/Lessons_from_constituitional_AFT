# ABOUTME: Builds the attack-variant training mixture: 9,284 Table2 + 716 four-framing rows.
# ABOUTME: Renders the attack rows with the verified Qwen template; Table2 rows carried verbatim.
"""Assemble the training mixture for the attack-variant (robustness) arm.

9,284 Table2 rows are taken verbatim from the da716 mixture. The 716 attack-variant rows
(179 scenarios x {original, hidden, incremental, authority}; the assistant reply is the same
across a scenario's four framings) are rendered to the same `text` template, round-trip
verified against the corpus before use.

Run: uv run python scratch/build_attack_mixture.py
"""
import collections
import json
from pathlib import Path

import fire

from src.utils import read_jsonl, timestamp, write_run_meta

DA716 = "data/t2_9284_da716_10k.jsonl"
ATTACK = "output/attack_variants/20260828_102411/attack_variants.jsonl"
OUT_ROOT = Path("output/attack_mixture")
TMPL = ("<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{answer}<|im_end|>\n")


def _render(row: dict) -> str:
    """Render one attack-variant row's messages into the verified Qwen chat template."""
    m = {x["role"]: x for x in row["messages"]}
    a = m["assistant"]
    return TMPL.format(system=m["system"]["content"], user=m["user"]["content"],
                       reasoning=a.get("reasoning_content", ""), answer=a["content"])


def main() -> None:
    """Write the 9,284 Table2 + 716 attack-variant mixture."""
    table2 = [d for d in read_jsonl(DA716) if not d.get("scenario_id")]
    assert len(table2) == 9284, f"expected 9284 Table2, got {len(table2)}"

    attack = list(read_jsonl(ATTACK))
    rendered = []
    for r in attack:
        m = r["metadata"]
        rendered.append({"source": "difficult_advice_v2_attack",
                         "text": _render(r),
                         "scenario_id": m["scenario_id"], "trait_id": m["trait_id"],
                         "variant": m["variant"]})
    assert len(rendered) == 716, f"expected 716 attack rows, got {len(rendered)}"

    out = table2 + rendered
    out_dir = OUT_ROOT / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "t2_9284_attack716_10k.jsonl"
    with path.open("w") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    stats = {"total": len(out), "table2": len(table2), "attack": len(rendered),
             "synth_source": "difficult_advice_v2_attack",
             "synth_fraction": round(len(rendered) / len(out), 4),
             "per_variant": dict(collections.Counter(r["variant"] for r in rendered)),
             "scenarios": len({r["scenario_id"] for r in rendered})}
    (path.with_suffix(".jsonl.stats.json")).write_text(json.dumps(stats, indent=2))
    write_run_meta(out_dir, {"da716": DA716, "attack": ATTACK}, stats)
    print(json.dumps(stats, indent=2))
    print(f"\n>>> {path}")


if __name__ == "__main__":
    fire.Fire(main)
