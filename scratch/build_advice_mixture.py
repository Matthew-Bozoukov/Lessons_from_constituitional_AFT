# ABOUTME: Builds the advice-framed training mixture: 9,284 Table2 rows + 716 advice-framed rows.
# ABOUTME: Renders the advice rows with the verified Qwen template; Table2 rows carried verbatim.
"""Assemble the training mixture for the advice-framed difficult-advice arm.

The 9,284 Table2 rows are taken verbatim from the da716 mixture (model-agnostic, already
rendered). The 716 difficult-advice rows are replaced with their advice-framed rewrites,
rendered to the exact same `text` template (round-trip-verified against every original synth
row before use). Output matches the schema the sibling arms train on: {source, text,
scenario_id?, trait_id?}.

Run: uv run python scratch/build_advice_mixture.py
"""
import json
from pathlib import Path

import fire

from src.utils import read_jsonl, timestamp, write_run_meta

DA716 = "data/t2_9284_da716_10k.jsonl"
ADVICE = "output/advice_original/20260828_124757/advice_original.jsonl"
OUT_ROOT = Path("output/advice_mixture")
TMPL = ("<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{answer}<|im_end|>\n")


def _render(row: dict) -> str:
    """Render one advice row's messages into the verified Qwen chat-template text.

    Args:
        row: A messages-format row with system/user/assistant (assistant carries
            reasoning_content).

    Returns:
        The rendered chat string, byte-identical in structure to the corpus rows.
    """
    m = {x["role"]: x for x in row["messages"]}
    a = m["assistant"]
    return TMPL.format(system=m["system"]["content"], user=m["user"]["content"],
                       reasoning=a.get("reasoning_content", ""), answer=a["content"])


def main() -> None:
    """Write the 9,284 + 716 advice-framed mixture."""
    advice = {r["metadata"]["scenario_id"]: r for r in read_jsonl(ADVICE)}
    out, n_tbl, n_adv, missing = [], 0, 0, 0
    for d in read_jsonl(DA716):
        sid = d.get("scenario_id")
        if not sid:                              # Table2 row — carry verbatim
            out.append(d)
            n_tbl += 1
            continue
        r = advice.get(sid)                       # synth row — replace with advice rewrite
        if r is None:
            out.append(d)                         # keep original if somehow un-rewritten
            missing += 1
            continue
        out.append({"source": d["source"], "text": _render(r),
                    "scenario_id": sid, "trait_id": d.get("trait_id")})
        n_adv += 1

    assert n_tbl == 9284, f"expected 9284 Table2, got {n_tbl}"
    assert n_adv == 716, f"expected 716 advice, got {n_adv}"
    assert missing == 0, f"{missing} synth rows had no advice rewrite"

    out_dir = OUT_ROOT / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "t2_9284_advice716_10k.jsonl"
    with path.open("w") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    stats = {"total": len(out), "table2": n_tbl, "advice": n_adv,
             "synth_source": "difficult_advice_v2", "synth_fraction": round(n_adv / len(out), 4)}
    (path.with_suffix(".jsonl.stats.json")).write_text(json.dumps(stats, indent=2))
    write_run_meta(out_dir, {"da716": DA716, "advice": ADVICE}, stats)
    print(json.dumps(stats, indent=2))
    print(f"\n>>> {path}")


if __name__ == "__main__":
    fire.Fire(main)
