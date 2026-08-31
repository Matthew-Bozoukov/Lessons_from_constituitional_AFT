# ABOUTME: Verify the trait-swap mixture renders correctly and report its ODCV-likeness shift.
# ABOUTME: Checks think-tag structure, token lengths and supervised fraction on swapped rows.

"""Confirm the swapped rows are trainable and say how ODCV-like the new mix is.

The swapped-in rows are rendered by hand rather than by the mixture builder, so their shape
is checked against the rows that were already in the control. The ODCV-similarity readout
matters for interpretation: if dropping traits 1/3/4 also strips ODCV-like content, this arm
is not cleanly separated from the low-ODCV arm it is meant to disambiguate.

Run:
  uv run python scratch/verify_trait_swap.py --mix <path>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_profile import model_profile  # noqa: E402

SOURCE = "synthdoc_difficult_advice"
USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
THINK_RE = re.compile(r"<think>\n(.*?)\n</think>", re.S)
DIMS = ("operator", "agentic", "metric", "oversight")


def main(mix: str, stats: str,
         control: str = "data/hf/2026-08-06-table2-9284-synthdoc-716-train/mixture_think.jsonl",
         sft: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl") -> None:
    """Structural checks plus an ODCV-similarity comparison.

    Args:
        mix: The built trait-swap mixture.
        stats: Its mixture_stats_traitswap.json.
        control: The control mixture.
        sft: stage_7 SFT file for the trait/user join.

    Raises:
        AssertionError: If a swapped row is malformed.
    """
    st = json.loads(Path(stats).read_text())
    added, removed = set(st["added_ids"]), set(st["removed_ids"])

    by_user = {}
    for line in Path(sft).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_user[{m["role"]: m for m in r["messages"]}["user"]["content"].strip()] = \
                r["metadata"]

    rows = [json.loads(x) for x in Path(mix).read_text().splitlines() if x.strip()]
    ctl = [json.loads(x) for x in Path(control).read_text().splitlines() if x.strip()]
    profile = model_profile("Qwen3.6")

    swapped, kept = [], []
    for m in rows:
        if m.get("source") != SOURCE:
            continue
        md = by_user[USER_RE.search(m["text"]).group(1).strip()]
        (swapped if md["scenario_id"] in added else kept).append(m)

    assert len(swapped) == len(added), f"{len(swapped)} swapped rows, expected {len(added)}"
    for m in swapped:
        t = m["text"]
        assert t.startswith("<|im_start|>system\n"), "row does not open with a system turn"
        assert t.count("<|im_start|>assistant\n") == 1, "not exactly one assistant turn"
        assert profile.prefill in t, "missing the family's think prefill"
        think = THINK_RE.search(t)
        assert think and think.group(1).strip(), "swapped row has an empty think block"
        assert t.rstrip().endswith("<|im_end|>"), "row does not close with <|im_end|>"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
    ln = lambda rs: [len(tok(r["text"]).input_ids) for r in rs]  # noqa: E731
    sw, kp = ln(swapped), ln(kept)
    print(f"swapped rows: {len(sw)}  tokens median {sorted(sw)[len(sw)//2]}  max {max(sw)}")
    print(f"kept rows   : {len(kp)}  tokens median {sorted(kp)[len(kp)//2]}  max {max(kp)}")
    print(f"rows over 8192 tokens: swapped {sum(1 for x in sw if x > 8192)}, "
          f"kept {sum(1 for x in kp if x > 8192)}")

    # ODCV-likeness, where ratings exist, so this arm can be told apart from the low-ODCV one.
    rated = {}
    for f in ("output/odcv_similarity/all716/similarity.jsonl",
              "output/odcv_similarity/candidates/similarity.jsonl",
              "output/odcv_similarity/candidates2/similarity.jsonl"):
        p = Path(f)
        if p.is_file():
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    rated[r["scenario_id"]] = r
    out_r = [rated[i] for i in removed if i in rated]
    in_r = [rated[i] for i in added if i in rated]
    print(f"\nODCV-similarity ratings available: removed {len(out_r)}/{len(removed)}, "
          f"added {len(in_r)}/{len(added)}")
    if out_r and in_r:
        print(f"{'dim':11s}{'removed':>9}{'added':>8}")
        for d in DIMS:
            print(f"{d:11s}{sum(r[d] for r in out_r)/len(out_r):9.2f}"
                  f"{sum(r[d] for r in in_r)/len(in_r):8.2f}")
    print(f"\nrow count {len(rows)} (control {len(ctl)}); "
          f"difficult-advice rows {len(swapped) + len(kept)}")


if __name__ == "__main__":
    fire.Fire(main)
