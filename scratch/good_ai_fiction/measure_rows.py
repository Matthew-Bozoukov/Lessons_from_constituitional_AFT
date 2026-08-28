# ABOUTME: Trainable-token accounting for a Good AI Fiction run, row by row, with the
# ABOUTME: trainer's own mask — so the corpus is compared to DA-716 in the trained unit.
"""Run: PYTHONPATH=. python scratch/good_ai_fiction/measure_rows.py --run <run dir>

Renders each interchange row exactly as the mixture builder would, runs
`src.train.masking.build_labels` over it, and attributes every supervised token to the
reasoning trace or to the visible answer. Writes `<run>/token_stats.json` — one entry per
row plus corpus totals — which `select_rows.py` and `build_browser.py` both read.

The renderer is IMPORTED from scratch/build_t2_9284_da716_mixture.py rather than copied:
the whole point of measuring in the trainer's unit is that the text measured here is
byte-identical to the text the trainer will see, and a second copy of the renderer is the
obvious way for that to stop being true.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch"))

from build_t2_9284_da716_mixture import render  # noqa: E402
from src.model_profile import model_profile  # noqa: E402
from src.train.masking import assistant_spans, build_labels, forced_spans  # noqa: E402

MODEL = "Qwen/Qwen3.6-27B"
MAX_LEN = 8192
CLOSE = "</think>"

# The slice this corpus is built to replace, measured the same way
# (scratch/good_ai_fiction/da_baseline.json).
DA716 = {
    "rows": 716, "trainable": 832_064, "reasoning": 421_163, "answer": 410_901,
    "per_row_trainable_median": 1141, "per_row_reasoning_median": 584,
    "per_row_answer_median": 557,
}


def ids_with_offsets(text: str, tok, prof):
    """The segmented tokenization build_labels uses, keeping the offset mapping.

    Same cuts at the forced-span boundaries, so token-for-token this is the stream the
    trainer sees; the offsets are what let a supervised token be attributed to the
    reasoning trace or to the answer.
    """
    kw = dict(header=prof.assistant_header, turn_end=prof.turn_end)
    prefills = forced_spans(text, assistant_spans(text, **kw), prof.prefill,
                            prof.empty_think)
    cuts = sorted({0, len(text), *(e for sp in prefills for e in sp)})
    ids, offs = [], []
    for a, b in zip(cuts, cuts[1:]):
        enc = tok(text[a:b], add_special_tokens=False, return_offsets_mapping=True)
        ids += enc["input_ids"]
        offs += [(a + s, a + e) for s, e in enc["offset_mapping"]]
    return ids[:MAX_LEN], offs[:MAX_LEN]


def quantiles(xs: list[int]) -> dict:
    xs = sorted(xs)
    p = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]  # noqa: E731
    return {"min": xs[0], "p10": p(.10), "p25": p(.25), "median": p(.50),
            "p75": p(.75), "p90": p(.90), "max": xs[-1],
            "mean": round(statistics.mean(xs), 1)}


def measure(rows: list[dict], tok, prof) -> list[dict]:
    """Per-row token counts, keyed by scenario_id, with the metadata a selector needs."""
    out = []
    for r in rows:
        text = render(r["messages"])
        labels = build_labels(text, tok, MAX_LEN, prof)["labels"]
        _, offs = ids_with_offsets(text, tok, prof)
        assert len(offs) == len(labels), "offset/label mismatch — segmentation drifted"
        close = text.find(CLOSE)
        close_end = close + len(CLOSE) if close != -1 else -1
        trainable = reasoning = answer = 0
        for (a, b), v in zip(offs, labels):
            if v == -100:
                continue
            trainable += 1
            if close_end != -1 and b <= close_end:
                reasoning += 1
            else:
                answer += 1
        md = r.get("metadata", {})
        out.append({
            "scenario_id": md.get("scenario_id", ""),
            "rendered": len(labels), "trainable": trainable,
            "reasoning": reasoning, "answer": answer,
            "ratio": round(reasoning / max(answer, 1), 3),
            "has_think": close != -1,
            **{k: md.get(k, "") for k in
               ("trait_id", "world", "stakes", "source_type", "source_archetype",
                "narrative_form", "length_band", "revise_status", "domain",
                "ai_name")},
            "judge_persona": (md.get("judge_persona") or {}).get("verdict", ""),
            "judge_pattern": (md.get("judge_pattern") or {}).get("verdict", ""),
        })
    return out


def main(run: str, dataset: str = "dataset.jsonl", out: str = "") -> None:
    """Measure a run and write its token sidecar.

    Args:
        run: Run directory holding the export (e.g. output/good_ai_fiction_pilot/<ts>).
        dataset: File within it, in interchange form (messages + metadata).
        out: Where to write; defaults to `<run>/token_stats.json`.
    """
    from transformers import AutoTokenizer

    run_dir = Path(run)
    path = run_dir / dataset
    if not path.exists():  # a run that stopped before export still has the stage file
        path = next(run_dir.glob("stage_*_export_sft.jsonl"))
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    tok = AutoTokenizer.from_pretrained(MODEL)
    prof = model_profile(MODEL)

    per_row = measure(rows, tok, prof)
    n = len(per_row)
    tot = {k: sum(r[k] for r in per_row)
           for k in ("rendered", "trainable", "reasoning", "answer")}
    stats = {
        "source": str(path), "model": MODEL, "max_seq_len": MAX_LEN, "rows": n,
        "totals": tot,
        "share_of_trainable": {
            "reasoning": round(tot["reasoning"] / max(tot["trainable"], 1), 4),
            "answer": round(tot["answer"] / max(tot["trainable"], 1), 4),
        },
        "per_row": {k: quantiles([r[k] for r in per_row])
                    for k in ("rendered", "trainable", "reasoning", "answer")},
        "rows_at_cap": sum(1 for r in per_row if r["rendered"] >= MAX_LEN),
        "rows_without_think": sum(1 for r in per_row if not r["has_think"]),
        "da716_baseline": DA716,
        "vs_da716": {
            "mean_trainable_per_row": round(tot["trainable"] / max(n, 1), 1),
            "da_mean_trainable_per_row": round(DA716["trainable"] / DA716["rows"], 1),
            "reasoning_share": round(tot["reasoning"] / max(tot["trainable"], 1), 4),
            "da_reasoning_share": round(DA716["reasoning"] / DA716["trainable"], 4),
            "projected_716_total": round(tot["trainable"] / max(n, 1) * 716),
            "da_716_total": DA716["trainable"],
        },
        "records": per_row,
    }
    dest = Path(out) if out else run_dir / "token_stats.json"
    dest.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    v = stats["vs_da716"]
    print(f"{path}  {n} rows")
    print(f"  trainable {tot['trainable']:,}  ({v['mean_trainable_per_row']}/row; "
          f"DA-716 is {v['da_mean_trainable_per_row']}/row)")
    print(f"  reasoning share {v['reasoning_share']:.1%}  (DA-716 "
          f"{v['da_reasoning_share']:.1%})")
    print(f"  projected over 716 rows: {v['projected_716_total']:,} trainable tokens "
          f"vs DA-716's {v['da_716_total']:,} "
          f"({100 * v['projected_716_total'] / v['da_716_total'] - 100:+.1f}%)")
    for k in ("trainable", "reasoning", "answer"):
        print(f"  per-row {k:<10} {stats['per_row'][k]}")
    if stats["rows_at_cap"]:
        print(f"  !! {stats['rows_at_cap']} rows hit the {MAX_LEN}-token cap")
    print(f"wrote {dest}")


if __name__ == "__main__":
    fire.Fire(main)
