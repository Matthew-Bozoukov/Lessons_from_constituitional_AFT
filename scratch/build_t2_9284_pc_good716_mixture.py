# ABOUTME: Builds the peer-critique GOOD-ARM-ONLY 716 mixture — the one-variable twin of the
# ABOUTME: 358 good / 358 flawed arm — reusing build_t2_9284_da716_mixture's helpers verbatim.
"""Build and verify the peer-critique good-arm ablation mixture.

Run: uv run python scratch/build_t2_9284_pc_good716_mixture.py [--out <path>] [--seed 0]

WHY THIS EXISTS. `LASR-Callum/qwen3.6-27b-lora-t2-9284-peercritique716-r64-dynbatch` trained on
716 peer-critique rows split 358 good / 358 flawed. Peer critique fails its `surface_shortcut`
gate at AUC 0.9973 (threshold 0.70) because the good arm's critiqued reply is written by Sonnet
and the flawed arm's by grok-4.3 / qwen3-32b / gemini-3.7-flash — the arm label is predictable
from authorship alone, and length by itself separates them at 0.8471.

A good-arm-only mixture sidesteps that leak entirely: with one class there is nothing to
discriminate. This builds that twin so the flawed arm's contribution can be measured directly.

WHY IT IMPORTS RATHER THAN COPIES. Everything about how a row is rendered — the exact
`<|im_start|>{role}\\n...<|im_end|>\\n` form, the `<think>` conventions, the empty marker on bare
history turns, the trait-balanced domain-round-robin selection — must be IDENTICAL to the arm
this is compared against, or the comparison measures the renderer instead of the data. So those
functions are imported from `build_t2_9284_da716_mixture` (Matthew's, unmodified) rather than
reimplemented here, and cannot drift from it.

WHAT THIS ADDS over calling that script directly:
  * `reply_quality == good` filtering of the synth half BEFORE selection, so trait/domain quotas
    are computed over the pool that can actually satisfy them;
  * the arm stamped onto every synth row, so a row's presence is explainable from the row;
  * the token-matching verification inlined (think census + sequence-length headroom), rather
    than left to a separate script that the published card references but that was never
    committed.
"""

import importlib.util
import json
import random
import re
from collections import Counter
from pathlib import Path

import fire
from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.model_profile import think_census

# Matthew's builder, loaded by path: `scratch/` is not a package, and this must be the SAME
# code the comparison arm was built with rather than a copy that can drift.
_SPEC = importlib.util.spec_from_file_location(
    "build_t2_9284_da716_mixture",
    Path(__file__).with_name("build_t2_9284_da716_mixture.py"))
_DA = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DA)

PC_REPO = "LASR-Callum/2026-08-14-peer-critique"
PC_FILE = "dataset.jsonl"
# Same replay source and exclusion the 358/358 arm used, so the two mixtures differ ONLY in
# which peer-critique rows they carry. That repo is itself a 10,000-row mixture whose 716
# synthdoc rows are the arm being replaced; dropping them leaves exactly 9,284 Table2 rows.
T2_REPO = "LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train"
T2_FILE = "mixture_think.jsonl"
EXCLUDE = ("synthdoc_difficult_advice",)
N_SYNTH = 716
ARM = "good"
TOKENIZER = "Qwen/Qwen3.6-27B"
MAX_SEQ_LEN = 8192  # the training config's truncation ceiling


def verify(rows: list[dict], tok_name: str, max_seq_len: int) -> dict:
    """Check the two things that make a mixture trainable, and report both.

    Think markers: under `thinking: true` every assistant turn must carry a block, so
    `absent` must be 0 — a bare turn has no forced span and would be supervised whole, i.e.
    under a different rule than every other turn in the file.

    Sequence length: `max_seq_len` is a truncation ceiling, and a row that exceeds it loses
    its tail. For a corpus whose trained content sits at the END of the row, silent truncation
    removes exactly the part that trains.

    Args:
        rows: The assembled mixture rows (each with `text`).
        tok_name: Tokenizer to measure with.
        max_seq_len: The training config's ceiling.

    Returns:
        The measured report.
    """
    census = think_census([r["text"] for r in rows])
    print(f"think_census: {census}")
    assert census["absent"] == 0, (
        f"{census['absent']} assistant turns carry no think block — under `thinking: true` "
        "they would be supervised under a different rule than the rest")

    bare = sum(1 for r in rows if "<|im_start|>assistant" not in r["text"])
    assert bare == 0, f"{bare} rows have no assistant turn at all"

    tok = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    lens = sorted(len(tok(r["text"], add_special_tokens=False)["input_ids"]) for r in rows)
    over = sum(1 for x in lens if x > max_seq_len)
    n = len(lens)
    report = {
        "think_census": census,
        "tokens_min": lens[0], "tokens_p50": lens[n // 2],
        "tokens_p90": lens[int(n * 0.90)], "tokens_p99": lens[int(n * 0.99)],
        "tokens_max": lens[-1],
        "max_seq_len": max_seq_len, "rows_over_max_seq_len": over,
    }
    print(f"tokens: min {lens[0]}  p50 {lens[n // 2]}  p90 {lens[int(n * 0.90)]}  "
          f"p99 {lens[int(n * 0.99)]}  max {lens[-1]}")
    print(f"rows over max_seq_len={max_seq_len}: {over}"
          f"{'  <-- these WILL be truncated' if over else '  (nothing truncated)'}")
    return report


def main(out: str = "data/t2_9284_pc_good716_10k.jsonl", seed: int = 0,
         n_synth: int = N_SYNTH, arm: str = ARM, tokenizer: str = TOKENIZER,
         max_seq_len: int = MAX_SEQ_LEN) -> None:
    """Assemble the good-arm mixture and verify it.

    Args:
        out: Output JSONL path.
        seed: Selection/shuffle seed. 0 matches the comparison arm.
        n_synth: Peer-critique rows to select.
        arm: `reply_quality` value to keep. `good` is the ablation; `flawed` builds the mirror.
        tokenizer: Tokenizer for the length check.
        max_seq_len: The training config's truncation ceiling.
    """
    load_dotenv()
    rng = random.Random(seed)

    pc = _DA.read_hf_jsonl(PC_REPO, PC_FILE)
    kept = [r for r in pc if r["metadata"].get("reply_quality") == arm]
    assert kept, f"no rows with reply_quality={arm!r}"
    print(f"peer_critique: {len(kept)}/{len(pc)} rows on the {arm!r} arm")

    t2 = _DA.read_hf_jsonl(T2_REPO, T2_FILE)
    before = len(t2)
    t2 = [r for r in t2 if r.get("source") not in set(EXCLUDE)]
    print(f"dropped {before - len(t2)} rows from {sorted(EXCLUDE)}; "
          f"instruction half: {len(t2)}")

    # Same guard Matthew's builder applies: a real trace on the replay side would put two
    # reasoning conventions in one mixture.
    nonempty = sum(1 for r in t2
                   if (m := re.search(r"<think>(.*?)</think>", r["text"], re.S))
                   and m.group(1).strip())
    print(f"table2 rows with a NON-empty <think> block: {nonempty} (expected 0)")
    assert nonempty == 0

    picked = _DA.pick_balanced(kept, n_synth, rng, None)
    pc_rows = [{"source": "peer_critique", "text": _DA.render(r["messages"]),
                "trait_id": r["metadata"]["trait_id"],
                "scenario_id": r["metadata"]["scenario_id"],
                "reply_quality": r["metadata"]["reply_quality"]}
               for r in picked]
    assert {r["reply_quality"] for r in pc_rows} == {arm}

    fixed = 0
    t2_rows = []
    for r in t2:
        text, added = _DA.ensure_think_on_every_turn(r["text"])
        fixed += added
        t2_rows.append({"source": r["source"], "text": text})
    print(f"empty markers inserted on bare HISTORY turns: {fixed}")

    mixture = pc_rows + t2_rows
    rng.shuffle(mixture)
    assert len(mixture) == 10000, f"expected 10,000 rows, got {len(mixture)}"

    report = verify(mixture, tokenizer, max_seq_len)

    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        for r in mixture:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tr = Counter(r["trait_id"] for r in pc_rows)
    stats = {
        "total": len(mixture), "synth": len(pc_rows), "table2": len(t2_rows),
        "synth_label": "peer_critique", "arm": arm,
        "synth_fraction": round(len(pc_rows) / len(mixture), 4),
        "seed": seed,
        "synth_source": f"{PC_REPO}::{PC_FILE}",
        "t2_source": f"{T2_REPO}::{T2_FILE}",
        "excluded_sources": list(EXCLUDE),
        "per_trait": dict(sorted(tr.items())),
        "distinct_domains_in_synth": len({r["metadata"].get("domain") for r in picked}),
        "distinct_scenarios_in_synth": len({r["scenario_id"] for r in pc_rows}),
        "per_source": dict(Counter(r["source"] for r in mixture).most_common()),
        "history_markers_inserted": fixed,
        "verification": report,
    }
    Path(str(out_p) + ".stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\ntotal {len(mixture)} = {len(pc_rows)} peer_critique ({arm}) + "
          f"{len(t2_rows)} table2  ({stats['synth_fraction'] * 100:.2f}% synth)")
    print(f"per trait: {stats['per_trait']}")
    print(f"distinct scenarios: {stats['distinct_scenarios_in_synth']} (no repeats)   "
          f"domains covered: {stats['distinct_domains_in_synth']}")
    print(f"wrote {out_p}  (+ .stats.json)")


if __name__ == "__main__":
    fire.Fire(main)
