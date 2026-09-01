# ABOUTME: Build the CoT-only supervision mixture: the control's 10,000 rows verbatim,
# ABOUTME: with `supervise: "cot"` flagged onto the 716 difficult-advice rows.

"""Flag the difficult-advice half of the Table2+DA mixture for CoT-only supervision.

Run: uv run python scratch/cot_only/build_mixture.py

The intervention is ONE per-row field. Every `text` is copied byte-for-byte from the
control mixture, so this arm and its control tokenize identically up to the point where
the CoT arm stops — the comparison is in the labels and the sequence length, nothing else.
What `supervise: "cot"` then means at train time (truncate the row at its reasoning close;
supervise the trace and that close, never the answer) lives in src/train/masking.py.

Everything the pod would discover the hard way is asserted here instead, because a data
error found after provisioning costs GPU-hours:

  - exactly N_DA rows carry the flag, and all of them are the difficult-advice source
  - every flagged row survives `cot_span` — real trace, thinking prefill, closed block
    (an EMPTY marker would be refused: supervising an empty close trains the collapse)
  - no `text` changed

The token accounting it prints is the experiment's headline number: how much of the
model's training signal, and how much of its forward pass, this removes.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model_profile import model_profile  # noqa: E402
from src.train.masking import answer_span, build_labels, cot_span  # noqa: E402
from src.utils import git_sha  # noqa: E402

CONTROL_REPO = "LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train"
CONTROL_FILE = "mixture_think.jsonl"
CONTROL_REVISION = "5b5d66dbd050a15493f38b500fe10f570b0b8f2b"
DA_SOURCE = "synthdoc_difficult_advice"
N_DA = 716
N_ROWS = 10_000

# Parameterised on 2026-08-31 to build the chunk-only (principle-scoped) arm without
# forking a near-identical script — the same move build_t2_9284_da716_mixture.py made for
# the courtroom arm. Defaults above are the ORIGINAL synthdoc values, so the provenance
# recorded on LASR-Callum/2026-08-31-cot-only-supervision-t2-9284-synthdoc-716 still
# reproduces.
#
# CHUNK_ONLY is the arm to prefer, and the reason is the control. Its base mixture is the
# exact file LASR-Callum/2026-08-21-qwen36-lora-table2-9284-difficult-advice-chunk-only-
# 702-rank-64-dynbatch was trained on (ODCV 11.5% [6.2, 19.6], 2 passes), so the CoT-only
# arm differs from a REAL, EVALUATED control in the `supervise` field alone. The synthdoc
# arm had no such control — that adapter was never trained (docs/LOG.md: it 404s).
#
# 702, not the corpus's 708: the builder requires equal per-trait quotas and trait 1
# finished with 78, so 9 x 78 = 702 is the largest perfectly balanced draw. Taking all 708
# would break both the trait balance and the pairing with the control.
CHUNK_ONLY = {
    "repo": "LASR-Callum/2026-08-21-table2-9284-difficult-advice-principle-scoped-702-train-mixture",
    "file": "t2_9284_da_chunk_only_702.jsonl",
    "revision": "fa98fadeee72",
    "da_source": "difficult_advice_chunk_only",
    "n_da": 702,
    "n_rows": 9_986,
}


def main(out_dir: str = "", model_id: str = "Qwen/Qwen3.6-27B",
         max_length: int = 8192, repo: str = CONTROL_REPO, file: str = CONTROL_FILE,
         revision: str = CONTROL_REVISION, da_source: str = DA_SOURCE,
         n_da: int = N_DA, n_rows: int = N_ROWS, mode: str = "cot") -> None:
    """Emit the CoT-only mixture, its stats and its provenance.

    Args:
        out_dir: Destination directory; defaults to a timestamped output/ path.
        model_id: Model whose tokenizer and profile define the token stream.
        max_length: Training sequence length, so the accounting matches the train config.
        repo: Control mixture's HF dataset repo.
        file: Control mixture's data file.
        revision: Control mixture's commit sha — pinned, because "same text as the
            control" is only meaningful against a fixed control.
        da_source: The `source` value marking the difficult-advice rows to flag, named
            per corpus (`synthdoc_difficult_advice`, `difficult_advice_chunk_only`).
            Pointing `repo` at another mixture without updating this fails the count
            assert below rather than silently flagging nothing.
        n_da: Expected difficult-advice row count (702 for chunk-only, 716 for synthdoc).
        n_rows: Expected total row count (9,986 for chunk-only, 10,000 for synthdoc).
        mode: The supervise mode written onto the difficult-advice rows -- "cot" (train
            the reasoning, truncate away the answer) or "answer" (train the visible
            answer, keep the trace as unsupervised context). The two are exact
            complements; see src/train/masking.py.

    Raises:
        AssertionError: The control is not the expected shape, or a flagged row has no
            trainable reasoning.
    """
    from dotenv import load_dotenv
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    load_dotenv()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_p = Path(out_dir or f"output/cot_only_mixture/{ts}")
    out_p.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(repo, file, repo_type="dataset", revision=revision)
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == n_rows, f"control has {len(rows)} rows, expected {n_rows}"
    by_source = Counter(r["source"] for r in rows)
    assert by_source[da_source] == n_da, (
        f"control has {by_source[da_source]} {da_source!r} rows, expected {n_da}; "
        f"sources present: {dict(by_source.most_common())}")

    profile = model_profile(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    out_rows, flagged = [], 0
    tok_control = tok_cot = sup_control = sup_cot = 0
    da_tok_control = da_tok_cot = da_sup_control = da_sup_cot = 0
    for i, r in enumerate(rows):
        is_da = r["source"] == da_source
        new = dict(r)
        if is_da:
            # Fail HERE, not on the pod: refuses an empty marker or an unclosed trace
            # (and, under "answer", an unterminated turn).
            if mode == "cot":
                cot_span(r["text"], header=profile.assistant_header,
                         prefill=profile.prefill, empty_think=profile.empty_think,
                         think_close=profile.think_close)
            else:
                answer_span(r["text"], header=profile.assistant_header,
                            prefill=profile.prefill, empty_think=profile.empty_think,
                            think_close=profile.think_close,
                            turn_end=profile.turn_end)
            new["supervise"] = mode
            flagged += 1
        assert new["text"] == r["text"], f"row {i}: text must not change"
        out_rows.append(new)

        control = build_labels(r["text"], tokenizer, max_length, profile)
        arm = build_labels(r["text"], tokenizer, max_length, profile,
                           supervise=mode if is_da else "all")
        n_c, n_a = len(control["input_ids"]), len(arm["input_ids"])
        s_c = sum(1 for v in control["labels"] if v != -100)
        s_a = sum(1 for v in arm["labels"] if v != -100)
        tok_control += n_c
        tok_cot += n_a
        sup_control += s_c
        sup_cot += s_a
        if is_da:
            da_tok_control += n_c
            da_tok_cot += n_a
            da_sup_control += s_c
            da_sup_cot += s_a

    assert mode in ("cot", "answer"), f"unknown mode {mode!r}"
    assert flagged == n_da, f"flagged {flagged} rows, expected {n_da}"

    data_file = out_p / "mixture_think_cotonly.jsonl"
    with data_file.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pct = lambda a, b: round(100 * (1 - a / b), 2)  # noqa: E731
    stats = {
        "rows": len(out_rows),
        "rows_supervise_cot": flagged,
        "supervise_mode": mode,
        "per_source": dict(by_source.most_common()),
        "max_length": max_length,
        "forward_tokens": {
            "control": tok_control, "cot_only": tok_cot,
            "reduction_pct": pct(tok_cot, tok_control),
            "da_control": da_tok_control, "da_cot_only": da_tok_cot,
            "da_reduction_pct": pct(da_tok_cot, da_tok_control),
        },
        "supervised_tokens": {
            "control": sup_control, "cot_only": sup_cot,
            "reduction_pct": pct(sup_cot, sup_control),
            "da_control": da_sup_control, "da_cot_only": da_sup_cot,
            "da_share_control_pct": round(100 * da_sup_control / sup_control, 1),
            "da_share_cot_only_pct": round(100 * da_sup_cot / sup_cot, 1),
        },
    }
    (out_p / "mixture_stats_cotonly.json").write_text(json.dumps(stats, indent=2))
    (out_p / "run_meta.json").write_text(json.dumps({
        "git_sha": git_sha(), "timestamp": ts, "model": model_id,
        "control": {"repo": repo, "file": file, "revision": revision},
        "script": "scratch/cot_only/build_mixture.py",
        "supervise_mode": mode,
    }, indent=2))

    print(f">>> wrote {data_file} ({len(out_rows)} rows, {flagged} flagged "
          f"supervise={mode})")
    print(f">>> forward tokens   : {tok_control:,} -> {tok_cot:,} "
          f"(-{stats['forward_tokens']['reduction_pct']}% overall, "
          f"-{stats['forward_tokens']['da_reduction_pct']}% on the DA rows)")
    print(f">>> supervised tokens: {sup_control:,} -> {sup_cot:,} "
          f"(-{stats['supervised_tokens']['reduction_pct']}%); DA share of signal "
          f"{stats['supervised_tokens']['da_share_control_pct']}% -> "
          f"{stats['supervised_tokens']['da_share_cot_only_pct']}%")


if __name__ == "__main__":
    fire.Fire(main)
