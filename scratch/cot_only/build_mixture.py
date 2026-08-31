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
from src.train.masking import build_labels, cot_span  # noqa: E402
from src.utils import git_sha  # noqa: E402

CONTROL_REPO = "LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train"
CONTROL_FILE = "mixture_think.jsonl"
CONTROL_REVISION = "5b5d66dbd050a15493f38b500fe10f570b0b8f2b"
DA_SOURCE = "synthdoc_difficult_advice"
N_DA = 716
N_ROWS = 10_000


def main(out_dir: str = "", model_id: str = "Qwen/Qwen3.6-27B",
         max_length: int = 8192, repo: str = CONTROL_REPO, file: str = CONTROL_FILE,
         revision: str = CONTROL_REVISION) -> None:
    """Emit the CoT-only mixture, its stats and its provenance.

    Args:
        out_dir: Destination directory; defaults to a timestamped output/ path.
        model_id: Model whose tokenizer and profile define the token stream.
        max_length: Training sequence length, so the accounting matches the train config.
        repo: Control mixture's HF dataset repo.
        file: Control mixture's data file.
        revision: Control mixture's commit sha — pinned, because "same text as the
            control" is only meaningful against a fixed control.

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
    assert len(rows) == N_ROWS, f"control has {len(rows)} rows, expected {N_ROWS}"
    by_source = Counter(r["source"] for r in rows)
    assert by_source[DA_SOURCE] == N_DA, \
        f"control has {by_source[DA_SOURCE]} {DA_SOURCE} rows, expected {N_DA}"

    profile = model_profile(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    out_rows, flagged = [], 0
    tok_control = tok_cot = sup_control = sup_cot = 0
    da_tok_control = da_tok_cot = da_sup_control = da_sup_cot = 0
    for i, r in enumerate(rows):
        is_da = r["source"] == DA_SOURCE
        new = dict(r)
        if is_da:
            # Fail HERE, not on the pod: refuses an empty marker or an unclosed trace.
            cot_span(r["text"], header=profile.assistant_header,
                     prefill=profile.prefill, empty_think=profile.empty_think,
                     think_close=profile.think_close)
            new["supervise"] = "cot"
            flagged += 1
        assert new["text"] == r["text"], f"row {i}: text must not change"
        out_rows.append(new)

        control = build_labels(r["text"], tokenizer, max_length, profile)
        arm = build_labels(r["text"], tokenizer, max_length, profile,
                           supervise="cot" if is_da else "all")
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

    assert flagged == N_DA, f"flagged {flagged} rows, expected {N_DA}"

    data_file = out_p / "mixture_think_cotonly.jsonl"
    with data_file.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pct = lambda a, b: round(100 * (1 - a / b), 2)  # noqa: E731
    stats = {
        "rows": len(out_rows),
        "rows_supervise_cot": flagged,
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
    }, indent=2))

    print(f">>> wrote {data_file} ({len(out_rows)} rows, {flagged} flagged supervise=cot)")
    print(f">>> forward tokens   : {tok_control:,} -> {tok_cot:,} "
          f"(-{stats['forward_tokens']['reduction_pct']}% overall, "
          f"-{stats['forward_tokens']['da_reduction_pct']}% on the DA rows)")
    print(f">>> supervised tokens: {sup_control:,} -> {sup_cot:,} "
          f"(-{stats['supervised_tokens']['reduction_pct']}%); DA share of signal "
          f"{stats['supervised_tokens']['da_share_control_pct']}% -> "
          f"{stats['supervised_tokens']['da_share_cot_only_pct']}%")


if __name__ == "__main__":
    fire.Fire(main)
