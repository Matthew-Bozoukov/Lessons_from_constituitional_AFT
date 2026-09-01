# ABOUTME: Build the empty-CoT mixture: the control's rows with each difficult-advice
# ABOUTME: trace REPLACED by the empty think marker. Run: uv run python scratch/empty_cot/build_mixture.py

"""Replace the difficult-advice reasoning with an empty think marker.

The third arm in the supervision series, and the only one that changes the TEXT.

  control      sys + user + <think>TRACE</think> + answer     loss: trace + answer
  cot-only     sys + user + <think>TRACE</think>              loss: trace        (truncated)
  answer-only  sys + user + <think>TRACE</think> + answer     loss: answer
  empty-cot    sys + user + <think></think>      + answer     loss: answer       (THIS)

So `answer-only` and `empty-cot` supervise the SAME answer tokens and differ in exactly
one thing: whether the reasoning is present in the context the answer is conditioned on.
That pair isolates the trace's value AS CONTEXT, which neither arm could do alone.

NO NEW MASK MODE IS NEEDED, and that is worth saying plainly: the generation-boundary
rule already masks the WHOLE empty marker and supervises only the visible answer (a
healthy Qwen3.6 never generates an empty close, so the marker is forced context in every
serving configuration). A row rewritten this way therefore trains correctly under the
DEFAULT `supervise: "all"` -- the mixture carries no `supervise` column at all. The
intervention lives entirely in the data.

WHAT THIS SCRIPT DOES NOT SHARE with scratch/cot_only/build_mixture.py: that script's
central invariant is that `text` is byte-identical to the control, which is what makes a
label-only ablation clean. This arm rewrites text by construction, so it is a sibling
rather than a mode of that script -- the two have opposite contracts and opposite
verification. What IS shared (the base mixture's identity) is imported, not copied.

Everything the pod would discover the hard way is asserted here:

  - exactly n_da rows change, and all of them are the difficult-advice source
  - each changed row's PROMPT and ANSWER are byte-identical to the control's; only the
    think block differs, and the new one is exactly the profile's empty marker
  - the census lands at 0 real / all empty / 0 absent
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scratch.cot_only.build_mixture import CHUNK_ONLY  # noqa: E402
from src.model_profile import model_profile, think_census  # noqa: E402
from src.train.masking import build_labels  # noqa: E402
from src.utils import git_sha  # noqa: E402


def blank_the_trace(text: str, *, header: str, prefill: str, empty_think: str,
                    think_close: str) -> str:
    """Replace the final assistant turn's reasoning with the empty marker.

    Returns the rewritten text. Refuses anything that is not a real, closed trace
    followed by the template's blank-line separator, because a silent no-op here would
    produce an arm identical to its control.
    """
    i = text.rfind(header)
    assert i != -1, "no assistant turn found"
    head = i + len(header)
    assert not text.startswith(empty_think, head), \
        "this turn already carries the empty marker; nothing to blank"
    assert text.startswith(prefill, head), \
        f"turn does not open with the thinking prefill {prefill!r}"
    close = text.find(think_close, head)
    assert close != -1, "the trace is never closed"
    end = close + len(think_close)
    # The template writes a blank line between the close and the answer; the empty
    # marker already ends with one, so it is consumed rather than duplicated.
    sep = empty_think[empty_think.rindex(think_close) + len(think_close):]
    assert text.startswith(sep, end), \
        f"expected {sep!r} after the reasoning close, found {text[end:end + 4]!r}"
    return text[:head] + empty_think + text[end + len(sep):]


def main(out_dir: str = "", model_id: str = "Qwen/Qwen3.6-27B", max_length: int = 8192,
         repo: str = CHUNK_ONLY["repo"], file: str = CHUNK_ONLY["file"],
         revision: str = CHUNK_ONLY["revision"],
         da_source: str = CHUNK_ONLY["da_source"],
         n_da: int = CHUNK_ONLY["n_da"], n_rows: int = CHUNK_ONLY["n_rows"]) -> None:
    """Emit the empty-CoT mixture, its stats and its provenance.

    Args:
        out_dir: Destination directory; defaults to a timestamped output/ path.
        model_id: Model whose tokenizer and profile define the token stream.
        max_length: Training sequence length, so the accounting matches the train config.
        repo: Control mixture's HF dataset repo.
        file: Control mixture's data file.
        revision: Control mixture's commit sha.
        da_source: `source` value marking the difficult-advice rows to rewrite.
        n_da: Expected difficult-advice row count.
        n_rows: Expected total row count.

    Raises:
        AssertionError: The control is not the expected shape, or a rewrite changed
            anything but the think block.
    """
    from dotenv import load_dotenv
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    load_dotenv()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_p = Path(out_dir or f"output/empty_cot_mixture/{ts}")
    out_p.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(repo, file, repo_type="dataset", revision=revision)
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == n_rows, f"control has {len(rows)} rows, expected {n_rows}"
    by_source = Counter(r["source"] for r in rows)
    assert by_source[da_source] == n_da, \
        f"control has {by_source[da_source]} {da_source!r} rows, expected {n_da}"

    profile = model_profile(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    kw = dict(header=profile.assistant_header, prefill=profile.prefill,
              empty_think=profile.empty_think, think_close=profile.think_close)

    out_rows, rewritten = [], 0
    tok_control = tok_arm = sup_control = sup_arm = 0
    da_tok_c = da_tok_a = da_sup_c = da_sup_a = 0
    for i, r in enumerate(rows):
        is_da = r["source"] == da_source
        new = dict(r)
        if is_da:
            new["text"] = blank_the_trace(r["text"], **kw)
            rewritten += 1
            # Prompt and answer must survive byte-for-byte: the ONLY difference is the
            # think block. Checked by splitting both texts at the assistant header and
            # at the marker/close, rather than trusting the rewrite.
            h = r["text"].rfind(profile.assistant_header) + len(profile.assistant_header)
            assert new["text"][:h] == r["text"][:h], f"row {i}: prompt changed"
            old_ans = r["text"][r["text"].index(profile.think_close, h)
                                + len(profile.think_close):]
            new_ans = new["text"][new["text"].index(profile.think_close, h)
                                  + len(profile.think_close):]
            assert old_ans == new_ans, f"row {i}: answer changed"
            assert new["text"].startswith(profile.empty_think, h), \
                f"row {i}: rewritten turn does not open with the empty marker"
        else:
            assert new["text"] == r["text"], f"row {i}: non-DA row must not change"
        out_rows.append(new)

        c = build_labels(r["text"], tokenizer, max_length, profile)
        a = build_labels(new["text"], tokenizer, max_length, profile)
        tok_control += len(c["input_ids"])
        tok_arm += len(a["input_ids"])
        sc = sum(1 for v in c["labels"] if v != -100)
        sa = sum(1 for v in a["labels"] if v != -100)
        sup_control += sc
        sup_arm += sa
        if is_da:
            da_tok_c += len(c["input_ids"])
            da_tok_a += len(a["input_ids"])
            da_sup_c += sc
            da_sup_a += sa

    assert rewritten == n_da, f"rewrote {rewritten} rows, expected {n_da}"
    census = think_census([r["text"] for r in out_rows])
    assert census["real"] == 0, f"{census['real']} real traces survived the rewrite"
    assert census["absent"] == 0, f"{census['absent']} turns lost their think block"

    data_file = out_p / "mixture_think_emptycot.jsonl"
    with data_file.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pct = lambda a, b: round(100 * (1 - a / b), 2)  # noqa: E731
    stats = {
        "rows": len(out_rows), "rows_rewritten": rewritten,
        "per_source": dict(by_source.most_common()), "max_length": max_length,
        "census": census,
        "forward_tokens": {"control": tok_control, "empty_cot": tok_arm,
                           "reduction_pct": pct(tok_arm, tok_control),
                           "da_control": da_tok_c, "da_empty_cot": da_tok_a,
                           "da_reduction_pct": pct(da_tok_a, da_tok_c)},
        "supervised_tokens": {"control": sup_control, "empty_cot": sup_arm,
                              "reduction_pct": pct(sup_arm, sup_control),
                              "da_control": da_sup_c, "da_empty_cot": da_sup_a,
                              "da_share_control_pct": round(100 * da_sup_c / sup_control, 1),
                              "da_share_arm_pct": round(100 * da_sup_a / sup_arm, 1)},
    }
    (out_p / "mixture_stats_emptycot.json").write_text(json.dumps(stats, indent=2))
    (out_p / "run_meta.json").write_text(json.dumps({
        "git_sha": git_sha(), "timestamp": ts, "model": model_id,
        "control": {"repo": repo, "file": file, "revision": revision},
        "script": "scratch/empty_cot/build_mixture.py",
    }, indent=2))

    print(f">>> wrote {data_file} ({len(out_rows)} rows, {rewritten} traces blanked)")
    print(f">>> census: {census['real']} real / {census['empty']} empty / "
          f"{census['absent']} absent")
    print(f">>> forward tokens   : {tok_control:,} -> {tok_arm:,} "
          f"(-{stats['forward_tokens']['reduction_pct']}% overall, "
          f"-{stats['forward_tokens']['da_reduction_pct']}% on the DA rows)")
    print(f">>> supervised tokens: {sup_control:,} -> {sup_arm:,} "
          f"(-{stats['supervised_tokens']['reduction_pct']}%); DA share of signal "
          f"{stats['supervised_tokens']['da_share_control_pct']}% -> "
          f"{stats['supervised_tokens']['da_share_arm_pct']}%")


if __name__ == "__main__":
    fire.Fire(main)
