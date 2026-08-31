# ABOUTME: Combines spec-filtered Table 2 instruction-tuning data with trait-balanced synthdoc
# ABOUTME: difficult-advice rows, adds empty-think markers, and writes a training-ready mixture.

"""Build a Table 2 + synthdoc SFT mixture.

Two halves with deliberately different think-block states:

  - **Table 2 rows** carry no reasoning. They get an empty ``<think></think>`` marker as
    inference-time CONTEXT; training masks those tokens out of the loss, because teaching a
    model to *emit* the marker is the documented reasoning-collapse pattern for Qwen3.x.
  - **synthdoc rows** carry a real reasoning trace and are fully supervised. Every row is
    asserted to still hold a real ``<think>`` after rendering, so a lost or emptied trace
    fails loudly rather than silently producing a corpus that trains reasoning suppression.

The synthdoc half is drawn as evenly as the count allows across the constitution's traits,
so no single principle dominates the difficult-advice signal.

    uv run python scratch/build_combined_mixture.py --n_table2 9284 --n_synthdoc 716
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path

import fire
from transformers import AutoTokenizer

# What Qwen3.6's template emits for a final assistant turn carrying no reasoning.
ASSISTANT_HEADER = "<|im_start|>assistant\n"
EMPTY_THINK = "<think>\n\n</think>\n\n"


def _add_marker(text: str) -> str:
    """Insert the empty think marker on EVERY assistant turn that lacks a think block.

    The preserve-thinking policy (src/train/mask_gate.py) requires a block on every
    assistant turn, not just the last: a multi-turn row whose earlier turns are bare is
    rejected before training starts. Marking only the final turn -- which is what a naive
    `rindex` does, and what the template's own no-think branch produces -- left 362 of
    10,362 turns bare in the first build of this mixture.
    """
    out, pos = [], 0
    while (i := text.find(ASSISTANT_HEADER, pos)) != -1:
        j = i + len(ASSISTANT_HEADER)
        out.append(text[pos:j])
        # Leave a turn alone if it already opens with a think block (real trace or marker).
        if not text.startswith("<think>", j):
            out.append(EMPTY_THINK)
        pos = j
    out.append(text[pos:])
    return "".join(out)


def _verify_marker_matches_template(tok) -> None:
    """Prove the marker surgery equals the chat template's own output, before real data."""
    probe = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    plain = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=False)
    padded = tok.apply_chat_template(probe + [{"role": "user", "content": "X"}],
                                     tokenize=False, add_generation_prompt=False)
    no_think = padded[: padded.rindex("<|im_start|>user")]
    assert _add_marker(no_think) == plain, "surgery does not reproduce the chat template"
    # And a two-assistant-turn probe: BOTH turns must end up with a block.
    multi = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
             {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]
    mp = tok.apply_chat_template(multi + [{"role": "user", "content": "X"}],
                                 tokenize=False, add_generation_prompt=False)
    mp = mp[: mp.rindex("<|im_start|>user")]
    assert _add_marker(mp).count("<think>") == 2, "multi-turn rows must get a block per turn"
    print(">>> verified: marker reproduces apply_chat_template, and marks every turn")


def _fit(rows: list[dict], render, tok, max_seq_len: int, label: str) -> list[dict]:
    """Keep only rows whose FINAL rendered form fits the training window.

    Length is measured on the text the trainer will actually see -- markers applied -- since
    marker repair adds tokens and a row measured before it can cross the cap afterwards.
    Rows are never truncated: a truncated row loses its closing <|im_end|> and so never
    teaches the stop token.

    Args:
        rows: Candidate pool.
        render: Row -> the exact string the trainer will tokenize.
        tok: Tokenizer.
        max_seq_len: Inclusive token cap.
        label: Pool name, for the report.

    Returns:
        The rows that fit, each carrying `marked` and `n`.
    """
    kept, over = [], 0
    for r in rows:
        marked = render(r)
        n = len(tok(marked, add_special_tokens=False)["input_ids"])
        if n > max_seq_len:
            over += 1
            continue
        kept.append({**r, "marked": marked, "n": n})
    print(f">>> {label}: {len(kept)}/{len(rows)} fit within {max_seq_len} tokens "
          f"({over} excluded)")
    return kept


def _even_quota(total: int, keys: list[str]) -> dict[str, int]:
    """Split `total` as evenly as possible over `keys`; the remainder goes to the first."""
    base, extra = divmod(total, len(keys))
    return {k: base + (1 if i < extra else 0) for i, k in enumerate(sorted(keys))}


def main(n_table2: int = 9284, n_synthdoc: int = 716, seed: int = 0, max_seq_len: int = 8192,
         table2: str = "output/mixture_table2_filtered_9284/mixture.jsonl",
         synthdoc: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
         out_dir: str = "output/mixture_t2_9284_synthdoc_716",
         tokenizer: str = "Qwen/Qwen3.6-27B") -> None:
    """Build the mixture and write both the plain and marker-carrying forms.

    Args:
        n_table2: Table 2 rows to take (stratified by source).
        n_synthdoc: synthdoc rows to take (split as evenly as possible across traits).
        seed: Sampling and shuffle seed.
        max_seq_len: Rows exceeding this WITH the marker are dropped, never truncated -- a
            truncated row loses its closing <|im_end|> and never teaches the stop token.
        table2: Spec-filtered Table 2 mixture (fields: text, source).
        synthdoc: stage_7_sft.jsonl from the synthdoc run.
        out_dir: Destination directory.
        tokenizer: Tokenizer used for rendering and length checks.
    """
    tok = AutoTokenizer.from_pretrained(tokenizer)
    _verify_marker_matches_template(tok)
    rng = random.Random(seed)

    # --- Table 2 half: stratified by source so the mixture's proportions survive --------
    t2 = [json.loads(line) for line in Path(table2).open()]
    # Length-filter the POOL, before selection. Gating after selection would drop whichever
    # long rows happened to be picked, quietly breaking both the exact counts and the trait
    # balance the caller asked for; filtering first spends the surplus in the pool instead.
    t2 = _fit(t2, lambda r: _add_marker(r["text"]), tok, max_seq_len, "table2")
    assert n_table2 <= len(t2), (
        f"asked for {n_table2} Table 2 rows, only {len(t2)} fit within {max_seq_len} tokens")
    by_source: dict[str, list] = collections.defaultdict(list)
    for r in t2:
        by_source[r["source"]].append(r)
    if n_table2 == len(t2):
        picked_t2 = list(t2)
    else:
        quota_s = {s: round(len(v) * n_table2 / len(t2)) for s, v in by_source.items()}
        picked_t2, leftover = [], []
        for s in sorted(by_source):
            rows = list(by_source[s])
            rng.shuffle(rows)
            picked_t2 += rows[: quota_s[s]]
            leftover += rows[quota_s[s]:]
        # Per-source quotas are rounded independently, so they sum to n_table2 only by
        # luck -- at 9,000/9,284 they sum to 8,999. Top up (or trim) to hit the count
        # exactly, taking the shortfall from the pooled leftovers so the proportions stay
        # as close to the source mix as rounding allows.
        if len(picked_t2) < n_table2:
            rng.shuffle(leftover)
            picked_t2 += leftover[: n_table2 - len(picked_t2)]
        rng.shuffle(picked_t2)
        picked_t2 = picked_t2[:n_table2]
    assert len(picked_t2) == n_table2, len(picked_t2)

    # --- synthdoc half: as even as the count allows across every trait ------------------
    recs = [json.loads(line) for line in Path(synthdoc).open()]

    def _render_syn(r: dict) -> str:
        """Render one synthdoc record; its real trace is never marker-substituted."""
        m = {x["role"]: x for x in r["messages"]}
        text = tok.apply_chat_template(
            [{"role": "system", "content": m["system"]["content"]},
             {"role": "user", "content": m["user"]["content"]},
             {"role": "assistant", "content": m["assistant"]["content"],
              "reasoning_content": m["assistant"]["reasoning_content"]}],
            tokenize=False, add_generation_prompt=False)
        assert "<think>" in text and EMPTY_THINK not in text, "trace lost or emptied"
        return text

    recs = _fit(recs, _render_syn, tok, max_seq_len, "synthdoc")
    by_trait: dict[str, list] = collections.defaultdict(list)
    for r in recs:
        by_trait[r["metadata"]["trait_id"]].append(r)
    quota_t = _even_quota(n_synthdoc, list(by_trait))
    print(f">>> synthdoc: {n_synthdoc} over {len(by_trait)} traits -> {quota_t}")
    picked_syn = []
    for t in sorted(by_trait):
        pool = sorted(by_trait[t], key=lambda r: r["metadata"]["scenario_id"])
        rng.shuffle(pool)
        assert len(pool) >= quota_t[t], f"{t}: have {len(pool)}, need {quota_t[t]}"
        picked_syn += pool[: quota_t[t]]
    assert len(picked_syn) == n_synthdoc, len(picked_syn)

    # --- assemble ------------------------------------------------------------------------
    # Both pools were rendered, marker-repaired and length-checked in _fit, so every row
    # already carries its final `marked` text and token count; nothing is dropped here.
    kept: list[dict] = [{"text": r["text"], "source": r["source"], "trait_id": None,
                         "marked": r["marked"], "n": r["n"]} for r in picked_t2]
    kept += [{"text": r["marked"], "source": "synthdoc_difficult_advice",
              "trait_id": r["metadata"]["trait_id"], "marked": r["marked"], "n": r["n"]}
             for r in picked_syn]
    rng.shuffle(kept)
    dropped: list[dict] = []
    assert len(kept) == n_table2 + n_synthdoc, len(kept)

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "mixture.jsonl").open("w") as f:
        for r in kept:
            f.write(json.dumps({"text": r["text"], "source": r["source"]},
                               ensure_ascii=False) + "\n")
    with (dest / "mixture_think.jsonl").open("w") as f:
        for r in kept:
            f.write(json.dumps({"text": r["marked"], "source": r["source"]},
                               ensure_ascii=False) + "\n")

    per: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for r in kept:
        per[r["source"]][0] += 1
        per[r["source"]][1] += r["n"]
    traits = collections.Counter(r["trait_id"] for r in kept if r["trait_id"])
    stats = {"examples": len(kept), "tokens_with_marker": sum(v[1] for v in per.values()),
             "max_seq_len": max_seq_len, "longest": max(r["n"] for r in kept),
             "dropped": len(dropped), "seed": seed,
             "by_source": {k: {"examples": v[0], "tokens": v[1]}
                           for k, v in sorted(per.items())},
             "synthdoc_per_trait": dict(sorted(traits.items()))}
    (dest / "mixture_stats.json").write_text(json.dumps(stats, indent=2))

    n_t2 = sum(v[0] for k, v in per.items() if k != "synthdoc_difficult_advice")
    print(f"\n>>> {len(kept):,} examples -> {dest}")
    print(f"    table2 {n_t2:,} | synthdoc {sum(traits.values()):,} "
          f"across {len(traits)} traits {dict(sorted(traits.items()))}")
    print(f"    longest row with marker: {stats['longest']:,} / {max_seq_len}")
    assert stats["longest"] <= max_seq_len


if __name__ == "__main__":
    fire.Fire(main)
