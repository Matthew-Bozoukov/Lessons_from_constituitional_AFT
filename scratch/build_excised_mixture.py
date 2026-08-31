# ABOUTME: Splice excised reasoning back into the training mixture and verify that only the
# ABOUTME: reasoning changed — and that the targeted property actually left.

"""Turn a set of reasoning rewrites into a trainable mixture, and check it.

Three claims have to hold before this file is worth a GPU, and each is tested here rather
than assumed:

* Only the reasoning moved. Every row's system prompt, user turn and assistant reply must be
  byte-identical to the control mixture; only the text inside <think> differs, and only on
  the rewritten rows.
* The property actually left. The lexical probes that found the move in the originals should
  drop sharply on the rewritten rows. If they do not, the rewrite failed and the arm is a
  control with extra steps.
* Nothing else drifted with it. Length and first-person rate are reported because a rewrite
  that quietly formalises the register would confound the ablation with a style change.

Run:
  uv run python scratch/build_excised_mixture.py --excisions <dir>/excisions.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scratch.probe_meta_reasoning import MOVE, SIGNATURE  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

SOURCE = "synthdoc_difficult_advice"
USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
THINK_RE = re.compile(r"(<think>\n?)(.*?)(\n?</think>)", re.S)
FIRST_PERSON = re.compile(r"\b(I|I'm|I'd|I'll|my|me)\b")


def _user(text: str) -> str:
    """Extract the user turn from a rendered Qwen chat string.

    Args:
        text: A rendered mixture row.

    Returns:
        The user message content, stripped.

    Raises:
        ValueError: If the row has no user turn.
    """
    m = USER_RE.search(text)
    if not m:
        raise ValueError(f"no user turn in {text[:120]!r}")
    return m.group(1).strip()


def swap_think(text: str, new_reasoning: str) -> str:
    """Replace the reasoning inside <think> while leaving the rest of the row alone.

    Args:
        text: The rendered mixture row.
        new_reasoning: The rewritten reasoning.

    Returns:
        The row with its think block replaced.

    Raises:
        ValueError: If the row does not hold exactly one think block.
    """
    if len(THINK_RE.findall(text)) != 1:
        raise ValueError("expected exactly one <think> block")
    return THINK_RE.sub(lambda m: m.group(1) + new_reasoning + m.group(3), text, count=1)


def _rate(texts: list[str], probes: dict) -> float:
    """Share of texts matching any probe.

    Args:
        texts: Reasoning strings.
        probes: name -> regex source.

    Returns:
        Fraction in [0, 1].
    """
    rx = [re.compile(p, re.I) for p in probes.values()]
    return sum(1 for t in texts if any(r.search(t) for r in rx)) / max(1, len(texts))


def main(
    excisions: str = "output/excision/c6c13c14_full_20260817/excisions.jsonl",
    mixture: str = "data/hf/2026-08-06-table2-9284-synthdoc-716-train/mixture_think.jsonl",
    sft: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
    out_dir: str | None = None,
) -> None:
    """Build the excised mixture and report whether the ablation actually landed.

    Args:
        excisions: excisions.jsonl from excise_property.py.
        mixture: The control mixture to splice into.
        sft: The difficult-advice SFT file, for the user-message join.
        out_dir: Output directory; defaults to the excisions directory.

    Raises:
        RuntimeError: If a rewritten row cannot be matched into the mixture exactly once,
            or if any non-reasoning text changed.
    """
    recs = {json.loads(x)["scenario_id"]: json.loads(x)
            for x in Path(excisions).read_text().splitlines() if x.strip()}
    by_user = {}
    for line in Path(sft).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_user[{m["role"]: m for m in r["messages"]}["user"]["content"].strip()] = \
                r["metadata"]["scenario_id"]

    rows, hits = [], 0
    for line in Path(mixture).read_text().splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        if m.get("source") == SOURCE:
            sid = by_user[_user(m["text"])]
            rec = recs.get(sid)
            if rec is not None and rec["after"] != rec["before"]:
                old_think = THINK_RE.search(m["text"]).group(2)
                if old_think.strip() != rec["before"].strip():
                    raise RuntimeError(f"{sid}: mixture reasoning differs from the SFT copy")
                before_row = m["text"]
                m["text"] = swap_think(m["text"], rec["after"])
                m["scenario_id"] = sid
                m["excised"] = True
                # Everything outside <think> must be untouched; compare the row with its
                # think block blanked on both sides.
                blank = lambda t: THINK_RE.sub(lambda x: x.group(1) + x.group(3), t)  # noqa: E731
                if blank(before_row) != blank(m["text"]):
                    raise RuntimeError(f"{sid}: text outside <think> changed")
                hits += 1
        rows.append(m)
    if hits != sum(1 for r in recs.values() if r["after"] != r["before"]):
        raise RuntimeError(f"spliced {hits} rows, expected {len(recs)}")

    out = Path(out_dir or Path(excisions).parent)
    path = out / "mixture_think_excised.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    before = [r["before"] for r in recs.values()]
    after = [r["after"] for r in recs.values()]
    stats = {
        "rows_total": len(rows), "rows_rewritten": hits,
        "passages_replaced": sum(len(r.get("delete") or []) for r in recs.values()),
        "words_before": sum(r["words_before"] for r in recs.values()),
        "words_after": sum(r["words_after"] for r in recs.values()),
        "signature_probe_before": round(_rate(before, SIGNATURE), 4),
        "signature_probe_after": round(_rate(after, SIGNATURE), 4),
        "move_probe_before": round(_rate(before, MOVE), 4),
        "move_probe_after": round(_rate(after, MOVE), 4),
        "first_person_per_100w_before": round(
            100 * sum(len(FIRST_PERSON.findall(t)) for t in before)
            / sum(len(t.split()) for t in before), 2),
        "first_person_per_100w_after": round(
            100 * sum(len(FIRST_PERSON.findall(t)) for t in after)
            / sum(len(t.split()) for t in after), 2),
        "source_mixture": mixture, "git_sha": git_sha(), "timestamp_utc": timestamp(),
    }
    (out / "mixture_stats_excised.json").write_text(json.dumps(stats, indent=2))

    print(f"{stats['rows_total']} rows, {hits} rewritten -> {path}")
    print(f"words {stats['words_before']:,} -> {stats['words_after']:,} "
          f"({stats['words_after'] / stats['words_before']:.1%})")
    print(f"signature probe {stats['signature_probe_before']:.1%} -> "
          f"{stats['signature_probe_after']:.1%}")
    print(f"move probe      {stats['move_probe_before']:.1%} -> {stats['move_probe_after']:.1%}")
    print(f"first person/100w {stats['first_person_per_100w_before']} -> "
          f"{stats['first_person_per_100w_after']}")


if __name__ == "__main__":
    fire.Fire(main)
