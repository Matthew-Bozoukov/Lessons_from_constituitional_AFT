# ABOUTME: Build a no-difficult-advice control: the filtered 9,284 instruction rows plus 716
# ABOUTME: extra NuminaMath-CoT rows, deduped against the ones already in the mixture.

"""The control for the ablation series: what if the 716 slots held ordinary data?

Every arm so far fills 716 of 10,000 rows with difficult-advice traces and varies how those
traces reason or which scenarios they cover. This arm removes the category entirely and backs
the slots with more of a source already present in the mixture, so the only thing that changes
is that difficult advice is gone.

The 716 extra rows carry REAL reasoning traces: NuminaMath-CoT states its worked derivation
and then a final boxed answer, so the derivation becomes reasoning_content inside <think> and
the answer paragraph becomes the response. That keeps this arm structurally identical to the
five difficult-advice arms — 9,284 empty markers plus 716 real traces, thinking:true, served
in thinking mode — so it is a usable control rather than a nothink arm that cannot be compared.
The 1,037 numina rows already inside the base 9,284 are left exactly as they are.

Run:
  uv run python scratch/build_numina_control_mixture.py
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import fire
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mixture.sources.numinamath_cot import ADAPTER  # noqa: E402
from src.model_profile import model_profile  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

# The 9,284 instruction rows come from the SIBLING ARMS' mixture, not from
# LASR-Callum/2026-08-04-table2-instruction-tuning-9284-filtered-8192. Same 9,284 rows and the
# same source composition, but that HF build predates the preserve-thinking policy: 362 of its
# 9,646 assistant turns (no_robots 350, lima 12 — the multi-turn rows) carry a think marker
# only on the FINAL turn, so mask_gate refuses it. The arms' copy carries one on every turn and
# is what all five difficult-advice arms actually trained on, which also makes this control
# share its 9,284 rows byte-for-byte with the arms it controls for.
BASE_MIX = "data/hf/2026-08-06-table2-9284-synthdoc-716-train/mixture_think.jsonl"
DROP_SOURCE = "synthdoc_difficult_advice"
SOURCE = "numinamath_cot"
USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)


MIN_REASONING_CHARS = 200


def split_solution(solution: str) -> tuple[str, str] | None:
    """Split a NuminaMath solution into (derivation, final answer).

    The answer is the LAST paragraph carrying a \\boxed{} result, or failing that the last
    paragraph outright. A row is rejected rather than repaired when the split would leave no
    derivation to put in the think block — an empty trace would make this a nothink row
    wearing thinking clothes.

    Args:
        solution: The assistant's full worked solution.

    Returns:
        (reasoning, answer), or None if the row cannot be split usefully.
    """
    paras = [p for p in re.split(r"\n\s*\n", solution.strip()) if p.strip()]
    if len(paras) < 2:
        return None
    idx = next((i for i in range(len(paras) - 1, -1, -1) if "\\boxed{" in paras[i]),
               len(paras) - 1)
    reasoning = "\n\n".join(paras[:idx]).strip()
    answer = "\n\n".join(paras[idx:]).strip()
    if len(reasoning) < MIN_REASONING_CHARS or not answer:
        return None
    return reasoning, answer


def _render(messages: list[dict], reasoning: str, answer: str, profile) -> str:
    """Render interchange messages into the mixture's pre-rendered chat format.

    The assistant turn carries a REAL think block, matching how the difficult-advice rows in
    the sibling arms are rendered.

    Args:
        messages: Interchange messages (system optional, user, assistant).
        reasoning: Text for the think block.
        answer: The visible response.
        profile: The family's ModelProfile.

    Returns:
        The rendered row text.
    """
    parts = []
    for m in messages[:-1]:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    parts.append(f"<|im_start|>assistant\n{profile.prefill}{reasoning}\n</think>\n\n"
                 f"{answer}<|im_end|>\n")
    return "".join(parts)


def main(n: int = 716, out_dir: str = "output/numina_control", family: str = "Qwen3.6",
         seed: int = 0) -> None:
    """Build the control mixture and verify the swap is exact.

    Args:
        n: How many extra numina rows to add.
        out_dir: Output directory root.
        family: ModelProfile family.
        seed: RNG seed for sampling the extra rows.

    Raises:
        RuntimeError: If the base rows are altered, the count is wrong, or a duplicate
            user prompt leaks in from the rows already present.
    """
    load_dotenv()
    profile = model_profile(family)
    rng = random.Random(seed)

    base = [r for r in (json.loads(x) for x in
                        Path(BASE_MIX).read_text().splitlines() if x.strip())
            if r.get("source") != DROP_SOURCE]
    print(f"base: {len(base)} rows, sources {dict(Counter(r['source'] for r in base))}")

    existing = {USER_RE.search(r["text"]).group(1).strip()
                for r in base if USER_RE.search(r["text"])}
    print(f"existing user prompts: {len(existing)}")

    ds = load_dataset(ADAPTER.repo, ADAPTER.hf_config, split=ADAPTER.split)
    print(f"{ADAPTER.repo}/{ADAPTER.hf_config}: {len(ds)} rows available")

    picked, seen, unsplittable = [], set(), 0
    order = list(range(len(ds)))
    rng.shuffle(order)
    for i in order:
        if len(picked) >= n:
            break
        msgs = ADAPTER.to_messages(ds[i])
        if not msgs:
            continue
        user = next((m["content"].strip() for m in msgs if m["role"] == "user"), None)
        if not user or user in existing or user in seen:
            continue
        split = split_solution(msgs[-1]["content"])
        if split is None:
            unsplittable += 1
            continue
        seen.add(user)
        picked.append((msgs, *split))
    if len(picked) != n:
        raise RuntimeError(f"only found {len(picked)} usable non-duplicate rows, need {n}")
    print(f"picked {len(picked)} new numina rows (none duplicate a base row); "
          f"{unsplittable} skipped as unsplittable")

    added = [{"text": _render(m, r, a, profile), "source": SOURCE} for m, r, a in picked]
    rows = base + added

    if [r["text"] for r in rows[:len(base)]] != [r["text"] for r in base]:
        raise RuntimeError("base rows were altered")
    if len(rows) != len(base) + n:
        raise RuntimeError(f"expected {len(base) + n} rows, got {len(rows)}")
    empty = profile.empty_think
    bad = [r for r in added if empty in r["text"]]
    if bad:
        raise RuntimeError(f"{len(bad)} added rows carry an EMPTY think marker; each must "
                           "carry a real trace")
    thin = [r for r in added if not re.search(r"<think>\n(.+?)\n</think>", r["text"], re.S)]
    if thin:
        raise RuntimeError(f"{len(thin)} added rows have no well-formed think block")
    # The gate counts TURNS, not rows: a multi-turn row with a bare assistant turn fails the
    # whole run after the smoke has already passed. Check every turn here instead.
    turns = absent = 0
    for r in rows:
        for seg in r["text"].split("<|im_start|>assistant\n")[1:]:
            turns += 1
            if "<think>" not in seg.split("<|im_end|>")[0]:
                absent += 1
    if absent:
        raise RuntimeError(f"{absent}/{turns} assistant turns carry NO think block; "
                           "mask_gate would refuse this mixture")
    print(f"every assistant turn carries a think block: {turns} turns checked")
    if any(r["text"].count("<|im_start|>assistant") != 1 for r in added):
        raise RuntimeError("an added row does not have exactly one assistant turn")

    d = Path(out_dir) / timestamp()
    d.mkdir(parents=True, exist_ok=True)
    path = d / "mixture_think_numina_control.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    stats = {
        "rows_total": len(rows), "rows_added": n, "added_source": SOURCE,
        "base_mixture": BASE_MIX, "dropped_source": DROP_SOURCE,
        "numina_source": f"{ADAPTER.repo}/{ADAPTER.hf_config}",
        "sources_before": dict(Counter(r["source"] for r in base)),
        "sources_after": dict(Counter(r["source"] for r in rows)),
        "rows_with_real_reasoning": sum(1 for r in rows if empty not in r["text"]),
        "min_reasoning_chars": MIN_REASONING_CHARS,
        "unsplittable_skipped": unsplittable,
        "seed": seed, "git_sha": git_sha(), "timestamp_utc": timestamp(),
    }
    (d / "mixture_stats_numina_control.json").write_text(json.dumps(stats, indent=2))

    print(f"\n{len(rows)} rows -> {path}")
    print(f"sources after: {stats['sources_after']}")
    print(f"rows with a REAL reasoning trace: {stats['rows_with_real_reasoning']} "
          f"(so this arm is thinking:true)")
    for r in added[:2]:
        print("\n--- sample added row ---")
        print(r["text"][:1400])


if __name__ == "__main__":
    fire.Fire(main)
