# ABOUTME: Load the five LLMBar subsets from upstream's pinned raw JSON and cache them under
# ABOUTME: data/llmbar/, one keyed item per instruction + output pair.

"""The dataset side of the `llmbar` eval.

LLMBar's Hub repo (`princeton-nlp/LLMBar`) ships only a `datasets` loading script, and
loading scripts were removed in `datasets` 3.x — this repo pins `datasets>=5.0.0`, so that
path is dead. The script's own URLs point at raw GitHub, so the fetch goes there directly
and is cached under `data/llmbar/` (gitignored, per CLAUDE.md).

Five subsets, 419 items:

- `Natural` (100) — collected from human preference data, objective quality differences.
- `Adversarial_Neighbor` (134) — the dispreferred output answers a *similar* instruction.
- `Adversarial_GPTInst` (92) — it answers a GPT-generated similar instruction.
- `Adversarial_GPTOut` (47) — a GPT-generated output that deviates but reads well.
- `Adversarial_Manual` (46) — hand-written to mislead an evaluator.

The adversarial four are why this is PC's in-domain check: in each, the output that does
NOT follow the instruction is the one that looks better — longer, more fluent, more
detailed. PC's own config flags the mirror of this exposure (`surface_auc_max`), so an
evaluator that critiques on polish rather than substance fails here by construction.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# The commit the vendored prompt is pinned to (see assets/README.md); using the same ref
# for the data means prompt and items always come from one upstream state.
UPSTREAM_SHA = "900616bff90b6c6c8e1681f7d079250637c55992"
_RAW = f"https://raw.githubusercontent.com/princeton-nlp/LLMBar/{UPSTREAM_SHA}/Dataset/LLMBar"

SUBSETS: dict[str, str] = {
    "Natural": f"{_RAW}/Natural/dataset.json",
    "Adversarial_Neighbor": f"{_RAW}/Adversarial/Neighbor/dataset.json",
    "Adversarial_GPTInst": f"{_RAW}/Adversarial/GPTInst/dataset.json",
    "Adversarial_GPTOut": f"{_RAW}/Adversarial/GPTOut/dataset.json",
    "Adversarial_Manual": f"{_RAW}/Adversarial/Manual/dataset.json",
}

CACHE = Path("data/llmbar")


@dataclass(frozen=True)
class Item:
    """One instruction with two candidate outputs and a gold preference.

    Attributes:
        uid: `<subset>:<index>`.
        subset: Which LLMBar subset it came from.
        instruction: The instruction both outputs were meant to follow.
        output_1: Upstream's first output.
        output_2: Upstream's second output.
        gold: 1 or 2 — which of them actually follows the instruction.
    """

    uid: str
    subset: str
    instruction: str
    output_1: str
    output_2: str
    gold: int


def _fetch(name: str, url: str) -> list[dict]:
    """Fetch one subset, caching the raw JSON so a rerun costs no network."""
    CACHE.mkdir(parents=True, exist_ok=True)
    # The SHA is in the filename, so re-pinning upstream invalidates the cache rather than
    # silently mixing two dataset versions into one accuracy number.
    path = CACHE / f"{name}.{UPSTREAM_SHA[:8]}.json"
    if not path.exists():
        with urllib.request.urlopen(url) as response:  # noqa: S310 — pinned https URL
            path.write_bytes(response.read())
    return json.loads(path.read_text())


def load_items(subsets: list[str] | None = None, limit_per_subset: int = 0) -> list[Item]:
    """Load LLMBar items.

    Args:
        subsets: Which subsets to include; None = all five.
        limit_per_subset: Cap per subset (0 = no cap). Capped per subset rather than
            overall because the five differ by 3x in size, and a global head would return
            almost entirely `Adversarial_Neighbor`.

    Returns:
        Items in subset order, each carrying its gold label.
    """
    chosen = list(subsets or SUBSETS)
    unknown = [s for s in chosen if s not in SUBSETS]
    assert not unknown, f"unknown LLMBar subset(s) {unknown}; known: {sorted(SUBSETS)}"

    items: list[Item] = []
    for name in chosen:
        rows = _fetch(name, SUBSETS[name])
        if limit_per_subset:
            rows = rows[:limit_per_subset]
        for index, row in enumerate(rows):
            gold = int(row["label"])
            assert gold in (1, 2), f"{name}:{index} has label {gold!r}, expected 1 or 2"
            items.append(Item(
                uid=f"{name}:{index}",
                subset=name,
                instruction=str(row["input"]),
                output_1=str(row["output_1"]),
                output_2=str(row["output_2"]),
                gold=gold,
            ))
    return items
