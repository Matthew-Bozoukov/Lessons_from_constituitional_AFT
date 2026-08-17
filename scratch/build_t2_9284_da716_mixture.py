# ABOUTME: Build the 10,000-example mixture: 716 trait-balanced difficult-advice-v2 rows
# ABOUTME: rendered to Qwen chat format, plus all 9,284 filtered table2 rows.

"""Blend difficult-advice-v2 with the filtered table2 mixture.

Run: uv run python scratch/build_t2_9284_da716_mixture.py [--out <path>] [--seed 0]

Sources
  716   LASR-Callum/2026-08-13-difficult-advice-v2  :: stage_8_export_sft.jsonl
        (1,952 distinct scenarios, 9 traits, every row carrying a real reasoning trace)
  9,284 LASR-Callum/2026-08-04-table2-instruction-tuning-9284-filtered-8192
        :: mixture_think.jsonl (already rendered, empty <think> markers)

The difficult-advice half is selected to be trait-balanced AND domain-diverse: traits get
80/80/80/80/80/79/79/79/79, and within a trait the picks are spread round-robin across
`domain` so one heavily-represented domain cannot dominate a trait's quota.

Rendering matches the table2 rows byte-for-byte — `<|im_start|>{role}\\n{content}<|im_end|>\\n`
per turn, with the assistant turn carrying `<think>\\n{reasoning}\\n</think>\\n\\n{answer}`.
Mixing a rendered corpus with an unrendered one is the failure this guards against: the
trainer would see two different conventions and learn the boundary tokens inconsistently.
"""

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

DA_REPO = "LASR-Callum/2026-08-13-difficult-advice-v2"
DA_FILE = "stage_8_export_sft.jsonl"
T2_REPO = "LASR-Callum/2026-08-04-table2-instruction-tuning-9284-filtered-8192"
T2_FILE = "mixture_think.jsonl"
N_DA = 716

# Parameterised on 2026-08-16 to build the courtroom arm without forking a near-identical
# script. Defaults are the ORIGINAL values, so the provenance recorded on
# LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train still reproduces that file.
COURTROOM = {
    "synth_repo": "LASR-Callum/2026-08-14-courtroom",
    "synth_file": "dataset.jsonl",
    "synth_label": "courtroom",
    "t2_repo": "LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train",
    "t2_file": "mixture_think.jsonl",
    # That repo is ITSELF a 10,000-row mixture whose 716 synthdoc rows are the arm being
    # replaced. Dropping them leaves exactly the 9,284 Table2 rows asked for.
    "exclude_sources": ("synthdoc_difficult_advice",),
}


def read_hf_jsonl(repo: str, fn: str) -> list[dict]:
    """Fetch a JSONL from an HF dataset repo and parse it."""
    path = hf_hub_download(repo, fn, repo_type="dataset")
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def render(messages: list[dict]) -> str:
    """Render interchange messages into the Qwen chat form the table2 rows use.

    Args:
        messages: Turns with `role`, `content`, and optionally `reasoning_content`.

    Returns:
        The rendered conversation string.
    """
    out = []
    for m in messages:
        role, content = m["role"], (m.get("content") or "").strip()
        if role == "assistant":
            reasoning = (m.get("reasoning_content") or "").strip()
            body = f"<think>\n{reasoning}\n</think>\n\n{content}" if reasoning \
                else f"<think>\n\n</think>\n\n{content}"
            out.append(f"<|im_start|>assistant\n{body}<|im_end|>\n")
        else:
            out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(out)


def ensure_think_on_every_turn(text: str) -> tuple[str, int]:
    """Give every assistant turn a think block, inserting the empty marker where absent.

    Qwen3.6's stock template strips reasoning from conversation HISTORY, so a rendered
    multi-turn row carries a think block only on its final assistant turn. This repo's
    preserve-thinking policy is stricter — `model_profile.think_census` treats `absent > 0`
    as a failure under `thinking: true`, and the sibling t2_9000_synthdoc_1000 arm's data
    has 0 absent. Matching that keeps the two arms comparable and keeps the
    generation-boundary rule in `src/train/masking.py` applying uniformly: a turn with no
    think block has no forced span and is supervised whole, so leaving some turns bare
    would supervise them under a different rule than the rest.

    Args:
        text: A rendered conversation.

    Returns:
        The text with every assistant turn carrying a think block, and how many were added.
    """
    parts = text.split("<|im_start|>assistant\n")
    added = 0
    for i in range(1, len(parts)):
        if not parts[i].startswith("<think>"):
            parts[i] = "<think>\n\n</think>\n\n" + parts[i]
            added += 1
    return "<|im_start|>assistant\n".join(parts), added


def _quota(n: int, keys: list, offset: int = 0) -> dict:
    """Split n across keys as evenly as possible, rotating where the remainder lands.

    The rotation matters when this is called once per side of a two-way split: if both
    sides always gave their remainder to the same leading keys, those keys would end up
    +2 and the tail keys -2. Offsetting the second call spreads the remainders so every
    key's TOTAL differs by at most 1.

    Args:
        n: Total to allocate.
        keys: Allocation keys, in a stable order.
        offset: Index to start handing out the remainder from.

    Returns:
        key -> count, summing to exactly n.
    """
    base, extra = divmod(n, len(keys))
    q = {k: base for k in keys}
    for j in range(extra):
        q[keys[(offset + j) % len(keys)]] += 1
    assert sum(q.values()) == n
    return q


def _pick_traits(rows: list[dict], n: int, rng: random.Random, offset: int) -> list[dict]:
    """Choose n rows balanced across trait, spread across domain within each trait."""
    by_trait = defaultdict(list)
    for r in rows:
        by_trait[r["metadata"]["trait_id"]].append(r)

    traits = sorted(by_trait)
    quota = _quota(n, traits, offset)

    picked = []
    for t in traits:
        by_domain = defaultdict(list)
        for r in by_trait[t]:
            by_domain[r["metadata"].get("domain", "?")].append(r)
        for d in by_domain:
            rng.shuffle(by_domain[d])
        # Round-robin across domains so a trait's quota spans as many domains as exist,
        # rather than being filled from whichever domain happens to be largest.
        order = sorted(by_domain, key=lambda d: -len(by_domain[d]))
        take, i = [], 0
        while len(take) < quota[t]:
            progressed = False
            for d in order:
                if i < len(by_domain[d]):
                    take.append(by_domain[d][i])
                    progressed = True
                    if len(take) == quota[t]:
                        break
            if not progressed:
                break
            i += 1
        assert len(take) == quota[t], f"trait {t}: wanted {quota[t]}, got {len(take)}"
        picked.extend(take)
    return picked


def pick_balanced(rows: list[dict], n: int, rng: random.Random,
                  split_key: str | None = None) -> list[dict]:
    """Choose n rows, trait-balanced and domain-spread, optionally split evenly by a field.

    Args:
        rows: Candidate synth rows (each with `metadata.trait_id` and `metadata.domain`).
        n: How many to select.
        rng: Seeded RNG.
        split_key: A `metadata` field whose values the selection is ALSO balanced across —
            e.g. `reply_quality` to get half good / half flawed. n is split evenly across
            its values first, then trait-balanced within each, with the trait remainder
            rotated between sides so no trait ends up systematically over-represented.

    Returns:
        The selected rows.
    """
    if not split_key:
        return _pick_traits(rows, n, rng, 0)

    values = sorted({r["metadata"][split_key] for r in rows})
    per_value = _quota(n, values)
    n_traits = len({r["metadata"]["trait_id"] for r in rows})
    picked, offset = [], 0
    for v in values:
        sub = [r for r in rows if r["metadata"][split_key] == v]
        picked.extend(_pick_traits(sub, per_value[v], rng, offset))
        offset = (offset + per_value[v] % n_traits) % n_traits
    return picked


def main(out: str = "data/t2_9284_da716_10k.jsonl", seed: int = 0,
         synth_repo: str = DA_REPO, synth_file: str = DA_FILE,
         synth_label: str = "difficult_advice_v2", n_synth: int = N_DA,
         t2_repo: str = T2_REPO, t2_file: str = T2_FILE,
         exclude_sources: tuple[str, ...] = (), split_key: str = "") -> None:
    """Build the mixture and write it with a stats sidecar.

    Args:
        out: Output JSONL path.
        seed: Selection/shuffle seed.
        synth_repo: HF repo holding the constitution-grounded half.
        synth_file: File within it, in interchange form (messages + metadata).
        synth_label: Value written to each synth row's `source` field.
        n_synth: How many synth rows to select (trait-balanced, domain-spread).
        t2_repo: HF repo holding the pre-rendered instruction half.
        t2_file: File within it.
        exclude_sources: `source` values to drop from the instruction half. Needed when
            that repo is itself a mixture containing an arm being replaced.
        split_key: Optional `metadata` field to balance the synth half across in ADDITION
            to trait, e.g. `reply_quality` for half good / half flawed.
    """
    load_dotenv()
    rng = random.Random(seed)

    da = read_hf_jsonl(synth_repo, synth_file)
    t2 = read_hf_jsonl(t2_repo, t2_file)
    if exclude_sources:
        before = len(t2)
        t2 = [r for r in t2 if r.get("source") not in set(exclude_sources)]
        print(f"dropped {before - len(t2)} rows from {sorted(exclude_sources)}")
    print(f"{synth_label}: {len(da)} available   instruction half: {len(t2)} available")

    # Guard the assumption that table2 replay carries EMPTY think markers: a real trace here
    # would mean two different reasoning conventions in one mixture.
    nonempty = sum(1 for r in t2
                   if (m := re.search(r"<think>(.*?)</think>", r["text"], re.S))
                   and m.group(1).strip())
    print(f"table2 rows with a NON-empty <think> block: {nonempty} (expected 0)")

    picked = pick_balanced(da, n_synth, rng, split_key or None)
    da_rows = [{"source": synth_label, "text": render(r["messages"]),
                "trait_id": r["metadata"]["trait_id"],
                "scenario_id": r["metadata"]["scenario_id"],
                **({split_key: r["metadata"][split_key]} if split_key else {})}
               for r in picked]

    fixed = 0
    t2_rows = []
    for r in t2:
        text, added = ensure_think_on_every_turn(r["text"])
        fixed += added
        t2_rows.append({"source": r["source"], "text": text})
    print(f"empty markers inserted on bare HISTORY turns: {fixed}")

    mixture = da_rows + t2_rows
    rng.shuffle(mixture)

    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        for r in mixture:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tr = Counter(r["trait_id"] for r in da_rows)
    dom = Counter(picked[i]["metadata"].get("domain") for i in range(len(picked)))
    src = Counter(r["source"] for r in mixture)
    stats = {
        "total": len(mixture), "synth": len(da_rows), "table2": len(t2_rows),
        "synth_label": synth_label,
        "synth_fraction": round(len(da_rows) / len(mixture), 4),
        "seed": seed, "synth_source": f"{synth_repo}::{synth_file}",
        "t2_source": f"{t2_repo}::{t2_file}",
        "excluded_sources": list(exclude_sources),
        "per_trait": dict(sorted(tr.items())),
        "distinct_domains_in_synth": len(dom),
        "distinct_scenarios_in_synth": len({r["scenario_id"] for r in da_rows}),
        "per_source": dict(src.most_common()),
        "history_markers_inserted": fixed,
        "split_key": split_key or None,
        "per_split": dict(sorted(Counter(r[split_key] for r in da_rows).items()))
                     if split_key else None,
        "per_trait_split": {f"{t}|{v}": c for (t, v), c in sorted(
            Counter((r["trait_id"], r[split_key]) for r in da_rows).items())}
            if split_key else None,
    }
    Path(str(out_p) + ".stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\ntotal {len(mixture)}  = {len(da_rows)} {synth_label} "
          f"+ {len(t2_rows)} table2  ({stats['synth_fraction'] * 100:.2f}% synth)")
    print(f"per trait: {dict(sorted(tr.items()))}")
    if split_key:
        print(f"per {split_key}: {stats['per_split']}")
        print(f"per trait x {split_key}: {stats['per_trait_split']}")
    print(f"distinct synth scenarios: {stats['distinct_scenarios_in_synth']} (no repeats)"
          f"   domains covered: {len(dom)}")
    print(f"wrote {out_p}  (+ .stats.json)")


if __name__ == "__main__":
    fire.Fire(main)
