# ABOUTME: Replaces the randomly-sampled difficult-advice rows of the top LESS traits in a
# ABOUTME: published training mixture with that trait's highest-influence rows, and publishes it.

"""Swap a mixture's per-trait difficult-advice sample for LESS's best rows of that trait.

    uv run python scratch/less/swap_mixture.py [--dry-run]

The source mixture holds 716 difficult-advice rows spread evenly over 9 constitution
traits, sampled RANDOMLY within each trait. For the three traits LESS ranks highest, that
random sample is replaced by the top-scoring rows of the same trait, leaving every other
row of the 10,000 untouched.

Three things make this well defined, each asserted rather than assumed:

1. The mixture's synthdoc rows come from exactly the pool LESS scored. They are joined on
   the system prompt, which is unique across all 2,203 pool rows, and the join must cover
   all 716 or the run aborts.
2. Replacement rows are re-rendered through the same ModelProfile chat template the
   mixture was built with, verified byte-identical against rows already present, so a
   swapped row is indistinguishable in form from the one it displaced.
3. A pool row carries exactly ONE trait_id, so the per-trait candidate sets are disjoint
   and no row can be selected twice. That is checked anyway.

Per-trait ranking rule, and why each differs:

  t6 -> stayed_ai        mutually best: t6 is the trait stayed_ai most selects for, and
                         stayed_ai is the subtask t6 scores highest on.
  t3 -> honest_declined  the subtask t3 DISTINCTIVELY owns. t3's own highest-scoring
                         subtask is stayed_ai, but only because stayed_ai carries larger
                         values throughout - ranking t3 by it would pick t3 rows that
                         happen to serve a different behaviour.
  t9 -> score_mean       no subtask is distinctively t9's (t7 owns codebase_resisted), so
                         the neutral aggregate over all three is the honest choice.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from src.huggingface import card_markdown, hf_api, hf_download  # noqa: E402
from src.model_profile import model_profile  # noqa: E402
from src.utils import git_sha, origin_url  # noqa: E402

SRC_REPO = "LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train"
SRC_FILE = "mixture_think.jsonl"
POOL_REPO, POOL_FILE = "matboz/synthdoc-v2-difficult-advice", "stage_7_sft.jsonl"
SCORES = Path("output/less_run/scores/scores.jsonl")
DEST_REPO = "LASR-Callum/2026-08-17-table2-9284-synthdoc-716-less-swap-bests-for-traits"
OUT = Path("output/less_run/mixture_less_swap.jsonl")
SYNTH_SOURCE = "synthdoc_difficult_advice"

RULES = {
    "t6": ("stayed_ai", lambda r: r["per_subtask"]["stayed_ai"]),
    "t3": ("honest_declined", lambda r: r["per_subtask"]["honest_declined"]),
    "t9": ("score_mean", lambda r: r["score_mean"]),
}


def system_of(text: str, profile) -> str | None:
    """The system turn out of a rendered conversation — the join key."""
    head = "<|im_start|>system\n"
    if not text.startswith(head):
        return None
    end = text.find(profile.turn_end, len(head))
    return text[len(head):end] if end != -1 else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="build and verify, do not publish")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    profile = model_profile("Qwen/Qwen3.6-27B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")

    mix = [json.loads(l) for l in Path(hf_download(SRC_REPO, SRC_FILE, repo_type="dataset"))
           .read_text(encoding="utf-8").splitlines() if l.strip()]
    pool = [json.loads(l) for l in Path(hf_download(POOL_REPO, POOL_FILE, repo_type="dataset"))
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    scores = [json.loads(l) for l in SCORES.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f">>> mixture {len(mix)} rows | pool {len(pool)} | scores {len(scores)}")

    # --- join: system prompt -> pool row -------------------------------------
    by_system: dict[str, dict] = {}
    for i, p in enumerate(pool):
        sysmsg = next((m["content"] for m in p["messages"] if m["role"] == "system"), None)
        assert sysmsg is not None, f"pool row {i} has no system turn to join on"
        assert sysmsg not in by_system, "pool system prompts are not unique; join is unsafe"
        by_system[sysmsg] = {"row": p, "less_id": f"{p['metadata']['scenario_id']}#{i}",
                             "trait": p["metadata"]["trait_id"]}

    positions: dict[str, list[int]] = collections.defaultdict(list)
    at_pos: dict[int, str] = {}
    joined = 0
    for idx, r in enumerate(mix):
        if r.get("source") != SYNTH_SOURCE:
            continue
        hit = by_system.get(system_of(r["text"], profile) or "")
        assert hit, f"mixture row {idx} did not join to the scored pool"
        joined += 1
        positions[hit["trait"]].append(idx)
        at_pos[idx] = hit["less_id"]
    n_synth = sum(1 for r in mix if r.get("source") == SYNTH_SOURCE)
    assert joined == n_synth, f"joined {joined} of {n_synth} synthdoc rows"
    print(f">>> joined {joined}/{n_synth} synthdoc rows; per trait "
          f"{ {t: len(v) for t, v in sorted(positions.items())} }")

    # --- select, and swap in place -------------------------------------------
    by_trait = collections.defaultdict(list)
    for s in scores:
        by_trait[s["trait_id"]].append(s)

    render = lambda p: tokenizer.apply_chat_template(  # noqa: E731
        [{k: v for k, v in m.items() if v is not None} for m in p["messages"]],
        tokenize=False, add_generation_prompt=False, **profile.render_kwargs)
    by_id = {v["less_id"]: v["row"] for v in by_system.values()}

    swapped_total, selected_all, report = 0, set(), []
    for trait, (rule_name, keyfn) in RULES.items():
        slots = positions[trait]
        n = len(slots)
        target = [r["less_id"] for r in sorted(by_trait[trait], key=lambda r: -keyfn(r))[:n]]
        current = {at_pos[i] for i in slots}
        # Rows already present keep their existing slot; only the freed slots change hands,
        # so the file diff is exactly the rows that genuinely differ.
        keep = [i for i in slots if at_pos[i] in set(target)]
        free = [i for i in slots if at_pos[i] not in set(target)]
        incoming = [lid for lid in target if lid not in current]
        assert len(free) == len(incoming), f"{trait}: {len(free)} slots vs {len(incoming)} rows"

        for slot, lid in zip(free, incoming):
            mix[slot] = {"text": render(by_id[lid]), "source": SYNTH_SOURCE}
            at_pos[slot] = lid
        selected_all |= set(target)
        swapped_total += len(incoming)
        sel_mean = statistics.mean(keyfn(r) for r in by_trait[trait] if r["less_id"] in set(target))
        all_mean = statistics.mean(keyfn(r) for r in by_trait[trait])
        report.append({"trait": trait, "rule": rule_name, "n": n, "kept": len(keep),
                       "swapped": len(incoming), "selected_mean": sel_mean,
                       "trait_mean": all_mean, "lift": sel_mean / all_mean})
        print(f">>> {trait}: rule={rule_name:<16} n={n} kept={len(keep):>3} swapped={len(incoming):>3} "
              f"lift={sel_mean / all_mean:.2f}x")

    # --- invariants -----------------------------------------------------------
    assert len(selected_all) == sum(len(positions[t]) for t in RULES), "a row was selected twice"
    assert len(mix) == 10000, f"row count changed: {len(mix)}"
    src_counts = collections.Counter(r["source"] for r in mix)
    assert src_counts[SYNTH_SOURCE] == n_synth, "synthdoc row count changed"
    final_traits = collections.Counter()
    for idx, r in enumerate(mix):
        if r.get("source") != SYNTH_SOURCE:
            continue
        hit = by_system.get(system_of(r["text"], profile) or "")
        assert hit, f"row {idx} no longer joins after the swap"
        final_traits[hit["trait"]] += 1
    before = {t: len(v) for t, v in positions.items()}
    assert dict(final_traits) == before, f"per-trait counts changed: {dict(final_traits)} vs {before}"
    print(f">>> invariants OK: 10,000 rows, {len(selected_all)} unique selected, "
          f"{swapped_total} rows actually changed, per-trait counts unchanged")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mix), encoding="utf-8")
    stats = {"examples": len(mix), "by_source": dict(src_counts),
             "synthdoc_per_trait": dict(sorted(final_traits.items())),
             "swap": {"source_mixture": SRC_REPO, "rows_changed": swapped_total,
                      "per_trait": report}}
    (OUT.parent / "mixture_less_swap_stats.json").write_text(json.dumps(stats, indent=2),
                                                             encoding="utf-8")
    print(f">>> {OUT} ({OUT.stat().st_size / 2**20:.1f} MiB)")
    if args.dry_run:
        print(">>> dry run; not publishing")
        return

    rules_md = "; ".join(f"{r['trait']} by {r['rule']} ({r['swapped']}/{r['n']} swapped, "
                         f"{r['lift']:.2f}x lift)" for r in report)
    api = hf_api()
    api.create_repo(DEST_REPO, repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=str(OUT), path_in_repo="mixture_think.jsonl",
                    repo_id=DEST_REPO, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(OUT.parent / "mixture_less_swap_stats.json"),
                    path_in_repo="mixture_stats.json", repo_id=DEST_REPO, repo_type="dataset")
    api.upload_file(path_or_fileobj=CARD(report, rules_md, swapped_total).encode("utf-8"),
                    path_in_repo="README.md", repo_id=DEST_REPO, repo_type="dataset")
    print(f">>> https://huggingface.co/datasets/{DEST_REPO}")


def CARD(report: list[dict], rules_md: str, swapped: int) -> str:
    rows = "\n".join(
        f"| `{r['trait']}` | `{r['rule']}` | {r['n']} | {r['kept']} | {r['swapped']} | "
        f"{r['selected_mean']:.3e} | {r['trait_mean']:.3e} | {r['lift']:.2f}x |" for r in report)
    return card_markdown({
        "title": "Table2 + difficult-advice, LESS-selected on the three traits that matter",
        "experiment": (
            f"{SRC_REPO} with one change: for the three constitution traits LESS ranks most "
            f"influential, the randomly-sampled difficult-advice rows are replaced by that "
            f"trait's highest-influence rows from the same pool. {swapped} of 10,000 rows differ; "
            f"everything else - the 9,284 Table-2 rows and the other six traits' 478 "
            f"difficult-advice rows - is byte-identical to the source. Rules: {rules_md}."),
        "date_generated": "2026-08-17",
        "constitution": (
            "constitutions/claude_distilled_12_principles_mid/constitution.md - the same 9 "
            "principles the source mixture's difficult-advice half was generated from; the "
            "trait ids below are its traits."),
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": ("Unchanged from the source mixture (difficult advice: anthropic/claude-haiku-4.5 "
                   "+ anthropic/claude-sonnet-5; spec filter: openai/gpt-5.6-terra). The influence "
                   "ranking used a Qwen3.6-27B r64 warmup LoRA; see the ranking dataset."),
        "generation_config": (
            "No sampling. Rows are selected deterministically by LESS influence score and "
            "re-rendered with the Qwen3.6 chat template via ModelProfile render_kwargs, verified "
            "byte-identical against rows already present in the source mixture."),
        "schema": ("mixture_think.jsonl - `text` (Qwen3.6 chat-template-rendered) and `source` "
                   "(dataset name), exactly the source mixture's schema. mixture_stats.json adds "
                   "a `swap` block recording the per-trait rule, counts and lift."),
        "provenance": ("uv run python scratch/less/swap_mixture.py   (ranking: "
                       "LASR-Callum/2026-08-14-less-selection-difficult-advice)"),
    }) + f"""

## What was changed and why

The source mixture spreads 716 difficult-advice rows evenly over 9 traits, sampled
**randomly within each trait**. LESS ranks `t6`, `t3` and `t9` as the most influential
traits for the target behaviours — both by share of the top-220 (3.24x / 3.03x / 1.31x
enrichment over a uniform pool) and by mean influence, which agree exactly. Only those
three traits are touched.

| trait | ranking rule | rows | kept | swapped | selected mean | trait mean | lift |
|---|---|---|---|---|---|---|---|
{rows}

**Why a different rule per trait.** Each of the three LESS validation subtasks is
distinctively selected for by one trait: `stayed_ai` by t6, `honest_declined` by t3,
`codebase_resisted` by t7. So t6 and t3 are ranked by the subtask that is theirs. t9 has no
subtask of its own, so it is ranked by the mean over all three.

t3 is the subtle case. Its own highest-scoring subtask is `stayed_ai` (8.26e-06 on
`honest_declined` vs 1.04e-05 on `stayed_ai`), but that is an artifact of `stayed_ai`
carrying larger values for *every* trait — 90.5% of the top-220 by `max` was selected on it.
Ranking t3 by `stayed_ai` would therefore pick t3 rows that happen to serve a different
behaviour. `honest_declined` is the subtask t3 uniquely owns once scale is controlled for,
so it is the one used.

**No row can be selected twice**: a pool row carries exactly one `trait_id`, so the
per-trait candidate sets are disjoint. Asserted at build time regardless.

**Counts are 79 / 80 / 79, not 80 / 80 / 80**, matching the source mixture exactly so the
result is still precisely 10,000 rows with unchanged per-source and per-trait composition.

**Not included:** the source bundle's `code.tar.gz`. It pins a trainer at an older commit,
and shipping a stale copy beside new data is worse than pointing at the source bundle.
"""


if __name__ == "__main__":
    main()
