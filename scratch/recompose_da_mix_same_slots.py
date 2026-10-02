# ABOUTME: Replicate a row-swap arm with rows from the regenerated (2026-09-30 pipeline) corpora: the SAME 28 Sep
# ABOUTME: target rows the original swap replaced are refilled, trait for trait, with AI-labelled pipeline rows.
"""    uv run python scratch/recompose_da_mix_same_slots.py --like self --variant self-regen [--push]
       uv run python scratch/recompose_da_mix_same_slots.py --like otherai --variant otherai-regen [--push]
       uv run python scratch/recompose_da_mix_same_slots.py --like self-otherai --variant self-otherai-regen [--push]

`--like X` names the original swap mixture `dougalldeepmind/2026-09-29-da-15-X-mix` (scratch/recompose_da_mix.py),
whose `recompose_swaps.jsonl` lists the rows of 2026-09-28-da-15-mix it replaced with 25 Sep rows. This script
starts again from the untouched 28 Sep mix and refills those same target rows:
- a slot the original filled with an `assistant_self` row gets a row of 2026-09-30-da-self-synth whose rotate label
  `ai` is "ai" (the assistant itself is part of the situation);
- a slot the original filled with any other AI category gets a row of 2026-09-30-da-otherai-synth whose rotate
  label `ai` is "ai" (another AI system is involved);
always of the slot's own trait. The label is the generation-time one (stage 2 `ai` field), not a judge's.
Where a trait has fewer such donors than slots, a seeded random subset of that trait's slots is refilled and the
rest keep their 28 Sep rows (listed in `unfilled`). Donor conversations move whole (system, user, assistant with
its reasoning), exactly as the mixture builder would have stored them; base rows are the target's byte for byte.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import random
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

load_dotenv(".env")
_spec = importlib.util.spec_from_file_location("recompose_da_mix", Path(__file__).with_name("recompose_da_mix.py"))
_rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rc)

ORG = "dougalldeepmind"
TARGET = (f"{ORG}/2026-09-28-da-15-mix", "ff52482340790eea9bf681348ffec6622b92057b")
DONORS = {"self": (f"{ORG}/2026-09-30-da-self-synth", "d123c17a96e3875e0ac31ed303498a247abeeb27"),
          "other": (f"{ORG}/2026-09-30-da-otherai-synth", "d1c2fc02610a013c38a38c51b1bc12b1c8fa604f")}
load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]


def donor_pool(kind: str) -> dict[str, list[dict]]:
    """trait -> the corpus rows whose generation-time rotate label is `ai`, in corpus order."""
    repo, rev = DONORS[kind]
    label = {r["scenario_id"]: r.get("ai") for r in load(hf_hub_download(repo, "stages/stage_2_write_scenarios.jsonl", repo_type="dataset", revision=rev))}
    pool = collections.defaultdict(list)
    for r in load(hf_hub_download(repo, "dataset.jsonl", repo_type="dataset", revision=rev)):
        if label[r["metadata"]["scenario_id"]] == "ai":
            pool[r["metadata"]["trait_id"]].append(r)
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", required=True, choices=["self", "otherai", "self-otherai"])
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--out", default="output/audits/2026-10-01_recompose_same_slots")
    a = ap.parse_args()
    api = HfApi()
    like_repo = f"{ORG}/2026-09-29-da-15-{a.like}-mix"
    like_sha = api.dataset_info(like_repo).sha
    orig = load(hf_hub_download(like_repo, "recompose_swaps.jsonl", repo_type="dataset", revision=like_sha))
    target_sha, files, rows, target_corpus = _rc.mixture(api, *TARGET)
    assert target_sha == TARGET[1]
    n_da = sum(r.get("source") == "da" for r in rows)
    # The original swap's target rows must be DA rows of the untouched 28 Sep mix.
    assert all(rows[s["target_row"]]["source"] == "da" for s in orig) and len({s["target_row"] for s in orig}) == len(orig)

    rng = random.Random(a.seed)
    slots = collections.defaultdict(list)   # (kind, trait) -> original swap records
    for s in orig:
        slots[("self" if s["donor_category"] == "assistant_self" else "other", s["trait_id"])].append(s)
    pools = {k: donor_pool(k) for k in sorted({k for k, _ in slots})}
    swaps, unfilled = [], []
    for (kind, trait) in sorted(slots):
        want = sorted(slots[(kind, trait)], key=lambda s: s["target_row"])
        have = list(pools[kind][trait]); rng.shuffle(have)
        n = min(len(want), len(have))
        chosen = set(rng.sample(range(len(want)), n))
        for i, s in enumerate(want):
            if i not in chosen:
                unfilled.append({"target_row": s["target_row"], "trait_id": trait, "kind": kind}); continue
            d = have.pop()
            assert d["metadata"]["trait_id"] == trait
            rows[s["target_row"]] = {"messages": json.loads(json.dumps(d["messages"])), "source": "da"}
            swaps.append({"target_row": s["target_row"], "target_scenario": s["target_scenario"], "trait_id": trait, "kind": kind,
                          "donor_corpus": DONORS[kind][0], "donor_revision": DONORS[kind][1], "donor_scenario": d["metadata"]["scenario_id"],
                          "original_swap_donor_scenario": s["donor_scenario"], "original_swap_donor_category": s["donor_category"]})
        print(f">>> {kind:5} {trait}: {len(want)} original slots, {len(pools[kind][trait])} donors -> {n} refilled")
    by_kind = collections.Counter(s["kind"] for s in swaps)
    by_trait = {k: dict(sorted(collections.Counter(s["trait_id"] for s in swaps if s["kind"] == k).items())) for k in by_kind}
    assert sum(r.get("source") == "da" for r in rows) == n_da
    print(f">>> refilled {len(swaps)} of {len(orig)} original slots ({dict(by_kind)}); {len(unfilled)} keep their 28 Sep rows; "
          f"{100 * len(swaps) / n_da:.0f}% of the {n_da} da rows (original swap: {100 * len(orig) / n_da:.0f}%)")

    from src.infra.huggingface import hf_repo_id, push_files, training_data_tags
    from src.naming import mix_name, mix_subject_from, split_mix_subject
    from src.utils import git_sha, origin_url
    styles, pct, variant = split_mix_subject(mix_subject_from(TARGET[0]))
    assert not variant
    name = mix_name(styles, pct, a.variant)
    work = Path(a.out) / name; work.mkdir(parents=True, exist_ok=True)
    (work / "mixture.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (work / "recompose_swaps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in swaps), encoding="utf-8")
    (work / "recompose_unfilled.jsonl").write_text("".join(json.dumps(s) + "\n" for s in unfilled), encoding="utf-8")
    for f in ("mixture_config.yaml", "mixture_stats.json", "run_meta.json"):
        (work / f).write_bytes(Path(files[f]).read_bytes())

    old = _rc.card_fields_of(Path(files["README.md"]).read_text(encoding="utf-8"))
    gen = json.loads(old["generation_config"])
    gen["recompose"] = {"target_mixture": {"repo": TARGET[0], "revision": target_sha},
                        "same_slots_as": {"repo": like_repo, "revision": like_sha, "n_slots": len(orig)},
                        "donor_corpora": {k: {"repo": DONORS[k][0], "revision": DONORS[k][1], "filter": "stage_2 rotate label ai == 'ai'"} for k in by_kind},
                        "n_swapped": len(swaps), "n_unfilled": len(unfilled), "swaps_by_kind": dict(by_kind), "swaps_by_trait": by_trait,
                        "seed": a.seed, "rule": "each original swap slot is refilled with a regenerated row of the same trait and AI kind; "
                                                "slots without a same-trait donor keep their 2026-09-28 row; donor rows move whole"}
    fields = {
        "experiment": (f"row-swap replication with regenerated rows: {TARGET[0]} @ {target_sha[:8]} with {len(swaps)} of its {n_da} da rows "
                       f"replaced, trait for trait, by AI-labelled rows of the 2026-09-30 pipeline corpora ({dict(by_kind)}), in the same "
                       f"target rows that {like_repo} @ {like_sha[:8]} filled with 2026-09-25 rows ({len(orig)} slots; {len(unfilled)} had no "
                       f"same-trait donor and keep their 2026-09-28 row); base rows identical to the target; mixture_stats.json is the "
                       f"target's, unrecomputed; swaps in recompose_swaps.jsonl (scratch/recompose_da_mix_same_slots.py)"),
        "title": f"difficult-advice arm, {len(swaps)} rows swapped for regenerated {a.like} rows",
        "date_generated": name[:10].replace("-", ""),
        "constitution": old["constitution"],
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": old["models"],
        "generation_config": json.dumps(gen),
        "schema": old["schema"] + ". recompose_swaps.jsonl: per swap {target_row, target_scenario, trait_id, kind, donor_corpus, donor_revision, "
                                  "donor_scenario, original_swap_donor_scenario, original_swap_donor_category}; recompose_unfilled.jsonl: original slots left as they were",
        "provenance": f"uv run python scratch/recompose_da_mix_same_slots.py --like {a.like} --variant {a.variant} --seed {a.seed} --push",
    }
    front = {"configs": [{"config_name": "default", "data_files": "mixture.jsonl", "default": True}],
             "tags": training_data_tags("mixture", f"da-{a.variant}", old["constitution"], extra=["stage:final"])}
    print(f">>> {name}: {len(rows)} rows -> {work}")
    if a.push:
        url = push_files([work / f for f in ("mixture.jsonl", "recompose_swaps.jsonl", "recompose_unfilled.jsonl", "mixture_config.yaml",
                                             "mixture_stats.json", "run_meta.json")], hf_repo_id(name), fields, private=False, front_matter=front)
        print(">>> pushed", url)
    else:
        print(">>> not pushed (--push)")


if __name__ == "__main__":
    main()
