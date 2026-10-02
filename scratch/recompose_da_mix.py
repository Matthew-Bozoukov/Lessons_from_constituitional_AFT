# ABOUTME: Recompose a published DA mixture by swapping N of its DA rows, trait for trait, with rows of a chosen
# ABOUTME: AI-subtype drawn at random from another published mixture; push as `<date>-da-<pct>-<variant>-mix`.
"""    uv run python scratch/recompose_da_mix.py --variant self --categories assistant_self --n 123 [--push]
       uv run python scratch/recompose_da_mix.py --variant otherai --categories classical_ml,ai_unspecified,llm_tool,other_llm_agent --n 123 [--push]

Target rows: the DA rows of --target-mix (default 2026-09-28-da-15-mix @ ff524823). Donor rows: the DA rows of
--donor-mix (default: the mixture the 2026-09-25-qwen36-0-da-15 arm trained on) whose seven-way AI-subtype label
(output/audits/2026-09-29_ai_subtype/<donor corpus>.jsonl, matched by user message) is in --categories. N donors
are drawn at random (seeded); each replaces a random, not yet replaced target row OF THE SAME TRAIT. Base rows and
sidecars are the target's, byte for byte. The card records every swap.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

load_dotenv(".env")
SUBTYPE_DIR = Path("output/audits/2026-09-29_ai_subtype")
CORPUS_LOCAL = {  # the local copies the subtype judge labelled (scenario_id -> label), keyed by corpus repo name
    "2026-09-25-da-synth": "output/audits/2026-09-28_corpus_runs/2026-09-25-da-synth/dataset.jsonl",
    "2026-09-28-da-synth": "output/synthdoc_v3/20260928_195614/dataset.jsonl",
}


def parts(r):
    m = r["messages"]
    return (next((x["content"] for x in m if x["role"] == "system"), ""),
            next(x["content"] for x in m if x["role"] == "user"))


def card_fields_of(readme: str) -> dict:
    fields = {}
    for m in re.finditer(r"^\| `([a-z_]+)` \| (.*) \|$", readme, re.M):
        fields[m.group(1)] = m.group(2).strip()
    assert {"experiment", "constitution", "models", "generation_config", "schema"} <= fields.keys(), fields.keys()
    return fields


def mixture(api, repo, revision):
    sha = api.dataset_info(repo, revision=revision).sha
    files = {f: hf_hub_download(repo, f, repo_type="dataset", revision=sha)
             for f in ("mixture.jsonl", "README.md", "mixture_config.yaml", "mixture_stats.json", "run_meta.json")}
    rows = [json.loads(l) for l in open(files["mixture.jsonl"], encoding="utf-8")]
    src = json.load(open(files["run_meta.json"]))["config"]["sources"]["da"]
    corpus = src["dataset"].split("/")[-1]
    return sha, files, rows, corpus


def corpus_labels(corpus):
    """user message -> {scenario_id, trait_id, category, ai_conduct_at_stake} for one labelled corpus."""
    labs = {l["scenario_id"]: l for l in map(json.loads, open(SUBTYPE_DIR / f"{corpus[:10]}.jsonl", encoding="utf-8"))}
    out = {}
    for r in map(json.loads, open(CORPUS_LOCAL[corpus], encoding="utf-8")):
        sid = r["metadata"]["scenario_id"]
        if sid in labs:
            out[parts(r)[1]] = {"scenario_id": sid, "trait_id": r["metadata"]["trait_id"], **{k: labs[sid][k] for k in ("category", "ai_conduct_at_stake")}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--categories", help="comma-separated seven-way categories the donors must have")
    ap.add_argument("--n", type=int, default=123)
    ap.add_argument("--group", action="append", default=[],
                    help="CATS:N, repeatable, in place of --categories/--n: N donors from each category set, drawn in "
                         "the order given (put the scarcer set first so it gets first claim on each trait's slots)")
    ap.add_argument("--target-mix", default="dougalldeepmind/2026-09-28-da-15-mix")
    ap.add_argument("--target-revision", default="ff52482340790eea9bf681348ffec6622b92057b")
    ap.add_argument("--donor-arm", default="dougalldeepmind/2026-09-25-qwen36-0-da-15", help="the donor mix is what this arm trained on")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--out", default="output/audits/2026-09-29_recompose")
    a = ap.parse_args()
    groups = ([(set(g.rsplit(":", 1)[0].split(",")), int(g.rsplit(":", 1)[1])) for g in a.group] if a.group
              else [(set(a.categories.split(",")), a.n)])
    cats = set().union(*(c for c, _ in groups))
    api = HfApi()

    tm = json.load(open(hf_hub_download(a.donor_arm, "training_meta.json")))["dataset"]
    donor_sha, _, donor_rows, donor_corpus = mixture(api, tm["repo"], tm["revision"])
    target_sha, files, rows, target_corpus = mixture(api, a.target_mix, a.target_revision)
    dlab, tlab = corpus_labels(donor_corpus), corpus_labels(target_corpus)

    donors = [(i, dlab[parts(r)[1]]) for i, r in enumerate(donor_rows) if r.get("source") == "da"]
    targets = {i: tlab[parts(r)[1]] for i, r in enumerate(rows) if r.get("source") == "da"}   # KeyError = unlabelled row: stop
    print(f">>> target {a.target_mix} @ {target_sha[:8]} ({target_corpus}): {len(targets)} da rows")
    rng = random.Random(a.seed)
    free = collections.defaultdict(list)
    for i, l in targets.items():
        free[l["trait_id"]].append(i)
    for t in free:
        rng.shuffle(free[t])
    chosen = []
    for gcats, gn in groups:
        pool = [(i, l) for i, l in donors if l["category"] in gcats]
        print(f">>> donor {tm['repo']} @ {donor_sha[:8]} ({donor_corpus}): {len(pool)} of {len(donors)} da rows in {sorted(gcats)}; "
              f"pool by trait {dict(sorted(collections.Counter(l['trait_id'] for _, l in pool).items()))}")
        rng.shuffle(pool)
        claimed = collections.Counter(dl["trait_id"] for _, dl in chosen)
        got = []
        for di, dl in pool:   # a donor whose trait has no target slot left is skipped, not forced elsewhere
            if len(got) == gn:
                break
            if claimed[dl["trait_id"]] < len(free[dl["trait_id"]]):
                got.append((di, dl)); claimed[dl["trait_id"]] += 1
        assert len(got) == gn, f"only {len(got)} donors fit the target's trait slots for {sorted(gcats)} (asked {gn})"
        chosen += got
    swaps = []
    for di, dl in chosen:
        t = dl["trait_id"]
        ti = free[t].pop()
        swaps.append({"target_row": ti, "target_scenario": targets[ti]["scenario_id"], "donor_row": di,
                      "donor_scenario": dl["scenario_id"], "trait_id": t, "donor_category": dl["category"],
                      "donor_ai_conduct_at_stake": dl["ai_conduct_at_stake"]})
        # The donor conversation moves whole (system, user, assistant): a row is one coherent exchange.
        # Unreplaced rows keep their 09-28 system prompts.
        rows[ti] = json.loads(json.dumps(donor_rows[di]))
        assert rows[ti]["source"] == "da"
    by_trait = collections.Counter(s["trait_id"] for s in swaps)
    by_cat = collections.Counter(s["donor_category"] for s in swaps)
    print(f">>> swapped {len(swaps)} rows; by trait {dict(sorted(by_trait.items()))}; by category {dict(by_cat)}")

    from src.infra.huggingface import hf_repo_id, push_files, training_data_tags
    from src.naming import mix_name, mix_subject_from, split_mix_subject
    from src.utils import git_sha, origin_url
    styles, pct, variant = split_mix_subject(mix_subject_from(a.target_mix))
    assert not variant
    name = mix_name(styles, pct, a.variant)
    work = Path(a.out) / name; work.mkdir(parents=True, exist_ok=True)
    (work / "mixture.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (work / "recompose_swaps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in swaps), encoding="utf-8")
    for f in ("mixture_config.yaml", "mixture_stats.json", "run_meta.json"):
        (work / f).write_bytes(Path(files[f]).read_bytes())

    old = card_fields_of(Path(files["README.md"]).read_text(encoding="utf-8"))
    gen = json.loads(old["generation_config"])
    gen["recompose"] = {"target_mixture": {"repo": a.target_mix, "revision": target_sha},
                        "donor_mixture": {"repo": tm["repo"], "revision": donor_sha, "arm": a.donor_arm},
                        "donor_categories": sorted(cats), "groups": [{"categories": sorted(c), "n": n} for c, n in groups], "subtype_judge": "google/gemini-3-flash-preview (scratch/da_ai_subtype.py, 2026-09-29)",
                        "n_swapped": len(swaps), "swaps_by_trait": dict(sorted(by_trait.items())), "swaps_by_category": dict(by_cat),
                        "seed": a.seed, "rule": "each donor row (system, user, assistant) replaces a random not-yet-replaced target DA row of the same trait"}
    fields = {
        "experiment": (f"difficult-advice arm recomposed for the assistant-self test: {a.target_mix} @ {target_sha[:8]} with "
                       f"{len(swaps)} of its {len(targets)} da rows replaced, trait for trait, by rows of {tm['repo']} @ {donor_sha[:8]} "
                       f"whose AI-subtype label is in {sorted(cats)} ({dict(by_cat)}); donor rows move whole (system, user, assistant); "
                       f"unreplaced rows keep their 2026-09-28 system prompts; base rows identical to the target; "
                       f"swaps listed in recompose_swaps.jsonl (scratch/recompose_da_mix.py)"),
        "title": f"difficult-advice arm, {len(swaps)} rows swapped for {a.variant} rows",
        "date_generated": name[:10].replace("-", ""),
        "constitution": old["constitution"],
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": old["models"],
        "generation_config": json.dumps(gen),
        "schema": old["schema"] + ". recompose_swaps.jsonl: per swap {target_row, target_scenario, donor_row, donor_scenario, trait_id, donor_category, donor_ai_conduct_at_stake}",
        "provenance": (f"uv run python scratch/recompose_da_mix.py --variant {a.variant} "
                       + (" ".join(f"--group {g}" for g in a.group) if a.group else f"--categories {a.categories} --n {a.n}") + " "
                       f"--target-mix {a.target_mix} --target-revision {target_sha} --donor-arm {a.donor_arm} --seed {a.seed} --push"),
    }
    front = {"configs": [{"config_name": "default", "data_files": "mixture.jsonl", "default": True}],
             "tags": training_data_tags("mixture", f"da-{a.variant}", old["constitution"], extra=["stage:final"])}
    print(f">>> {name}: {len(rows)} rows -> {work}")
    if a.push:
        url = push_files([work / f for f in ("mixture.jsonl", "recompose_swaps.jsonl", "mixture_config.yaml", "mixture_stats.json", "run_meta.json")],
                         hf_repo_id(name), fields, private=False, front_matter=front)
        print(">>> pushed", url)
    else:
        print(">>> not pushed (--push)")


if __name__ == "__main__":
    main()
