# ABOUTME: Build the explicit-ask vs advice-request pair of DA mixtures: the same slots of the 28 Sep da-15 mix are filled
# ABOUTME: with 8 Sep rows matched on trait and AI type, explicit asks in one arm and advice requests in the other.
"""    uv run python scratch/build_ask_arms.py [--push]

Rows come from ONE corpus (2026-09-08-da-synth), so recipe and date are fixed across the arms. Per (AI type, trait)
cell -- AI type from scratch/da_assistant_subject.py (assistant itself / other AI / no AI), ask type from the
pressure judge (explicit = push|override, advice = none|lean) -- n = min(explicit, advice) rows are drawn from each
side. Both arms replace the SAME target slots, slot for slot with the same trait and AI type, so the arms differ only
in whether the 8 Sep row asks the assistant to act or asks it for advice. Base rows and the other DA rows are the
target's, byte for byte.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from recompose_da_mix import card_fields_of, mixture, parts  # noqa: E402  (scratch-to-scratch import)

load_dotenv(".env")
DONOR = ("dougalldeepmind/2026-09-08-da-synth", "42107bde00cd7f4360a3a6c581aac23a540dbfea")
TARGET = ("dougalldeepmind/2026-09-28-da-15-mix", "ff52482340790eea9bf681348ffec6622b92057b")
SUBJECT = Path("output/audits/2026-09-30_assistant_subject/2026-09-08.jsonl")
PRESSURE = Path("output/da_pressure/2026-09-29_2026-09-08-da-synth_pressure_labels.jsonl")
TYPES = ("assistant itself", "other AI", "no AI")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--out", default="output/audits/2026-09-30_ask_arms")
    a = ap.parse_args()
    api = HfApi()

    donor_rows = {json.loads(l)["metadata"]["scenario_id"]: json.loads(l) for l in
                  open(hf_hub_download(*DONOR[:1], "dataset.jsonl", repo_type="dataset", revision=DONOR[1]), encoding="utf-8")}
    subj = {l["scenario_id"]: l for l in map(json.loads, open(SUBJECT)) if "unlabelled" not in l}
    push = {l["scenario_id"]: l["push"] for l in map(json.loads, open(PRESSURE)) if "push" in l}
    for sid, l in subj.items():   # the labels were made on a local copy: prove it is this revision's rows
        assert sid in donor_rows and donor_rows[sid]["metadata"]["trait_id"] == l["trait_id"], sid
    ai_type = lambda l: "assistant itself" if l["assistant_subject"] else ("other AI" if l["other_ai"] else "no AI")
    ask = lambda p: "explicit" if p in ("push", "override") else ("advice" if p in ("none", "lean") else None)
    cells = collections.defaultdict(list)
    for sid, l in subj.items():
        k = ask(push.get(sid))
        if k:
            cells[(ai_type(l), l["trait_id"], k)].append(sid)

    target_sha, files, rows, target_corpus = mixture(api, *TARGET)
    from recompose_da_mix import corpus_labels
    tlab = corpus_labels(target_corpus)
    slots_by_trait = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("source") == "da":
            slots_by_trait[tlab[parts(r)[1]]["trait_id"]].append(i)

    rng = random.Random(a.seed)
    plan = []   # (slot, trait, ai type, explicit sid, advice sid)
    for t in sorted(slots_by_trait):
        free = sorted(slots_by_trait[t]); rng.shuffle(free)
        for ty in TYPES:
            ex, ad = sorted(cells[(ty, t, "explicit")]), sorted(cells[(ty, t, "advice")])
            n = min(len(ex), len(ad))
            for e_sid, a_sid in zip(rng.sample(ex, n), rng.sample(ad, n)):
                plan.append((free.pop(), t, ty, e_sid, a_sid))
    print(f">>> {len(plan)} matched slots; by type {dict(collections.Counter(p[2] for p in plan))}; "
          f"by trait {dict(sorted(collections.Counter(p[1] for p in plan).items()))}")

    from src.infra.huggingface import hf_repo_id, push_files, training_data_tags
    from src.naming import mix_name, mix_subject_from, split_mix_subject
    from src.utils import git_sha, origin_url
    styles, pct, _ = split_mix_subject(mix_subject_from(TARGET[0]))
    old = card_fields_of(Path(files["README.md"]).read_text(encoding="utf-8"))
    for arm, col in (("explicit", 3), ("advice", 4)):
        out_rows = [json.loads(json.dumps(r)) for r in rows]
        swaps = []
        for p in plan:
            sid = p[col]
            out_rows[p[0]] = {"messages": json.loads(json.dumps(donor_rows[sid]["messages"])), "source": "da"}
            swaps.append({"target_row": p[0], "trait_id": p[1], "ai_type": p[2], "donor_scenario": sid,
                          "donor_push": push[sid], "pair_scenario": p[4 if col == 3 else 3]})
        name = mix_name(styles, pct, arm)
        work = Path(a.out) / name; work.mkdir(parents=True, exist_ok=True)
        (work / "mixture.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
        (work / "recompose_swaps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in swaps), encoding="utf-8")
        for f in ("mixture_config.yaml", "mixture_stats.json", "run_meta.json"):
            (work / f).write_bytes(Path(files[f]).read_bytes())
        gen = json.loads(old["generation_config"])
        gen["recompose"] = {"target_mixture": {"repo": TARGET[0], "revision": target_sha},
                            "donor_corpus": {"repo": DONOR[0], "revision": DONOR[1]}, "arm": arm, "n_swapped": len(swaps),
                            "swaps_by_type": dict(collections.Counter(s["ai_type"] for s in swaps)),
                            "swaps_by_trait": dict(sorted(collections.Counter(s["trait_id"] for s in swaps).items())),
                            "labels": {"ai_type": "scratch/da_assistant_subject.py (gemini-3-flash, 2026-09-30)",
                                       "ask_type": "scratch/classify_da_pressure.py push: explicit=push|override, advice=none|lean"},
                            "seed": a.seed}
        fields = {
            "experiment": (f"difficult-advice arm for the explicit-ask vs advice-request test ({arm}): {TARGET[0]} @ "
                           f"{target_sha[:8]} with {len(swaps)} da rows replaced by {DONOR[0]} @ {DONOR[1][:8]} rows whose "
                           f"user {'asks the assistant to do or write the thing' if arm == 'explicit' else 'asks for advice'}; "
                           f"the paired arm fills the same slots with rows of the same trait and AI type; donor rows move whole; "
                           f"swaps in recompose_swaps.jsonl (scratch/build_ask_arms.py)"),
            "title": f"difficult-advice arm, {len(swaps)} rows swapped for 8 Sep {arm} rows",
            "date_generated": name[:10].replace("-", ""),
            "constitution": old["constitution"],
            "source_repo": f"{origin_url()} @ {git_sha()}",
            "models": old["models"],
            "generation_config": json.dumps(gen),
            "schema": old["schema"] + ". recompose_swaps.jsonl: per swap {target_row, trait_id, ai_type, donor_scenario, donor_push, pair_scenario}",
            "provenance": f"uv run python scratch/build_ask_arms.py --seed {a.seed} --push",
        }
        front = {"configs": [{"config_name": "default", "data_files": "mixture.jsonl", "default": True}],
                 "tags": training_data_tags("mixture", f"da-{arm}", old["constitution"], extra=["stage:final"])}
        print(f">>> {name}: {len(out_rows)} rows -> {work}")
        if a.push:
            print(">>> pushed", push_files([work / f for f in ("mixture.jsonl", "recompose_swaps.jsonl", "mixture_config.yaml",
                                                               "mixture_stats.json", "run_meta.json")],
                                          hf_repo_id(name), fields, private=False, front_matter=front))


if __name__ == "__main__":
    main()
