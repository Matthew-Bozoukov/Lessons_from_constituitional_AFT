# ABOUTME: Third arm — verbose difficult-advice subsampled so its TRAINABLE TOKEN budget
# ABOUTME: matches the control's, instead of its row share. Same 9,284 table2 rows.

"""Hold difficult-advice TOKENS fixed instead of difficult-advice ROWS.

The 7.16%-rows arm answers "what if we make the traces longer". Holding rows fixed while the
traces grew nearly doubled difficult advice's share of the trainable tokens, so that arm
cannot separate "more deliberation" from "more difficult-advice signal".

This arm fixes the other variable. It keeps the same 9,284 table2 rows and takes a SUBSET of
the verbose difficult-advice rows whose assistant tokens sum to the control's difficult-
advice total. Same token budget, same share of the loss, spent on fewer and deeper traces
rather than more and shallower ones. Row count is what floats.

Two choices worth stating, because both are defensible the other way:

  * SUBSET DRAWN FROM THE EXPANDED ROWS ONLY. 79 of the 716 kept their original trace (50
    failed the fidelity judge, 29 were refused by the content filter). Spending budget on
    those would spend it on control-length rows, which is the opposite of what this arm is
    for.
  * STRATIFIED BY TRAIT, randomly within trait. A flat random draw would preserve the nine
    traits only in expectation; the corpus was built trait-balanced on purpose, and letting
    that balance wander would add a second difference between the arms.

Run: uv run python scratch/verbose_cot/build_token_matched_mixture.py [--push]
"""

from __future__ import annotations

import collections
import json
import random
import re
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.infra.huggingface import hf_download, push_files
from src.utils import git_sha, origin_url

load_dotenv()

CONTROL = "LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train"
CONTROL_FILE = "t2_9284_da716_10k.jsonl"
VERBOSE_LOCAL = Path("output/verbose_cot/t2_9284_da716_verbose_10k.jsonl")
RUN_DIR = Path("output/verbose_cot/20260825_042004")
OUT_REPO = "LASR-Callum/2026-08-25-table2-9284-difficult-advice-verbose-token-matched-train-mixture"
OUT_FILE = "t2_9284_da_verbose_tokenmatched.jsonl"
DA = "difficult_advice_v2"
SEED = 0


def _rows(path) -> list[dict]:
    return [json.loads(line) for line
            in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def choose(vda: dict, traits: dict, eligible: set, target: int) -> list[str]:
    """Trait-stratified random draw whose assistant tokens land closest to `target`.

    Round-robin across traits so the balance holds at EVERY prefix, not only at the end --
    which is what lets the accumulation stop the moment it is closest to the target without
    the stopping point itself skewing the traits.
    """
    rng = random.Random(SEED)
    by_trait = collections.defaultdict(list)
    for sid in sorted(eligible):
        by_trait[traits[sid]].append(sid)
    for pool in by_trait.values():
        rng.shuffle(pool)

    order, cursor = [], {t: 0 for t in by_trait}
    while len(order) < len(eligible):
        for t in sorted(by_trait):
            if cursor[t] < len(by_trait[t]):
                order.append(by_trait[t][cursor[t]])
                cursor[t] += 1

    total, picked = 0, []
    for sid in order:
        # Stop where the running total is CLOSEST to the target: overshooting by less than
        # the shortfall of stopping early is the better match, so this is not a ceiling.
        if abs(total + vda[sid] - target) > abs(total - target):
            break
        picked.append(sid)
        total += vda[sid]
    return picked


def main(push: bool = False) -> None:
    """Build (and optionally publish) the token-matched arm."""
    control = _rows(hf_download(CONTROL, CONTROL_FILE, repo_type="dataset"))
    verbose = _rows(VERBOSE_LOCAL)
    cache = json.loads(
        Path("output/verbose_cot/_tok_cache.json").read_text(encoding="utf-8"))
    vda, target, t2_tok = cache["verbose_da"], cache["target"], cache["t2"]

    status = {json.loads(line)["scenario_id"]: json.loads(line).get("expansion_status")
              for line in (RUN_DIR / "stage_2_expand.jsonl").read_text(encoding="utf-8")
              .splitlines() if line.strip()}
    traits = {r["scenario_id"]: r["trait_id"] for r in verbose if r.get("source") == DA}
    eligible = {s for s, st in status.items() if st == "expanded"}
    print(f"eligible (expanded only): {len(eligible)} of {len(traits)}")

    picked = set(choose(vda, traits, eligible, target))
    got = sum(vda[s] for s in picked)
    print(f"picked {len(picked)} rows, {got:,} assistant tokens vs target {target:,} "
          f"({got / target:.4f}x, {got - target:+,})")

    built = [r for r in verbose if r.get("source") != DA or r["scenario_id"] in picked]
    da_rows = [r for r in built if r.get("source") == DA]

    # --- checks that catch a build which only looks right -------------------------------
    ctl_t2 = [r for r in control if r.get("source") != DA]
    new_t2 = [r for r in built if r.get("source") != DA]
    assert len(ctl_t2) == len(new_t2) == 9284, "table2 row count changed"
    assert all(a["text"] == b["text"] for a, b in zip(ctl_t2, new_t2)), \
        "a table2 row differs from the control's"
    assert len(da_rows) == len(picked), "difficult-advice row count mismatch"
    assert all(status[r["scenario_id"]] == "expanded" for r in da_rows), \
        "an unexpanded row got into the subset"
    assert [r["scenario_id"] for r in da_rows] == [
        r["scenario_id"] for r in verbose
        if r.get("source") == DA and r["scenario_id"] in picked], "row order changed"

    per_trait = collections.Counter(r["trait_id"] for r in da_rows)
    print(f"VERIFIED: 9,284 table2 rows byte-identical to the control; "
          f"{len(da_rows)} difficult-advice rows, all expanded")
    print(f"\nrows      {len(built):,}  ({len(da_rows)} DA "
          f"= {len(da_rows) / len(built):.2%} of rows, was 7.16%)")
    print(f"DA tokens {got:,} = {got / (got + t2_tok):.2%} of trainable "
          f"(control {target:,} = {target / (target + t2_tok):.2%})")
    print(f"per trait {dict(sorted(per_trait.items()))}")

    out = Path("output/verbose_cot") / OUT_FILE
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in built) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    if not push:
        print("\n(dry run - pass --push to publish)")
        return

    url = push_files([out], OUT_REPO, private=False, fields={
        "experiment":
            "Token-matched verbose difficult-advice arm. Holds difficult advice's share of "
            "the TRAINABLE TOKENS at the control's value while the traces are ~3x longer, "
            "by keeping only a subset of the expanded rows. Its sibling arm holds the ROW "
            "share instead; together they separate more deliberation from more "
            "difficult-advice signal.",
        "date_generated": "2026-08-25",
        "constitution":
            "constitutions/claude_distilled_12_principles_mid/constitution.md (inherited "
            "from the source run; never rendered into any prompt of the expansion itself)",
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models":
            "expansion anthropic/claude-sonnet-5 (temp 0.7); fidelity and coverage judges "
            "openai/gpt-5.6-terra (temp 0.0); pinned to first-party endpoints via "
            "configs/endpoints/providers.yaml",
        "generation_config":
            f"scratch/verbose_cot/build_token_matched_mixture.py, seed {SEED}. Subset drawn "
            "trait-stratified and randomly within trait, from the EXPANDED rows only, "
            "accumulating in a trait round-robin until the assistant-token total is closest "
            "to the control's.",
        "schema":
            f"{OUT_FILE}: text (rendered Qwen chat), source, scenario_id, trait_id - "
            f"identical schema to {CONTROL}.",
        "provenance":
            "uv run python scratch/verbose_cot/build_token_matched_mixture.py --push",
        "composition":
            f"{len(built):,} rows = {len(da_rows)} difficult-advice + 9,284 table2. "
            f"Difficult-advice trainable tokens {got:,} against the control's {target:,} "
            f"({got / target:.4f}x), i.e. {got / (got + t2_tok):.2%} of trainable tokens "
            f"against the control's {target / (target + t2_tok):.2%}. Row share falls to "
            f"{len(da_rows) / len(built):.2%} from 7.16% - that is the variable this arm "
            "lets float, deliberately. Per trait: "
            + ", ".join(f"{k} {v}" for k, v in sorted(per_trait.items())) + ".",
        "control_arm":
            f"{CONTROL} - same table2 rows, same difficult-advice token budget, original "
            "short traces.",
        "sibling_arm":
            "LASR-Callum/2026-08-25-table2-9284-difficult-advice-verbose-716-train - all "
            "716 verbose rows, row share held at 7.16%, token share allowed to rise.",
    })
    print(f"pushed: {url}")


if __name__ == "__main__":
    sys.exit(fire.Fire(main))
