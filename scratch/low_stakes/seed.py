# ABOUTME: Build the low-stakes arm's seed run: the exact 716 difficult-advice-v2 rows the
# ABOUTME: comparator was trained on, each stamped with the setting it will be rewritten into.

"""Stage the 716 source rows as a run directory the synth engine can load.

Run: uv run python scratch/low_stakes/seed.py [--out data/low_stakes_source] [--n N]

`load_source_run` reads a completed run from `hf_repo` or `local_dir`, and the engine has
no trait-balanced sampler -- so the selection happens here, once, and the engine consumes
the result. Two things are baked in rather than left to the config:

  * WHICH 716. The scenario_ids the published training mixture actually carries, read back
    out of it rather than re-derived (see the note on MIXTURE_REPO below).
  * WHICH SETTING each row is rewritten into. Dealt round-robin over the trait-grouped
    order, so every principle lands in every setting 4-5 times with no bias (80 rows per
    trait against 18 settings does not divide, so the deal rotates between traits). Doing
    it here makes the assignment part of the seed's provenance instead of a side effect of
    whatever order the engine happened to iterate in, and it puts the whole `[static
    instruction + principle]` prefix in front of the per-row text, which is what lets the
    rewrite stage cache it across a trait's ~80 consecutive calls.

The manifest carries the source run's `constitution_sha256` so `load_source_run`'s
cross-arm assertion still means something: it will refuse to run this seed against a
config pointing at a different constitution.
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scratch.low_stakes.prompts import LOW_STAKES_SETTINGS  # noqa: E402

SOURCE_REPO = "LASR-Callum/2026-08-13-difficult-advice-v2"
# Stage 7, not stage 8: stage 7 is the engine's flat record (system, user, situation,
# shortcut, trait_*, scenario_id), which is what the rewrite prompt's placeholders read.
# Stage 8 is the chat-format export and has none of those at the top level.
SNAPSHOT = "stage_7_revise_responses.jsonl"

# WHICH 716: read back out of the published mixture, not re-derived.
#
# `pick_balanced(seed=0)` does reproduce this selection -- verified 2026-08-26, 716/716 of
# its picks appear verbatim in the mixture and 0/716 non-picks do. But that is a property
# of the current selector, not a guarantee, and a future change to it would silently
# repoint this arm at a different 716 while every check here still passed. The mixture
# carries `scenario_id` on its difficult-advice rows, so the ids are read directly and the
# arm covers exactly the scenarios the comparator covers by construction.
#
# Same approach as scratch/verbose_cot/prepare_source.py, which reached it first.
MIXTURE_REPO = "LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train"
MIXTURE_FILE = "t2_9284_da716_10k.jsonl"
MIXTURE_SOURCE_TAG = "difficult_advice_v2"


def _load(repo: str, name: str, tok: str | None) -> list[dict]:
    return [json.loads(line) for line in open(
        hf_hub_download(repo, name, repo_type="dataset", token=tok),
        encoding="utf-8") if line.strip()]


def main(out: str = "data/low_stakes_source", n: int | None = None,
         seed: int = 0) -> None:
    tok = os.environ.get("HF_TOKEN")
    rows = _load(SOURCE_REPO, SNAPSHOT, tok)
    mix = _load(MIXTURE_REPO, MIXTURE_FILE, tok)
    src_manifest = json.loads(Path(hf_hub_download(
        SOURCE_REPO, "manifest.json", repo_type="dataset",
        token=tok)).read_text(encoding="utf-8"))

    by_id = {r["scenario_id"]: r for r in rows}
    wanted = [r["scenario_id"] for r in mix if r.get("source") == MIXTURE_SOURCE_TAG]
    missing = [s for s in wanted if s not in by_id]
    assert not missing, (
        f"{len(missing)} mixture scenario_ids are absent from {SNAPSHOT} "
        f"(first: {missing[:3]}) -- mixture and source run have drifted apart")

    # Trait-grouped, so the rewrite stage's [instruction + principle] cache prefix survives
    # a whole trait's consecutive calls, and so a smoke slice spans traits rather than
    # taking whichever ids the mixture happened to list first.
    wanted.sort(key=lambda s: (by_id[s]["trait_id"], s))
    if n is not None:
        keep, per = [], max(1, n // len({by_id[s]["trait_id"] for s in wanted}))
        seen: dict[str, int] = {}
        for s in wanted:
            t = by_id[s]["trait_id"]
            if seen.get(t, 0) < per and len(keep) < n:
                keep.append(s)
                seen[t] = seen.get(t, 0) + 1
        wanted = keep
    picks = [by_id[s] for s in wanted]
    for i, r in enumerate(picks):
        r["setting"] = LOW_STAKES_SETTINGS[i % len(LOW_STAKES_SETTINGS)]
        r["setting_id"] = i % len(LOW_STAKES_SETTINGS)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = Path(out) / ts
    d.mkdir(parents=True, exist_ok=True)
    (d / SNAPSHOT).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in picks) + "\n",
        encoding="utf-8")
    (d / "manifest.json").write_text(json.dumps({
        "seeded_from": f"{SOURCE_REPO}::{SNAPSHOT}",
        "selector": (f"scenario_ids read from {MIXTURE_REPO}::{MIXTURE_FILE}"
                     + (f", first {n} trait-spread" if n is not None else "")),
        "n": len(picks),
        "constitution_sha256": src_manifest.get("constitution_sha256"),
        "git_sha": src_manifest.get("git_sha"),
        "settings": LOW_STAKES_SETTINGS,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    traits = Counter(r["trait_id"] for r in picks)
    spread = Counter((r["trait_id"], r["setting_id"]) for r in picks)
    print(f"{len(rows)} source rows -> {len(picks)} seeded")
    print(f"traits: {dict(sorted(traits.items()))}")
    print(f"trait x setting cells filled: {len(spread)} of "
          f"{len(traits) * len(LOW_STAKES_SETTINGS)}   "
          f"per-cell min {min(spread.values())} max {max(spread.values())}")
    print(f"constitution_sha256 carried: {src_manifest.get('constitution_sha256', '')[:20]}")
    print(f"\nwrote {d}")
    print(f"point the config at:  source: {{local_dir: \"{d.as_posix()}\", "
          f"snapshot: \"{SNAPSHOT}\"}}")


if __name__ == "__main__":
    fire.Fire(main)
