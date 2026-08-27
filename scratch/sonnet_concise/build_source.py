# ABOUTME: Stage the da716 arm's exact 716 prompts WITH their Haiku drafts (the baseline's
# ABOUTME: stage 6) as a local source run, so a rewrite-only arm pays for the rewrite alone.

"""Prepare the input for the length-capped Sonnet rewrite arm.

Run: uv run python scratch/sonnet_concise/build_source.py [--out data/da716_draft_source]
       [--local_run_dir /path/to/output/synthdoc_v2/20260814_112121]

The responder-swap arm (scratch/build_da716_prompt_source.py) staged the baseline's
stage-5 PROMPTS and regenerated both response stages with grok. This arm goes one stage
further: it stages the baseline's stage-6 rows -- the same prompts plus the Haiku DRAFTS
Sonnet rewrote in the da716 corpus -- and regenerates only `revise_responses`. The arm
therefore shares everything with da716 up to and including the draft, and differs from
it in one sentence of the rewrite prompt.

  1. reproduces the exact 716-row selection the da716 training arm uses, by the same
     call the responder-swap arm uses (`pick_balanced(rows, 716, Random(0))` over the
     published `stage_8_export_sft.jsonl`, no RNG draw before it);
  2. filters the published `stage_6_draft_responses.jsonl` to those ids, in selection
     order, and asserts every row carries a draft and NO final response (the final is
     what this arm regenerates -- a leaked one would silently reproduce da716);
  3. writes them plus a `manifest.json` carrying the baseline's `constitution_sha256`,
     which `op_load_source_run` asserts against the consuming config's constitution;
  4. writes a `<out>_smoke/` sibling holding the first `smoke_per_trait` rows of every
     trait, for a paid smoke that covers all nine principles (`--smoke` does not truncate
     a loaded source, so the smoke run is pointed at this directory instead).

`--local_run_dir` cross-checks the staged rows against a local copy of the same run
(the resumable cache `synth run` left behind) and reports whether they are identical;
the published copy is what is staged either way, because it is the provenance the card
names.
"""

import json
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scratch.build_da716_prompt_source import BASE_REPO, SELECT_FILE, read  # noqa: E402
from scratch.build_t2_9284_da716_mixture import pick_balanced  # noqa: E402

DRAFT_FILE = "stage_6_draft_responses.jsonl"
N = 716
FINAL_FIELDS = ("reasoning", "response", "rewrite_changes")
DRAFT_FIELDS = ("draft_reasoning", "draft_response")


def _read_stage(repo: str, fn: str) -> list[dict]:
    """A stage snapshot from the root (pre-layout repos) or under stages/ (new layout)."""
    try:
        return read(repo, fn)
    except EntryNotFoundError:
        return read(repo, f"stages/{fn}")


def _write_run(d: Path, rows: list[dict], manifest: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    with (d / DRAFT_FILE).open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (d / "manifest.json").write_text(json.dumps({**manifest, "n": len(rows)}, indent=2))


def _describe(label: str, rows: list[dict]) -> None:
    traits = Counter(r["trait_id"] for r in rows)
    doms = {str(r.get("domain")).lower() for r in rows}
    wr = st.median(len(r["draft_reasoning"].split()) for r in rows)
    wa = st.median(len(r["draft_response"].split()) for r in rows)
    print(f"{label}: {len(rows)} rows")
    print(f"  traits: {dict(sorted(traits.items()))}")
    print(f"  distinct domains: {len(doms)}")
    print(f"  draft words, median: reasoning {wr:.0f}  response {wa:.0f}")


def main(
    out: str = "data/da716_draft_source",
    seed: int = 0,
    repo: str = BASE_REPO,
    n: int = N,
    smoke_per_trait: int = 3,
    local_run_dir: str = "",
) -> None:
    """Stage the da716 arm's exact prompts + Haiku drafts as a source run.

    Args:
        out: Output run directory. A `<out>_smoke` sibling is written beside it.
        seed: Selection seed. 0 reproduces the trained arm; anything else does not.
        repo: Baseline synth run to take the drafts from.
        n: How many rows to stage.
        smoke_per_trait: Rows per trait in the smoke directory (9 traits -> 27 by default).
        local_run_dir: Optional local copy of the same run, to confirm the published
            drafts and the cached ones are identical.
    """
    load_dotenv()
    selectable = read(repo, SELECT_FILE)
    drafts = _read_stage(repo, DRAFT_FILE)
    print(f"{repo}: {len(selectable)} exported rows, {len(drafts)} stage-6 drafts")

    picked = pick_balanced(selectable, n, random.Random(seed))
    ids = [r["metadata"]["scenario_id"] for r in picked]
    assert len(set(ids)) == len(ids) == n, "selection is not n distinct scenarios"

    by_id = {r["scenario_id"]: r for r in drafts}
    missing = [i for i in ids if i not in by_id]
    assert not missing, (
        f"{len(missing)} selected scenarios have no stage-6 draft "
        f"(first: {missing[:3]}). The export and the draft snapshot disagree."
    )
    staged = [by_id[i] for i in ids]

    # Every row must carry the draft this arm rewrites, and none may carry the final
    # that it regenerates: a leaked `reasoning`/`response` would make the rewrite stage
    # a no-op copy of da716 under a different name.
    for r in staged:
        for f in DRAFT_FIELDS:
            assert r.get(f), f"{r['scenario_id']} has no {f}"
    leaked = sorted({k for r in staged for k in r if k in FINAL_FIELDS})
    assert not leaked, f"stage-6 snapshot carries final-response fields {leaked}"

    if local_run_dir:
        local = {
            r["scenario_id"]: r
            for r in (
                json.loads(line)
                for line in (Path(local_run_dir) / DRAFT_FILE).open(encoding="utf-8")
                if line.strip()
            )
        }
        same = sum(local.get(r["scenario_id"]) == r for r in staged)
        print(
            f"local cache {local_run_dir}: {same}/{len(staged)} staged rows identical"
        )
        assert same == len(staged), "published drafts differ from the local cache"

    src_manifest = json.loads(
        Path(hf_hub_download(repo, "manifest.json", repo_type="dataset")).read_text()
    )
    sha = src_manifest.get("constitution_sha256")
    assert sha, "source manifest has no constitution_sha256 to carry forward"
    manifest = {
        "constitution_sha256": sha,
        "git_sha": src_manifest.get("git_sha"),
        "staged_from": f"{repo}::{DRAFT_FILE}",
        "selected_by": (
            f"pick_balanced(read('{repo}::{SELECT_FILE}'), {n}, "
            f"Random({seed})) -- the da716 training arm's own selection"
        ),
    }

    d = Path(out)
    _write_run(d, staged, manifest)
    _describe(f"staged -> {d}/{DRAFT_FILE}", staged)

    per_trait: dict[str, list[dict]] = defaultdict(list)
    for r in staged:
        if len(per_trait[r["trait_id"]]) < smoke_per_trait:
            per_trait[r["trait_id"]].append(r)
    smoke = [r for rows in per_trait.values() for r in rows]
    ds = Path(f"{out}_smoke")
    _write_run(ds, smoke, {**manifest, "smoke_of": str(d)})
    _describe(f"smoke  -> {ds}/{DRAFT_FILE}", smoke)
    print(f"  constitution_sha256 carried: {sha[:16]}...")


if __name__ == "__main__":
    fire.Fire(main)
