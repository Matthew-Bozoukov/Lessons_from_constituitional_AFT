# ABOUTME: Stage the EXACT 716 baseline prompts the da716 arm trains on, as a local
# ABOUTME: source run the grok responder-swap arm consumes via `load_source_run`.

"""Prepare the input for the responder-swap arm.

Run: uv run python scratch/build_da716_prompt_source.py [--out data/da716_prompt_source]

The all-grok arm (`difficult_advice_grok_716.yaml`) regenerates its own scenarios and
prompts, which makes it differ from the baseline in the situations it is about, the
domains it covers, its trait balance and its user-turn length -- four confounds on top
of the one it exists to measure. The responder-swap arm removes all four by running
grok over the baseline's OWN prompts and swapping nothing but the model that answers.

This script stages those prompts:

  1. reproduces the exact 716-row selection the da716 training arm uses --
     `pick_balanced(rows, 716, Random(0))` from build_t2_9284_da716_mixture.py, on the
     same `stage_8_export_sft.jsonl`, with no RNG draw before it, so the ids match the
     trained arm row for row rather than merely matching its recipe;
  2. filters the baseline's published `stage_5_revise_prompts.jsonl` to those ids, so
     what is staged is the PROMPTS only -- no baseline responses leak into the arm;
  3. writes them plus a `manifest.json` carrying the baseline's `constitution_sha256`,
     which `op_load_source_run` asserts against the consuming config's constitution.
     That assert is the guard against silently crossing arms, so the sha is copied from
     the source manifest and never recomputed here.

Output layout is a run directory (`<out>/stage_5_revise_prompts.jsonl` + manifest.json),
which is what `source: {local_dir: ...}` expects.
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scratch.build_t2_9284_da716_mixture import pick_balanced  # noqa: E402

BASE_REPO = "LASR-Callum/2026-08-13-difficult-advice-v2"
SELECT_FILE = "stage_8_export_sft.jsonl"      # what the mixture selects over
PROMPT_FILE = "stage_5_revise_prompts.jsonl"  # what we actually stage
N = 716


def read(repo: str, fn: str) -> list[dict]:
    p = hf_hub_download(repo, fn, repo_type="dataset")
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


def main(out: str = "data/da716_prompt_source", seed: int = 0,
         repo: str = BASE_REPO, n: int = N) -> None:
    """Stage the da716 arm's exact prompts as a source run.

    Args:
        out: Output run directory.
        seed: Selection seed. 0 reproduces the trained arm; anything else does not.
        repo: Baseline synth run to take prompts from.
        n: How many prompts to stage.
    """
    load_dotenv()
    selectable = read(repo, SELECT_FILE)
    prompts = read(repo, PROMPT_FILE)
    print(f"{repo}: {len(selectable)} exported rows, {len(prompts)} stage-5 prompts")

    picked = pick_balanced(selectable, n, random.Random(seed))
    ids = [r["metadata"]["scenario_id"] for r in picked]
    assert len(set(ids)) == len(ids) == n, "selection is not n distinct scenarios"

    by_id = {r["scenario_id"]: r for r in prompts}
    missing = [i for i in ids if i not in by_id]
    assert not missing, (
        f"{len(missing)} selected scenarios have no stage-5 prompt "
        f"(first: {missing[:3]}). The export and the prompt snapshot disagree.")
    staged = [by_id[i] for i in ids]

    # The responses must NOT come along: this arm regenerates them. Fail loudly rather
    # than silently shipping baseline answers into a grok corpus.
    leaked = sorted({k for r in staged for k in r
                     if k in ("reasoning", "response", "draft_reasoning",
                              "draft_response")})
    assert not leaked, f"stage-5 snapshot carries response fields {leaked}"

    src_manifest = json.loads(Path(hf_hub_download(
        repo, "manifest.json", repo_type="dataset")).read_text())
    sha = src_manifest.get("constitution_sha256")
    assert sha, "source manifest has no constitution_sha256 to carry forward"

    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    with (d / PROMPT_FILE).open("w", encoding="utf-8") as fh:
        for r in staged:
            fh.write(json.dumps(r) + "\n")
    (d / "manifest.json").write_text(json.dumps({
        "constitution_sha256": sha,
        "git_sha": src_manifest.get("git_sha"),
        "staged_from": f"{repo}::{PROMPT_FILE}",
        "selected_by": (f"pick_balanced(read('{repo}::{SELECT_FILE}'), {n}, "
                        f"Random({seed})) -- the da716 training arm's own selection"),
        "n": len(staged),
    }, indent=2))

    traits = Counter(r["trait_id"] for r in staged)
    doms = {str(r.get("domain")).lower() for r in staged}
    ulen = sorted(len(r.get("user") or "") for r in staged)
    print(f"staged {len(staged)} prompts -> {d}/{PROMPT_FILE}")
    print(f"  traits: {dict(sorted(traits.items()))}")
    print(f"  distinct domains: {len(doms)}")
    print(f"  user-turn chars: med={ulen[len(ulen)//2]} "
          f"min={ulen[0]} max={ulen[-1]}")
    print(f"  constitution_sha256 carried: {sha[:16]}...")


if __name__ == "__main__":
    fire.Fire(main)
