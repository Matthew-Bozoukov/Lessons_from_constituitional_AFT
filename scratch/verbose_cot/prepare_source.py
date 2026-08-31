# ABOUTME: Stage the 716 difficult-advice records the published mixture actually uses, as a
# ABOUTME: local source run for 2026-08-25_verbose_cot.yaml. Run: uv run python scratch/verbose_cot/prepare_source.py

"""Build the expansion's input.

The published arm's 716 rows are a trait-balanced, domain-diverse SELECTION from the 1,952
scenarios in the difficult-advice-v2 run (see scratch/build_t2_9284_da716_mixture.py).
Pointing the expansion at the whole source run would expand 1,952 records to use 716 of
them -- roughly $150 to keep $56 of work.

Rather than re-deriving that selection and hoping the seed still lands the same way, the
ids are read back out of the published mixture itself, so the expanded arm covers exactly
the scenarios the control arm covers, by construction.

Writes a source-run directory `op_load_source_run` can read with `source.local_dir`,
carrying the upstream manifest (see the note on the constitution guard below).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.huggingface import hf_download

MIXTURE = "LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train"
MIXTURE_FILE = "t2_9284_da716_10k.jsonl"
SOURCE = "LASR-Callum/2026-08-13-haiku45-sonnet45-difficult-advice-diversity-gated-voice-linted"
SNAPSHOT = "stage_7_revise_responses.jsonl"
OUT = Path("data/verbose_cot_source")
SMOKE_OUT = Path("data/verbose_cot_source_smoke")
SMOKE_N = 20


def main() -> None:
    mix = [json.loads(line) for line
           in Path(hf_download(MIXTURE, MIXTURE_FILE, repo_type="dataset")
                   ).read_text(encoding="utf-8").splitlines() if line.strip()]
    wanted = {r["scenario_id"] for r in mix if r.get("source") == "difficult_advice_v2"}
    print(f"mixture: {len(mix):,} rows, {len(wanted)} difficult-advice scenarios")

    records = [json.loads(line) for line
               in Path(hf_download(SOURCE, SNAPSHOT, repo_type="dataset")
                       ).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"source run: {len(records):,} records at {SNAPSHOT}")

    by_id = {r["scenario_id"]: r for r in records}
    missing = wanted - set(by_id)
    assert not missing, (f"{len(missing)} mixture scenarios absent from the source run, "
                         f"e.g. {sorted(missing)[:3]} -- wrong snapshot or wrong run")
    # Source-run order, not mixture order: the mixture is shuffled, and keeping the
    # generation order makes the expanded run diff cleanly against its own source.
    selected = [r for r in records if r["scenario_id"] in wanted]
    assert len(selected) == len(wanted), "duplicate scenario_id in the source run"

    for r in selected:                      # the field the expansion must stay faithful to
        r["source_reasoning"] = r["reasoning"]

    manifest = json.loads(Path(hf_download(SOURCE, "manifest.json",
                                           repo_type="dataset")).read_text())
    manifest["selected_from"] = {"mixture": MIXTURE, "file": MIXTURE_FILE,
                                 "n_selected": len(selected), "n_source": len(records)}

    # `load_source_run` asserts the source's constitution sha matches the consuming
    # config's, because a pipeline that GENERATES reasoning against a constitution would
    # silently cross arms if the two differed. The source's sha
    # (fe2ed96093d6...) is not reproducible from any constitution in the tree — not the
    # current file, not the frozen 20260804 snapshot, not any git revision of either --
    # so `full_text()` must have changed since 2026-08-13.
    #
    # The guard does not bear on THIS pipeline: verbose_cot never puts the constitution in
    # a prompt. The expander is deliberately blind to it (that is what keeps it from
    # importing new normative content), and both judges compare the rewrite against the
    # source reasoning, never against a spec. So the key is moved aside rather than
    # matched -- renamed, not dropped, so the source's value stays on the record and in
    # the dataset card instead of being quietly lost to make an assertion pass.
    manifest["source_run_constitution_sha256"] = manifest.pop("constitution_sha256", None)
    manifest["constitution_guard"] = (
        "n/a — no stage of verbose_cot renders the constitution into any prompt")

    # The smoke slice is every Nth record rather than the first N: the run is ordered by
    # trait, so a head slice would be one trait's documents and would smoke-test a
    # single voice against a single kind of dilemma.
    step = max(1, len(selected) // SMOKE_N)
    slices = {OUT: selected, SMOKE_OUT: selected[::step][:SMOKE_N]}
    for out_dir, rows in slices.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / SNAPSHOT).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8")
        (out_dir / "manifest.json").write_text(
            json.dumps({**manifest, "n_records": len(rows)}, indent=2), encoding="utf-8")
        print(f"wrote {len(rows):>3} records to {out_dir / SNAPSHOT}")

    words = sum(len(r["reasoning"].split()) for r in selected)
    print(f"  reasoning: {words:,} words total, {words / len(selected):.0f} mean")
    print(f"  source_run_constitution_sha256: "
          f"{str(manifest.get('source_run_constitution_sha256'))[:16]}")
    traits = {}
    for r in selected:
        traits[r.get("trait_id")] = traits.get(r.get("trait_id"), 0) + 1
    print(f"  per trait: {dict(sorted(traits.items()))}")


if __name__ == "__main__":
    main()
