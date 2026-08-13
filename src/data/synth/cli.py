# ABOUTME: Fire entrypoint for the synth pipeline (scripts/data/synth/build_dataset.py is the
# ABOUTME: canonical runner; this console script adds topup/check/estimate/segment).

from __future__ import annotations

import json
from pathlib import Path

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf

from . import pipeline
from .constitution import full_text
from .core import Checkpoint, Ctx, Usage


def _load(config: str) -> dict:
    """Load a run YAML into a plain dict."""
    return OmegaConf.to_container(OmegaConf.load(config), resolve=True)


def _csv(arg) -> list[str]:
    """Split a Fire argument into a list, however Fire chose to hand it over.

    Fire passes `a,b` as a tuple but `a` as a string, so every comma-separated flag
    needs both cases; one place to get that right rather than four.
    """
    items = arg if isinstance(arg, (list, tuple)) else str(arg).split(",")
    return [str(x).strip() for x in items if str(x).strip()]


def run(config: str, smoke: bool = False, resume: str | None = None,
        ablate: str | None = None, overrides: str | None = None) -> None:
    """Run the pipeline the config declares (its `stages:` list).

    Args:
        config: Path to the run YAML (configs/data/synth/<document_type>.yaml).
        smoke: Merge the config's `smoke:` overrides -- tiny slice, full wiring.
        resume: Existing run directory to continue instead of starting fresh.
        overrides: Comma-separated OmegaConf dotlist applied over the YAML, e.g.
            "total_scenarios=144,id_prefix=b" -- keeps a one-off variant (a top-up, a
            different size) as one reproducible command rather than a forked config.
            Recorded in the run manifest either way.
        ablate: Comma-separated stage names to ablate, merged over the config's
            `ablate:` list and recorded in the manifest.
    """
    load_dotenv()
    loaded = OmegaConf.load(config)
    if overrides:
        loaded = OmegaConf.merge(loaded, OmegaConf.from_dotlist(_csv(overrides)))
        print(f">>> overrides: {overrides}")
    cfg = OmegaConf.to_container(loaded, resolve=True)
    if ablate:
        cfg["ablate"] = sorted(set(cfg.get("ablate") or []) | set(_csv(ablate)))
    pipeline.exit_if_gate_failed(pipeline.run(cfg, smoke=smoke, resume=resume))


def topup(config: str, resume: str, traits, n: int = 25) -> None:
    """Re-run the `final` stage for specific traits until each has `n` completed records.

    A run stopped early covers only the traits its scenarios happened to reach, since
    scenarios are ordered by trait. Reuses the final stage's checkpoint so nothing
    already paid for is repeated. Assumes the difficult-advice stage layout
    (`draft_responses` feeding `final`).

    Args:
        config: Path to the run YAML.
        resume: Run directory holding the stage snapshots.
        traits: Trait ids -- Fire hands this over as a tuple when it contains commas,
            so both "t5,t6" and a tuple are accepted.
        n: Target completed records per trait.
    """
    load_dotenv()
    cfg = _load(config)
    stage_list = pipeline.build_stages(cfg)
    names = [s.name for s in stage_list]
    assert "draft_responses" in names and "final" in names, \
        "topup assumes the difficult-advice stage layout (draft_responses -> final)"
    run_dir = Path(resume)
    assert run_dir.exists(), f"run dir does not exist: {run_dir}"

    src_idx = names.index("draft_responses") + 1
    fin_idx = names.index("final") + 1
    drafted = [json.loads(line) for line in
               (run_dir / f"stage_{src_idx}_draft_responses.jsonl").open()]
    ckpt = Checkpoint(run_dir / f"stage_{fin_idx}_final.partial.jsonl")

    have: dict[str, int] = {}
    for r in ckpt.done.values():
        have[r["trait_id"]] = have.get(r["trait_id"], 0) + 1
    ids = _csv(traits)
    print(">>> current per-trait counts:", {t: have.get(t, 0) for t in ids})

    todo = []
    for t in ids:
        need = n - have.get(t, 0)
        if need <= 0:
            continue
        pool = [d for d in drafted
                if d["trait_id"] == t and d["scenario_id"] not in ckpt.done]
        todo += pool[:need]
        print(f"    {t}: need {need}, taking {len(pool[:need])}")
    if not todo:
        print(">>> nothing to do; every named trait already meets the target")
        return

    usage = Usage()
    ctx = Ctx(cfg=cfg, usage=usage, workers=int(cfg.get("workers", 8)),
              run_dir=run_dir, smoke=False,
              vars={"constitution": full_text(cfg["constitution"])})
    print(f">>> rewriting {len(todo)} responses")
    stage_list[names.index("final")].fn(ctx, todo, ckpt)

    after: dict[str, int] = {}
    for r in ckpt.done.values():
        after[r["trait_id"]] = after.get(r["trait_id"], 0) + 1
    print(f">>> per-trait counts now: {dict(sorted(after.items()))}")
    print(f">>> top-up spend ${usage.usd:.2f}")


def check(config: str, run_dir: str, sample: int | None = None,
          stage: str | None = None) -> None:
    """Re-check a finished run without regenerating it, and gate on its thresholds.

    Two independent halves, either of which may be absent: each `corpus_check` stage in
    `stages:` (its properties run over that stage's input snapshot, writing
    `<stage>_report.json` -- any document type can have this) and a `checks:` block (the
    model-eval-model validity checks, writing `checks_report.json`).

    Args:
        config: Path to the run YAML.
        run_dir: The run directory holding the stage snapshots.
        sample: Override the number of documents the LLM-judged checks sample.
        stage: Name of the `corpus_check` stage to run. Naming one also means "only
            this": the `checks:` half is skipped, so a corpus can be re-checked
            without re-paying for the judged model-eval-model checks.

    Raises:
        SystemExit: Nonzero when any gated check fails. Every report is written in
            full first, so a failed run is still inspectable.
    """
    load_dotenv()
    cfg = _load(config)
    corpus_stages = [s for s in cfg.get("stages") or []
                     if s.get("kind") == "corpus_check"
                     and (stage is None or s["name"] == stage)]
    assert cfg.get("checks") or corpus_stages, (
        f"{config} declares neither a `checks:` block nor a `corpus_check` stage"
        + (f" named {stage!r}" if stage else "") + " -- there is nothing to check")

    ok = True
    for sc in corpus_stages:
        ok = _check_corpus_stage(cfg, sc, Path(run_dir), sample) and ok
    if cfg.get("checks") and stage is None:
        from .checks import run_checks

        _, checks_ok = run_checks(run_dir, cfg, sample=sample)
        ok = ok and checks_ok
    if not ok:
        raise SystemExit(1)


def _check_corpus_stage(cfg: dict, sc: dict, run_dir: Path,
                        sample: int | None) -> bool:
    """Run one `corpus_check` stage entry over a finished run's snapshots."""
    from .corpus import is_paid, print_summary, run_corpus_checks
    from .hf_cache import read_jsonl
    from .pipeline import build_stages, snapshot_positions

    # A corpus check is an observer: it writes no snapshot of its own and inspects the
    # last real stage before it, so that is the snapshot to re-read.
    positions = snapshot_positions(build_stages(cfg))
    pos = positions[sc["name"]]
    source = next((s["name"] for s in cfg["stages"]
                   if positions.get(s["name"]) == pos and s["name"] != sc["name"]), None)
    snap = run_dir / f"stage_{pos}_{source}.jsonl" if source else None
    assert snap is not None and snap.exists(), (
        f"the corpus check {sc['name']!r} inspects "
        + (f"stage_{pos}_{source}.jsonl, which is not in {run_dir}"
           if source else "the stage before it, and it is first in the list"))

    if sample is not None:
        sc = {**sc, "properties": [{**p, "params": {**(p.get("params") or {}),
                                                    "sample": sample}}
                                   for p in sc["properties"]]}
    ctx = None
    if is_paid(sc):
        ctx = Ctx(cfg=cfg, usage=Usage(), workers=int(cfg.get("workers", 8)),
                  run_dir=run_dir, smoke=False,
                  vars={"constitution": full_text(cfg["constitution"])})

    print(f">>> corpus checks over {snap.name}")
    report = run_corpus_checks(read_jsonl(snap), sc, run_dir=run_dir,
                               seed=int(cfg.get("seed", 0)),
                               workers=int(cfg.get("workers", 8)), ctx=ctx)
    dest = run_dir / f"{sc['name']}_report.json"
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(report)
    print(f">>> report at {dest}")
    return bool(report["pass"])


def compare(reports, key: str = "group_size", out: str | None = None,
            stage: str | None = None) -> None:
    """Compare several runs' corpus reports as arms of one experiment.

    One run is one arm, so a claim like "the conflict rate rises with group size" can
    only be checked across run directories. Computes nothing new.

    Args:
        reports: Run directories, one per arm. Fire accepts "a,b,c" or a tuple.
        key: Manifest/config field ordering the arms (e.g. group_size).
        out: Write the markdown here; the JSON lands beside it. Default: print only.
        stage: Corpus-check stage name, when a run holds more than one report.
    """
    from .compare import compare_arms, markdown

    result = compare_arms(_csv(reports), key=key, stage=stage)
    text = markdown(result)
    print(text)
    if out:
        dest = Path(out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        dest.with_suffix(".json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n>>> {dest} (+ {dest.with_suffix('.json').name})")


def estimate(config: str, measured: str | None = None) -> None:
    """Print a cost estimate for a full run of the config's pipeline.

    Args:
        config: Path to the run YAML.
        measured: Optional manifest.json from a smoke run, to price from real token
            counts instead of assumptions.
    """
    print(json.dumps(pipeline.estimate(_load(config), measured), indent=2))


def chunkings() -> None:
    """List the chunking methods a dataset config's `chunking:` flag may name."""
    from .constitution import CHUNKINGS, DEFAULT_CHUNKING

    for name, c in CHUNKINGS.items():
        mark = " (default)" if name == DEFAULT_CHUNKING else ""
        shape = (f"{c.granularity} x {c.strategy}"
                 + (f", {c.n_clusters} clusters" if c.strategy == "cluster"
                    else f", {c.size} per unit" if c.size > 1 else ""))
        print(f"{name}{mark}\n     {shape}\n     {c.summary}")
    print("\nUse in a dataset config:  chunking: <name>\n"
          "Preview one:              uv run synth segment --chunking <name>")


def segment(constitution: str = "constitutions/claude_distilled_12_principles_mid/constitution.md",
            chunking: str | None = None, seed: int = 0, full: bool = False) -> None:
    """Print the units a chunking method produces, without calling any model.

    The dry-run for every chunking choice: inspectable for free, offline and with no
    API key, before a cent is spent generating against it.

    Args:
        constitution: Path to the constitution markdown.
        chunking: A method name (`uv run synth chunkings` lists them). Defaults to the
            one every existing corpus was built with.
        seed: Run seed, used by methods that shuffle.
        full: Print each unit's whole text instead of a one-line preview.
    """
    from .constitution import chunk as _chunk
    from .constitution import full_text, group as _group, preamble as _preamble
    from .constitution import resolve_chunking

    spec = resolve_chunking(chunking)
    granularity = spec.granularity
    chunks, style = _chunk(constitution, granularity=granularity,
                           min_words=spec.min_words)
    units = _group(chunks, size=spec.size, strategy=spec.strategy, seed=seed,
                   n_clusters=spec.n_clusters)

    for u in units:
        members = f"  <- {', '.join(u.chunk_ids)}" if u.n_chunks > 1 else ""
        print(f"{u.unit_id:<14} {u.name}{members}")
        print(u.text if full else f"     {u.text[:150].replace(chr(10), ' ')}")

    words = [len(u.text.split()) for u in units]
    doc_words = len(full_text(constitution).split())
    print(f"\nchunking: {spec.name} -- {len(chunks)} {granularity} chunks -> "
          f"{len(units)} units ({spec.strategy})")
    # Below `principle` the sum exceeds the document: every sub-chunk repeats its
    # principle title so it stays self-contained. Say so rather than look like a bug.
    note = " incl. repeated titles" if sum(words) > doc_words else ""
    print(f"words/unit: min {min(words)}  median {sorted(words)[len(words) // 2]}  "
          f"max {max(words)}  |  {sum(words)} words across units{note}, "
          f"{doc_words} in the document")
    print(f"{len(style)} chars of shared style guidance (injected everywhere)")

    unchunked = _preamble(constitution)
    if unchunked and granularity != "whole":
        # Names the blind spot rather than hiding it: this is where the constitution
        # says how principles trade off, and no unit is built around it.
        print(f"{len(unchunked.split())} words of preamble belong to NO unit at this "
              f"granularity (title + priority/conflict-resolution); they reach the "
              f"model only via {{constitution}}")

    if len(chunks) > 1:
        from .checks import _hashed_features

        X = _hashed_features([c.text for c in chunks])
        sims = X @ X.T
        n = len(chunks)
        central = (sims.sum(axis=1) - 1.0) / (n - 1)
        order = sorted(range(n), key=lambda i: -float(central[i]))
        print(f"chunk centrality (mean cosine to the rest): "
              f"{float(central.min()):.2f}-{float(central.max()):.2f}; "
              f"most central {chunks[order[0]].chunk_id}, "
              f"most peripheral {chunks[order[-1]].chunk_id}")


def main() -> None:
    fire.Fire({"run": run, "topup": topup, "check": check, "compare": compare,
               "estimate": estimate, "segment": segment, "chunkings": chunkings})


if __name__ == "__main__":
    main()
