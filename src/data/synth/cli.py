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
from .stage_runtime import Checkpoint, Ctx, Usage


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
        ablate: str | None = None, overrides: str | None = None,
        batch: bool = False) -> None:
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
        batch: Route the bulk of every paid stage through OpenRouter's async batch
            API (50% token pricing, results within 24h): llm_json/llm_tagged batch
            their records, scenarios batches per steering wave (diversity preserved),
            and parse/lint rejects mop up interactively. Equivalent to `batch: true`
            in the config; a stage opts out with `batch: false` on its entry.
    """
    load_dotenv()
    loaded = OmegaConf.load(config)
    if overrides:
        loaded = OmegaConf.merge(loaded, OmegaConf.from_dotlist(_csv(overrides)))
        print(f">>> overrides: {overrides}")
    cfg = OmegaConf.to_container(loaded, resolve=True)
    # The stem is the style-type and the style-type names the corpus (src/naming.py), so
    # the stem is what `pipeline` holds — not a label beside it that can drift from it.
    cfg["pipeline"] = Path(config).stem
    if batch:
        cfg["batch"] = True
    if ablate:
        cfg["ablate"] = sorted(set(cfg.get("ablate") or []) | set(_csv(ablate)))
    pipeline.exit_if_gate_failed(pipeline.run(cfg, smoke=smoke, resume=resume))


def topup(config: str, resume: str, traits, n: int = 25,
          draft_stage: str = "draft_responses",
          revise_stage: str = "revise_responses") -> None:
    """Re-run the revision stage for specific traits until each has `n` completed records.

    A run stopped early covers only the traits its scenarios happened to reach, since
    scenarios are ordered by trait. Reuses the revision stage's checkpoint so nothing
    already paid for is repeated.

    Args:
        config: Path to the run YAML.
        resume: Run directory holding the stage snapshots.
        traits: Trait ids -- Fire hands this over as a tuple when it contains commas,
            so both "t5,t6" and a tuple are accepted.
        n: Target completed records per trait.
        draft_stage: Stage whose snapshot supplies the records to revise.
        revise_stage: Stage to re-run. The defaults are the difficult-advice /
            pre-action-deliberation layout; pass both when topping up a run whose stages are
            named otherwise, including one from before the 2026-08-13 rename
            (`draft_responses`/`final`).
    """
    load_dotenv()
    cfg = _load(config)
    stage_list = pipeline.build_stages(cfg)
    names = [s.name for s in stage_list]
    missing = [s for s in (draft_stage, revise_stage) if s not in names]
    assert not missing, (
        f"topup: stage(s) {missing} are not in this config's pipeline ({names}). Pass "
        f"--draft_stage/--revise_stage to name the two stages to top up.")
    run_dir = Path(resume)
    assert run_dir.exists(), f"run dir does not exist: {run_dir}"

    src_idx = names.index(draft_stage) + 1
    fin_idx = names.index(revise_stage) + 1
    drafted = [json.loads(line) for line in
               (run_dir / f"stage_{src_idx}_{draft_stage}.jsonl").open()]
    ckpt = Checkpoint(run_dir / f"stage_{fin_idx}_{revise_stage}.partial.jsonl")

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
    stage_list[names.index(revise_stage)].fn(ctx, todo, ckpt)

    after: dict[str, int] = {}
    for r in ckpt.done.values():
        after[r["trait_id"]] = after.get(r["trait_id"], 0) + 1
    print(f">>> per-trait counts now: {dict(sorted(after.items()))}")
    print(f">>> top-up spend ${usage.usd:.2f}")
    # A top-up changes what the run dir holds without being the run that made it, so it
    # writes itself into the manifest: what was asked for, and what the counts became.
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("topups", []).append({
        "command": " ".join(sys.argv), "traits": list(traits), "n": n,
        "draft_stage": draft_stage, "revise_stage": revise_stage,
        "counts_after": dict(sorted(after.items())), "usd": round(usage.usd, 4)})
    manifest_path.write_text(json.dumps(manifest, indent=2))


def checks(tier: str | None = None) -> None:
    """List the registered corpus properties, their tier and what each flags.

    The counterpart to `synth chunkings`: the names `--only`/`--skip` accept, so a
    selection can be composed without opening the registry.
    """
    from .check_corpus import CORPUS_CHECKS, TIERS

    for t in TIERS:
        if tier and t != tier:
            continue
        rows = [c for c in CORPUS_CHECKS.values() if c.tier == t]
        cost = "free, offline, every run" if t == "surface" else "calls a model, sampled"
        print(f"\n{t} ({cost}):")
        for c in rows:
            print(f"  {c.name:22s} min_docs={c.min_docs:<4d} {c.doc}")
    print(f"\nSelect with: synth check --config <cfg> --run_dir <dir> "
          f"[--only a,b] [--skip c] [--tier surface]")


def check(config: str, run_dir: str, sample: int | None = None,
          stage: str | None = None, only: str | None = None,
          skip: str | None = None, tier: str | None = None) -> None:
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
        only: Comma-separated properties/aliases to run, to the exclusion of the rest.
        skip: Comma-separated properties/aliases to exclude.
        tier: `surface` (free, offline) or `judged` (paid). `--tier surface` is the
            cheap re-check; `--tier judged` re-runs only what costs money, resuming
            from the sidecar so already-judged documents are not re-paid for.

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
    # A selection is about the corpus properties; combining it with the model-eval-model
    # `checks:` half would silently run that (paid) half in full.
    selecting = any(x is not None for x in (only, skip, tier))
    assert not (selecting and stage is None and cfg.get("checks")), (
        "--only/--skip/--tier select corpus properties, so name the corpus stage too "
        "(--stage corpus); otherwise the `checks:` half would run unselected")

    ok = True
    for sc in corpus_stages:
        ok = _check_corpus_stage(cfg, sc, Path(run_dir), sample,
                                 only=only, skip=skip, tier=tier) and ok
    if cfg.get("checks") and stage is None:
        from .check_model_eval_model import run_checks

        _, checks_ok = run_checks(run_dir, cfg, sample=sample)
        ok = ok and checks_ok
    if not ok:
        raise SystemExit(1)


def _check_corpus_stage(cfg: dict, sc: dict, run_dir: Path, sample: int | None,
                        only: str | None = None, skip: str | None = None,
                        tier: str | None = None) -> bool:
    """Run one `corpus_check` stage entry over a finished run's snapshots."""
    from .check_corpus import (is_paid, pattern_table, print_summary, run_corpus_checks,
                         select_properties)
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
    # Before `is_paid`: deselecting every judged property must also mean no model context
    # is built, so `--tier surface` needs no key and costs nothing.
    sc = select_properties(sc, only=only, skip=skip, tier=tier)
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
    # The markdown mirror the repo requires beside any result: pattern frequencies must
    # be greppable without opening a 2MB JSON.
    table = pattern_table(report)
    if table:
        mirror = run_dir / f"{sc['name']}_patterns.md"
        mirror.write_text(f"# {sc['name']} — recurring patterns\n{table}\n",
                          encoding="utf-8")
        print(table)
        print(f">>> pattern table at {mirror}")
    print_summary(report)
    print(f">>> report at {dest}")
    return bool(report["pass"])


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
        from .check_corpus import hashed_features

        X = hashed_features([c.text for c in chunks])
        sims = X @ X.T
        n = len(chunks)
        central = (sims.sum(axis=1) - 1.0) / (n - 1)
        order = sorted(range(n), key=lambda i: -float(central[i]))
        print(f"chunk centrality (mean cosine to the rest): "
              f"{float(central.min()):.2f}-{float(central.max()):.2f}; "
              f"most central {chunks[order[0]].chunk_id}, "
              f"most peripheral {chunks[order[-1]].chunk_id}")


def _refuse_unknown_flags(commands: dict) -> None:
    """Exit BEFORE dispatch when a flag names no parameter of the chosen command.

    Fire's chaining semantics run the command first and then try to apply leftover
    args to its RETURN VALUE, so `synth run --config x --estimate` executes the whole
    PAID pipeline and only afterwards prints "Could not consume arg: --estimate"
    (observed 2026-08-14: a full write_scenarios stage spent before the flag error
    surfaced). A CLI whose commands cost money must refuse a flag it does not
    understand before spending anything.
    """
    import inspect
    import sys

    argv = sys.argv[1:]
    if not argv or argv[0] not in commands:
        return  # unknown command: Fire's own usage error covers it, nothing runs
    params = inspect.signature(commands[argv[0]]).parameters
    # Fire accepts --no<flag> for booleans and hyphens for underscores.
    known = set(params) | {f"no{p}" for p in params} | {"help"}
    for tok in argv[1:]:
        if tok == "--":
            break  # everything after the separator is Fire's own flag namespace
        if tok.startswith("--"):
            name = tok[2:].split("=", 1)[0].replace("-", "_")
            if name and name not in known:
                raise SystemExit(
                    f"synth {argv[0]}: unknown flag --{name}. Accepted: "
                    + ", ".join(f"--{p}" for p in params)
                    + ". Refusing before dispatch: Fire would otherwise run the "
                      "command first and only fail on the flag afterwards.")


def main() -> None:
    commands = {"run": run, "topup": topup, "check": check, "checks": checks,
                "estimate": estimate, "segment": segment, "chunkings": chunkings}
    _refuse_unknown_flags(commands)
    fire.Fire(commands)


if __name__ == "__main__":
    main()
