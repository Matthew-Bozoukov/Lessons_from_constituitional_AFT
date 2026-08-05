# ABOUTME: Fire entrypoint for the synthdoc pipeline (scripts/data/synthdoc/build_dataset.py is the
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


def run(config: str, smoke: bool = False, resume: str | None = None,
        ablate: str | None = None, overrides: str | None = None) -> None:
    """Run the pipeline the config declares (its `stages:` list).

    Args:
        config: Path to the run YAML (configs/data/synthdoc/<document_type>.yaml).
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
        loaded = OmegaConf.merge(loaded, OmegaConf.from_dotlist(
            [o.strip() for o in overrides.split(",") if o.strip()]))
        print(f">>> overrides: {overrides}")
    cfg = OmegaConf.to_container(loaded, resolve=True)
    if ablate:
        cfg["ablate"] = sorted(set(cfg.get("ablate") or [])
                               | {a.strip() for a in str(ablate).split(",") if a.strip()})
    pipeline.run(cfg, smoke=smoke, resume=resume)


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
    ids = list(traits) if isinstance(traits, (list, tuple)) else \
        [x.strip() for x in str(traits).split(",")]
    ids = [x for x in ids if x]
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


def check(config: str, run_dir: str, sample: int | None = None) -> None:
    """Run the corpus validity checks over a run and gate on the config's thresholds.

    Args:
        config: Path to the run YAML (its `checks:` block supplies judges + gates;
            model-eval-model configs declare one).
        run_dir: The run directory holding the stage snapshots.
        sample: Override the number of documents the LLM-judged checks sample.

    Raises:
        SystemExit: Nonzero when any gated check fails; the full report is still
            written to <run_dir>/checks_report.json first.
    """
    load_dotenv()
    from .checks import run_checks

    cfg = _load(config)
    assert cfg.get("checks"), "this config declares no `checks:` block"
    _, ok = run_checks(run_dir, cfg, sample=sample)
    if not ok:
        raise SystemExit(1)


def estimate(config: str, measured: str | None = None) -> None:
    """Print a cost estimate for a full run of the config's pipeline.

    Args:
        config: Path to the run YAML.
        measured: Optional manifest.json from a smoke run, to price from real token
            counts instead of assumptions.
    """
    print(json.dumps(pipeline.estimate(_load(config), measured), indent=2))


def segment(constitution: str = "constitutions/claude_distilled_12_principles_mid/constitution.md") -> None:
    """Print the traits the constitution segments into, without calling any model.

    Args:
        constitution: Path to the constitution markdown.
    """
    from .constitution import segment as _segment

    traits, style = _segment(constitution)
    for t in traits:
        print(f"{t.trait_id}  {t.name}")
        print(f"     {t.text[:150]}")
    print(f"\n{len(traits)} traits, {len(style)} chars of shared style guidance")


def main() -> None:
    fire.Fire({"run": run, "topup": topup, "check": check,
               "estimate": estimate, "segment": segment})


if __name__ == "__main__":
    main()
