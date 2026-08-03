# ABOUTME: The six-stage runner replicating the Teaching Claude Why difficult-advice pipeline.
# ABOUTME: Each stage writes a complete snapshot and mirrors it to HF before the next begins.

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openrouter import OpenRouterClient  # noqa: E402
from utils import git_sha, timestamp  # noqa: E402

from . import stages  # noqa: E402
from .constitution import full_text, segment  # noqa: E402
from .hf_cache import StageCache  # noqa: E402

# (index, name, which config block supplies the model) in execution order.
STAGES = [
    (1, "traits", None),
    (2, "scenarios", "scenarios"),
    (3, "draft_prompts", "draft"),
    (4, "refined_prompts", "refine"),
    (5, "draft_responses", "respond"),
    (6, "final", "rewrite"),
]


def _model_cfg(cfg: dict, key: str) -> dict[str, Any]:
    """Return the merged model settings for one stage.

    Args:
        cfg: Full run config.
        key: Stage key under `models`.

    Returns:
        Dict with model, temperature and max_tokens.
    """
    defaults = cfg.get("defaults", {})
    block = cfg["models"][key]
    return {
        "model": block["model"],
        "temperature": float(block.get("temperature", defaults.get("temperature", 1.0))),
        "max_tokens": int(block.get("max_tokens", defaults.get("max_tokens", 4096))),
    }


def run(cfg: dict, smoke: bool = False, resume: str | None = None) -> dict:
    """Run the full pipeline, caching every stage.

    Args:
        cfg: Run config (see configs/synthdoc_v2.yaml).
        smoke: Restrict to 2 traits x 1 scenario and shrink the budget, to validate wiring.
        resume: Existing run directory to continue. Completed stage snapshots are reused
            and partially-completed stages pick up from their checkpoint, so a crash or
            an interrupt only costs the items that were in flight.

    Returns:
        A manifest dict describing the run.
    """
    started = time.time()
    ts = timestamp()
    if resume:
        run_dir = Path(resume)
        assert run_dir.exists(), f"resume dir does not exist: {run_dir}"
        print(f">>> resuming into {run_dir}")
    else:
        run_dir = Path(cfg["output_dir"]) / (f"smoke_{ts}" if smoke else ts)
    repo = cfg.get("hf_repo")
    if smoke:
        repo = cfg.get("hf_repo_smoke")  # never pollute the real dataset from a smoke run
    cache = StageCache(run_dir, repo, private=bool(cfg.get("hf_private", False)))

    workers = int(cfg.get("workers", 8))
    budget = float(cfg.get("budget_usd", 0)) or None
    client = OpenRouterClient()
    usage = stages.Usage()
    # Per-stage wall clock, so a later run can be scheduled from real numbers rather
    # than from a guess about per-call latency.
    durations: dict[str, float] = {}

    def timed(name: str, fn):
        """Run a stage, recording its wall clock."""
        t0 = time.time()
        out = fn()
        durations[name] = round(time.time() - t0, 1)
        return out

    def guard(stage: str) -> None:
        """Stop before the next stage if the budget is already spent."""
        if budget is not None and usage.usd > budget:
            raise RuntimeError(
                f"budget_usd=${budget:.2f} exceeded (${usage.usd:.2f}) after {stage}. "
                f"Snapshots up to this stage are in {run_dir}; raise budget_usd and re-run "
                f"to resume."
            )

    # --- stage 1: segment the constitution (deterministic) --------------------------
    traits, style_guidance = segment(cfg["constitution"])
    constitution = full_text(cfg["constitution"])
    if smoke:
        traits = traits[:2]
    print(f">>> stage 1: {len(traits)} traits -> {[t.trait_id for t in traits]}")
    cache.save(1, "traits", [t.as_dict() for t in traits])

    per_trait = 1 if smoke else int(cfg["scenarios_per_trait"])
    per_call = 1 if smoke else int(cfg.get("scenarios_per_call", per_trait))

    # --- stage 2: scenarios ---------------------------------------------------------
    if cache.has(2, "scenarios"):
        scenarios = cache.load(2, "scenarios")
        print(f">>> stage 2: reused {len(scenarios)} cached scenarios")
    else:
        m = _model_cfg(cfg, "scenarios")
        scenarios = timed("scenarios", lambda: stages.generate_scenarios(
            traits, client, usage, per_trait=per_trait, per_call=per_call,
            workers=workers, **m))
        cache.save(2, "scenarios", scenarios)
    print(f">>> stage 2: {len(scenarios)} scenarios")
    print(f"    FIRST: [{scenarios[0]['trait_name']}] {scenarios[0]['situation'][:220]}")
    guard("stage 2")

    # --- stage 3: draft the prompt --------------------------------------------------
    if cache.has(3, "draft_prompts"):
        drafts = cache.load(3, "draft_prompts")
        print(f">>> stage 3: reused {len(drafts)} cached drafts")
    else:
        drafts = timed("draft", lambda: stages.draft_prompts(
            scenarios, client, usage, workers=workers, **_model_cfg(cfg, "draft")))
        cache.save(3, "draft_prompts", drafts)
    print(f"    FIRST DRAFT USER: {drafts[0]['draft_user'][:220]}")
    guard("stage 3")

    # --- stage 4: refine the prompt -------------------------------------------------
    if cache.has(4, "refined_prompts"):
        refined = cache.load(4, "refined_prompts")
        print(f">>> stage 4: reused {len(refined)} cached refinements")
    else:
        refined = timed("refine", lambda: stages.refine_prompts(
            drafts, client, usage, constitution=constitution, workers=workers,
            ckpt=stages.Checkpoint(run_dir / "stage_4_refined_prompts.partial.jsonl"),
            **_model_cfg(cfg, "refine")))
        cache.save(4, "refined_prompts", refined)
    print(f"    FIRST REFINED USER: {refined[0]['user'][:220]}")
    guard("stage 4")

    # --- stage 5: generate the response ---------------------------------------------
    if cache.has(5, "draft_responses"):
        drafted = cache.load(5, "draft_responses")
        print(f">>> stage 5: reused {len(drafted)} cached responses")
    else:
        drafted = timed("respond", lambda: stages.generate_responses(
            refined, client, usage, style_guidance=style_guidance, workers=workers,
            ckpt=stages.Checkpoint(run_dir / "stage_5_draft_responses.partial.jsonl"),
            **_model_cfg(cfg, "respond")))
        cache.save(5, "draft_responses", drafted)
    print(f"    FIRST REASONING: {drafted[0]['draft_reasoning'][:220]}")
    guard("stage 5")

    # --- stage 6: rewrite against the constitution ----------------------------------
    if cache.has(6, "final"):
        final = cache.load(6, "final")
        print(f">>> stage 6: reused {len(final)} cached rewrites")
    else:
        final = timed("rewrite", lambda: stages.rewrite_responses(
            drafted, client, usage, constitution=constitution, workers=workers,
            ckpt=stages.Checkpoint(run_dir / "stage_6_final.partial.jsonl"),
            **_model_cfg(cfg, "rewrite")))
        cache.save(6, "final", final)
    print(f"    FIRST FINAL RESPONSE: {final[0]['response'][:220]}")

    # --- training-ready export ------------------------------------------------------
    sft = stages.to_sft(final)
    cache.save(7, "sft", sft)

    manifest = {
        "run_id": ts,
        "git_sha": git_sha(),
        "smoke": smoke,
        "config": cfg,
        # What this run ACTUALLY used, which differs from cfg under --smoke. The cost
        # estimator rescales the scenario stage by these, so they must be the real values.
        "effective": {"n_traits": len(traits), "scenarios_per_trait": per_trait,
                      "scenarios_per_call": per_call},
        "counts": {
            "traits": len(traits), "scenarios": len(scenarios), "drafts": len(drafts),
            "refined": len(refined), "responses": len(drafted), "final": len(final),
            "sft": len(sft),
        },
        "usage": usage.as_dict(),
        "wall_clock_s": round(time.time() - started, 1),
        "stage_seconds": durations,
        "workers": workers,
        "hf_repo": repo,
        "run_dir": str(run_dir),
    }
    cache.save_json("manifest.json", manifest)

    assert len(sft) == len(final), f"sft={len(sft)} != final={len(final)}"
    kept = 100 * len(final) / max(len(scenarios), 1)
    print(f">>> {len(final)}/{len(scenarios)} scenarios survived all stages ({kept:.1f}%)")
    print(f"\n>>> {len(sft)} training records in {run_dir}")
    print(f">>> spend ${usage.usd:.2f} | {manifest['wall_clock_s']}s")
    if repo:
        print(f">>> https://huggingface.co/datasets/{repo}")
    return manifest


def topup(cfg: dict, resume: str, traits: list[str], n_per_trait: int) -> dict:
    """Run stage 6 for specific traits only, until each has `n_per_trait` rewrites.

    A run stopped early covers only the traits its scenarios happened to reach, since
    scenarios are ordered by trait. This rewrites just enough additional stage-5 responses
    to bring the named traits up to a target count, reusing the stage-6 checkpoint so
    nothing already paid for is repeated.

    Args:
        cfg: Run config.
        resume: Run directory holding the stage snapshots.
        traits: Trait ids to top up, e.g. ["t5", "t6"].
        n_per_trait: Target completed rewrites per trait.

    Returns:
        A dict of per-trait counts after the top-up.
    """
    run_dir = Path(resume)
    assert run_dir.exists(), f"run dir does not exist: {run_dir}"
    constitution = full_text(cfg["constitution"])
    client = OpenRouterClient()
    usage = stages.Usage()

    drafted = [__import__("json").loads(line)
               for line in (run_dir / "stage_5_draft_responses.jsonl").open()]
    ckpt = stages.Checkpoint(run_dir / "stage_6_final.partial.jsonl")

    have: dict[str, int] = {}
    for r in ckpt.done.values():
        have[r["trait_id"]] = have.get(r["trait_id"], 0) + 1
    print(">>> current per-trait counts:", {t: have.get(t, 0) for t in traits})

    todo = []
    for t in traits:
        need = n_per_trait - have.get(t, 0)
        if need <= 0:
            continue
        pool = [d for d in drafted
                if d["trait_id"] == t and d["scenario_id"] not in ckpt.done]
        todo += pool[:need]
        print(f"    {t}: need {need}, taking {len(pool[:need])}")
    if not todo:
        print(">>> nothing to do; every named trait already meets the target")
        return have

    print(f">>> rewriting {len(todo)} responses")
    stages.rewrite_responses(todo, client, usage, constitution=constitution,
                             workers=int(cfg.get("workers", 8)), ckpt=ckpt,
                             **_model_cfg(cfg, "rewrite"))

    after: dict[str, int] = {}
    for r in ckpt.done.values():
        after[r["trait_id"]] = after.get(r["trait_id"], 0) + 1
    print(f">>> per-trait counts now: {dict(sorted(after.items()))}")
    print(f">>> top-up spend ${usage.usd:.2f}")
    return after
