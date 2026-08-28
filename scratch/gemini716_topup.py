# ABOUTME: Top-up driver for the gemini-716 run: regenerates flash-surviving replacement
# ABOUTME: scenarios for filter-dropped records until the corpus holds exactly 716 rows.

"""Top the gemini difficult-advice run up to its per-trait targets.

Google's non-configurable safety layer blocks a few percent of difficult-advice
records on gemini-3.7-flash — persistently, so retries cannot save them (measured
2026-08-20: 26-30/716 draft calls filtered on every one of six resamples, ~25 of them
on the harm-weighing trait t4). The run therefore finishes short after phase A
(`--overrides max_fail_pct=5`). This driver closes the gap per the decision to stay
all-flash: generate fresh scenarios for the shortfall traits (stage-2 prompts
verbatim, avoid-list seeded with the existing corpus, the same 0.86 reject-cosine
gate against every existing situation), run them through the real stage functions
(draft -> refine -> respond -> rewrite, with the run dir's own checkpoints, lint and
retries), and keep whatever survives the filter. Repeats until every trait meets its
target, then rebuilds the stage snapshots + export and re-publishes dataset.jsonl.

The corpus this produces systematically avoids what Google refuses to generate —
that is the accepted, recorded trade-off of the all-flash arm (vs re-drafting the
blocked records with gemini-3.1-pro, which passes them).

    uv run python scratch/gemini716_topup.py \
        --config configs/data/synth/difficult_advice_gemini_716.yaml \
        --run_dir output/synthdoc_gemini716/20260820_143809 [--push]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import fire
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synth import pipeline  # noqa: E402
from src.data.synth.constitution import full_text, units_from_config  # noqa: E402
from src.data.synth.embeddings import DEFAULT_MODEL, embed  # noqa: E402
from src.data.synth.hf_cache import StageCache, read_jsonl  # noqa: E402
from src.data.synth.stage_operators import _gist, _too_close, scenario_batches  # noqa: E402
from src.data.synth.stage_runtime import Checkpoint, Ctx, Usage, call_json, model_cfg  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

# stage name -> snapshot position in this config (see the run dir's filenames)
POS = {"write_scenarios": 2, "dedupe_scenarios": 3, "draft_prompts": 4,
       "revise_prompts": 5, "draft_responses": 6, "revise_responses": 7,
       "export_sft": 8}
PIPE = ("draft_prompts", "revise_prompts", "draft_responses", "revise_responses")


def main(config: str, run_dir: str, max_rounds: int = 8, budget_usd: float = 15.0,
         push: bool = True) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    # The driver owns attrition: a filtered call inside a small top-up batch must
    # reduce the accepted count, never abort the loop.
    cfg["max_fail_pct"] = 100.0
    rd = Path(run_dir)
    assert (rd / "stage_7_revise_responses.jsonl").exists(), (
        "run phase A to completion first — the driver tops up a finished run")

    stages = {s.name: s for s in pipeline.build_stages(cfg)}
    units, style = units_from_config(cfg)
    traits = [u.as_trait() for u in units]
    prov = {r["trait_id"]: {k: r.get(k) for k in
                            ("chunk_ids", "granularity", "grouping_strategy", "n_chunks")}
            for r in read_jsonl(rd / "stage_1_chunk_constitution.jsonl")}

    # Per-trait targets: the run's own split, recomputed identically.
    target: dict[str, int] = {}
    for ti, _bi, n in scenario_batches(len(traits), cfg):
        target[traits[ti].trait_id] = target.get(traits[ti].trait_id, 0) + n

    snaps = {name: read_jsonl(rd / f"stage_{i}_{name}.jsonl")
             for name, i in POS.items() if name != "export_sft"}
    have = Counter(r["trait_id"] for r in snaps["revise_responses"])
    shortfall = {t: target[t] - have.get(t, 0) for t in target if target[t] > have.get(t, 0)}
    print(f">>> targets {dict(sorted(target.items()))}")
    print(f">>> shortfall {dict(sorted(shortfall.items()))} "
          f"({sum(shortfall.values())} records)")
    if not shortfall:
        print(">>> nothing to do")
        return

    ctx = Ctx(cfg=cfg, usage=Usage(), workers=int(cfg.get("workers", 16)), run_dir=rd,
              smoke=False, vars={"constitution": full_text(cfg["constitution"]),
                                 "style_guidance": style})
    ckpts = {n: Checkpoint(rd / f"stage_{POS[n]}_{n}.partial.jsonl") for n in PIPE}

    scen_entry = next(s for s in cfg["stages"] if s["name"] == "write_scenarios")
    scen_m = model_cfg(cfg, scen_entry["model"])
    ban = [_gist(r) for r in snaps["write_scenarios"]]
    seen_vecs = embed([r["situation"] for r in snaps["write_scenarios"]],
                      model=DEFAULT_MODEL)
    by_trait = {t.trait_id: t for t in traits}
    tag = timestamp()[-6:]  # unique id prefix per driver invocation
    new_scen_by_id: dict[str, dict] = {}
    accepted: list[dict] = []

    for rnd in range(max_rounds):
        need = {t: n for t, n in shortfall.items() if n > 0}
        if not need:
            break
        assert ctx.usage.usd < budget_usd, (
            f"top-up spend ${ctx.usage.usd:.2f} passed the ${budget_usd} guard with "
            f"{sum(need.values())} records still short — the filter may be refusing "
            "faster than replacements can be found; reconsider the all-flash choice")
        print(f">>> round {rnd + 1}: need {dict(sorted(need.items()))}, "
              f"spent ${ctx.usage.usd:.2f}")

        candidates: list[dict] = []
        for t_id, n in sorted(need.items()):
            t = by_trait[t_id]
            k = min(8, n + 2)  # small overgen for gate + filter losses
            avoid = ("These situations already exist in this corpus. Do not write "
                     "another version of any of them -- the same story in a different "
                     "domain, or with the roles renamed, still counts as a repeat:\n"
                     + "\n".join(f"- {x}" for x in ban[-150:]))
            user = scen_entry["prompts"]["user"].format(
                trait_name=t.name, trait_text=t.text, n=k, avoid=avoid,
                overrepresented="")
            try:
                parsed, _ = call_json(
                    ctx.client, ctx.usage, scen_m["model"],
                    scen_entry["prompts"]["system"], user, scen_m["temperature"],
                    scen_m["max_tokens"], stage="topup_scenarios",
                    extra=scen_m.get("extra_body"))
            except Exception as exc:  # noqa: BLE001 - a lost batch just retries next round
                print(f"    !! {t_id} scenario call failed ({type(exc).__name__}); "
                      "retrying next round")
                continue
            for j, s in enumerate(parsed if isinstance(parsed, list) else []):
                if not isinstance(s, dict) or "situation" not in s:
                    continue
                candidates.append({
                    "scenario_id": f"{t_id}_tu{tag}_r{rnd}_s{j:03d}",
                    "trait_id": t_id, "trait_name": t.name, "trait_text": t.text,
                    "domain": str(s.get("domain", "")),
                    "situation": str(s["situation"]),
                    "shortcut": str(s.get("shortcut", "")),
                    **prov.get(t_id, {}),
                })

        if not candidates:
            continue
        bad, X = _too_close([r["situation"] for r in candidates], seen_vecs, 0.86,
                            None)
        survivors, rows = [], []
        counts = Counter()
        for i, r in enumerate(candidates):
            # Never overshoot a trait's target, mirroring op_scenarios.
            if i in bad or counts[r["trait_id"]] >= need.get(r["trait_id"], 0):
                continue
            counts[r["trait_id"]] += 1
            survivors.append(r)
            rows.append(X[i])
        print(f"    {len(candidates)} generated, {len(bad)} rejected as near-dupes, "
              f"{len(survivors)} entering the pipeline")
        if rows:
            seen_vecs = np.vstack([seen_vecs, np.array(rows)])
        ban.extend(_gist(r) for r in survivors)

        recs = survivors
        for name in PIPE:
            if not recs:
                break
            recs = stages[name].fn(ctx, recs, ckpts[name])
        for r in recs:
            shortfall[r["trait_id"]] -= 1
        for r in survivors:
            new_scen_by_id[r["scenario_id"]] = r
        accepted.extend(recs)
        print(f"    round {rnd + 1}: {len(recs)}/{len(survivors)} survived all stages")

    remaining = {t: n for t, n in shortfall.items() if n > 0}
    print(f">>> top-up complete: +{len(accepted)} records, ${ctx.usage.usd:.2f}, "
          f"remaining shortfall {remaining or 'NONE'}")
    if not accepted:
        return

    # --- rebuild snapshots, export, and the published dataset -------------------------
    cache = StageCache(rd, repo_id=(cfg.get("hf_repo") if push else None),
                       private=bool(cfg.get("hf_private", False)))
    new_scens = [new_scen_by_id[i] for i in sorted(new_scen_by_id)]
    cache.save(2, "write_scenarios", snaps["write_scenarios"] + new_scens)
    cache.save(3, "dedupe_scenarios", snaps["dedupe_scenarios"] + new_scens)
    for name in PIPE:
        # Top-up records only ("_tu" ids), and only ones the snapshot lacks — a
        # re-run of the driver must not double-append a prior run's rows.
        snap_ids = {r["scenario_id"] for r in snaps[name]}
        fresh = [r for k, r in ckpts[name].done.items()
                 if "_tu" in k and k not in snap_ids]
        cache.save(POS[name], name, snaps[name] + fresh)
    all_final = snaps["revise_responses"] + accepted
    export_rows = stages["export_sft"].fn(ctx, all_final, None)
    cache.save(8, "export_sft", export_rows)
    cache.publish_final(export_rows)

    manifest = json.loads((rd / "manifest.json").read_text())
    manifest.setdefault("topup", []).append({
        "timestamp": timestamp(), "git_sha": git_sha(), "accepted": len(accepted),
        "remaining_shortfall": remaining, "usage": ctx.usage.as_dict(),
        "note": "flash-surviving replacements for records blocked by Google's "
                "non-configurable safety filter; corpus composition therefore avoids "
                "what gemini-3.7-flash refuses to generate",
    })
    cache.save_json("manifest.json", manifest)
    print(f">>> published {len(export_rows)} records"
          + (f" to {cfg['hf_repo']}" if push else " (local only)"))
    if remaining:
        raise SystemExit(f"still short {remaining} after {max_rounds} rounds — "
                         "rerun the driver, or accept the count")


if __name__ == "__main__":
    fire.Fire(main)
