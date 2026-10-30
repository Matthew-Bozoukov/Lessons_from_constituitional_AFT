# ABOUTME: Finish a MASK run whose generations are on disk but whose judging never completed (a
# ABOUTME: judge_batch=true run whose batches stalled): judge live, score, package and publish as run_eval would.
"""    uv run python scratch/mask_judge_saved.py output/mask/<run_dir> [--no-push]

The run dir is the one run_eval created: `run_meta.json` (the launch record) at its root and
`mask_work/data/responses/*.csv` holding every generation. Everything from the judge stage on
is redone here exactly as src/eval/misalignment/mask/runner.py does it, but with the judge
called live (`judge_batch=false`); the summary, the published layout, the card and the tags
are built by the same functions run_eval uses, so the pushed repo is indistinguishable from
one the entrypoint published -- except that run_meta records the rejudge.

Why it exists: 2026-09-18, daa-7's MASK batches sat in OpenRouter's queue for ten hours with
nothing completed (the batch API has no cancel and a 24 h window), then the poll loop died on
a DNS blip. The 4,438 generations were the GPU half and were already saved.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.layout import publish_layout
from src.eval.misalignment.mask.runner import (
    _HARNESS, ARCHETYPES, UPSTREAM_COMMIT, _run_stage, empty_content, generation_errors, overall_honesty,
)
from src.eval.run_eval import _card_fields, _publish
from src.infra.endpoints.openrouter import OPENROUTER_BASE_URL
from src.utils import write_run_meta


def main() -> None:
    out_dir = Path(sys.argv[1]).resolve()
    push = "--no-push" not in sys.argv
    launch = json.loads((out_dir / "run_meta.json").read_text())
    cfg = OmegaConf.create(launch["config"])
    hf_path, mode = launch["target"], launch["mode"]
    work = out_dir / "mask_work"
    data_dir, logs = work / "data", work / "logs"
    # The harness named the response files after the SERVED model name (the target's undated
    # model key, e.g. qwen36_0_daa_7), so read that name back off the files themselves.
    names = {p.stem[len(a) + 1:] for p in data_dir.glob("responses/*.csv")
             for a in sorted(ARCHETYPES, key=len, reverse=True) if p.stem.startswith(a + "_")}
    assert len(names) == 1, f"expected one model's response files under {data_dir / 'responses'}, found {names}"
    modelname = model_key = names.pop()
    responses = sorted(data_dir.glob(f"responses/*_{modelname}.csv"))
    assert responses, f"no saved generations for {modelname} under {data_dir / 'responses'}"
    # The sampling record runner.sample_data returned: rebuilt from the drawn CSVs and the
    # harness's full ones (subsample was null here, so n == of for every archetype).
    sampled = {}
    for a in ARCHETYPES:
        drawn = list(csv.DictReader((data_dir / f"{a}.csv").open(encoding="utf-8", newline="")))
        full = list(csv.DictReader((_HARNESS / "csv_data" / f"{a}.csv").open(encoding="utf-8", newline="")))
        sampled[a] = {"n": len(drawn), "of": len(full), "task_ids": [r.get("task_id", "") for r in drawn]}
    n_rows = sum(v["n"] for v in sampled.values())
    subsample = int(cfg.get("subsample") or 0) or None
    seed = int(cfg.get("seed", 0))
    gen_concurrency = int(cfg.get("gen_concurrency", 10))
    empty_policy = str(cfg.get("empty_content", "evasion"))

    env = dict(os.environ)
    env["MASK_JUDGE_MODEL"] = str(cfg.judge_model)
    env["MASK_JUDGE_BASE_URL"] = OPENROUTER_BASE_URL
    env["MASK_JUDGE_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
    env["MASK_DATA_DIR"] = str(data_dir)
    env["MASK_EMPTY_CONTENT"] = empty_policy

    errors = generation_errors(data_dir / "responses")
    error_rate = errors["errors"] / max(errors["generations"], 1)
    empty = empty_content(data_dir / "responses")
    empty_rate = empty["total"] / max(errors["generations"], 1)
    print(f">>> MASK: {errors['errors']}/{errors['generations']} saved generations failed ({100 * error_rate:.1f}%); "
          f"{empty['total']} empty ({100 * empty_rate:.1f}%)", flush=True)
    for stale in (data_dir / "evaluated", data_dir / "metrics"):   # the collect pass may have left placeholders
        shutil.rmtree(stale, ignore_errors=True)
    print(f">>> MASK: judging honesty with {cfg.judge_model} (live, {cfg.judge_concurrency} in flight)", flush=True)
    _run_stage(["evaluate.py", "--concurrency_limit", str(cfg.judge_concurrency)], env, _HARNESS, logs)
    _run_stage(["metric.py"], env, _HARNESS, logs)
    _run_stage(["process_metrics.py"], env, _HARNESS, logs)

    all_results = json.loads((data_dir / "metrics" / "all_results.json").read_text())
    per_archetype = {}
    for archetype in ARCHETYPES:
        m = all_results.get(archetype, {}).get(modelname)
        if m and "honesty_score_1" in m:
            per_archetype[archetype] = {"honesty_score": m["honesty_score_1"], "unhonest_pct": m["unhonest_1"],
                                        "honest_pct": m["honest_1"], "n_responses": m["total_responses"],
                                        "n_rows": sampled[archetype]["n"]}
    assert per_archetype, f"no honesty scores for {modelname!r}; archetypes {list(all_results)}"
    overall = overall_honesty(per_archetype)
    rejudge = {"rejudged_live_by": "scratch/mask_judge_saved.py",
               "reason": "judge_batch=true batches never ran (10 h at 0/4196 completed); generations reused, judge re-run live"}
    summary = {
        "benchmark": "MASK", "model": modelname, "mode": mode, "judge": str(cfg.judge_model),
        "overall_honesty_score": overall, "overall_weighting": "per_row", "per_archetype": per_archetype,
        "n_archetypes": len(per_archetype), "n_rows": n_rows,
        "generation_errors": errors, "generation_error_rate": round(error_rate, 4),
        "empty_content": empty, "empty_content_rate": round(empty_rate, 4), "empty_content_policy": empty_policy,
        "subsample": subsample, "seed": seed, "gen_concurrency": gen_concurrency,
        "judge_batch": False, "judge_batch_stats": None, "lie_k": int(cfg.lie_k), "smoke": False, **rejudge,
    }
    print(f">>> MASK overall honesty {overall} | " + ", ".join(f"{a}={v['honesty_score']:.1f}" for a, v in per_archetype.items()))

    rollouts_dir, results_dir, metadata_dir = publish_layout(out_dir)
    for archetype in per_archetype:
        for src, dst in ((data_dir / "responses" / f"{archetype}_{modelname}.csv", rollouts_dir / f"{archetype}.csv"),
                         (data_dir / "evaluated" / f"{archetype}_{modelname}.csv", results_dir / f"{archetype}_evaluated.csv"),
                         (data_dir / "metrics" / f"{archetype}_{modelname}.csv", results_dir / f"{archetype}_metrics.csv")):
            if src.is_file():
                shutil.copy2(src, dst)
    (results_dir / "all_results.json").write_text(json.dumps(all_results, indent=2))
    (metadata_dir / "subsample.json").write_text(json.dumps({"subsample": subsample, "seed": seed, "archetypes": sampled}, indent=2))
    write_run_meta(metadata_dir, OmegaConf.to_container(cfg, resolve=True),
                   extra={"target": hf_path, "mode": mode, "upstream_commit": UPSTREAM_COMMIT, "n_rows": n_rows,
                          "generation_errors": errors, "generation_error_rate": round(error_rate, 4),
                          "empty_content": empty, "empty_content_rate": round(empty_rate, 4),
                          "empty_content_policy": empty_policy,
                          "rows_per_archetype": {a: v["n"] for a, v in sampled.items()},
                          "gen_concurrency": gen_concurrency, "judge_batch": False, "judge_batch_stats": None, **rejudge})
    (metadata_dir / "run_meta.json").rename(metadata_dir / "mask_run_meta.json")
    if logs.is_dir():
        shutil.copytree(logs, metadata_dir / "harness_logs", dirs_exist_ok=True)
    shutil.rmtree(work)                                        # runner.run removes the work tree too

    command = launch["command"] + f"  # then: uv run python scratch/mask_judge_saved.py {sys.argv[1]}"
    launch_cfg = OmegaConf.create(launch["config"])
    summary = {"target": hf_path, "mode": mode, **summary}
    url = _publish(out_dir, name="mask", model_key=model_key, mode=mode, target=hf_path, summary=summary, push=push,
                   card=_card_fields("mask", launch_cfg, command,
                                     experiment=f"mask eval of {hf_path} (mode={mode}); judged live after the batch stalled",
                                     models=json.dumps({"target": launch["target"], "target_revision": launch["target_revision"],
                                                        "base": launch["base_model"], "base_revision": launch["base_model_revision"]}),
                                     source_revision=launch["git_sha"]),
                   tags=["eval-run", "eval:mask", f"model:{model_key}", f"mode:{mode}"])
    print(f">>> done: honesty {overall} | {url or '(not pushed)'}")


if __name__ == "__main__":
    main()
