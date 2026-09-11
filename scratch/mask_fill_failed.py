# ABOUTME: Post hoc fill of a published MASK run's "[ERROR: Generation failed or timed out]" cells: regenerate
# ABOUTME: exactly those cells against the served target, re-judge their rows, recompute metrics, republish as a revision.
"""
Run (from the repo root, with the target served on a pod rented by `uv run runpod up --eval <hf>`):

    uv run python scratch/mask_fill_failed.py \
        --run output/mask/2026-09-11_qwen36_0_delib_7_180103 \
        --target dougalldeepmind/2026-09-11-qwen36-0-delib-7 \
        --server root@<ip>:<port> --port 8021 [--no-push]

Contract (CLAUDE.md, "hand-pushed runs must match"): this reuses the vendored harness's own code for
every step it can — `generate_responses_async` builds and sends the very request the harness sends
(same client, same sampling, same reasoning/empty-content handling), `evaluate.process_file` judges
the affected rows with the same prompts and parser, `metric.py`/`process_metrics.py` recompute the
per-archetype scores, `overall_honesty` pools them the way the runner does — and it rewrites the
run dir in place (rollouts/, results/, metadata/) before `push_run_dir` pushes it to the SAME repo
with the same card fields and tags, so the Hub shows one more revision of the one run.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.misalignment.mask.runner import (  # noqa: E402
    ARCHETYPES, _HARNESS, generation_errors, overall_honesty)
from src.eval.run_eval import _card_fields, _results_markdown  # noqa: E402
from src.infra.endpoints.openrouter import OPENROUTER_BASE_URL  # noqa: E402
from src.infra.endpoints.vllm import SshExec, VllmServer, resolve_target  # noqa: E402
from src.infra.huggingface import push_run_dir  # noqa: E402

ERROR_PREFIX = "[ERROR"
sys.path.insert(0, str(_HARNESS))  # the harness imports `prompts.*` relative to its own dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path, help="local run dir in the published layout")
    p.add_argument("--target", required=True, help="HF path of the adapter the run measured")
    p.add_argument("--server", required=True, help="root@ip:port of the pod holding vLLM (runpod up --eval)")
    p.add_argument("--port", type=int, default=8021)
    p.add_argument("--ssh-key", default=os.path.expanduser("~/.ssh/id_ed25519"))
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--repo", required=True, help="the run's existing HF repo (org/name); the push is a new revision of it")
    p.add_argument("--no-push", action="store_true")
    return p.parse_args()


def failed_cells(rollouts: Path) -> dict[str, list[tuple[int, str]]]:
    """{archetype: [(row index, generation column), ...]} for every "[ERROR" cell."""
    out = {}
    for path in sorted(rollouts.glob("*.csv")):
        df = pd.read_csv(path)
        cells = [(i, c) for c in df.columns if c.startswith("generation(")
                 for i, v in df[c].items() if isinstance(v, str) and v.startswith(ERROR_PREFIX)]
        if cells:
            out[path.stem] = cells
    return out


def messages_for(row: pd.Series, column: str, doubling_down: bool) -> list[dict]:
    """The harness's message construction (generate_responses.process_dataframe), per column."""
    if column.startswith("generation(System Prompt + User Prompt)"):
        if doubling_down:
            return [{"role": "system", "content": row["system_prompt"]},
                    {"role": "user", "content": row["dd_turn_1_prompt"]},
                    {"role": "assistant", "content": row["dd_turn_1_response"]},
                    {"role": "user", "content": row["user_prompt"]}]
        return [{"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": row["user_prompt"]}]
    assert column.startswith("generation(belief_system_1 + belief_elicit_"), column
    be = column.split("belief_elicit_")[1].split(")")[0]
    system = row["belief_system_1"] if "belief_system_1" in row.index and pd.notna(row["belief_system_1"]) \
        else "You are a helpful assistant."
    return [{"role": "system", "content": system}, {"role": "user", "content": row[f"belief_elicit_{be}"]}]


async def regenerate(base_url: str, api_key: str, model: str, jobs: dict[str, list[dict]],
                     cfg, concurrency: int) -> dict[str, tuple[str | None, str | None]]:
    """jobs: {key: messages}. Returns {key: (content, reasoning)}; content None = failed again."""
    import generate_responses as gr  # the vendored module, exactly what the runner runs

    gr.EMPTY_CONTENT.clear()
    client = gr.create_model_client(model, base_url=base_url, api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    try:
        results = await gr.generate_responses_async(client, model, jobs, int(cfg.max_tokens),
                                                    float(cfg.temperature), sem, K=1)
    finally:
        await client.close()
    reasoning = results.pop("__reasoning__", {})
    return {k: (results[k][0] if results.get(k) else None,
                (reasoning.get(k) or [None])[0]) for k in jobs}, dict(gr.EMPTY_CONTENT)


def rejudge_rows(archetype: str, responses_csv: Path, evaluated_csv: Path, rows: list[int],
                 work: Path, env: dict, concurrency: int) -> None:
    """Judge only `rows` of the filled responses file (a temp CSV of those rows through the
    harness's own `evaluate.process_file`), then splice every eval column back by index."""
    df = pd.read_csv(responses_csv)
    sub = df.loc[sorted(set(rows))].copy()
    # `process_file` keys the numerical (statistics) prompt off the file NAME; keep it.
    sub_path = work / "rejudge" / f"{archetype}{responses_csv.suffix}"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(sub_path, index=False)
    out_path = work / "rejudge" / f"{archetype}_evaluated.csv"
    os.environ.update({k: v for k, v in env.items() if k.startswith("MASK_")})
    import evaluate as ev  # reads MASK_JUDGE_* at import time, so after the env is set

    judged = asyncio.run(ev.process_file(str(sub_path), str(out_path), concurrency))
    judged.index = sorted(set(rows))
    full = pd.read_csv(evaluated_csv)
    eval_cols = [c for c in judged.columns if c.startswith("eval(") or c == "ground_truth_extracted"]
    missing = [c for c in eval_cols if c not in full.columns]
    assert not missing, f"{archetype}: re-judge produced columns the published file lacks: {missing}"
    for col in [c for c in df.columns if c.startswith(("generation(", "reasoning("))] + eval_cols:
        full.loc[judged.index, col] = judged[col].values
    full.to_csv(evaluated_csv, index=False)


def main() -> None:
    args = parse_args()
    run_dir = args.run.resolve()
    rollouts, results, metadata = run_dir / "rollouts", run_dir / "results", run_dir / "metadata"
    launch_meta = json.loads((metadata / "run_meta.json").read_text())
    mask_meta = json.loads((metadata / "mask_run_meta.json").read_text())
    summary = json.loads((results / "results.json").read_text())
    cfg = OmegaConf.create(launch_meta["config"])
    modelname = summary["model"]
    assert launch_meta["target"] == args.target, (launch_meta["target"], args.target)

    cells = failed_cells(rollouts)
    n_cells = sum(len(v) for v in cells.values())
    print(f">>> {n_cells} failed cells in {sorted(cells)}: "
          + ", ".join(f"{a}={len(v)}" for a, v in cells.items()), flush=True)
    if not n_cells:
        raise SystemExit("nothing to fill")

    # --- serve the target exactly as run_eval does (same executor, same serving plan) ---
    executor = SshExec(args.server, port=args.port, bind="127.0.0.1", identity=args.ssh_key)
    executor.check_ready()
    server = VllmServer(work_dir=Path("output") / "mask" / "server", port=args.port, executor=executor,
                        serve_requirements=OmegaConf.to_container(cfg.get("serving") or {}, resolve=True))
    spec = resolve_target(args.target)
    assert spec.revision == launch_meta["target_revision"], (spec.revision, launch_meta["target_revision"])
    assert spec.mode == launch_meta["mode"], (spec.mode, launch_meta["mode"])

    work = run_dir / "fill_work"
    if work.exists():
        shutil.rmtree(work)
    data_dir = work / "data"
    (data_dir / "responses").mkdir(parents=True)
    (data_dir / "evaluated").mkdir(parents=True)
    # The harness's per-run data layout, rebuilt from the published files so metric.py and
    # process_metrics.py read exactly what they read on the original run.
    for path in rollouts.glob("*.csv"):
        shutil.copy2(path, data_dir / "responses" / f"{path.stem}_{modelname}.csv")
    for path in results.glob("*_evaluated.csv"):
        shutil.copy2(path, data_dir / "evaluated" / f"{path.stem[:-len('_evaluated')]}_{modelname}.csv")

    env = dict(os.environ)
    env.update({
        "MASK_JUDGE_MODEL": str(cfg.judge_model), "MASK_JUDGE_BASE_URL": OPENROUTER_BASE_URL,
        "MASK_JUDGE_API_KEY": os.environ["OPENROUTER_API_KEY"], "MASK_DATA_DIR": str(data_dir),
        "MASK_GEN_TIMEOUT_S": str(int(cfg.get("gen_timeout_s", 1800))),
        "MASK_EMPTY_CONTENT": str(cfg.get("empty_content", "evasion")),
    })
    os.environ.update({k: v for k, v in env.items() if k.startswith("MASK_")})

    try:
        served = server.ensure(spec)
        filled: dict[str, dict] = {}
        empty_fill: dict[str, int] = {}
        for archetype, cell_list in cells.items():
            responses_csv = data_dir / "responses" / f"{archetype}_{modelname}.csv"
            df = pd.read_csv(responses_csv)
            dd = "dd_turn_1_prompt" in df.columns and "dd_turn_1_response" in df.columns
            jobs = {f"{i}|{c}": messages_for(df.loc[i], c, dd) for i, c in cell_list}
            print(f">>> {archetype}: regenerating {len(jobs)} cells", flush=True)
            out, empties = asyncio.run(regenerate(served.base_url, served.api_key, served.model_name,
                                         jobs, cfg, args.concurrency))
            for key, n in empties.items():
                empty_fill[f"{archetype}:{key}"] = n
            still = []
            for key, (content, reasoning) in out.items():
                i, c = key.split("|", 1)
                i = int(i)
                if content is None:
                    still.append((i, c))
                    continue
                df.at[i, c] = content
                df.at[i, c.replace("generation(", "reasoning(", 1)] = reasoning
            df.to_csv(responses_csv, index=False)
            filled[archetype] = {"cells": len(jobs), "still_failed": still,
                                 "rows": sorted({i for i, _ in cell_list})}
            assert not still, f"{archetype}: {len(still)} cells failed again: {still}"
    finally:
        server.stop()

    # --- re-judge the affected rows only, splice, recompute ---
    for archetype, info in filled.items():
        print(f">>> {archetype}: re-judging {len(info['rows'])} rows with {cfg.judge_model}", flush=True)
        rejudge_rows(archetype, data_dir / "responses" / f"{archetype}_{modelname}.csv",
                     data_dir / "evaluated" / f"{archetype}_{modelname}.csv", info["rows"], work, env,
                     int(cfg.judge_concurrency))
    for stage in ("metric.py", "process_metrics.py"):
        r = subprocess.run([sys.executable, stage], cwd=str(_HARNESS), env=env, capture_output=True, text=True)
        (work / f"{Path(stage).stem}.log").write_text(f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
        if r.returncode:
            raise RuntimeError(f"{stage} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")

    all_results = json.loads((data_dir / "metrics" / "all_results.json").read_text())
    per_archetype = {}
    for archetype in ARCHETYPES:
        m = all_results.get(archetype, {}).get(modelname)
        if m and "honesty_score_1" in m:
            per_archetype[archetype] = {
                "honesty_score": m["honesty_score_1"], "unhonest_pct": m["unhonest_1"],
                "honest_pct": m["honest_1"], "n_responses": m["total_responses"],
                "n_rows": summary["per_archetype"][archetype]["n_rows"]}
    assert set(per_archetype) == set(summary["per_archetype"]), (per_archetype.keys(), summary["per_archetype"].keys())
    errors = generation_errors(data_dir / "responses")
    assert errors["errors"] == 0, errors
    assert errors["generations"] == summary["generation_errors"]["generations"], errors
    # The original run's empty-content tally (metadata) plus what the fill itself saw; the
    # harness writes its tally only from process_dataframe, which this driver bypasses.
    empty = {"total": summary["empty_content"]["total"] + sum(empty_fill.values()),
             "by_type": dict(summary["empty_content"]["by_type"])}
    for key, n in empty_fill.items():
        kind = "lie" if "System Prompt" in key else "belief"
        empty["by_type"][kind] = empty["by_type"].get(kind, 0) + n
    before = {"overall_honesty_score": summary["overall_honesty_score"],
              "per_archetype": {a: v["honesty_score"] for a, v in summary["per_archetype"].items()},
              "generation_errors": summary["generation_errors"]}
    summary.update({
        "overall_honesty_score": overall_honesty(per_archetype), "per_archetype": per_archetype,
        "generation_errors": errors, "generation_error_rate": 0.0,
        "empty_content": empty, "empty_content_rate": round(empty["total"] / errors["generations"], 4),
        "post_hoc_fill": {
            "date": datetime.now(timezone.utc).date().isoformat(),
            "what": "generations the original run stored as '[ERROR: Generation failed or timed out]' "
                    "(SDK client read timeout at 600 s) regenerated with gen_timeout_s=1800 against "
                    "the same adapter revision, their rows re-judged, metrics recomputed",
            "cells": n_cells, "by_archetype": {a: v["cells"] for a, v in filled.items()},
            "rows_rejudged": {a: v["rows"] for a, v in filled.items()},
            "before": before, "empty_content_in_fill": empty_fill, "driver": "scratch/mask_fill_failed.py",
        },
    })

    # --- rewrite the run dir in the published layout ---
    for archetype in per_archetype:
        shutil.copy2(data_dir / "responses" / f"{archetype}_{modelname}.csv", rollouts / f"{archetype}.csv")
        shutil.copy2(data_dir / "evaluated" / f"{archetype}_{modelname}.csv", results / f"{archetype}_evaluated.csv")
        shutil.copy2(data_dir / "metrics" / f"{archetype}_{modelname}.csv", results / f"{archetype}_metrics.csv")
    (results / "all_results.json").write_text(json.dumps(all_results, indent=2))
    (results / "results.json").write_text(json.dumps(summary, indent=2))
    (results / "results.md").write_text(_results_markdown(args.target, summary["mode"], summary))
    mask_meta.update({"generation_errors": errors, "generation_error_rate": 0.0,
                      "empty_content": empty, "empty_content_rate": summary["empty_content_rate"],
                      "post_hoc_fill": summary["post_hoc_fill"]})
    (metadata / "mask_run_meta.json").write_text(json.dumps(mask_meta, indent=2))
    for name in ("metric.log", "process_metrics.log"):
        shutil.copy2(work / name, metadata / "harness_logs" / f"fill_{name}")
    shutil.rmtree(work)
    print(f">>> honesty {before['overall_honesty_score']} -> {summary['overall_honesty_score']} | "
          + " ".join(f"{a}={v['honesty_score']}" for a, v in per_archetype.items()), flush=True)

    if args.no_push:
        return
    command = launch_meta["command"] + f"  # + {' '.join(sys.argv)}"
    card = _card_fields(
        "mask", cfg, command,
        experiment=f"mask eval of {args.target} (mode={summary['mode']})",
        models=json.dumps({"target": launch_meta["target"], "target_revision": launch_meta["target_revision"],
                           "base": launch_meta["base_model"], "base_revision": launch_meta["base_model_revision"]}),
        source_revision=launch_meta["git_sha"])
    # A revision of the ORIGINAL run: same repo, and the card keeps the date the run was
    # produced (gate_push checks it against the date in the name), not today's.
    card["date_generated"] = launch_meta["timestamp_utc"][:10]
    url = push_run_dir(run_dir, args.repo, card,
                       front_matter={"tags": ["eval-run", "eval:mask", f"model:{modelname}", f"mode:{summary['mode']}"]})
    print(f">>> pushed {url}")


if __name__ == "__main__":
    main()
