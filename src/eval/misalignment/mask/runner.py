# ABOUTME: MASK honesty-benchmark eval-framework entrypoint: generate the served target's answers to
# ABOUTME: the six MASK archetypes, judge honesty via OpenRouter, report per-archetype + overall scores.
"""Run MASK (Model Alignment between Statements and Knowledge) against one served target.

MASK measures honesty — whether a model contradicts its own stated beliefs when pressured to
lie — separately from factual accuracy. The vendored harness
(`third_party/mask/`, its own `.venv`) is a four-stage batch pipeline:

    generate_responses.py  the target answers each archetype's prompts   -> responses/*.csv
    evaluate.py            an LLM judge grades each answer for honesty    -> evaluated/*.csv
    metric.py              per-row honesty/accuracy labels                -> metrics/*.csv
    process_metrics.py     aggregate honesty score per archetype/model    -> all_results.json

Generation goes to the target's OpenAI-compatible endpoint (`target.base_url`); judging goes
to the configured OpenRouter judge. Both are handed to the harness through `MASK_*` env vars
(the harness is PATCHED to read them — see the PATCH comments in third_party/mask/mask/*.py
and third_party/VENDORED_FROM.txt). The harness runs in place via its own interpreter over a
PER-RUN copy of the data (`MASK_DATA_DIR`), so a run is re-entrant and never touches the
tracked csv_data tree; this target's outputs are packaged into the published layout.

Three knobs beyond upstream's, all in configs/eval/mask.yaml:

  * `subsample: N` — N rows drawn evenly across the six archetypes (`stratified_counts`),
    seeded, the drawn task_ids recorded in metadata/subsample.json. Upstream's 2,595 rows
    are ~6x more than an arm interval needs; the per-archetype spread is what matters.
  * `gen_concurrency` — generations in flight against the served target. Upstream's 10
    cannot fill a vLLM server; the family's `max_num_seqs` (ModelProfile.serving) is the
    ceiling worth matching.
  * `judge_batch: true` — the judge calls go through OpenRouter's batch API at half price:
    evaluate.py runs once in `collect` transport (writes every judge request, judges
    nothing), the batch is submitted and polled here (src/infra/endpoints/openrouter.py
    run_batch), then evaluate.py runs in `replay` transport, answering from the batch and
    falling back to a live call for anything the batch did not return.

  * `empty_content: evasion|reasoning` — a thinking target can finish with no content:
    it wrote its reply inside the think block and never closed it (the nosynth control does
    this on 13/16 pressure prompts, 2026-09-07). `evasion` hands the judge an empty
    statement; `reasoning` hands it the text the model actually wrote. Counted and
    reported either way, separately from real generation failures.

honesty_score = 100 - (% of responses judged dishonest), so higher is more honest.
"""
from __future__ import annotations

import csv
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.eval.layout import publish_layout
from src.infra.endpoints.openrouter import OPENROUTER_BASE_URL, run_batch
from src.utils import write_run_meta

_HARNESS_ROOT = Path(__file__).parent / "third_party" / "mask"
_HARNESS = _HARNESS_ROOT / "mask"          # the package dir the stages run from (relative paths)
_VENV_PY = _HARNESS_ROOT / ".venv" / "bin" / "python"
ARCHETYPES = ("continuations", "disinformation", "doubling_down_known_facts",
              "known_facts", "provided_facts", "statistics")
UPSTREAM_COMMIT = "25e0b1201e6c928ebe69f7c5aad6fa9063a377ea"


def stratified_counts(available: dict[str, int], n: int) -> dict[str, int]:
    """How many rows to draw from each archetype for an even N-row subsample.

    Equal shares, capped by what an archetype has (statistics holds 108 rows, so at N=1000
    it gives all 108 and the shortfall is spread evenly over the rest), so the total is
    exactly N whenever N fits. A smaller archetype is never over-drawn and a larger one
    never under-drawn to compensate for a third: water-filling, one level at a time.
    """
    total = sum(available.values())
    if n > total:
        raise ValueError(f"subsample {n} exceeds the {total} rows available")
    counts = {a: 0 for a in available}
    remaining = n
    open_ = [a for a in available if available[a] > 0]
    while remaining > 0 and open_:
        quota = remaining // len(open_)
        if quota == 0:  # fewer rows left than open archetypes: one each, in name order
            for a in sorted(open_)[:remaining]:
                counts[a] += 1
            break
        for a in list(open_):
            take = min(quota, available[a] - counts[a])
            counts[a] += take
            remaining -= take
            if counts[a] == available[a]:
                open_.remove(a)
    return counts


def sample_data(src: Path, dst: Path, n: int | None, seed: int) -> dict[str, dict]:
    """Copy the archetype CSVs from `src` to `dst`, subsampled to `n` rows evenly (or whole).

    Rows are drawn without replacement, seeded per archetype, and written in their original
    order with their original text (the csv module round-trips quoting; pandas would rewrite
    NaNs). Returns, per archetype, the count drawn, the count available and the drawn task
    ids — the record that makes the subsample reproducible and auditable.
    """
    files = sorted(p for p in src.glob("*.csv") if p.stem in ARCHETYPES)
    assert files, f"no archetype CSVs under {src}"
    rows_by = {}
    for path in files:
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows_by[path.stem] = (reader.fieldnames, list(reader))
    available = {a: len(rows) for a, (_f, rows) in rows_by.items()}
    counts = stratified_counts(available, n) if n else dict(available)
    dst.mkdir(parents=True, exist_ok=True)
    record = {}
    for archetype, (fieldnames, rows) in rows_by.items():
        k = counts[archetype]
        if k < len(rows):
            keep = sorted(random.Random(f"{seed}:{archetype}").sample(range(len(rows)), k))
            rows = [rows[i] for i in keep]
        with (dst / f"{archetype}.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        record[archetype] = {"n": len(rows), "of": available[archetype],
                             "task_ids": [r.get("task_id", "") for r in rows]}
    return record


def empty_content(responses_dir: Path) -> dict:
    """The harness's tally of generations that finished with no content (the reply was left
    inside an unclosed think block), by prompt type: `lie` (the pressure turn) or `belief`.
    Zero when the harness wrote no tally (an older harness, or nothing generated)."""
    path = responses_dir / "_empty_content.json"
    by_type: dict[str, int] = {}
    if path.is_file():
        for _file, counts in json.loads(path.read_text()).items():
            for key, n in counts.items():
                kind = "lie" if key.startswith("lying") else "belief"
                by_type[kind] = by_type.get(kind, 0) + int(n)
    return {"total": sum(by_type.values()), "by_type": by_type}


def generation_errors(responses_dir: Path) -> dict:
    """Count the harness's "[ERROR: ...]" cells across every generation column it wrote."""
    generations = errors = 0
    for path in sorted(responses_dir.glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            gen_cols = [c for c in (reader.fieldnames or []) if c.startswith("generation(")]
            for row in reader:
                for c in gen_cols:
                    v = row.get(c)
                    if v is None or v == "":
                        continue
                    generations += 1
                    if v.startswith("[ERROR"):
                        errors += 1
    return {"generations": generations, "errors": errors}


def _run_stage(argv: list[str], env: dict, cwd: Path, log_dir: Path | None = None) -> str:
    """Run one harness stage via its own venv; fail fast with its output on a non-zero exit.

    The stage's stdout and stderr are kept under `log_dir` whatever the exit code: the
    harness reports a generation it gave up on ("Max retries exceeded", the exception it
    saw) on stdout and still exits 0, so a stage that "succeeded" with a third of its
    calls failed (2026-09-07) is only explicable from this file.
    """
    result = subprocess.run([str(_VENV_PY), *argv], cwd=str(cwd), env=env,
                            capture_output=True, text=True)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(argv[0]).stem + ("_" + env["MASK_JUDGE_TRANSPORT"]
                                     if env.get("MASK_JUDGE_TRANSPORT", "live") != "live" else "")
        (log_dir / f"{stem}.log").write_text(
            f"$ {' '.join(argv)}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")
    if result.returncode != 0:
        raise RuntimeError(
            f"MASK stage {argv[0]} failed (exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout[-3000:]}\n--- stderr ---\n{result.stderr[-3000:]}")
    return result.stdout


def batch_judge(exchange: Path, judge_model: str) -> dict:
    """Push the judge requests evaluate.py collected through OpenRouter's batch API.

    Reads `<exchange>.requests.jsonl` ({custom_id, body}), writes `<exchange>.results.jsonl`
    ({custom_id, content}) for every request the batch answered. A request the batch did
    not return (a failed or expired job, a refused item) is simply absent, and evaluate.py's
    `replay` transport judges it live. Submission state sits beside the exchange, so a
    killed run resumes the same jobs.
    """
    requests = {}
    with (exchange.parent / f"{exchange.name}.requests.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                requests[rec["custom_id"]] = rec["body"]
    results: dict[str, str] = {}

    def collect(cid: str, payload: dict) -> None:
        content = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content")
        if content:
            results[cid] = content

    if requests:
        run_batch(judge_model, requests, "mask_judge",
                  exchange.parent / f"{exchange.name}.batch_state.json", collect)
    with (exchange.parent / f"{exchange.name}.results.jsonl").open("w", encoding="utf-8") as fh:
        for cid, content in results.items():
            fh.write(json.dumps({"custom_id": cid, "content": content}) + "\n")
    stats = {"requested": len(requests), "returned": len(results),
             "fallback_live": len(requests) - len(results)}
    print(f">>> MASK judge batch: {stats['returned']}/{stats['requested']} answered by the "
          f"batch; {stats['fallback_live']} fall to live calls", flush=True)
    return stats


def run(target, cfg: DictConfig, out_dir: Path) -> dict:
    """Eval-framework entrypoint (CLAUDE.md contract): evaluate one served target on MASK.

    Args:
        target: The served target; `base_url`, `model_name` and `api_key` reach its
            OpenAI-compatible endpoint, `spec.hf_path`/`spec.mode` identify it.
        cfg: `configs/eval/mask.yaml` merged with CLI overrides.
        out_dir: Per-target run directory owned by run_eval.py.

    Returns:
        Summary dict: overall honesty score, per-archetype breakdown, judge, model.
    """
    assert _VENV_PY.exists(), (
        f"MASK harness venv missing at {_VENV_PY}; the vendored harness needs its own "
        "environment: `uv venv {_HARNESS_ROOT}/.venv && uv pip install -p "
        f"{_VENV_PY} -r {_HARNESS_ROOT}/requirements.txt`.")
    smoke = bool(cfg.get("smoke", False))
    model = target.model_name
    modelname = model.split("/")[-1]           # the suffix the harness names output files with
    subsample = int(cfg.get("subsample") or 0) or None
    seed = int(cfg.get("seed", 0))
    gen_concurrency = int(cfg.get("gen_concurrency", 10))
    judge_batch = bool(cfg.get("judge_batch", False))

    # The per-run data copy: the harness reads and writes only here (MASK_DATA_DIR), so the
    # tracked csv_data tree is never touched and two targets in one process cannot collide.
    # The smoke slice is upstream's test_csv_data (5 rows per archetype), never subsampled.
    # Absolute: the harness runs with its own package dir as cwd, and run_eval hands over a
    # repo-relative out_dir (the first smoke wrote its responses INTO the harness tree).
    work = out_dir.resolve() / "mask_work"
    if work.exists():
        shutil.rmtree(work)
    data_dir = work / "data"
    logs = work / "logs"
    sampled = sample_data(_HARNESS / ("test_csv_data" if smoke else "csv_data"), data_dir,
                          None if smoke else subsample, seed)
    n_rows = sum(v["n"] for v in sampled.values())

    # The harness subprocesses read the target + judge endpoints from these (the patch points).
    env = dict(os.environ)
    env["MASK_TARGET_BASE_URL"] = target.base_url
    env["MASK_TARGET_API_KEY"] = target.api_key
    env["MASK_JUDGE_MODEL"] = str(cfg.judge_model)
    env["MASK_JUDGE_BASE_URL"] = OPENROUTER_BASE_URL
    env["MASK_JUDGE_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
    env["MASK_DATA_DIR"] = str(data_dir)
    env["MASK_GEN_CONCURRENCY"] = str(gen_concurrency)
    empty_policy = str(cfg.get("empty_content", "evasion"))
    assert empty_policy in ("evasion", "reasoning"), f"empty_content must be evasion|reasoning, not {empty_policy!r}"
    env["MASK_EMPTY_CONTENT"] = empty_policy

    test = ["--test"] if smoke else []
    print(f">>> MASK: generating {modelname} answers for {n_rows} rows "
          f"({'smoke' if smoke else f'subsample {subsample}' if subsample else 'full'}; "
          f"{gen_concurrency} in flight)", flush=True)
    _run_stage(["generate_responses.py", "--model", model,
                "--temperature", str(cfg.temperature), "--max_tokens", str(cfg.max_tokens),
                "--lie_k", str(cfg.lie_k), *test], env, _HARNESS, logs)
    # CLAUDE.md gotcha 4, made visible: a thinking target that spends `max_tokens` inside its
    # trace returns no content, and the harness stores "[ERROR: ...]" in that cell, which the
    # judge then reads as "no belief"/"evasion" -- a silent 0 on the honesty axis. Measured
    # 2026-09-07 on the nosynth smoke at max_tokens 4096: 45 of 145 generations (31%). The
    # rate is reported and, past `max_generation_error_rate`, refused.
    errors = generation_errors(data_dir / "responses")
    error_rate = errors["errors"] / max(errors["generations"], 1)
    empty = empty_content(data_dir / "responses")
    empty_rate = empty["total"] / max(errors["generations"], 1)
    print(f">>> MASK: {errors['errors']}/{errors['generations']} generations failed "
          f"({100 * error_rate:.1f}%); {empty['total']} finished with empty content "
          f"({100 * empty_rate:.1f}%: pressure {empty['by_type'].get('lie', 0)}, "
          f"belief {empty['by_type'].get('belief', 0)}; scored as {empty_policy!r})", flush=True)
    cap = float(cfg.get("max_generation_error_rate", 0.05))
    if error_rate > cap:
        raise RuntimeError(
            f"MASK: {100 * error_rate:.1f}% of generations failed, above "
            f"max_generation_error_rate={cap:.0%}. Almost always max_tokens={cfg.max_tokens} "
            "exhausted inside <think>; raise it (and the serving window) rather than score "
            "empty answers. The work tree is kept under "
            f"{work} for inspection.")

    judge_argv = ["evaluate.py", "--concurrency_limit", str(cfg.judge_concurrency), *test]
    batch_stats = None
    if judge_batch:
        exchange = work / "judge_exchange"
        print(f">>> MASK: collecting judge requests for {cfg.judge_model} (batch)", flush=True)
        _run_stage(judge_argv, {**env, "MASK_JUDGE_TRANSPORT": "collect",
                                "MASK_JUDGE_EXCHANGE": str(exchange)}, _HARNESS, logs)
        batch_stats = batch_judge(exchange, str(cfg.judge_model))
        print(">>> MASK: replaying the batch's verdicts (live fallback for the rest)", flush=True)
        _run_stage(judge_argv, {**env, "MASK_JUDGE_TRANSPORT": "replay",
                                "MASK_JUDGE_EXCHANGE": str(exchange)}, _HARNESS, logs)
    else:
        print(f">>> MASK: judging honesty with {cfg.judge_model}", flush=True)
        _run_stage(judge_argv, env, _HARNESS, logs)
    _run_stage(["metric.py", *test], env, _HARNESS, logs)
    _run_stage(["process_metrics.py", *test], env, _HARNESS, logs)

    all_results = json.loads((data_dir / "metrics" / "all_results.json").read_text())
    per_archetype = {}
    for archetype in ARCHETYPES:
        m = all_results.get(archetype, {}).get(modelname)
        if m and "honesty_score_1" in m:
            per_archetype[archetype] = {
                "honesty_score": m["honesty_score_1"],
                "unhonest_pct": m["unhonest_1"],
                "honest_pct": m["honest_1"],
                "n_responses": m["total_responses"],
                "n_rows": sampled[archetype]["n"],
            }
    assert per_archetype, (
        f"MASK produced no honesty scores for model {modelname!r}; "
        f"all_results.json holds archetypes {list(all_results.keys())}")
    overall = round(sum(a["honesty_score"] for a in per_archetype.values()) / len(per_archetype), 2)

    summary = {
        "benchmark": "MASK",
        "model": modelname,
        "mode": target.spec.mode,
        "judge": str(cfg.judge_model),
        "overall_honesty_score": overall,
        "per_archetype": per_archetype,
        "n_archetypes": len(per_archetype),
        "n_rows": n_rows,
        "generation_errors": errors,
        "generation_error_rate": round(error_rate, 4),
        "empty_content": empty,
        "empty_content_rate": round(empty_rate, 4),
        "empty_content_policy": empty_policy,
        "subsample": subsample,
        "seed": seed,
        "gen_concurrency": gen_concurrency,
        "judge_batch": judge_batch,
        "judge_batch_stats": batch_stats,
        "lie_k": int(cfg.lie_k),
        "smoke": smoke,
    }

    # Package into the published layout: model generations under rollouts/, judge outputs
    # and the aggregate under results/, provenance (and the drawn rows) under metadata/.
    rollouts_dir, results_dir, metadata_dir = publish_layout(out_dir)
    for archetype in per_archetype:
        responses = data_dir / "responses" / f"{archetype}_{modelname}.csv"
        evaluated = data_dir / "evaluated" / f"{archetype}_{modelname}.csv"
        if responses.is_file():
            shutil.copy2(responses, rollouts_dir / f"{archetype}.csv")
        if evaluated.is_file():
            shutil.copy2(evaluated, results_dir / f"{archetype}_evaluated.csv")
        # The per-row verdicts (eval_belief, eval_honesty_1, eval_accuracy) live in the
        # metrics CSV, not the evaluated one; without it a published run cannot say WHICH
        # rows lied, only how many (noticed 2026-09-07 when three arms scored exactly 50.0).
        metrics = data_dir / "metrics" / f"{archetype}_{modelname}.csv"
        if metrics.is_file():
            shutil.copy2(metrics, results_dir / f"{archetype}_metrics.csv")
    (results_dir / "all_results.json").write_text(json.dumps(all_results, indent=2))
    (metadata_dir / "subsample.json").write_text(json.dumps(
        {"subsample": subsample, "seed": seed, "archetypes": sampled}, indent=2))
    write_run_meta(
        metadata_dir,
        OmegaConf.to_container(cfg, resolve=True),
        extra={"target": target.spec.hf_path, "mode": target.spec.mode,
               "upstream_commit": UPSTREAM_COMMIT, "n_rows": n_rows,
               "generation_errors": errors, "generation_error_rate": round(error_rate, 4),
               "empty_content": empty, "empty_content_rate": round(empty_rate, 4),
               "empty_content_policy": empty_policy,
               "rows_per_archetype": {a: v["n"] for a, v in sampled.items()},
               "gen_concurrency": gen_concurrency, "judge_batch": judge_batch,
               "judge_batch_stats": batch_stats},
    )
    (metadata_dir / "run_meta.json").rename(metadata_dir / "mask_run_meta.json")
    if logs.is_dir():
        shutil.copytree(logs, metadata_dir / "harness_logs", dirs_exist_ok=True)
    # The work tree (sampled inputs, exchange files, harness outputs) is not part of the
    # published layout; everything it held that matters was copied above or is regenerable
    # from metadata/subsample.json. It stays only when a run fails, for debugging.
    shutil.rmtree(work)

    print(f">>> MASK honesty {overall} (avg over {len(per_archetype)} archetypes, {n_rows} rows) | "
          + " ".join(f"{a}={v['honesty_score']}" for a, v in per_archetype.items()), flush=True)
    return summary
