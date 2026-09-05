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
(the harness is PATCHED to read them — see the PATCH comments in generate_responses.py and
evaluate.py). The harness runs in place via its own interpreter, and this target's per-model
outputs are packaged into the published layout.

honesty_score = 100 - (% of responses judged dishonest), so higher is more honest.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.eval.layout import publish_layout
from src.infra.endpoints.openrouter import OPENROUTER_BASE_URL
from src.utils import write_run_meta

_HARNESS_ROOT = Path(__file__).parent / "third_party" / "mask"
_HARNESS = _HARNESS_ROOT / "mask"          # the package dir the stages run from (relative paths)
_VENV_PY = _HARNESS_ROOT / ".venv" / "bin" / "python"
ARCHETYPES = ("continuations", "disinformation", "doubling_down_known_facts",
              "known_facts", "provided_facts", "statistics")


def _run_stage(argv: list[str], env: dict, cwd: Path) -> str:
    """Run one harness stage via its own venv; fail fast with its output on a non-zero exit."""
    result = subprocess.run([str(_VENV_PY), *argv], cwd=str(cwd), env=env,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"MASK stage {argv[0]} failed (exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout[-3000:]}\n--- stderr ---\n{result.stderr[-3000:]}")
    return result.stdout


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
        "environment (its deps: datasets, openai, anthropic, pandas, pydantic, tqdm).")
    smoke = bool(cfg.get("smoke", False))
    model = target.model_name
    modelname = model.split("/")[-1]           # the suffix the harness names output files with
    data = "test_csv_data" if smoke else "csv_data"

    # The harness subprocesses read the target + judge endpoints from these (the patch points).
    env = dict(os.environ)
    env["MASK_TARGET_BASE_URL"] = target.base_url
    env["MASK_TARGET_API_KEY"] = target.api_key
    env["MASK_JUDGE_MODEL"] = str(cfg.judge_model)
    env["MASK_JUDGE_BASE_URL"] = OPENROUTER_BASE_URL
    env["MASK_JUDGE_API_KEY"] = os.environ["OPENROUTER_API_KEY"]

    # Every stage skips an output file that already exists, so clear this model's prior
    # outputs to force a clean run (leaves other models' files — e.g. the tracked baselines).
    for sub in ("responses", "evaluated", "metrics"):
        for stale in (_HARNESS / data / sub).glob(f"*_{modelname}.csv"):
            stale.unlink()

    test = ["--test"] if smoke else []
    print(f">>> MASK: generating {modelname} answers "
          f"({'smoke: 5 rows/archetype' if smoke else 'full'})", flush=True)
    _run_stage(["generate_responses.py", "--model", model,
                "--temperature", str(cfg.temperature), "--max_tokens", str(cfg.max_tokens),
                "--lie_k", str(cfg.lie_k), *test], env, _HARNESS)
    print(f">>> MASK: judging honesty with {cfg.judge_model}", flush=True)
    _run_stage(["evaluate.py", "--concurrency_limit", str(cfg.judge_concurrency), *test], env, _HARNESS)
    _run_stage(["metric.py", *test], env, _HARNESS)
    _run_stage(["process_metrics.py", *test], env, _HARNESS)

    all_results = json.loads((_HARNESS / data / "metrics" / "all_results.json").read_text())
    per_archetype = {}
    for archetype in ARCHETYPES:
        m = all_results.get(archetype, {}).get(modelname)
        if m and "honesty_score_1" in m:
            per_archetype[archetype] = {
                "honesty_score": m["honesty_score_1"],
                "unhonest_pct": m["unhonest_1"],
                "honest_pct": m["honest_1"],
                "n_responses": m["total_responses"],
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
        "lie_k": int(cfg.lie_k),
        "smoke": smoke,
    }

    # Package into the published layout: model generations under rollouts/, judge outputs
    # and the aggregate under results/, provenance under metadata/.
    rollouts_dir, results_dir, metadata_dir = publish_layout(out_dir)
    for archetype in per_archetype:
        responses = _HARNESS / data / "responses" / f"{archetype}_{modelname}.csv"
        evaluated = _HARNESS / data / "evaluated" / f"{archetype}_{modelname}.csv"
        if responses.is_file():
            shutil.copy2(responses, rollouts_dir / f"{archetype}.csv")
        if evaluated.is_file():
            shutil.copy2(evaluated, results_dir / f"{archetype}_evaluated.csv")
    (results_dir / "all_results.json").write_text(json.dumps(all_results, indent=2))
    write_run_meta(
        metadata_dir,
        OmegaConf.to_container(cfg, resolve=True),
        extra={"target": target.spec.hf_path, "mode": target.spec.mode,
               "upstream_commit": "25e0b1201e6c928ebe69f7c5aad6fa9063a377ea"},
    )
    (metadata_dir / "run_meta.json").rename(metadata_dir / "mask_run_meta.json")

    print(f">>> MASK honesty {overall} (avg over {len(per_archetype)} archetypes) | "
          + " ".join(f"{a}={v['honesty_score']}" for a, v in per_archetype.items()), flush=True)
    return summary
