# ABOUTME: Eval-framework entrypoint for one Arena-Hard ARM: get this model's answers —
# ABOUTME: generated, or fetched from a prior run — and publish them. Judging is pool.py.

"""One Arena-Hard arm, which is a set of answers and nothing else.

Arena-Hard is a comparison, so a single arm has no result: a win rate is a fact about
(arm, baseline, exam), never about a model on its own. An arm therefore publishes
`rollouts/` (its answers) and `metadata/` (where they came from), and no `results/` at
all. Every judgment lives in the comparison, `<date>-ah-vs-<baseline>` (pool.py).

That is what makes an arm reusable. Its answers are an artifact in their own right, so the
same model is a target this week and the baseline next, and `run()` does not care which:

* the target is a MODEL (`org/2026-09-04-qwen36-difficult-advice-0`) — generate, serve,
  and publish the answers;
* the target is a PRIOR ARM (`org/2026-09-05-ah-qwen36-difficult-advice-0`) — the answers
  already exist, so fetch them and start no server at all.

The baseline is named by `--reference` and is an ordinary arm: run_eval runs it first
(`EvalSpec.arm_kwargs`), and the only thing that marks it out is a line in its own
metadata saying so, which is how pool.py later knows what everything was measured against.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.capabilities.arena_hard import arena_hard_gen
from src.eval.layout import publish_layout
from src.infra.huggingface import hf_api, hf_download


def answers_from_run(repo: str, dest: Path) -> dict:
    """Place a prior arm's published answers where the vendored bench reads them.

    Args:
        repo: HF dataset repo of a prior arm of this eval (its `rollouts/answers.jsonl`).
        dest: `<vendor_dir>/data/<bench>/model_answer/<arm>.jsonl`.

    Returns:
        `{"repo", "revision"}` — pinned to the exact commit, so a pointer published today
        cannot come to mean a different answer set later.
    """
    sha = hf_api().repo_info(repo, repo_type="dataset").sha
    src = Path(hf_download(repo, "rollouts/answers.jsonl", repo_type="dataset",
                           revision=sha))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)
    return {"repo": repo, "revision": sha}


def register(arms: list[dict], name: str, adapter: str, role: str, cfg) -> list[dict]:
    """Add an arm to the config's ladder if it is not already declared there.

    A CLI arm is dynamic, so it carries none of the per-arm prompt counts the static
    ladder spells out — and the judge reads `n_hard_prompt` off the arm it is judging.
    `arm_defaults` supplies them, which is also the fix for a target appended without
    them crashing the judge with `Missing key n_hard_prompt`.
    """
    if any(a["name"] == name for a in arms):
        return arms
    defaults = OmegaConf.to_container(cfg.arm_defaults, resolve=True)
    return arms + [{**defaults, "name": name, "adapter": adapter, "role": role,
                    "synthetic_fraction": None}]


def bench_answers_dir(cfg) -> Path:
    """`<vendor_dir>/data/<bench>/model_answer` — where the harness reads answers."""
    return Path(str(cfg.vendor_dir)) / "data" / str(cfg.bench_name) / "model_answer"


def run(target, cfg, out_dir: Path, *, reference: str = "") -> dict:
    """Produce one arm's answers (CLAUDE.md contract).

    Args:
        target: The arm. `target.spec.answers` set means its generations already exist,
            and nothing here serves a model.
        cfg: The eval config.
        out_dir: This arm's run directory.
        reference: HF path naming the baseline of the comparison this arm belongs to.
            Recorded, not used: an arm is not judged here.

    Returns:
        What this arm produced and where it came from. No scores — see the module
        docstring.
    """
    cfg = OmegaConf.merge(cfg)  # private copy
    arm_name = target.spec.model_key
    assert reference, (
        "arena_hard is a comparison: pass --reference <hf path> (a model, or a prior "
        "arena_hard arm whose answers are reused). run_eval runs it first, as an ordinary "
        "arm, and pool.py judges everything against it.")

    _, _, metadata_dir = publish_layout(out_dir)
    rollouts_dir = out_dir / "rollouts"
    arms = register(OmegaConf.to_container(cfg.arms, resolve=True),
                    arm_name, target.spec.hf_path, "target", cfg)
    cfg.arms = arms
    cfg.output_dir = str(out_dir)
    cfg_path = metadata_dir / "arena_hard_config.yaml"
    OmegaConf.save(cfg, cfg_path)

    bench = bench_answers_dir(cfg)
    if target.spec.answers:
        source = answers_from_run(target.spec.answers, bench / f"{arm_name}.jsonl")
        print(f">>> answers reused from {target.spec.answers} — nothing to generate")
    else:
        arena_hard_gen.main(config=str(cfg_path), arm=arm_name,
                            served_model=target.model_name, endpoint=target.base_url,
                            api_key=target.api_key, smoke=bool(cfg.get("smoke", False)))
        gen_dir = max((out_dir / arm_name).glob("*/"), key=lambda p: p.name)
        source = {"generated": target.spec.hf_path}
        # Generation health (empty-think rate, token counts) is a fact about producing
        # these answers, not a result of any comparison, so it travels with them.
        for name in ("gen_metrics.json", "raw_samples.md", "run_meta.json"):
            if (gen_dir / name).exists():
                (gen_dir / name).rename(metadata_dir / f"gen_{name}")
        shutil.rmtree(out_dir / arm_name)

    # COPY the answers out — the vendor tree's original is a resume/staging cache read
    # back by exact path, so it must stay where it is.
    shutil.copy2(bench / f"{arm_name}.jsonl", rollouts_dir / "answers.jsonl")
    is_reference = target.spec.hf_path == reference
    (metadata_dir / "sources.json").write_text(json.dumps(
        {"arm": arm_name, "answers": source, "reference_arm": is_reference,
         "reference": reference}, indent=2))
    return {"arm": arm_name, "reference_arm": is_reference,
            "answers": "rollouts/answers.jsonl", "sources": "metadata/sources.json"}
