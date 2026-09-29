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
import hashlib
import shutil
from pathlib import Path

from omegaconf import OmegaConf
from huggingface_hub.errors import EntryNotFoundError

from src.eval.capabilities.arena_hard import arena_hard_gen
from src.eval.layout import publish_layout
from src.infra.huggingface import hf_api, hf_download
from src.utils import read_jsonl


PROTOCOL_VERSION = 1


def _generation_settings(cfg, mode: str) -> dict:
    return {"thinking_mode": mode,
            "temperature": float(cfg.generation.temperature),
            "top_p": float(cfg.generation.top_p),
            "max_tokens": int(cfg.generation.max_tokens),
            "configured_context_window": int(cfg.serving.context_window),
            "answer_extraction": {"parser": "split-think-visible-answer-v1",
                                  "strip_think_for_judging": bool(cfg.generation.strip_think_for_judging)}}


def _questions(cfg, arm_name: str) -> list[dict]:
    rows = read_jsonl(Path(str(cfg.vendor_dir)) / "data" / str(cfg.bench_name) / "question.jsonl")
    return arena_hard_gen._select_questions(rows, arena_hard_gen._arm(cfg, arm_name))


def _question_identity(question: dict) -> dict:
    return {"category": question["category"],
            "prompt_sha256": hashlib.sha256(question["prompt"].encode("utf-8")).hexdigest()}


def _answer_records(path: Path) -> dict:
    rows = read_jsonl(path)
    records = {str(row["uid"]): row for row in rows}
    if len(rows) != len(records):
        raise ValueError(f"Arena answers have duplicate question IDs: {path}")
    return records


def _validate_arm_protocol(protocol: dict, cfg, mode: str, arm_name: str, answer_file: Path) -> None:
    if protocol.get("schema_version") != PROTOCOL_VERSION or protocol.get("status") != "recorded":
        raise ValueError(f"Arena {arm_name}: generation protocol is missing or uncertified")
    expected = _generation_settings(cfg, mode)
    actual = protocol.get("settings", {})
    mismatches = [key for key in expected if actual.get(key) != expected[key]]
    if mismatches:
        raise ValueError(f"Arena {arm_name}: incompatible generation settings: {', '.join(mismatches)}")
    context = protocol.get("effective_context_window")
    if not isinstance(context, int) or context <= 0:
        raise ValueError(f"Arena {arm_name}: effective generation context was not recorded")
    if protocol.get("output_budget_policy") != arena_hard_gen.OUTPUT_BUDGET_POLICY:
        raise ValueError(f"Arena {arm_name}: output budget policy differs or is missing")
    if protocol.get("bench_name") != str(cfg.bench_name):
        raise ValueError(f"Arena {arm_name}: benchmark identity differs")
    records = _answer_records(answer_file)
    questions = _questions(cfg, arm_name)
    if not questions:
        raise ValueError(f"Arena {arm_name}: no requested comparison questions")
    for question in questions:
        uid = str(question["uid"])
        item = protocol.get("questions", {}).get(uid)
        if not isinstance(item, dict) or item.get("identity") != _question_identity(question):
            raise ValueError(f"Arena {arm_name}: prompt identity missing or changed for {uid}")
        if not isinstance(item.get("max_tokens"), int) or item["max_tokens"] <= 0:
            raise ValueError(f"Arena {arm_name}: actual output budget missing for {uid}")
        record = records.get(uid)
        if not record:
            raise ValueError(f"Arena {arm_name}: requested answer is missing for {uid}")
        messages = record.get("messages") or []
        if not messages or messages[0].get("role") != "user" or messages[0].get("content") != question["prompt"]:
            raise ValueError(f"Arena {arm_name}: answer contains a different prompt for {uid}")
        generation = record.get("generation_record") or {}
        if generation.get("max_tokens") != item["max_tokens"]:
            raise ValueError(f"Arena {arm_name}: answer/protocol output budget differs for {uid}")


def generation_protocol(cfg, mode: str, arm_name: str, answer_file: Path, metrics: dict) -> dict:
    """Record measured generation settings and each prompt's actual output allowance."""
    records = _answer_records(answer_file)
    questions = {}
    for question in _questions(cfg, arm_name):
        uid = str(question["uid"])
        questions[uid] = {"identity": _question_identity(question),
                          "max_tokens": records.get(uid, {}).get("generation_record", {}).get("max_tokens")}
    protocol = {"schema_version": PROTOCOL_VERSION, "status": "recorded",
                "settings": _generation_settings(cfg, mode), "bench_name": str(cfg.bench_name),
                "effective_context_window": metrics.get("effective_context_window"),
                "output_budget_policy": metrics.get("output_budget_policy"), "questions": questions}
    _validate_arm_protocol(protocol, cfg, mode, arm_name, answer_file)
    return protocol


def validate_generation_protocols(runs: list[dict], cfg) -> dict:
    """Refuse incomparable or historical unrecorded generations before paid judging.

    Extra cached questions are allowed, but every requested comparison question must
    have the same prompt and actual generation allowance in every participating arm.
    """
    protocols = {}
    for run in runs:
        key, directory = run["model_key"], Path(run["out_dir"])
        path = directory / "metadata" / "generation_protocol.json"
        if not path.exists():
            raise ValueError(f"Arena {key}: historical generation protocol missing; compatibility is uncertified")
        protocol = json.loads(path.read_text(encoding="utf-8"))
        _validate_arm_protocol(protocol, cfg, str(run["mode"]), key, directory / "rollouts" / "answers.jsonl")
        protocols[key] = protocol
    contexts = {p["effective_context_window"] for p in protocols.values()}
    settings = {json.dumps(p["settings"], sort_keys=True) for p in protocols.values()}
    if len(contexts) != 1 or len(settings) != 1:
        raise ValueError("Arena generation protocols differ across arms (thinking, sampling, context or extraction)")
    requested = {str(q["uid"]) for run in runs for q in _questions(cfg, run["model_key"])}
    for uid in requested:
        identities = [p["questions"].get(uid) for p in protocols.values()]
        if any(item is None for item in identities) or any(item != identities[0] for item in identities[1:]):
            raise ValueError(f"Arena generations have different prompt coverage or output budgets for {uid}")
    return {"status": "matched", "schema_version": PROTOCOL_VERSION,
            "arms": sorted(protocols), "n_questions": len(requested),
            "settings": next(iter(protocols.values()))["settings"],
            "effective_context_window": next(iter(contexts))}


def answers_from_run(repo: str, dest: Path, *, revision: str | None = None,
                     metadata_dir: Path | None = None) -> dict:
    """Place a prior arm's published answers where the vendored bench reads them.

    Args:
        repo: HF dataset repo of a prior arm of this eval (its `rollouts/answers.jsonl`).
        dest: `<vendor_dir>/data/<bench>/model_answer/<arm>.jsonl`.

    Returns:
        `{"repo", "revision"}` — pinned to the exact commit, so a pointer published today
        cannot come to mean a different answer set later.
    """
    sha = revision or hf_api().repo_info(repo, repo_type="dataset").sha
    try:
        protocol_file = Path(hf_download(repo, "metadata/generation_protocol.json",
                                        repo_type="dataset", revision=sha))
    except EntryNotFoundError as exc:
        raise ValueError(f"Historical Arena answers {repo}@{sha} have no generation protocol; "
                         "compatibility is uncertified. Generate matched answers before comparing.") from exc
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    src = Path(hf_download(repo, "rollouts/answers.jsonl", repo_type="dataset",
                           revision=sha))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)
    if metadata_dir is not None:
        metadata_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(protocol_file, metadata_dir / "generation_protocol.json")
        for name in ("gen_gen_metrics.json", "gen_run_meta.json"):
            try:
                metric = Path(hf_download(repo, f"metadata/{name}", repo_type="dataset", revision=sha))
            except EntryNotFoundError:
                continue  # absence remains explicit; never invent historical health
            shutil.copy2(metric, metadata_dir / name)
    return {"repo": repo, "revision": sha, "generation_protocol": protocol}


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


def isolate_harness(cfg, out_dir: Path) -> None:
    """Copy immutable harness inputs; keep answers, credentials and judges run-local."""
    source = Path(str(cfg.vendor_dir)).resolve()
    staging = out_dir.resolve() / "arena_hard_harness"
    if source == staging:
        return
    if not source.is_dir():
        raise FileNotFoundError(f"Arena-Hard harness missing: {source}")
    shutil.copytree(source, staging, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("model_answer", "model_judgment",
                                                  "generated_*.yaml", "__pycache__"))
    cfg.vendor_dir = str(staging)


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
    source_vendor = str(cfg.vendor_dir)
    isolate_harness(cfg, out_dir)
    cfg.target_identity = {"target": target.spec.hf_path,
                           "revision": target.spec.revision,
                           "base_revision": target.spec.base_revision,
                           "mode": target.spec.mode}
    cfg.base_model = target.spec.base_model
    cfg.generation.enable_thinking = target.spec.mode != "nothink"
    if target.spec.hf_path.startswith("tinker:"):
        cfg.generation.stream = False
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
    if cfg.get("smoke", False):
        for arm in cfg.arms:
            arm.n_hard_prompt = min(4, int(arm.n_hard_prompt))
            arm.n_creative_writing = min(4, int(arm.n_creative_writing))
    cfg.output_dir = str(out_dir)
    cfg_path = metadata_dir / "arena_hard_config.yaml"
    OmegaConf.save(cfg, cfg_path)

    bench = bench_answers_dir(cfg)
    if target.spec.answers:
        if not target.spec.revision:
            raise ValueError("Reused Arena answers require the dataset revision resolved on TargetSpec")
        source = answers_from_run(target.spec.answers, bench / f"{arm_name}.jsonl",
                                  revision=target.spec.revision, metadata_dir=metadata_dir)
        protocol = source.pop("generation_protocol")
        _validate_arm_protocol(protocol, cfg, target.spec.mode, arm_name, bench / f"{arm_name}.jsonl")
        print(f">>> answers reused from {target.spec.answers} — nothing to generate")
    else:
        arena_hard_gen.main(config=str(cfg_path), arm=arm_name,
                            served_model=target.model_name, endpoint=target.base_url,
                            api_key=target.api_key, smoke=bool(cfg.get("smoke", False)))
        gen_dir = max((out_dir / arm_name).glob("*/"), key=lambda p: p.name)
        source = {"generated": target.spec.hf_path}
        metrics = json.loads((gen_dir / "gen_metrics.json").read_text(encoding="utf-8"))
        protocol = generation_protocol(cfg, target.spec.mode, arm_name, bench / f"{arm_name}.jsonl", metrics)
        # Generation health (empty-think rate, token counts) is a fact about producing
        # these answers, not a result of any comparison, so it travels with them.
        for name in ("gen_metrics.json", "raw_samples.md", "run_meta.json"):
            if (gen_dir / name).exists():
                (gen_dir / name).rename(metadata_dir / f"gen_{name}")
        for diagnostic in gen_dir.glob("partial_checkpoint_recovery*.jsonl"):
            diagnostic.rename(metadata_dir / f"gen_{diagnostic.name}")
        shutil.rmtree(out_dir / arm_name)

    (metadata_dir / "generation_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    # Preserve the run-local staging cache until the answers have been published.
    shutil.copy2(bench / f"{arm_name}.jsonl", rollouts_dir / "answers.jsonl")
    shutil.rmtree(Path(str(cfg.vendor_dir)))
    cfg.vendor_dir = source_vendor
    OmegaConf.save(cfg, cfg_path)
    is_reference = target.spec.hf_path == reference
    (metadata_dir / "sources.json").write_text(json.dumps(
        {"arm": arm_name, "answers": source, "reference_arm": is_reference,
         "reference": reference}, indent=2))
    return {"arm": arm_name, "reference_arm": is_reference,
            "answers": "rollouts/answers.jsonl", "sources": "metadata/sources.json"}
