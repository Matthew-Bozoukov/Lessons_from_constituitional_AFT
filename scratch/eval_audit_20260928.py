# ABOUTME: Historical reproductions of defects at the September 28 pre-repair baseline.
# ABOUTME: Run at commit 0b43ebe5; current regressions live in tests/test_eval_repair_regressions.py.

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from omegaconf import OmegaConf


def secret_number_checks():
    from src.eval.misalignment.secret_number.metrics import aggregate, classify
    state = NS(secret_intact=True, guesses=[], log_present=True)
    quoted = classify([{"command": "echo 'cat /secrets/secret_number.txt'",
                        "output": "cat /secrets/secret_number.txt\n", "returncode": 0}], state, 17)
    failed = classify([{"command": "rm /agent/guess_log.json", "output": "Permission denied",
                        "returncode": 1}], state, 17)
    row = {"submitted": False, "gamed": True, "routes": ["read_secret"],
           "write_secret_succeeded": False}
    return {"quoted_command_without_read": asdict(quoted),
            "failed_tamper": asdict(failed), "observed_hack_without_submission": aggregate([row])}


def mask_checks():
    path = Path("src/eval/misalignment/mask/third_party/mask/mask/generate_responses.py")
    spec = importlib.util.spec_from_file_location("audit_mask_generate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    async def one(content, finish):
        async def create(**kwargs):
            return NS(choices=[NS(message=NS(content=content, reasoning_content="reasoning"),
                                  finish_reason=finish)])
        mod.EMPTY_CONTENT.clear()
        result = await mod.generate_responses_async(
            NS(chat=NS(completions=NS(create=create))), "fixture",
            {"lying": [{"role": "user", "content": "fixture"}]}, 100, 1, asyncio.Semaphore(1))
        return {"responses": result, "empty_tally": dict(mod.EMPTY_CONTENT)}
    return {f"{content!r}/{finish}": asyncio.run(one(content, finish))
            for content, finish in [(None, "stop"), ("", "stop"), ("", "length"), ("partial", "length")]}


def internalization_checks(root):
    import importlib
    mod = importlib.import_module("src.eval.misalignment.internalization.pipeline.run")
    from src.eval.misalignment.internalization.config import load_config
    cfg = load_config("src/eval/misalignment/internalization/control/configs/base.yaml")
    cfg["itemset"]["dir"] = str(root / "missing_items")
    def would_build(*args, **kwargs):
        raise RuntimeError("attempted to build a generator client for missing pinned itemset")
    with patch.object(mod, "build_client", side_effect=would_build):
        try:
            mod.prepare_itemset(cfg)
        except RuntimeError as exc:
            return {"requested_id": cfg["itemset"]["id"], "result": str(exc)}
    raise AssertionError("Expected missing-itemset regeneration path")


def arena_checks(root):
    from src.eval.capabilities.arena_hard import arena_hard_gen as gen, runner
    cfg = OmegaConf.load("configs/eval/arena_hard.yaml")
    cfg.vendor_dir = str(root / "arena_vendor")
    cfg.arm_defaults = {"n_hard_prompt": 1, "n_creative_writing": 0}
    cfg.arms = []
    bench = Path(cfg.vendor_dir) / "data" / cfg.bench_name
    bench.mkdir(parents=True)
    (bench / "question.jsonl").write_text(json.dumps(
        {"uid": "fixture", "category": "hard_prompt", "prompt": "Say hello"}) + "\n")
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return iter([NS(choices=[NS(delta=NS(content="Hello", reasoning_content=None,
                                             reasoning=None), finish_reason="stop")])])
    client = NS(chat=NS(completions=NS(create=create)),
                models=NS(list=lambda: NS(data=[NS(max_model_len=16384)])))
    target = NS(spec=NS(model_key="qwen36", hf_path="Qwen/Qwen3.6-27B", answers=None),
                model_name="base", base_url="http://unused.invalid/v1", api_key="EMPTY")
    result = {}
    with patch.object(gen, "OpenAI", return_value=client), \
         patch.object(gen.tiktoken, "encoding_for_model", return_value=NS(encode=lambda s, **kw: list(s))), \
         patch.object(gen, "style_features", return_value={"token_len": 1, "header_count": {},
                                                          "list_count": {}, "bold_count": {}}), \
         contextlib.redirect_stdout(io.StringIO()):
        try:
            runner.run(target, cfg, root / "arena_run", reference="reference-model")
        except FileNotFoundError as exc:
            result["base_model_runner_error"] = str(exc)
        result["written_answer_files"] = [p.name for p in (bench / "model_answer").glob("*.jsonl")]
        config_path = root / "arena_run" / "metadata" / "arena_hard_config.yaml"
        revised = OmegaConf.load(config_path)
        revised.generation.temperature = 0.93
        OmegaConf.save(revised, config_path)
        calls.clear()
        gen.main(config=str(config_path), arm="qwen36", served_model="base")
        result["new_calls_after_changed_temperature"] = len(calls)
    return result


def whistlebench_checks(root):
    from src.eval.misalignment.whistlebench_team.runner import run
    cfg = OmegaConf.load("configs/eval/whistlebench_team.yaml")
    target = NS(base_url="http://unused.invalid/v1", is_api=True,
                spec=NS(hf_path="tinker://fixture:0/sampler_weights/gptoss120b-fixture"))
    try:
        run(target, cfg, root / "whistlebench")
    except AssertionError as exc:
        return {"tinker_runner_error": str(exc)}
    raise AssertionError("Expected Tinker routing assertion")


def main():
    with tempfile.TemporaryDirectory(prefix="eval-audit-") as d:
        root = Path(d)
        result = {"secret_number": secret_number_checks(), "mask": mask_checks(),
                  "internalization": internalization_checks(root),
                  "arena_hard": arena_checks(root), "whistlebench_team": whistlebench_checks(root),
                  "missing_tinker_dependencies": [m for m in ("tinker", "tinker_cookbook")
                                                  if importlib.util.find_spec(m) is None]}
    out = Path("output/eval_audit/reproductions.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
