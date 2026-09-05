# ABOUTME: The launch contract of `uv run train` (src/train/launch.py): a recipe carries no arm,
# ABOUTME: retired keys are refused with the fix, `model=` resolves, and W&B is checked before GPU spend.

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.train.launch import (
    LAUNCH_ARGS,
    check_retired_keys,
    require_launch_args,
    resolve_model,
    wandb_preflight,
    write_back_pins,
)

RECIPE = "configs/train.yaml"
LAUNCH = ["model=qwen36", "data_repo=o/2026-09-05-nosynth-mix", "thinking=true"]


def test_the_recipe_carries_no_arm_identity_and_no_retired_key():
    cfg = OmegaConf.load(RECIPE)
    check_retired_keys(cfg)
    for k in LAUNCH_ARGS:
        assert k not in cfg, f"{k} is a launch argument, not a recipe field"
    assert "lora" in cfg and "train" in cfg
    for gone in ("dynamic_batching", "packing", "assistant_only_loss", "loss_type"):
        assert gone not in cfg.train, gone


def test_there_is_one_train_config_and_it_reports_nowhere_by_default():
    """configs/train.yaml is THE recipe; configs/train/ holds only the archive; W&B is a
    launch decision (`wandb=true`), never the file's default."""
    assert Path(RECIPE).is_file()
    assert [p.name for p in Path("configs/train").iterdir()] == ["archive"]
    cfg = OmegaConf.load(RECIPE)
    assert cfg.wandb is False and "report_to" not in cfg.train
    cfg.merge_with_dotlist(["wandb=true"])
    assert cfg.wandb is True


def test_launch_args_are_required_and_the_error_names_them_all():
    cfg = OmegaConf.load(RECIPE)
    with pytest.raises(ValueError, match=r"model.*data_repo.*thinking"):
        require_launch_args(cfg, RECIPE)
    cfg.merge_with_dotlist(LAUNCH)
    require_launch_args(cfg, RECIPE)


@pytest.mark.parametrize("key, value", [
    ("train.dynamic_batching", {}), ("train.packing", False), ("train.assistant_only_loss", True),
    ("train.loss_type", "nll"), ("train.mask_empty_think", True), ("train.load_in_4bit", False),
    ("train.report_to", ["wandb"]),
    ("model_class", "image_text_to_text"), ("lora.target_modules", "q_proj"),
    ("data_path", "data/x.jsonl"),
])
def test_an_archived_per_arm_shape_is_refused_and_told_where_the_fact_went(key, value):
    cfg = OmegaConf.load(RECIPE)
    OmegaConf.update(cfg, key, value, merge=True)
    with pytest.raises(ValueError, match=key.replace(".", r"\.")):
        check_retired_keys(cfg)


def test_the_archived_configs_really_do_carry_retired_keys():
    """The archive is the pre-recipe shape; if one of them passed unchanged, the refusal
    would be protecting nothing."""
    archived = sorted(Path("configs/train/archive").glob("*dynbatch*.yaml"))
    assert archived
    for path in archived:
        with pytest.raises(ValueError, match="no longer reads"):
            check_retired_keys(OmegaConf.load(path))


def test_model_resolves_from_a_key_an_id_a_path_or_a_stamp():
    profile, model_id = resolve_model(OmegaConf.create({"model": "qwen36"}))
    assert (profile.key, model_id) == ("qwen36", "Qwen/Qwen3.6-27B")
    _, model_id = resolve_model(OmegaConf.create({"model": "/root/qwen36"}))
    assert model_id == "/root/qwen36", "a path is loaded as typed, never rewritten to the id"
    # A stamped config carries the profile its run used, and that wins over the directory.
    stamp = profile.to_dict()
    stamp["train"] = {**stamp["train"], "lora_target_modules": "custom_regex$"}
    p2, _ = resolve_model(OmegaConf.create({"model": "Qwen/Qwen3.6-27B", "profile": stamp}))
    assert p2.lora_target_modules == "custom_regex$"
    with pytest.raises(AssertionError, match="different runs"):
        resolve_model(OmegaConf.create({"model": "Qwen/Qwen3-32B", "profile": profile.to_dict()}))
    with pytest.raises(ValueError, match="no verified thinking profile"):
        resolve_model(OmegaConf.create({"model": "qwen3"}))


def test_wandb_is_checked_before_any_gpu_spend():
    assert "no reporter" in wandb_preflight(False)
    assert "disabled" in wandb_preflight(True, {"WANDB_MODE": "disabled"})
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        wandb_preflight(True, {})
    env = {"WANDB_API_KEY": "k"}
    assert "project=lasr" in wandb_preflight(True, env) and env["WANDB_PROJECT"] == "lasr"
    env = {"WANDB_API_KEY": "k", "WANDB_PROJECT": "p", "WANDB_ENTITY": "e"}
    assert "project=p entity=e" in wandb_preflight(True, env)


def test_the_saved_config_is_a_complete_rerun():
    """What a run writes back is enough: the saved YAML passes every launch check and
    resolves to the same model, pins and profile with no command-line argument."""
    cfg = OmegaConf.load(RECIPE)
    cfg.merge_with_dotlist(LAUNCH)
    profile, model_id = resolve_model(cfg)
    write_back_pins(cfg, model_id=model_id, base_revision="abc123",
                    dataset_ref={"repo": "o/2026-09-05-nosynth-mix", "file": "mixture.jsonl",
                                 "revision": "def456"}, profile=profile)
    cfg.train.token_budget = 8000
    again = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))
    check_retired_keys(again)
    require_launch_args(again, "train_config.yaml")
    p2, mid = resolve_model(again)
    assert p2 == profile and mid == "Qwen/Qwen3.6-27B"
    assert (again.base_model_revision, again.data_file, again.data_revision,
            again.train.token_budget, again.thinking) == ("abc123", "mixture.jsonl", "def456",
                                                          8000, True)
