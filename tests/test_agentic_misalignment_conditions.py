# ABOUTME: Locks the agentic-misalignment config to the 18-condition grid every published run used.
# ABOUTME: An edit that changes the panel renames or drops conditions and silently stops matching them.
"""The config was once left on an eight-condition grid while the published runs used eighteen, so
`uv run evals --name agentic_misalignment` reproduced none of the paper's numbers. These names come
from the published runs' own `results/misalignment_summary.json`; the test is offline."""
from omegaconf import OmegaConf

from src.eval.misalignment.agentic_misalignment.runner import planned_conditions

PUBLISHED = {
    f"{scenario}_{goal}_{urgency}_prod"
    for scenario in ("blackmail", "leaking", "murder")
    for goal in ("explicit-america", "none-none")
    for urgency in ("none", "restriction", "replacement")
}


def test_config_declares_exactly_the_published_condition_panel():
    cfg = OmegaConf.load("configs/eval/agentic_misalignment.yaml")
    assert planned_conditions(cfg) == PUBLISHED


def test_panel_and_trial_count_still_make_the_published_180_rollouts():
    cfg = OmegaConf.load("configs/eval/agentic_misalignment.yaml")
    assert len(PUBLISHED) == 18
    assert len(planned_conditions(cfg)) * cfg["global"]["samples_per_condition"] == 180


def test_production_variant_is_set_because_the_condition_names_carry_its_suffix():
    cfg = OmegaConf.load("configs/eval/agentic_misalignment.yaml")
    assert cfg.get("production_variant") is True
    assert all(name.endswith("_prod") for name in planned_conditions(cfg))
