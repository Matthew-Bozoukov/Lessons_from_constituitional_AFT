# ABOUTME: The managed Colosseum harness (src/eval/misalignment/colosseum/harness.py) and the Hospital
# ABOUTME: constitution arm's config: patch lists, cache key, subprocess env, preamble seats. No network.

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.eval import EVALS
from src.eval.misalignment.colosseum import harness
from src.eval.misalignment.colosseum.hospital.config import PAIR, resolve_fixes
from src.eval.preamble import ENV_VAR

CONSTITUTION = "constitutions/claude_distilled_09_principles/constitution.md"


def test_every_listed_patch_exists_and_each_list_is_unique():
    for name in (*harness.COLOSSEUM_PATCHES, *harness.TERRARIUM_PATCHES):
        assert (harness.THIRD_PARTY / name).is_file(), name
    assert len(set(harness.COLOSSEUM_PATCHES)) == len(harness.COLOSSEUM_PATCHES)
    assert len(set(harness.TERRARIUM_PATCHES)) == len(harness.TERRARIUM_PATCHES)


def test_the_preamble_patches_come_last_but_the_solo_patches_cut_on_top_of_them():
    # 2026-11-11: the Solo Hospital Test's patches were cut on top of the preamble patches
    # and are applied after them; everything before is unchanged.
    assert harness.COLOSSEUM_PATCHES[-3:] == (
        "hospital_preamble.patch",
        "hospital_solo.patch",
        "hospital_scripted_seats.patch",
    )
    assert harness.TERRARIUM_PATCHES[-2:] == (
        "terrarium_hospital_preamble.patch",
        "terrarium_hospital_solo.patch",
    )


def test_the_cache_key_changes_when_any_patch_byte_changes(tmp_path):
    copy = tmp_path / "third_party"
    shutil.copytree(harness.THIRD_PARTY, copy)
    before = harness.build_key(copy)
    assert before == harness.build_key(copy), "deterministic"
    target = copy / harness.TERRARIUM_PATCHES[2]
    target.write_bytes(target.read_bytes() + b"\n")
    assert harness.build_key(copy) != before


def test_the_subprocess_env_puts_the_patched_copy_first(tmp_path):
    h = harness.Harness(
        root=tmp_path / "c", terrarium_site=tmp_path / "t", managed=True
    )
    env = h.env({"PYTHONPATH": "/elsewhere", "X": "1"})
    assert env["PYTHONPATH"].split(os.pathsep) == [str(tmp_path / "t"), "/elsewhere"]
    assert env["X"] == "1"
    assert (
        harness.Harness(root=tmp_path, terrarium_site=None, managed=False).env({}) == {}
    )


def test_a_named_checkout_wins_and_nothing_is_built(tmp_path, monkeypatch):
    monkeypatch.setenv("COLOSSEUM_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("COLOSSEUM_ROOT", str(tmp_path / "by-hand"))
    h = harness.ensure_harness("/ignored/when/env/set")
    assert h.root == tmp_path / "by-hand" and h.terrarium_site is None and not h.managed
    monkeypatch.delenv("COLOSSEUM_ROOT")
    h = harness.ensure_harness(str(tmp_path / "from-config"))
    assert h.root == tmp_path / "from-config" and not h.managed
    assert not (tmp_path / "cache").exists()


def test_existing_build_is_none_until_built_and_never_builds(tmp_path, monkeypatch):
    monkeypatch.setenv("COLOSSEUM_CACHE", str(tmp_path))
    assert harness.existing_build() is None
    final = tmp_path / f"hospital-{harness.build_key()}"
    (final / "colosseum").mkdir(parents=True)
    (final / "BUILT.json").write_text("{}")
    built = harness.existing_build()
    assert (
        built.root == final / "colosseum"
        and built.terrarium_site == final / "terrarium_site"
    )


def test_the_hospital_config_defaults_to_the_managed_build():
    cfg = OmegaConf.load("configs/eval/colosseum_hospital.yaml")
    assert cfg.colosseum_root is None


# ── the constitution arm ──────────────────────────────────────────────────────


def test_no_preamble_leaves_the_seats_null_and_the_name_plain():
    cfg = OmegaConf.load("configs/eval/colosseum_hospital.yaml")
    assert cfg.protocol is None and cfg.system_preamble_file is None
    assert resolve_fixes(cfg)["system_preamble_agents"] is None
    assert "protocol" in EVALS["colosseum_hospital"].name_facets


def test_a_preamble_defaults_its_seats_to_the_pair():
    cfg = OmegaConf.merge(
        OmegaConf.load("configs/eval/colosseum_hospital.yaml"),
        OmegaConf.create(
            {"protocol": "constitution", "system_preamble_file": CONSTITUTION}
        ),
    )
    assert resolve_fixes(cfg)["system_preamble_agents"] == list(PAIR)
    cfg.fixes.system_preamble_agents = ["General_Hospital_Triage"]
    assert resolve_fixes(cfg)["system_preamble_agents"] == ["General_Hospital_Triage"]


def test_seats_without_a_preamble_are_refused():
    cfg = OmegaConf.create({"fixes": {"system_preamble_agents": list(PAIR)}})
    with pytest.raises(AssertionError, match="no system_preamble_file"):
        resolve_fixes(cfg)


def test_the_patches_read_the_shared_preamble_not_a_private_copy():
    t = (harness.THIRD_PARTY / "terrarium_hospital_preamble.patch").read_text()
    assert "prepend_from_env" in t and ENV_VAR in t
    assert [l for l in t.splitlines() if l.startswith("+++ ")] == [
        "+++ b/envs/dcops/hospital/hospital_prompts.py"
    ]
    c = (harness.THIRD_PARTY / "hospital_preamble.patch").read_text()
    assert "agent_name=agent_name" in c
    assert [l for l in c.splitlines() if l.startswith("+++ ")] == [
        "+++ b/experiments/agent_misalignment/prompts.py"
    ]


def test_the_patched_prompt_prepends_only_for_the_named_seats(monkeypatch):
    built = harness.existing_build()
    if built is None:
        pytest.skip("needs the managed harness built (run hospital/smoke.py once)")
    import sys

    monkeypatch.syspath_prepend(str(built.terrarium_site))
    sys.modules.pop("envs.dcops.hospital.hospital_prompts", None)
    from envs.dcops.hospital.hospital_prompts import HospitalPrompts

    from src.eval.preamble import Preamble

    pre = Preamble.from_file(CONSTITUTION)
    monkeypatch.setenv(ENV_VAR, str(Path(CONSTITUTION).resolve()))
    p = HospitalPrompts(None, {"fixes": {"system_preamble_agents": list(PAIR)}})
    stock = HospitalPrompts(None, {}).get_system_prompt()
    assert p.get_system_prompt(agent_name=PAIR[0]) == pre.prepend(stock)
    assert p.get_system_prompt(agent_name="St_Marys_Center_Surgery") == stock
    assert p.get_system_prompt() == stock


# ── the Jira set, and nothing on a cluster ───────────────────────────────────


def test_the_jira_set_is_the_routing_patch_on_the_stock_wheel(tmp_path, monkeypatch):
    assert harness.JIRA.colosseum == ("per_agent_models.patch",)
    assert harness.JIRA.terrarium == ()
    assert harness.build_key(patches=harness.JIRA) != harness.build_key(patches=harness.HOSPITAL)
    monkeypatch.setenv("COLOSSEUM_CACHE", str(tmp_path))
    final = tmp_path / f"jira-{harness.build_key(patches=harness.JIRA)}"
    (final / "colosseum").mkdir(parents=True)
    (final / "BUILT.json").write_text("{}")
    built = harness.existing_build(harness.JIRA)
    assert built.root == final / "colosseum" and built.terrarium_site is None
    assert built.env({"A": "1"}) == {"A": "1"}, "the stock wheel: PYTHONPATH untouched"


def test_both_colosseum_configs_default_to_the_managed_build():
    for path in ("configs/eval/colosseum_hospital.yaml", "configs/eval/colosseum_jira.yaml"):
        assert OmegaConf.load(path).colosseum_root is None, path


def test_the_hospital_eval_has_one_config_and_every_variant_resolves():
    from src.eval import EVALS
    from src.eval.misalignment.colosseum.hospital.smoke import VARIANTS, variant_cfg

    assert EVALS["colosseum_hospital"].config == "configs/eval/colosseum_hospital.yaml"
    assert not Path("configs/eval/colosseum_hospital").exists(), "one config, no protocol folder"
    cfg = OmegaConf.load("configs/eval/colosseum_hospital.yaml")
    # the paper's self-sacrificial arm with the 2026-11-04/05 settings baked in
    assert cfg.condition == "self_sacrificial"
    assert cfg.peer == "dougalldeepmind/2026-10-05-qwen36-0-plain"
    assert cfg.fixes.public_actions_without_rationale is False
    assert cfg.serving.context_window == 131072
    for stem in VARIANTS:
        v = variant_cfg(stem)
        assert v.get("colosseum_root") is None, stem
        resolve_fixes(v)  # every switch known and consistent