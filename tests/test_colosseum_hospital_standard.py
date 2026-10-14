# ABOUTME: Offline tests for the 2026-10-09 Hospital "standard conversation" switches — keep_briefings
# ABOUTME: and board_deltas: the switch table, the config, the patch file, the guards, and the loop itself.
"""No network, no GPU.

The agent loop (terrarium `BaseAgent`) is a third-party package patched in site-packages,
so its behaviour is exercised end to end by scratch/colosseum_hospital/fixes_smoke.py
(`--variants standard`). Here the two methods the patch adds are driven directly when
TERRARIUM_SITE points at a patched tree; the patch and the configs are checked as text
always.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.eval.misalignment.colosseum.hospital.config import (
    FIXES,
    TERRARIUM_FIXES_STAMP,
    resolve_fixes,
)

THIRD_PARTY = Path("src/eval/misalignment/colosseum/third_party")
PATCH = THIRD_PARTY / "terrarium_hospital_fixes_4.patch"
BASE = "configs/eval/colosseum_hospital.yaml"
DIRECTION_A = "scratch/colosseum_hospital/configs/2026-09-09_colosseum_hospital_carried_history.yaml"
DIRECTION_B = (
    "scratch/colosseum_hospital/configs/2026-09-09_colosseum_hospital_board_access.yaml"
)
HEADLINE = "scratch/colosseum_hospital/configs/2026-09-25_colosseum_hospital_da15_nosynth_peer.yaml"
STANDARD = (
    "scratch/colosseum_hospital/configs/2026-10-09_colosseum_hospital_standard.yaml"
)
SWITCHES = ("keep_briefings", "board_deltas")


def _with(cfg, *dotlist):
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(dotlist)))


# ── the switch table ──────────────────────────────────────────────────────────


def test_the_switches_exist_default_off_and_the_stamp_is_theirs():
    assert all(FIXES[s] is False for s in SWITCHES)
    assert TERRARIUM_FIXES_STAMP == "2026-10-09"


@pytest.mark.parametrize("path", [BASE, DIRECTION_A, DIRECTION_B, HEADLINE])
def test_every_earlier_config_has_the_standard_switches_off(path):
    fixes = resolve_fixes(OmegaConf.load(path))
    assert all(fixes[s] is False for s in SWITCHES), path


def test_the_switches_need_each_other_in_order():
    cfg = OmegaConf.load(BASE)
    with pytest.raises(AssertionError, match="carry_agent_history"):
        resolve_fixes(
            _with(cfg, "fixes.carry_agent_history=false", "fixes.keep_briefings=true")
        )
    with pytest.raises(AssertionError, match="keep_briefings"):
        resolve_fixes(_with(cfg, "fixes.board_deltas=true"))
    both = resolve_fixes(
        _with(cfg, "fixes.keep_briefings=true", "fixes.board_deltas=true")
    )
    assert (
        both["keep_briefings"] and both["board_deltas"] and both["carry_agent_history"]
    )
    # keep_briefings alone is legal: the marker goes, the boards stay whole.
    alone = resolve_fixes(_with(cfg, "fixes.keep_briefings=true"))
    assert alone["keep_briefings"] and not alone["board_deltas"]


# ── the config ────────────────────────────────────────────────────────────────


def test_the_standard_config_is_the_headline_harness_plus_the_two_switches():
    cfg, parent = OmegaConf.load(STANDARD), OmegaConf.load(HEADLINE)
    assert resolve_fixes(cfg) == dict(
        resolve_fixes(parent), keep_briefings=True, board_deltas=True
    )
    c, p = OmegaConf.to_container(cfg), OmegaConf.to_container(parent)
    for key in (
        "condition",
        "peer",
        "partner",
        "partner_seat",
        "seeds",
        "num_agents",
        "num_patients",
        "max_iterations",
        "max_planning_rounds",
        "max_conversation_steps",
        "max_tokens",
        "temperature",
        "request_timeout",
        "max_concurrent_runs",
        "judge",
        "serving",
        "colosseum_root",
    ):
        assert c[key] == p[key], key
    assert c["max_tokens"] == 8192
    # No typed labels: the name is built by run_eval from the run's metadata.
    assert "arm_labels" not in c
    head = Path(STANDARD).read_text(encoding="utf-8").splitlines()[:6]
    assert head[0].startswith("# ABOUTME:") and head[1].startswith("# ABOUTME:")
    assert any(line.startswith("# Run:") for line in head)
    assert any(STANDARD in line for line in head)


# ── the patch file, the README, the guards ────────────────────────────────────


def test_the_patch_touches_the_loop_and_the_stamps_only():
    text = PATCH.read_text(encoding="utf-8")
    touched = {line.split()[1] for line in text.splitlines() if line.startswith("+++ ")}
    assert touched == {
        "b/terrarium/agents/base.py",
        "b/envs/dcops/hospital/hospital_env.py",
    }
    assert '-TERRARIUM_FIXES = "2026-09-15"' in text
    assert '+TERRARIUM_FIXES = "2026-10-09"' in text
    assert '-HOSPITAL_FIXES = "2026-09-15"' in text
    assert '+HOSPITAL_FIXES = "2026-10-09"' in text
    for needle in (
        "keep_briefings",
        "board_deltas",
        "BOARD_DELTA_NEW",
        "BOARD_DELTA_NONE",
        "+def _split_events(",
        "+    def _board_deltas(",
        '"board_events_shown"',
        "blackboard_context = self._board_deltas(blackboard_context)",
    ):
        assert needle in text, needle
    # The 2026-09-09 shape survives, switched off: the marker and the cut are kept.
    assert "[Your earlier turn: {where}." in text
    assert "_HISTORY_TOOL_RESULT_CHARS" in text


def test_the_readme_the_bootstrap_and_the_runner_name_the_patch():
    readme = (THIRD_PARTY / "README.md").read_text(encoding="utf-8")
    assert "terrarium_hospital_fixes_4.patch" in readme
    assert all(s in readme for s in SWITCHES)
    boot = Path("scratch/colosseum_hospital/pod_bootstrap.sh").read_text(
        encoding="utf-8"
    )
    assert "terrarium_hospital_fixes_4.patch" in boot
    assert f"'{TERRARIUM_FIXES_STAMP}'" in boot
    assert boot.index("TPATCH_NOSIM") < boot.index("TPATCH4=")
    from src.eval.misalignment.colosseum.hospital import runner

    assert runner.TERRARIUM_PATCH_4.endswith("terrarium_hospital_fixes_4.patch")


# ── the loop, when a patched tree is importable ───────────────────────────────


class _FakeClient:
    @staticmethod
    def init_context(system_prompt, user_prompt):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]


@pytest.fixture
def base_module():
    site = os.environ.get("TERRARIUM_SITE")
    if site:
        sys.path.insert(0, site)
    try:
        from terrarium.agents import base
    except ImportError:
        pytest.skip("terrarium is not importable; set TERRARIUM_SITE to a patched tree")
    if getattr(base, "TERRARIUM_FIXES", None) != TERRARIUM_FIXES_STAMP:
        pytest.skip(
            f"the importable terrarium carries stamp {getattr(base, 'TERRARIUM_FIXES', None)!r}"
        )
    return base


def _agent(base, **fixes):
    agent = base.BaseAgent.__new__(base.BaseAgent)
    agent.turn_options = dict(carry_agent_history=True, **fixes)
    agent.turn_history = []
    agent._board_seen = {}
    agent._board_shown = {}
    agent.current_iteration, agent.current_phase, agent.current_round = 1, "planning", 1
    agent.client = _FakeClient()
    return agent


def test_keep_briefings_keeps_the_turn_as_the_model_saw_it_minus_reasoning(base_module):
    agent = _agent(base_module, keep_briefings=True)
    ctx = _FakeClient.init_context("SYS", "BRIEFING 1")
    agent._carry_history_into(ctx)
    start = len(ctx)
    ctx += [
        {
            "role": "assistant",
            "content": "reply",
            "reasoning": "secret",
            "tool_calls": [{"id": "c1"}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "x" * 700},
        {"role": "user", "content": "[Note from the harness: ...]"},
        {"role": "assistant", "content": "again", "reasoning_content": "more"},
    ]
    agent._remember_turn(ctx, start)
    turn = agent.turn_history[-1]["messages"]
    assert [m["role"] for m in turn] == [
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
    ]
    assert turn[0] == {"role": "user", "content": "BRIEFING 1"}
    assert turn[1] == {
        "role": "assistant",
        "content": "reply",
        "tool_calls": [{"id": "c1"}],
    }
    assert len(turn[2]["content"]) == 700
    assert turn[3]["content"].startswith("[Note from the harness")
    assert turn[4] == {"role": "assistant", "content": "again"}
    nxt = _FakeClient.init_context("SYS", "BRIEFING 2")
    agent._carry_history_into(nxt)
    assert [m["role"] for m in nxt] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
        "user",
    ]
    assert nxt[1]["content"] == "BRIEFING 1" and nxt[-1]["content"] == "BRIEFING 2"


def test_without_keep_briefings_the_2026_09_09_shape_is_unchanged(base_module):
    agent = _agent(base_module, keep_briefings=False)
    ctx = _FakeClient.init_context("SYS", "BRIEFING 1")
    start = len(ctx)
    ctx += [
        {"role": "assistant", "content": "reply", "reasoning": "secret"},
        {"role": "tool", "tool_call_id": "c1", "content": "x" * 700},
        {"role": "user", "content": "[Note from the harness: ...]"},
    ]
    agent._remember_turn(ctx, start)
    turn = agent.turn_history[-1]["messages"]
    assert [m["role"] for m in turn] == ["user", "assistant", "tool"]
    assert turn[0]["content"].startswith(
        "[Your earlier turn: iteration 1, planning phase"
    )
    assert turn[2]["content"].endswith(" ...[cut]") and len(turn[2]["content"]) == 609


def test_board_deltas_show_each_event_once_numbered_as_on_the_board(base_module):
    agent = _agent(base_module, keep_briefings=True, board_deltas=True)
    boards = {
        "0": "[1] [context] welcome\n[2] X: SURPLUS\nsecond line",
        "1": "No recent activity",
    }
    first = agent._board_deltas(dict(boards))
    assert first == boards
    assert agent._board_shown == {
        "0": {"seen_before": 0, "total": 2, "new": 2},
        "1": {"seen_before": 0, "total": 0, "new": 0},
    }
    boards2 = {"0": boards["0"] + "\n[3] Y: DEFICIT", "1": "[1] Z: plan"}
    second = agent._board_deltas(dict(boards2))
    assert (
        second["0"] == base_module.BOARD_DELTA_NEW.format(seen=2) + "\n[3] Y: DEFICIT"
    )
    assert second["1"] == "[1] Z: plan", "a board first seen now is shown whole"
    third = agent._board_deltas(dict(boards2))
    assert third == {
        "0": base_module.BOARD_DELTA_NONE.format(seen=3),
        "1": base_module.BOARD_DELTA_NONE.format(seen=1),
    }
    # Off: the context passes through untouched and nothing is recorded.
    plain = _agent(base_module, keep_briefings=True)
    assert plain._board_deltas(dict(boards2)) == boards2 and plain._board_shown == {}
