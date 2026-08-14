# ABOUTME: Tests for the synth CLI's pre-dispatch flag guard: a paid command must refuse
# ABOUTME: an unknown flag BEFORE running, because Fire itself only errors after execution.
# ABOUTME: Run: uv run pytest tests/test_synth_cli.py -q

from __future__ import annotations

import pytest

from src.data.synth.cli import _refuse_unknown_flags, run


def _argv(monkeypatch, *argv: str) -> None:
    monkeypatch.setattr("sys.argv", ["synth", *argv])


def test_unknown_flag_is_refused_before_dispatch(monkeypatch) -> None:
    """The 2026-08-14 incident: `run --estimate` spent a whole stage before Fire's
    "Could not consume arg" surfaced. The guard must fire first, naming the flag."""
    _argv(monkeypatch, "run", "--config", "x.yaml", "--estimate")
    with pytest.raises(SystemExit, match="unknown flag --estimate"):
        _refuse_unknown_flags({"run": run})


def test_known_flags_pass(monkeypatch) -> None:
    _argv(monkeypatch, "run", "--config", "x.yaml", "--smoke",
          "--resume", "out/d", "--overrides", "total_scenarios=8")
    _refuse_unknown_flags({"run": run})


def test_fire_spellings_pass(monkeypatch) -> None:
    """Fire accepts `--flag=value`, `--noflag` for booleans, and hyphens for
    underscores; the guard must not reject what Fire would accept."""
    _argv(monkeypatch, "run", "--config=x.yaml", "--nosmoke", "--help")
    _refuse_unknown_flags({"run": run})


def test_fires_own_namespace_after_separator_is_ignored(monkeypatch) -> None:
    _argv(monkeypatch, "run", "--config", "x.yaml", "--", "--trace")
    _refuse_unknown_flags({"run": run})


def test_unknown_command_is_left_to_fire(monkeypatch) -> None:
    """Fire's usage error for a bad command runs nothing, so the guard stays out."""
    _argv(monkeypatch, "nope", "--whatever")
    _refuse_unknown_flags({"run": run})
