# ABOUTME: Offline dispatch and preflight regression checks for the shared synth command.
# ABOUTME: Exercises both methods without remote reads, paid generation or publication.

import sys

import pytest
import yaml

from src.data.synth import cli
from src.data.synth.deliberative_alignment import pipeline


def config(tmp_path, method=None):
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump({"method": method} if method else {}))
    return str(path)


@pytest.mark.parametrize("method", [None, "ours"])
def test_existing_config_routes_to_ours(tmp_path, monkeypatch, method):
    calls = []
    monkeypatch.setattr(cli.ours, "run", lambda *a, **kw: calls.append((a, kw)))
    path = config(tmp_path, method)
    cli.run(path, smoke=True, ablate="revise_responses", batch=True)
    assert calls == [((path,), {"smoke": True, "resume": None, "ablate": "revise_responses",
                              "overrides": None, "batch": True})]


def test_deliberative_method_routes_resolved_config(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda *a, **kw: calls.append((a, kw)))
    cli.run(config(tmp_path, "deliberative_alignment"), smoke=True, overrides="workers=2")
    assert calls == [(({"method": "deliberative_alignment", "pipeline": "recipe", "workers": 2},),
                     {"smoke": True, "resume": None})]


@pytest.mark.parametrize("kwargs", [{"batch": True}, {"ablate": "response"}])
def test_deliberative_unsupported_options_fail_before_dispatch(tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(pipeline, "run", lambda *a, **kw: pytest.fail("must not dispatch"))
    with pytest.raises(ValueError, match="neither --ablate nor --batch"):
        cli.run(config(tmp_path, "deliberative_alignment"), **kwargs)


def test_unknown_flag_is_rejected_before_paid_run(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run", lambda **kw: pytest.fail("must not dispatch"))
    monkeypatch.setattr(sys, "argv", ["synth", "run", "--config", config(tmp_path), "--typo"])
    with pytest.raises(SystemExit, match="unknown flag"):
        cli.main()


def test_unknown_method_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown synth method"):
        cli.run(config(tmp_path, "deliberativ_alignment"))


def test_help_after_config_never_runs_generation(tmp_path, monkeypatch, capsys):
    def paid_run(config=None):
        pytest.fail("help must not dispatch generation")
    monkeypatch.setattr(cli, "run", paid_run)
    monkeypatch.setattr(sys, "argv", ["synth", "run", "--config", config(tmp_path), "--help"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0
    captured = capsys.readouterr()
    assert "NAME" in captured.out + captured.err


@pytest.mark.parametrize("field,value", [("max_tokens", 1.5), ("max_tokens", "8192"),
                                         ("seed", True), ("temperature", float("nan"))])
def test_sampling_types_fail_during_preflight(field, value):
    cfg = cli._config("configs/data/synth/delib.yaml")
    cfg["sampling"][field] = value
    with pytest.raises(ValueError):
        pipeline.validate_config(cfg)
