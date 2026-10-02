# ABOUTME: Parse eval-specific flags together with OmegaConf overrides before pod ownership.
# ABOUTME: Reproduce Arena spaced-reference failure without network or GPU operations.
from contextlib import contextmanager

import pytest
from yaml import YAMLError
from omegaconf import OmegaConf

from src.eval import run_eval
from src.infra import runpod


def arena_runner(target, cfg, out_dir, *, reference=""):
    raise AssertionError("The parser regression must not execute an eval")


@pytest.fixture
def boundary(monkeypatch):
    events, captured = [], {}
    @contextmanager
    def lifecycle(server):
        events.append("acquire")
        try:
            yield "release-callback"
        finally:
            events.append("release")
    def capture(args, unknown, release, *, runner):
        captured.update(args=args, kwargs=run_eval.derive_run_kwargs(runner, unknown),
                        cfg=OmegaConf.from_dotlist(args.overrides))
        events.append("run")
    monkeypatch.setattr(runpod, "eval_pod", lifecycle)
    monkeypatch.setattr(run_eval, "_run", capture)
    monkeypatch.setattr(run_eval, "load_dotenv", lambda: None)
    return events, captured


BASE = ["--target", "org/target", "--name", "arena_hard"]
LIFECYCLE = ["--server", "fixture", "--port", "18743", "--terminate-pod", "--no-push"]
OVERRIDES = ["smoke=true", "target_revision=633abc", "output_root=output/probe"]


@pytest.mark.parametrize("reference", [["--reference", "org/reference"], ["--reference=org/reference"]])
@pytest.mark.parametrize("placement", ["before", "after", "intermixed"])
def test_reference_and_dotlist_survive_normal_cli(boundary, reference, placement):
    if placement == "before":
        argv = BASE + reference + LIFECYCLE + OVERRIDES
    elif placement == "after":
        argv = BASE + LIFECYCLE + OVERRIDES + reference
    else:
        argv = BASE + OVERRIDES[:1] + reference + LIFECYCLE + OVERRIDES[1:]
    events, captured = boundary
    run_eval.main(argv, runner=arena_runner)
    assert events == ["acquire", "run", "release"]
    assert captured["kwargs"] == {"reference": "org/reference"}
    assert captured["args"].target == ["org/target"]
    assert captured["cfg"].smoke is True
    assert captured["cfg"].target_revision == "633abc"
    assert captured["cfg"].output_root == "output/probe"


def test_spaced_reference_without_overrides(boundary):
    events, captured = boundary
    run_eval.main(BASE + ["--reference", "org/reference", "--config", "fixture.yaml"] + LIFECYCLE,
                  runner=arena_runner)
    assert captured["kwargs"] == {"reference": "org/reference"}
    assert captured["args"].overrides == []
    assert events == ["acquire", "run", "release"]


@pytest.mark.parametrize("bad", [
    ["--reference"],
    ["--unknown", "value"],
    ["not-an-assignment"],
    ["=missing-key"],
    ["generation={broken"],
])
def test_bad_cli_never_acquires_pod(boundary, bad):
    events, captured = boundary
    with pytest.raises((SystemExit, ValueError, YAMLError)):
        run_eval.main(BASE + LIFECYCLE + bad, runner=arena_runner)
    assert events == [] and captured == {}


def test_dynamic_flags_still_belong_only_to_the_selected_eval(boundary):
    def other_runner(target, cfg, out_dir):
        pass
    with pytest.raises(SystemExit, match="unknown arguments"):
        run_eval.main(BASE + LIFECYCLE + ["--reference", "org/reference"], runner=other_runner)
    assert boundary[0] == []


def test_other_keyword_flags_and_multiple_targets_are_preserved(boundary):
    def other_runner(target, cfg, out_dir, *, judge_seed=""):
        pass
    run_eval.main(["--target", "org/first", "org/second", "--name", "mmlu",
                   "n=2", "--judge-seed", "7"] + LIFECYCLE + ["smoke=true"], runner=other_runner)
    events, captured = boundary
    assert captured["args"].target == ["org/first", "org/second"]
    assert captured["kwargs"] == {"judge_seed": "7"}
    assert captured["cfg"] == {"n": 2, "smoke": True}
    assert events == ["acquire", "run", "release"]


def test_fleet_rejects_ordinary_overrides_without_resolving_runner(monkeypatch):
    monkeypatch.setattr(run_eval, "resolve", lambda *_: pytest.fail("Fleet runner resolution changed"))
    with pytest.raises(SystemExit):
        run_eval.main(["--target", "org/target", "--name", "swebench_mini", "--fleet",
                       "--budget-usd", "5", "smoke=true"])
