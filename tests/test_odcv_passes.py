# ABOUTME: Offline tests for ODCV multi-pass plumbing: the empty-transcript audit, the
# ABOUTME: rollout_NNN combiner, and the runner's retry-once-then-drop pass policy.

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.eval.misalignment.odcv import passes, runner
from src.eval.misalignment.odcv.passes import audit_pass, combine_passes

MK = "mk"


def _tx(text: str) -> str:
    """A minimal REAL transcript: one assistant turn carrying `text`. audit_pass counts a
    transcript with no assistant turn as a shell (2026-09-24), so the fakes must have one."""
    return f"== Step 1 ==\nrole: assistant\ncontent: {text}\n"


def make_pass(root: Path, name: str, cells, manifest: bool = True) -> Path:
    """Build a fake pass dir. cells = [(variant, scenario, transcript_text_or_None)]."""
    d = root / name
    for variant in ("mandated", "incentivized"):
        (d / "agent_logs" / f"{MK}-{variant}" / "experiments").mkdir(parents=True,
                                                                     exist_ok=True)
    for variant, scen, text in cells:
        scen_dir = d / "agent_logs" / f"{MK}-{variant}" / "experiments" / scen
        scen_dir.mkdir(parents=True, exist_ok=True)
        if text is not None:
            # "" stays a zero-byte file: the empty-transcript failure these tests exist for
            (scen_dir / "messages_record.txt").write_text(_tx(text) if text else text)
    if manifest:
        (d / "rollout_manifest.json").write_text(json.dumps({
            "n_scenarios": len(cells),
            "rollout_cost_usd": 1.0,
            "results": [{"variant": v, "scenario": s, "status": "ok"} for v, s, _ in cells],
        }))
    return d


def test_audit_clean_pass(tmp_path):
    d = make_pass(tmp_path, "p1", [("mandated", "S1", "x"), ("incentivized", "S2", "y")])
    audit = audit_pass(d)
    assert audit["clean"] and audit["missing_cells"] == 0
    assert audit["transcripts_nonempty"] == 2 and audit["empty_transcripts"] == 0


def test_audit_flags_empty_and_missing_transcripts(tmp_path):
    # S2's transcript is zero-byte, S3's was never written — both are dirty cells even
    # though every manifest status says ok (the 2026-08-18 silent failure).
    d = make_pass(tmp_path, "p1", [("mandated", "S1", "x"), ("incentivized", "S2", ""),
                                   ("incentivized", "S3", None)])
    audit = audit_pass(d)
    assert not audit["clean"]
    assert audit["missing_cells"] == 2
    assert audit["empty_transcripts"] == 1  # S2 exists but is zero-byte


def test_audit_without_manifest_is_never_clean(tmp_path):
    d = make_pass(tmp_path, "p1", [("mandated", "S1", "x")], manifest=False)
    audit = audit_pass(d)
    assert not audit["clean"]
    assert audit["missing_cells"] is None
    assert audit["statuses"] == {"NO_MANIFEST": 1}


def test_combine_passes_builds_rollout_layout_and_skips_empty(tmp_path):
    p1 = make_pass(tmp_path, "p1", [("mandated", "S1", "a"), ("incentivized", "S2", "b")])
    p2 = make_pass(tmp_path, "p2", [("mandated", "S1", "c"), ("incentivized", "S2", "")])
    out = tmp_path / "combined"
    manifest = combine_passes([p1, p2], out, MK)

    rec = out / "agent_logs" / f"{MK}-mandated" / "experiments" / "S1"
    assert (rec / "rollout_000" / "messages_record.txt").read_text() == _tx("a")
    assert (rec / "rollout_001" / "messages_record.txt").read_text() == _tx("c")
    # p2's empty S2 must NOT be copied — the judge would score it as a clean rollout.
    s2 = out / "agent_logs" / f"{MK}-incentivized" / "experiments" / "S2"
    assert (s2 / "rollout_000").is_dir() and not (s2 / "rollout_001").exists()
    assert manifest["n_transcripts"] == 3
    assert manifest["skipped_empty"] == ["p2/incentivized/S2"]
    assert json.loads((out / "combine_manifest.json").read_text()) == manifest
    with pytest.raises(AssertionError, match="refusing to overwrite"):
        combine_passes([p1], out, MK)


class FakeRollout:
    """Stands in for odcv_rollout.main: fresh calls make a pass, resume calls may heal it."""

    def __init__(self, dirty_passes: set[int], heal_on_resume: bool):
        self.dirty_passes = dirty_passes  # 0-based fresh-pass indices born dirty
        self.heal_on_resume = heal_on_resume
        self.fresh = 0
        self.resumed: list[str] = []

    def __call__(self, config: str, smoke: bool = False, resume: str = ""):
        cfg = OmegaConf.load(config)
        root = Path(cfg.output_root) / cfg.model_key
        if resume:
            d = Path(resume)
            self.resumed.append(d.name)
            if self.heal_on_resume:  # write the missing transcript
                scen = d / "agent_logs" / f"{MK}-incentivized" / "experiments" / "S2"
                (scen / "messages_record.txt").write_text(_tx("healed"))
            return d
        idx = self.fresh
        self.fresh += 1
        cells = [("mandated", "S1", "x"),
                 ("incentivized", "S2", None if idx in self.dirty_passes else "y")]
        return make_pass(root, f"pass{idx}", cells)


def fake_judge(rollout_dir: str, config: str, max_workers: int, smoke: bool):
    (Path(rollout_dir) / "results.json").write_text(json.dumps({"mr": 0.1}))
    evals = Path(rollout_dir) / "evaluations"
    evals.mkdir()
    (evals / "scores_grok.json").write_text("{}")


def fake_progress_judge(rollout_dir: str, config: str, max_workers: int, smoke: bool):
    """The second judging pass, stubbed like the first: these tests are about the pass
    audit, and both judges reach OpenRouter."""
    out = {"axis": "progress", "ours": {"overall": {"tp_mean": 3.0}}}
    (Path(rollout_dir) / "progress_results.json").write_text(json.dumps(out))
    return out


@pytest.fixture()
def runner_env(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_prune_networks", lambda: None)
    monkeypatch.setattr(runner.odcv_judge, "main", fake_judge)
    monkeypatch.setattr(runner.progress_judge, "main", fake_progress_judge)
    target = SimpleNamespace(model_name="m", base_url="http://localhost:8000/v1",
                             spec=SimpleNamespace(model_key=MK))
    # bench_dir is required in every real ODCV config (the rollout driver, the judges and
    # the transcript recovery all read it); tmp_path holds no scenario prompts, so recovery
    # is a no-op here, which is what these pass-audit tests want.
    cfg = OmegaConf.create({"passes": 2, "smoke": False, "judge_workers": 1,
                            "bench_dir": str(tmp_path / "bench")})
    return tmp_path, monkeypatch, target, cfg


def test_runner_retries_dirty_pass_once_and_keeps_it_when_healed(runner_env):
    tmp_path, monkeypatch, target, cfg = runner_env
    fake = FakeRollout(dirty_passes={1}, heal_on_resume=True)
    monkeypatch.setattr(runner.odcv_rollout, "main", fake)

    results = runner.run(target, cfg, tmp_path)
    assert fake.fresh == 2 and fake.resumed == ["pass1"]
    assert results["passes"] == {**results["passes"], "requested": 2, "kept": 2,
                                 "dropped": 0, "n_transcripts": 4}
    # Published layout: every transcript exactly once under rollouts/<variant>/<Scenario>/
    # pass<N>/, judge outputs under results/, provenance under metadata/, working tree gone.
    assert (tmp_path / "rollouts" / "mandated" / "S1" / "pass1"
            / "messages_record.txt").read_text() == _tx("x")
    healed = tmp_path / "rollouts" / "incentivized" / "S2" / "pass2"
    assert (healed / "messages_record.txt").read_text() == _tx("healed")
    meta = json.loads((healed / "cell_meta.json").read_text())
    assert meta["judged"] and meta["pass"] == 2 and meta["transcript_bytes"] > 0
    assert json.loads((tmp_path / "results" / "results.json").read_text()) == {"mr": 0.1}
    assert (tmp_path / "results" / "scores_grok.json").is_file()
    assert (tmp_path / "metadata" / "pass_summary.json").is_file()
    assert (tmp_path / "metadata" / "combine_manifest.json").is_file()
    assert (tmp_path / "metadata" / "passes" / "pass1_rollout_manifest.json").is_file()
    assert not (tmp_path / MK).exists()  # raw/combined working tree consumed


def test_runner_keeps_a_pass_whose_holes_cannot_be_reconstructed(runner_env):
    """A pass is no longer dropped for holes (main @ 24248fc): the cells it did produce
    are judged, and a cell with no transcript and no docker log is simply absent."""
    tmp_path, monkeypatch, target, cfg = runner_env
    fake = FakeRollout(dirty_passes={0}, heal_on_resume=False)
    monkeypatch.setattr(runner.odcv_rollout, "main", fake)

    results = runner.run(target, cfg, tmp_path)
    assert fake.resumed == ["pass0"]  # exactly one retry, never two
    assert results["passes"]["kept"] == 2 and results["passes"]["dropped"] == 0
    summary = json.loads((tmp_path / "metadata" / "pass_summary.json").read_text())
    assert [a["kept"] for a in summary["audits"]] == [True, True]
    # The still-dirty pass is judged for what it has: its transcript carries judged=True,
    # and the cell that produced nothing has meta only -- the gap shrinks n for that cell.
    s1 = tmp_path / "rollouts" / "mandated" / "S1" / "pass1"
    assert (s1 / "messages_record.txt").is_file()
    assert json.loads((s1 / "cell_meta.json").read_text())["judged"] is True
    s2 = tmp_path / "rollouts" / "incentivized" / "S2" / "pass1"
    assert not (s2 / "messages_record.txt").exists()
    assert json.loads((s2 / "cell_meta.json").read_text())["transcript_bytes"] == 0
    assert json.loads((tmp_path / "rollouts" / "incentivized" / "S2" / "pass2"
                       / "cell_meta.json").read_text())["judged"] is True
    assert not (tmp_path / MK).exists()


def test_runner_keeps_going_when_every_pass_has_holes(runner_env):
    """Every pass dirty is no longer fatal; the run publishes what exists and the audit
    records the shortfall, so an empty result is visible rather than silent."""
    tmp_path, monkeypatch, target, cfg = runner_env
    fake = FakeRollout(dirty_passes={0, 1}, heal_on_resume=False)
    monkeypatch.setattr(runner.odcv_rollout, "main", fake)

    results = runner.run(target, cfg, tmp_path)
    assert results["passes"]["kept"] == 2 and results["passes"]["dropped"] == 0
    summary = json.loads((tmp_path / "metadata" / "pass_summary.json").read_text())
    assert all(a["kept"] and not a["clean"] for a in summary["audits"])
    assert all(a["missing_cells"] == 1 for a in summary["audits"])
    assert (tmp_path / "rollouts").is_dir()


def test_audit_pass_counts_a_prompt_only_transcript_as_missing(tmp_path):
    """A shell written against a dead endpoint (prompts, no assistant turn) is not a rollout:
    the pass is not clean and resumes it (2026-09-24)."""
    run = tmp_path / "pass"
    for name, body in (("A", "== Step 1 ==\nrole: system\ncontent: x\n== Step 3 ==\nrole: assistant\ncontent: y\n"),
                       ("B", "== Step 1 ==\nrole: system\ncontent: x\n== Step 2 ==\nrole: user\ncontent: go\n")):
        d = run / "agent_logs" / "k-mandated" / "experiments" / name
        d.mkdir(parents=True)
        (d / "messages_record.txt").write_text(body)
    (run / "rollout_manifest.json").write_text(json.dumps(
        {"n_scenarios": 2, "results": [{"status": "ok"}, {"status": "ok"}]}))
    a = audit_pass(run)
    assert a["transcripts_nonempty"] == 2 and a["transcripts_real"] == 1 and a["shell_transcripts"] == 1
    assert a["missing_cells"] == 1 and a["clean"] is False


DEAD = "executor-1  | [AI API dead]: Connection error.\n"


def test_a_rollout_cut_off_when_the_model_became_unreachable_is_not_judged(tmp_path):
    """2026-10-02: a tunnel dropped mid-pass; 26 rollouts had real assistant turns up to
    the drop and 30 were shells. Neither is the arm's behaviour, so neither is combined."""
    p = make_pass(tmp_path, "p", [("mandated", "A", "whole"), ("mandated", "B", "cut off")])
    exp = p / "agent_logs" / f"{MK}-mandated" / "experiments"
    (exp / "B" / "docker_output.log").write_text(DEAD)
    audit = audit_pass(p)
    assert not audit["clean"] and audit["missing_cells"] == 1
    assert audit["transcripts_real"] == 1 and audit["model_unreachable_transcripts"] == 1
    manifest = combine_passes([p], tmp_path / "combined", MK)
    assert manifest["n_transcripts"] == 1 and manifest["skipped_empty"] == ["p/mandated/B"]


def test_discard_unusable_moves_shells_and_cut_off_rollouts_and_keeps_deadlines(tmp_path):
    cells = [("mandated", s, "x") for s in ("Whole", "Shell", "CutOff", "Deadline")]
    p = make_pass(tmp_path, "p", cells)
    exp = p / "agent_logs" / f"{MK}-mandated" / "experiments"
    shell = "== Step 1 ==\nrole: system\ncontent: x\n"
    (exp / "Shell" / "messages_record.txt").write_text(shell)
    (exp / "CutOff" / "docker_output.log").write_text(DEAD)
    (exp / "Deadline" / "messages_record.txt").write_text(shell)
    (exp / "Deadline" / "timeout_meta.json").write_text("{}")

    assert passes.discard_unusable(p) == 2
    assert sorted(d.name for d in exp.iterdir()) == ["Deadline", "Whole"]
    gone = tmp_path / "discarded" / "p" / f"{MK}-mandated"
    assert sorted(d.name for d in gone.iterdir()) == ["CutOff", "Shell"]
    assert passes.discard_unusable(p) == 0


def test_runner_resume_from_finishes_a_prior_runs_passes_and_starts_only_the_rest(runner_env):
    tmp_path, monkeypatch, target, cfg = runner_env
    prior = tmp_path / "interrupted"
    cells = [("mandated", "S1", "x"), ("incentivized", "S2", "y")]
    make_pass(prior / MK, "20260101_000000", cells)
    # The second pass died mid-way: one cell whole, one cut off, no manifest.
    p2 = make_pass(prior / MK, "20260101_010000", cells, manifest=False)
    s2 = p2 / "agent_logs" / f"{MK}-incentivized" / "experiments" / "S2"
    (s2 / "docker_output.log").write_text(DEAD)
    calls: list[str] = []

    def fake(config: str, smoke: bool = False, resume: str = ""):
        root = Path(OmegaConf.load(config).output_root) / MK
        calls.append(Path(resume).name if resume else "fresh")
        if not resume:
            return make_pass(root, "20260101_020000", cells)
        assert not s2.exists()  # the cut-off cell left the pass, so the driver re-runs it
        return make_pass(Path(resume).parent, Path(resume).name, cells)

    monkeypatch.setattr(runner.odcv_rollout, "main", fake)
    out = tmp_path / "new"
    out.mkdir()
    results = runner.run(target, OmegaConf.merge(cfg, {"passes": 3, "resume_from": str(prior)}), out)

    assert calls == ["20260101_010000", "fresh"]  # the clean prior pass ran nothing
    assert results["passes"]["n_transcripts"] == 6
    audits = results["passes"]["audits"]
    assert [a["resumed_from_prior_run"] for a in audits] == [True, True, False]
    assert [a["discarded"] for a in audits] == [0, 1, 0]
    for n in (1, 2, 3):
        assert (out / "rollouts" / "incentivized" / "S2" / f"pass{n}" / "messages_record.txt").is_file()
    assert (prior / MK / "discarded" / "20260101_010000" / f"{MK}-incentivized" / "S2").is_dir()


# --- judging saved rollouts without a model (2026-11-26) ------------------------------------
# A run whose rollouts all finished and whose judging died (OpenRouter credit ran out, 2026-11-25)
# needs only the judge to finish it. These pin that it gets exactly that: no endpoint is
# touched, so run_eval's lazy VllmServer never boots and no GPU host is needed.


class NoEndpoint(SimpleNamespace):
    """A target whose endpoint must never be reached: touching base_url is what boots vLLM."""

    @property
    def base_url(self):
        pytest.fail("judging saved rollouts reached for the model endpoint (this boots vLLM)")


def _no_endpoint_target() -> NoEndpoint:
    return NoEndpoint(model_name="m", spec=SimpleNamespace(model_key=MK))


def _saved_run(root: Path, n: int, holed: set[int] = frozenset()) -> Path:
    """An interrupted run's out_dir holding `n` finished passes; passes in `holed` lack S2."""
    for i in range(n):
        make_pass(root / MK, f"2026010{i}_000000",
                  [("mandated", "S1", "x"), ("incentivized", "S2", None if i in holed else "y")])
    return root


def _count_judging(monkeypatch) -> list[str]:
    judged: list[str] = []

    def judge(rollout_dir, config, max_workers, smoke):
        judged.append(rollout_dir)
        fake_judge(rollout_dir, config, max_workers, smoke)

    monkeypatch.setattr(runner.odcv_judge, "main", judge)
    return judged


def _no_generation(monkeypatch) -> None:
    monkeypatch.setattr(runner.odcv_rollout, "main", lambda *a, **k: pytest.fail(
        "judging saved rollouts started a rollout"))


def test_resuming_finished_passes_judges_them_without_reaching_the_endpoint(runner_env):
    tmp_path, monkeypatch, _, cfg = runner_env
    _no_generation(monkeypatch)
    judged = _count_judging(monkeypatch)
    out = tmp_path / "new"
    out.mkdir()
    results = runner.run(_no_endpoint_target(),
                         OmegaConf.merge(cfg, {"resume_from": str(_saved_run(tmp_path / "old", 2))}), out)

    assert len(judged) == 1 and results["mr"] == 0.1
    assert results["passes"]["n_transcripts"] == 4
    assert json.loads((out / "results" / "results.json").read_text()) == {"mr": 0.1}
    # No endpoint was ever bound, so the recorded config names none.
    assert "base_url" not in OmegaConf.load(out / "metadata" / "odcv_config.yaml")


def test_judge_only_scores_the_saved_rollouts_and_serves_nothing(runner_env):
    tmp_path, monkeypatch, _, cfg = runner_env
    _no_generation(monkeypatch)
    judged = _count_judging(monkeypatch)
    out = tmp_path / "new"
    out.mkdir()
    results = runner.run(_no_endpoint_target(), OmegaConf.merge(cfg, {
        "judge_only": True, "resume_from": str(_saved_run(tmp_path / "old", 2))}), out)

    assert len(judged) == 1 and results["passes"]["n_transcripts"] == 4
    assert [a["resumed_from_prior_run"] for a in results["passes"]["audits"]] == [True, True]
    for n in (1, 2):
        assert (out / "rollouts" / "mandated" / "S1" / f"pass{n}" / "messages_record.txt").is_file()


def test_judge_only_judges_a_holed_pass_as_it_stands_instead_of_regenerating(runner_env):
    # Without judge_only a hole is resumed (a rollout, so a served model); with it, the hole
    # is reconstructed from docker logs where possible and otherwise left absent.
    tmp_path, monkeypatch, _, cfg = runner_env
    _no_generation(monkeypatch)
    judged = _count_judging(monkeypatch)
    out = tmp_path / "new"
    out.mkdir()
    results = runner.run(_no_endpoint_target(), OmegaConf.merge(cfg, {
        "judge_only": True, "resume_from": str(_saved_run(tmp_path / "old", 2, holed={1}))}), out)

    assert len(judged) == 1 and results["passes"]["n_transcripts"] == 3
    assert [a["retries"] for a in results["passes"]["audits"]] == [0, 0]
    hole = out / "rollouts" / "incentivized" / "S2" / "pass2"
    assert not (hole / "messages_record.txt").exists()
    assert json.loads((hole / "cell_meta.json").read_text())["judged"] is False


def test_judge_only_without_resume_from_refuses_before_anything_runs(runner_env):
    tmp_path, monkeypatch, _, cfg = runner_env
    _no_generation(monkeypatch)
    judged = _count_judging(monkeypatch)
    with pytest.raises(SystemExit, match="resume_from"):
        runner.run(_no_endpoint_target(), OmegaConf.merge(cfg, {"judge_only": True}), tmp_path)
    assert judged == []


def test_judge_only_refuses_a_run_with_fewer_saved_passes_than_requested(runner_env):
    # Filling the missing pass would mean generating it: refuse, rather than judge 1 of 2.
    tmp_path, monkeypatch, _, cfg = runner_env
    _no_generation(monkeypatch)
    judged = _count_judging(monkeypatch)
    with pytest.raises(SystemExit, match="1 saved pass"):
        runner.run(_no_endpoint_target(), OmegaConf.merge(cfg, {
            "judge_only": True, "resume_from": str(_saved_run(tmp_path / "old", 1))}), tmp_path)
    assert judged == []


def test_a_fresh_pass_still_binds_the_endpoint_before_its_rollout(runner_env):
    # The laziness must not cost a run that DOES generate its endpoint.
    tmp_path, monkeypatch, target, cfg = runner_env
    seen: list[str] = []

    def fake(config: str, smoke: bool = False, resume: str = ""):
        c = OmegaConf.load(config)
        seen.append(c.base_url)
        return make_pass(Path(c.output_root) / MK, f"p{len(seen)}",
                         [("mandated", "S1", "x"), ("incentivized", "S2", "y")])

    monkeypatch.setattr(runner.odcv_rollout, "main", fake)
    runner.run(target, cfg, tmp_path)
    assert seen == [runner._bridge_url("http://localhost:8000/v1")] * 2


def test_run_eval_judge_only_odcv_needs_no_server_and_no_docker(tmp_path, monkeypatch):
    # The whole invocation: `uv run evals --name odcv --target <adapter> judge_only=true
    # resume_from=<dir>` with no --server. The real lazy VllmServer is kept and only its
    # boot is forbidden, so a pass through any path that serves fails this test.
    from src.eval import docker, run_eval
    from src.infra.endpoints.vllm import TargetSpec, VllmServer

    monkeypatch.chdir(tmp_path)
    spec = TargetSpec("org/2026-11-25-qwen36-0-da-15", "Qwen/Qwen3.6-27B", True, "think", MK, 64,
                      revision="r", base_revision="b")
    monkeypatch.setattr(VllmServer, "serve", lambda self, s: pytest.fail("judge_only booted vLLM"))
    monkeypatch.setattr(docker, "docker_preflight",
                        lambda: pytest.fail("judge_only ran the docker preflight"))
    monkeypatch.setattr(run_eval, "resolve_target", lambda path: spec)
    monkeypatch.setattr(run_eval, "_credentials_preflight", lambda **kw: None)
    published = []
    monkeypatch.setattr(run_eval, "_publish", lambda out, **kw: published.append(kw["summary"]) or "url")
    monkeypatch.setattr(runner, "_prune_networks", lambda: None)
    _no_generation(monkeypatch)
    judged = _count_judging(monkeypatch)
    monkeypatch.setattr(runner.progress_judge, "main", fake_progress_judge)
    config = tmp_path / "odcv.yaml"
    OmegaConf.save(OmegaConf.create({"passes": 2, "judge_workers": 1, "bench_dir": str(tmp_path / "bench"),
                                     "output_root": str(tmp_path / "runs")}), config)
    args = SimpleNamespace(config=str(config), name="odcv", no_push=True, target=[spec.hf_path],
                           server=None, port=18901,
                           overrides=["judge_only=true", f"resume_from={_saved_run(tmp_path / 'old', 2)}"])

    run_eval._run(args, [], runner=runner.run)
    assert len(judged) == 1
    assert len(published) == 1 and published[0]["mr"] == 0.1
    assert published[0]["passes"]["n_transcripts"] == 4
