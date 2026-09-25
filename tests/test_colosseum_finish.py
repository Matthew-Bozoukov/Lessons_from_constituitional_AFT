# ABOUTME: Offline tests for the Hospital eval finishing itself inside one `uv run evals`: the
# ABOUTME: runner's judge step (judge_arm), the up-front key check, and the finisher's parity.

"""No network, no GPU.

Until 2026-09-25 the Hospital eval was run with `--no-push` and a second script judged the
episodes and pushed the run dir under a hand-typed label. Now the runner judges its own arm
(judge_arm, right after the harvest) and run_eval publishes it under the law's name; the
finisher (publish.py) exists only for a run dir that invocation could not push, and must
produce the same files, name and tags. The judge itself is a fake here — a canned verdict
per seed — because what is under test is the plumbing around it, not the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.eval.misalignment.colosseum import publish
from src.eval.misalignment.colosseum.hospital import judge as hjudge
from src.eval.misalignment.colosseum.hospital.config import PAIR
from src.eval.misalignment.colosseum.hospital.judge import JUDGE_MEASURES, judge_arm

TARGET = "dougalldeepmind/2026-09-24-qwen36-0-da-multiparty-human-15"
KEY = "qwen36_0_da_multiparty_human_15"
PROV, TRIAGE = PAIR


def _verdicts(seeds, refusal):
    """A judge.json as judge_run_root returns it: one rating per channel and aspect."""
    per_run = {}
    for s in seeds:
        rec = {
            "public": {
                "refusal": {"rating": refusal, "reason": "r"},
                "simple": {"rating": 1, "reason": "r"},
            },
            "secret": {
                "refusal": {"rating": refusal, "reason": "r"},
                "simple": {"rating": 2, "reason": "r"},
            },
            f"reasoning:{PROV}": {
                "refusal": {"rating": refusal, "reason": "r"},
                "refusal_held": {"rating": refusal, "reason": "r"},
            },
            f"reasoning:{TRIAGE}": {
                "refusal": {"rating": 0, "reason": "r"},
                "refusal_held": {"rating": 0, "reason": "r"},
            },
            "all": {
                "refusal": {"rating": refusal, "reason": "r"},
                "refusal_held": {"rating": refusal, "reason": "r"},
                "complex": {"rating": 0, "reason": "r"},
            },
        }
        per_run[f"self_sacrificial_seed{s}"] = rec
    return {
        "judge_model": "fake/judge",
        "per_run": per_run,
        "unparsed": 0,
        "n_calls": 5 * len(seeds),
        "n_episodes": len(seeds),
        "admin_runs": {},
    }


def _arm_dir(tmp_path, seeds=(1, 2, 3), condition="self_sacrificial"):
    """A harvested arm as the runner leaves it before the judge: per_seed.json with the
    action measures, the Colosseum tree under rollouts/, run_meta under metadata/."""
    run_dir = tmp_path / f"2026-09-25_{condition}_{KEY}_120000"
    for d in ("rollouts/colosseum/20260925-120000/runs", "results", "metadata"):
        (run_dir / d).mkdir(parents=True)
    (run_dir / "results" / "per_seed.json").write_text(
        json.dumps(
            {
                "advantage_pair": {condition: {str(s): 0.6 for s in seeds}},
            }
        )
    )
    cfg = OmegaConf.load("configs/eval/colosseum_hospital.yaml")
    cfg = OmegaConf.merge(cfg, {"condition": condition, "seeds": list(seeds)})
    (run_dir / "metadata" / "run_meta.json").write_text(
        json.dumps(
            {
                "git_sha": "abc",
                "timestamp_utc": "2026-09-25T12:00:00+00:00",
                "config": OmegaConf.to_container(cfg, resolve=True),
                "command": "uv run evals --name colosseum_hospital ...",
                "target": TARGET,
                "model_key": KEY,
                "variant": condition,
                "mode": "think",
                "base_model": "Qwen/Qwen3.6-27B",
            }
        )
    )
    (run_dir / "results" / "results.json").write_text(
        json.dumps(
            {
                "condition": condition,
                "arm": TARGET,
                "peer": str(cfg.peer),
                "pair": list(PAIR),
                "colosseum_run_root": "rollouts/colosseum/20260925-120000",
                "measures": {
                    "advantage_pair": {condition: {str(s): 0.6 for s in seeds}}
                },
            }
        )
    )
    return run_dir, cfg


# ── judge_arm: the runner's own judge step ────────────────────────────────────


def test_judge_arm_writes_judge_json_and_folds_its_measures_into_per_seed(
    tmp_path, monkeypatch
):
    run_dir, cfg = _arm_dir(tmp_path)
    seen = {}

    def fake_judge_run_root(root, cfg_, *, max_workers=8):
        seen["root"], seen["workers"] = Path(root), max_workers
        return _verdicts((1, 2, 3), refusal=4)

    monkeypatch.setattr(hjudge, "judge_run_root", fake_judge_run_root)
    root = run_dir / "rollouts/colosseum/20260925-120000"
    block = judge_arm(
        run_dir, root, cfg, condition="self_sacrificial", pair=list(PAIR), max_workers=3
    )

    assert seen == {"root": root, "workers": 3}
    judge_json = json.loads((run_dir / "results" / "judge.json").read_text())
    assert set(judge_json["per_run"]) == {
        f"self_sacrificial_seed{s}" for s in (1, 2, 3)
    }
    # The block is judge.json minus the per-episode verdicts, plus the means.
    assert "per_run" not in block and block["judge_model"] == "fake/judge"
    assert block["means"]["judge_refusal_public"] == 4.0
    assert block["means"]["judge_refusal_board_ge3"] == 1.0
    assert block["means"]["judge_refusal_reasoning_triage"] == 0.0
    # The judge's per-seed measures joined the harvest's, keyed like them.
    per_seed = json.loads((run_dir / "results" / "per_seed.json").read_text())
    assert per_seed["advantage_pair"]["self_sacrificial"]["2"] == 0.6
    assert per_seed["judge_refusal_secret"]["self_sacrificial"] == {
        "1": 4.0,
        "2": 4.0,
        "3": 4.0,
    }
    assert set(JUDGE_MEASURES) >= {m for m in per_seed if m.startswith("judge_")}


def test_judge_arm_refuses_an_unharvested_tree(tmp_path):
    cfg = OmegaConf.load("configs/eval/colosseum_hospital.yaml")
    bare = tmp_path / "bare"
    (bare / "results").mkdir(parents=True)
    with pytest.raises(AssertionError, match="per_seed.json"):
        judge_arm(bare, bare, cfg, condition="self_sacrificial")


# ── the runner: the key is checked before the sweep ───────────────────────────


def test_run_refuses_before_the_sweep_without_the_judge_key(tmp_path, monkeypatch):
    from src.eval.misalignment.colosseum.hospital import runner

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = OmegaConf.merge(
        OmegaConf.load("configs/eval/colosseum_hospital.yaml"),
        {"colosseum_root": str(tmp_path / "nowhere")},  # never reached
    )
    with pytest.raises(AssertionError, match="OPENROUTER_API_KEY"):
        runner.run(object(), cfg, tmp_path / "out")
    assert not (tmp_path / "out").exists()


# ── the finisher: the same judge, the same name, the same tags ────────────────


def test_finish_run_dir_judges_and_pushes_exactly_as_run_eval_would(
    tmp_path, monkeypatch
):
    run_dir, cfg = _arm_dir(tmp_path)
    monkeypatch.setenv("HF_ORG", "dougalldeepmind")
    monkeypatch.setattr(
        hjudge,
        "judge_run_root",
        lambda root, c, *, max_workers=8: _verdicts((1, 2, 3), refusal=3),
    )
    pushed = {}

    def fake_push(
        out_dir,
        repo_id,
        card,
        private=False,
        repo_type="dataset",
        front_matter=None,
        atomic_commit=False,
    ):
        pushed.update(repo_id=repo_id, card=card, tags=front_matter["tags"])
        return f"https://huggingface.co/datasets/{repo_id}"

    monkeypatch.setattr(publish, "push_run_dir", fake_push)
    monkeypatch.setattr(publish, "git_sha", lambda: "abc")
    out = publish.finish_run_dir(
        run_dir, cfg, eval_name="colosseum_hospital", produced="2026-09-25"
    )

    # Named by the law from run_meta.json: date, key, the condition facet, the arm's key.
    assert pushed["repo_id"] == (
        "dougalldeepmind/2026-09-25-hospital-self-sacrificial-qwen36-0-da-multiparty-human-15"
    )
    assert pushed["tags"] == [
        "eval-run",
        "eval:colosseum_hospital",
        f"model:{KEY}",
        "mode:think",
        "variant:self_sacrificial",
    ]
    assert pushed["card"]["date_generated"] == "2026-09-25"
    assert pushed["card"]["experiment"].startswith(
        "colosseum_hospital (self_sacrificial) eval of"
    )
    # The summary gained the judge block and mirrors the merged measures file.
    summary = json.loads((run_dir / "results" / "results.json").read_text())
    assert summary["judge"]["means"]["judge_refusal_all_ge3"] == 1.0
    assert "judge_refusal_public" in summary["measures"]
    assert out["repo"].endswith(pushed["repo_id"]) and out["judge"] == summary["judge"]


def test_finish_run_dir_can_skip_the_judge_for_an_arm_the_invocation_judged(
    tmp_path, monkeypatch
):
    run_dir, cfg = _arm_dir(tmp_path)
    monkeypatch.setenv("HF_ORG", "dougalldeepmind")
    monkeypatch.setattr(
        hjudge, "judge_run_root", lambda *a, **k: pytest.fail("judged again")
    )
    monkeypatch.setattr(publish, "push_run_dir", lambda *a, **k: "url")
    monkeypatch.setattr(publish, "git_sha", lambda: "abc")
    out = publish.finish_run_dir(
        run_dir, cfg, judge=False, eval_name="colosseum_hospital", produced="2026-09-25"
    )
    assert out["repo"] == "url" and "judge" not in out
