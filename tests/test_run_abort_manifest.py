# ABOUTME: Offline test that a stage which raises mid-run still leaves a manifest behind --
# ABOUTME: with `aborted` naming the stage and the per-stage failure tallies -- instead of
# ABOUTME: taking the run's record with it (added 2026-08-26 after a gate trip left none).
# ABOUTME: Run: uv run pytest tests/test_run_abort_manifest.py -q

from __future__ import annotations

import json

import pytest

from src.data.synth import pipeline, stage_runtime
from src.infra.endpoints.openrouter import ChatResult, ProviderRejectionError


class _Client:
    """Every call is rejected by the provider: the stage's failure gate must trip."""

    def chat(self, model, messages, temperature=1.0, max_tokens=4096, **kw):
        raise ProviderRejectionError("blocked by content filter")


def test_a_stage_that_raises_still_writes_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_runtime, "OpenRouterClient", lambda: _Client())
    cfg = {
        "pipeline": "abort_test",
        "constitution": "constitutions/claude_distilled_12_principles_mid/constitution.md",
        "chunking": "principle",
        "output_dir": str(tmp_path),
        "hf_repo": None,
        "hf_private": False,
        "workers": 1,
        "max_fail_pct": 2.0,
        "defaults": {"temperature": 1.0, "max_tokens": 100},
        "models": {"m": {"model": "x/y", "max_tokens": 100}},
        "stages": [
            {"name": "chunk_constitution", "kind": "segment"},
            {
                "name": "label",
                "kind": "llm_tagged",
                "model": "m",
                "tags": ["reply"],
                "save": {"reply": "reply"},
                "prompts": {"system": "s", "user": "{name}"},
            },
        ],
    }
    with pytest.raises(RuntimeError, match="above max_fail_pct"):
        pipeline.run(cfg)
    run_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1
    m = json.loads((run_dirs[0] / "manifest.json").read_text())
    assert m["aborted"]["stage"].endswith("(label)")
    assert "above max_fail_pct" in m["aborted"]["error"]
    assert m["dataset"] is None
    # The tally that motivated this: which exception class lost the rows, and how many.
    assert m["failures"]["label"]["by_type"] == {"ProviderRejectionError": 9}
    assert m["failures"]["label"]["n"] == 9 and m["failures"]["label"]["of"] == 9
    # The stages that did complete are counted.
    assert m["counts"]["chunk_constitution"] == 9
