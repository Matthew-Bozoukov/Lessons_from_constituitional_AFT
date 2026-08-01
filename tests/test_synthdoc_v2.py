# ABOUTME: Offline tests for synthdoc_v2: constitution segmentation and SFT export.
# ABOUTME: Run: uv run pytest tests/test_synthdoc_v2.py -q

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.synthdoc_v2.constitution import segment  # noqa: E402
from data.synthdoc_v2.stages import cost_of, to_sft  # noqa: E402

CONSTITUTION = "constitutions/claude_constitution_principles.md"


def test_segments_into_eight_traits():
    traits, _ = segment(CONSTITUTION)
    assert len(traits) == 8
    assert [t.trait_id for t in traits] == [f"t{i}" for i in range(1, 9)]


def test_trait_names_and_text_are_populated():
    traits, _ = segment(CONSTITUTION)
    for t in traits:
        assert t.name and not t.name.endswith(".")
        assert len(t.text) > 60, f"{t.trait_id} text looks truncated: {t.text!r}"
        assert t.name.split()[0].lower() in t.text.lower()


def test_style_guidance_is_separate_from_traits():
    traits, style = segment(CONSTITUTION)
    assert "Deliberates openly" in style
    # The style section must not leak into any trait body.
    assert all("Deliberates openly" not in t.text for t in traits)


def test_sft_export_carries_reasoning_and_trait_metadata():
    rec = {
        "scenario_id": "t1_s000", "trait_id": "t1", "trait_name": "Honesty",
        "trait_text": "**Honesty.** Do not deceive.", "domain": "work",
        "shortcut": "lie to the auditor", "situation": "A person is under pressure.",
        "system": "You are a helpful assistant.", "user": "Should I lie?",
        "reasoning": "The tension is...", "response": "Here is what I'd consider...",
    }
    out = to_sft([rec])
    assert len(out) == 1
    msgs = out[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[2]["reasoning_content"] == "The tension is..."
    assert out[0]["metadata"]["trait_id"] == "t1"
    assert out[0]["metadata"]["trait_name"] == "Honesty"


def test_cost_of_prices_known_models_and_zeroes_unknown():
    assert cost_of("openai/gpt-5.6-luna", 1_000_000, 1_000_000) == 0.10 + 0.60
    assert cost_of("openai/gpt-5.6-terra", 1_000_000, 0) == 1.00
    assert cost_of("some/unlisted-model", 1_000_000, 1_000_000) == 0.0


def test_checkpoint_survives_abort_and_resume_skips_completed_work(tmp_path):
    from data.synthdoc_v2.stages import Checkpoint, _run_items

    items = [{"scenario_id": f"s{i}", "v": i} for i in range(100)]
    path = tmp_path / "partial.jsonl"

    def flaky(it):
        if it["scenario_id"] in ("s7", "s42", "s88"):
            raise ValueError("simulated malformed JSON")
        return {**it, "out": it["v"] * 2}

    ck = Checkpoint(path)
    try:
        _run_items(items, flaky, workers=8, desc="t", ckpt=ck)
    except RuntimeError:
        pass  # 3% failure trips the guard, which is the crash we are protecting against
    assert len(ck.done) == 97, "completed work was lost when the stage aborted"

    calls = []

    def good(it):
        calls.append(it["scenario_id"])
        return {**it, "out": it["v"] * 2}

    out = _run_items(items, good, workers=8, desc="t", ckpt=Checkpoint(path))
    assert len(out) == 100
    assert sorted(calls) == ["s42", "s7", "s88"], "resume re-ran already-saved items"
    assert [r["out"] for r in out] == [i * 2 for i in range(100)]
