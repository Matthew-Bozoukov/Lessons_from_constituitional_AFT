# ABOUTME: Offline tests for the difficult-advice pipeline and shared synth core.
# ABOUTME: Run: uv run pytest tests/test_difficult_advice.py -q

from __future__ import annotations



from src.data.synth.constitution import segment  # noqa: E402
from src.data.synth.stage_runtime import cost_of  # noqa: E402

CONSTITUTION = "constitutions/archive/claude_distilled_8_principles_v1/constitution.md"


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


def test_segments_specgen_heading_format():
    # The default constitution since 2026-08-03: specgen's mid arm, whose units are
    # numbered H2 headings rather than v1's bolded list items.
    import re
    from pathlib import Path

    path = "constitutions/claude_distilled_12_principles_mid/constitution.md"
    # Counted from the document rather than hardcoded: this file has been re-cut once
    # already (twelve units -> ten on 2026-08-04, while keeping its folder name), and a
    # literal here just goes red without telling anyone whether segmentation still works.
    expected = sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines()
                   if re.match(r"^##\s+\d+\.", line))
    traits, style = segment(path)
    assert len(traits) == expected
    assert [t.trait_id for t in traits] == [f"t{i}" for i in range(1, expected + 1)]
    for t in traits:
        assert t.name and len(t.text) > 60, f"{t.trait_id} looks truncated"
        assert "*Why:*" in t.text, f"{t.trait_id} lost its rationale block"
    # The closing aligned-response section is style guidance, not a trait; the
    # un-numbered preamble heading must not become a trait either.
    assert "Engages with the pressure" in style
    assert all("holistic, not strict" not in t.text for t in traits)


def test_sft_export_carries_reasoning_and_trait_metadata():
    # The export is now the real config's chat_export stage -- test through it.
    import yaml

    from src.data.synth.stage_operators import op_chat_export

    sft_spec = next(s for s in yaml.safe_load(
        open("configs/data/synth/difficult_advice.yaml"))["stages"] if s["name"] == "export_sft")
    rec = {
        "scenario_id": "t1_s000", "trait_id": "t1", "trait_name": "Honesty",
        "trait_text": "**Honesty.** Do not deceive.", "domain": "work",
        "shortcut": "lie to the auditor", "situation": "A person is under pressure.",
        "system": "You are a helpful assistant.", "user": "Should I lie?",
        "reasoning": "The tension is...", "response": "Here is what I'd consider...",
    }
    out = op_chat_export(sft_spec, {}).fn(None, [rec], None)
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
    from src.data.synth.stage_runtime import Checkpoint, run_items

    items = [{"scenario_id": f"s{i}", "v": i} for i in range(100)]
    path = tmp_path / "partial.jsonl"

    def flaky(it):
        if it["scenario_id"] in ("s7", "s42", "s88"):
            raise ValueError("simulated malformed JSON")
        return {**it, "out": it["v"] * 2}

    ck = Checkpoint(path)
    try:
        run_items(items, flaky, workers=8, desc="t", ckpt=ck)
    except RuntimeError:
        pass  # 3% failure trips the guard, which is the crash we are protecting against
    assert len(ck.done) == 97, "completed work was lost when the stage aborted"

    calls = []

    def good(it):
        calls.append(it["scenario_id"])
        return {**it, "out": it["v"] * 2}

    out = run_items(items, good, workers=8, desc="t", ckpt=Checkpoint(path))
    assert len(out) == 100
    assert sorted(calls) == ["s42", "s7", "s88"], "resume re-ran already-saved items"
    assert [r["out"] for r in out] == [i * 2 for i in range(100)]
