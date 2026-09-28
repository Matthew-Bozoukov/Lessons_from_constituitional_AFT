# ABOUTME: The scenarios stage's `diversity:` block -- the ban list carried between waves
# ABOUTME: and the embedding gate that refuses a scenario too close to one already kept.

"""What these pin down.

Plain fan-out generates every batch concurrently and blind, which is why the baseline
corpus reached 46.8% of its mass in ten domains and kept a 0.886-cosine clone pair. The
`diversity:` block adds two mechanisms and both are tested here:

- the PROMPT-side ban list, which must actually reach the generator and must grow with
  what earlier waves produced (asking);
- the EMBEDDING gate, which must drop a near-duplicate the generator wrote anyway, and
  must re-queue the shortfall so rejection cannot quietly unbalance the corpus
  (enforcing).

The gate is the half that matters: the baseline corpus proves asking alone does not work.
"""

from __future__ import annotations

import pytest

from src.data.synth.ours import stage_operators as ops
from src.data.synth.ours.stage_runtime import Ctx, Usage

TRAITS = [{"trait_id": "t1", "index": 0, "name": "Oversight", "text": "..."},
          {"trait_id": "t2", "index": 1, "name": "Honesty", "text": "..."}]

# Deliberately unrelated so nothing collides except where a test makes it.
DISTINCT = ["A nurse must decide whether to log an overtime shift she did not work",
            "A cyclist is offered a courier contract that requires ignoring red lights",
            "A baker is asked to relabel yesterday's bread as fresh for a food festival",
            "A violinist is pressed to claim a prize entry was composed unaided",
            "A surveyor is told to sign off on a boundary he never physically walked",
            "A translator is asked to soften a witness statement for a sympathetic client",
            "A brewer is offered cheaper hops if he keeps the origin off the label",
            "A librarian is asked to quietly discard donations from a disliked patron"]


def _stage(**div):
    return {"name": "scenarios", "model": "scenarios",
            "prompts": {"system": "sys {avoid}",
                        "user": "make {n} {avoid} {overrepresented}"},
            "diversity": div}


def _cfg(**over):
    return {"seed": 0, "scenarios_per_trait": 4, "scenarios_per_call": 2,
            "models": {"scenarios": {"model": "m", "temperature": 1.0,
                                     "max_tokens": 100}}, **over}


def _run(monkeypatch, tmp_path, stage, cfg, replies, capture=None, domains=None):
    """Drive the operator with a scripted generator. `replies` is a list of lists of
    situation strings, consumed one per call in call order; `domains` optionally pins the
    domain label every scenario carries, for the concentration tests."""
    box = list(replies)

    def fake_call_json(client, usage, model, system, user, temp, max_tokens, stage=None,
                       extra=None):
        if capture is not None:
            capture.append({"system": system, "user": user})
        batch = box.pop(0) if box else []
        return [{"domain": domains or f"d{i}", "situation": s, "shortcut": "x"}
                for i, s in enumerate(batch)], {}

    monkeypatch.setattr(ops, "call_json", fake_call_json)
    st = ops.OPERATORS["scenarios"](stage, cfg)
    ctx = Ctx(cfg=cfg, usage=Usage(), workers=1, run_dir=tmp_path, smoke=False,
              _client=object())  # Scripted calls must not require API credentials.
    return st.fn(ctx, TRAITS, None), ctx


def test_no_diversity_block_keeps_plain_fanout(monkeypatch, tmp_path):
    """The unchanged path: no block, no gate, every scenario kept."""
    stage = {"name": "scenarios", "model": "scenarios",
             "prompts": {"system": "sys", "user": "make {n}"}}
    out, ctx = _run(monkeypatch, tmp_path, stage, _cfg(),
                    [DISTINCT[:2]] * 4)
    assert len(out) == 8
    assert "scenario_diversity" not in ctx.manifest_extra


def test_seeded_avoid_list_reaches_the_generator(monkeypatch, tmp_path):
    seen: list[dict] = []
    stage = _stage(avoid=["academic research: a postdoc facing a funding cliff"],
                   wave_size=4)
    _run(monkeypatch, tmp_path, stage, _cfg(), [DISTINCT[:2]] * 4, capture=seen)

    assert seen, "the generator was never called"
    assert "a postdoc facing a funding cliff" in seen[0]["user"]
    assert "Do not write another version" in seen[0]["user"]


def test_the_ban_list_grows_with_what_earlier_waves_produced(monkeypatch, tmp_path):
    """Wave 2 must be told what wave 1 actually wrote -- the whole point of waves."""
    seen: list[dict] = []
    stage = _stage(wave_size=2, reject_cosine=0.0)
    _run(monkeypatch, tmp_path, stage, _cfg(),
         [DISTINCT[0:2], DISTINCT[2:4], DISTINCT[4:6], DISTINCT[6:8]], capture=seen)

    first_wave_prompt, later_wave_prompt = seen[0]["user"], seen[-1]["user"]
    assert "A nurse must decide" not in first_wave_prompt
    assert "A nurse must decide" in later_wave_prompt


def test_the_gate_rejects_a_near_duplicate_and_refills(monkeypatch, tmp_path):
    """A generator that ignores the ban list must still not get its duplicate in."""
    dup = DISTINCT[0]
    # Wave 1 writes two distinct; wave 2 re-writes the first one verbatim plus a new one;
    # the make-up round then supplies a genuinely new one.
    replies = [DISTINCT[0:2], [dup, DISTINCT[2]], DISTINCT[3:5], DISTINCT[5:7],
               DISTINCT[7:8], DISTINCT[6:7]]
    stage = _stage(wave_size=1, reject_cosine=0.9, max_regen_rounds=3)
    out, ctx = _run(monkeypatch, tmp_path, stage, _cfg(), replies)

    entry = ctx.manifest_extra["scenario_diversity"]["scenarios"]
    assert entry["rejected"] >= 1, entry
    situations = [r["situation"] for r in out]
    assert situations.count(dup) == 1, "the duplicate survived the gate"


def test_rejection_never_silently_unbalances_the_traits(monkeypatch, tmp_path):
    """Every kept trait must reach its target, or the shortfall must be reported."""
    stage = _stage(wave_size=2, reject_cosine=0.9, max_regen_rounds=3)
    out, ctx = _run(monkeypatch, tmp_path, stage, _cfg(),
                    [DISTINCT[i:i + 2] for i in range(0, 8, 2)])

    per_trait: dict[str, int] = {}
    for r in out:
        per_trait[r["trait_id"]] = per_trait.get(r["trait_id"], 0) + 1
    assert per_trait == {"t1": 4, "t2": 4}
    assert ctx.manifest_extra["scenario_diversity"]["scenarios"]["shortfall"] == 0


def test_a_generator_stuck_on_one_scenario_shrinks_the_corpus(monkeypatch, tmp_path):
    """It must never pad with near-duplicates to hit the target count."""
    stage = _stage(wave_size=4, reject_cosine=0.9, max_regen_rounds=1)
    out, ctx = _run(monkeypatch, tmp_path, stage, _cfg(),
                    [[DISTINCT[0], DISTINCT[0]]] * 12)

    entry = ctx.manifest_extra["scenario_diversity"]["scenarios"]
    assert len(out) < 8, "duplicates were padded in to hit the target"
    assert entry["shortfall"] > 0
    assert [r["situation"] for r in out] == [DISTINCT[0]]


def test_an_overrepresented_domain_is_fed_back_into_the_next_wave(monkeypatch, tmp_path):
    """The corrective loop: nothing is banned up front, but a domain running hot in this
    run's own output gets named to the generator that keeps producing it."""
    seen: list[dict] = []
    cfg = _cfg(scenarios_per_trait=8, scenarios_per_call=2)
    stage = _stage(wave_size=1, reject_cosine=0.0, over_share=0.4, over_min_docs=4)
    _run(monkeypatch, tmp_path, stage, cfg,
         [DISTINCT[i % 8:i % 8 + 2] or DISTINCT[:2] for i in range(8)],
         capture=seen, domains="small business")

    assert "over-represented" not in seen[0]["user"], "flagged before any evidence"
    late = seen[-1]["user"]
    assert "over-represented" in late and "small business" in late
    assert "% of the corpus so far" in late


def test_no_domain_is_named_before_over_min_docs(monkeypatch, tmp_path):
    """Early shares are noise: 1 of 2 scenarios is 50% and means nothing."""
    seen: list[dict] = []
    stage = _stage(wave_size=1, reject_cosine=0.0, over_share=0.1, over_min_docs=100)
    _run(monkeypatch, tmp_path, stage, _cfg(), [DISTINCT[i:i + 2] for i in range(0, 8, 2)],
         capture=seen, domains="small business")

    assert all("over-represented" not in c["user"] for c in seen)


def test_a_domain_under_the_share_is_never_named(monkeypatch, tmp_path):
    seen: list[dict] = []
    stage = _stage(wave_size=1, reject_cosine=0.0, over_share=0.9, over_min_docs=2)
    _run(monkeypatch, tmp_path, stage, _cfg(), [DISTINCT[i:i + 2] for i in range(0, 8, 2)],
         capture=seen)

    assert all("over-represented" not in c["user"] for c in seen)


def test_the_manifest_records_the_concentration_that_resulted(monkeypatch, tmp_path):
    """The number the block exists to move, recorded beside the settings that moved it."""
    stage = _stage(wave_size=4, reject_cosine=0.0, over_share=0.04)
    _, ctx = _run(monkeypatch, tmp_path, stage, _cfg(),
                  [DISTINCT[i:i + 2] for i in range(0, 8, 2)], domains="small business")

    entry = ctx.manifest_extra["scenario_diversity"]["scenarios"]
    assert entry["distinct_domains"] == 1
    assert entry["top10_domain_share"] == 1.0
    assert entry["top_domains"] == {"small business": 8}
    assert entry["over_share"] == 0.04


def test_gist_names_the_family_without_quoting_the_whole_scenario():
    line = ops._gist({"domain": "academic research",
                      "situation": "A postdoctoral researcher has spent three years on a "
                                   "promising cancer drug study with borderline results"})
    assert line.startswith("academic research: A postdoctoral researcher")
    assert "borderline" not in line


# A reworded DISTINCT[0]: no shared 4-gram, so lexical shingles cannot see it. Measured
# at cosine 0.644 against its original under potion-base-8M (unrelated pairs here score
# ~0.05). These strings are ~13 words and cosine is length-dependent -- the SAME pair of
# 68-word scenarios sits far higher, which is why the shipped `cosine_min` is 0.86 and
# these thresholds are local to the fixture rather than copied from the config.
PARAPHRASE = "A nurse is weighing whether to record hours of overtime she never actually worked"


@pytest.mark.parametrize("cosine,expected_rejects", [(0.9, 0), (0.5, 1)])
def test_the_gate_threshold_is_what_decides(monkeypatch, tmp_path, cosine,
                                            expected_rejects):
    """The same reworded scenario is kept or refused purely on `reject_cosine`."""
    stage = _stage(wave_size=1, reject_cosine=cosine, max_regen_rounds=0)
    replies = [DISTINCT[0:2], [PARAPHRASE, DISTINCT[2]], DISTINCT[3:5], DISTINCT[5:7]]
    _, ctx = _run(monkeypatch, tmp_path, stage, _cfg(), replies)
    assert ctx.manifest_extra["scenario_diversity"]["scenarios"]["rejected"] == \
        expected_rejects


def test_the_gate_catches_a_reword_that_shares_no_ngram(monkeypatch, tmp_path):
    """The case lexical dedup provably misses: same story, no shared 4-gram."""
    from src.data.synth.ours.check_corpus import ngrams, words

    assert not (ngrams(words(DISTINCT[0]), 4) & ngrams(words(PARAPHRASE), 4))

    stage = _stage(wave_size=1, reject_cosine=0.5, max_regen_rounds=0)
    out, _ = _run(monkeypatch, tmp_path, stage, _cfg(),
                  [DISTINCT[0:2], [PARAPHRASE, DISTINCT[2]], DISTINCT[3:5],
                   DISTINCT[5:7]])
    assert PARAPHRASE not in [r["situation"] for r in out]


def test_batched_waves_feed_the_same_diversity_machinery(monkeypatch, tmp_path):
    """With --batch, a wave's calls go through the async batch API instead of live
    calls, but the SAME reject/steer logic runs on what comes back. This drives the
    batch fetch path and asserts every scenario lands in the diversity machinery: the
    interactive call_json is never touched, and the kept set + manifest match."""
    import json as _json

    from src.data.synth.ours import stage_runtime
    from src.infra.endpoints import openrouter

    # 16 genuinely distinct situations, one per request (per_call=1, per_trait=8, 2
    # traits) — unrelated so the real reject gate keeps all of them.
    sits = list(DISTINCT) + [
        "A pilot is pressured to skip a preflight check to keep the schedule",
        "A pharmacist is asked to backdate a prescription for a regular customer",
        "A referee is offered a bonus to overlook a foul in the closing minutes",
        "A tailor is told to stitch a designer label into an unbranded coat",
        "A gardener is asked to spray a banned pesticide before an estate sale",
        "An electrician is pressed to certify wiring he was not allowed to inspect",
        "A sommelier is asked to pass off a cheaper vintage at a charity gala",
        "A cartographer is told to redraw a disputed border to please a patron"]

    monkeypatch.setattr(openrouter, "build_request_body",
                        lambda model, messages, temperature, max_tokens,
                        extra_body=None: {"messages": messages}, raising=True)

    class _R:
        def __init__(self, content):
            self.content, self.prompt_tokens, self.completion_tokens = content, 1, 1
            self.finish_reason, self.cached_tokens, self.provider = "stop", 0, ""

    monkeypatch.setattr(openrouter, "result_from_payload",
                        lambda model, payload: _R(payload["content"]), raising=True)

    batch_calls = {"n": 0}

    def fake_run_batch(model, requests, stage, state_path, collect, **kw):
        batch_calls["n"] += 1
        for cid in requests:
            arr = [{"domain": "d", "situation": sits.pop(0), "shortcut": "x"}] if sits else []
            collect(cid, {"content": _json.dumps(arr)})

    monkeypatch.setattr(stage_runtime, "run_batch", fake_run_batch, raising=True)

    def no_interactive(*a, **k):
        raise AssertionError("interactive call_json used despite --batch")

    monkeypatch.setattr(ops, "call_json", no_interactive)

    stage = _stage(wave_size=16, reject_cosine=0.86, max_regen_rounds=0)
    cfg = _cfg(scenarios_per_trait=8, scenarios_per_call=1, batch=True)
    st = ops.OPERATORS["scenarios"](stage, cfg)
    ctx = Ctx(cfg=cfg, usage=Usage(), workers=4, run_dir=tmp_path, smoke=False)
    out = st.fn(ctx, TRAITS, None)

    assert batch_calls["n"] >= 1, "the batch path was never taken"
    assert len(out) == 16, f"expected 16 kept, got {len(out)}"
    assert ctx.manifest_extra["scenario_diversity"]["scenarios"]["kept"] == 16
    assert "scenarios" in ctx.manifest_extra["batched_stages"]



def test_a_trait_note_reaches_only_its_own_unit(monkeypatch, tmp_path):
    """`trait_notes` is how a fix for one principle stays out of every other unit's prompt."""
    seen: list[dict] = []
    stage = {"name": "scenarios", "model": "scenarios",
             "prompts": {"system": "sys", "user": "make {n}\n{trait_note}\nfor {trait_name}"}}
    _run(monkeypatch, tmp_path, stage, _cfg(trait_notes={"t2": "- ONLY-FOR-HONESTY"}),
         [DISTINCT[:2]] * 4, capture=seen)
    for_t2 = [c["user"] for c in seen if "for Honesty" in c["user"]]
    for_t1 = [c["user"] for c in seen if "for Oversight" in c["user"]]
    assert for_t1 and for_t2
    assert all("ONLY-FOR-HONESTY" in u for u in for_t2)
    assert not any("ONLY-FOR-HONESTY" in u for u in for_t1)


def test_the_default_note_reaches_every_unit_without_its_own(monkeypatch, tmp_path):
    """A unit with its own note gets only that one; every other unit gets `default`."""
    seen: list[dict] = []
    stage = {"name": "scenarios", "model": "scenarios",
             "prompts": {"system": "sys", "user": "make {n}\n{trait_note}\nfor {trait_name}"}}
    _run(monkeypatch, tmp_path, stage,
         _cfg(trait_notes={"default": "- GENERIC", "t2": "- ONLY-FOR-HONESTY"}),
         [DISTINCT[:2]] * 4, capture=seen)
    for_t1 = [c["user"] for c in seen if "for Oversight" in c["user"]]
    for_t2 = [c["user"] for c in seen if "for Honesty" in c["user"]]
    assert all("GENERIC" in u and "ONLY-FOR-HONESTY" not in u for u in for_t1)
    assert all("ONLY-FOR-HONESTY" in u and "GENERIC" not in u for u in for_t2)


def test_a_trait_note_slot_with_no_notes_configured_is_refused():
    """Refused when the stage is built, before any call is paid for."""
    stage = {"name": "scenarios", "model": "scenarios",
             "prompts": {"system": "sys", "user": "make {n} {trait_note}"}}
    with pytest.raises(ValueError, match="trait_note"):
        ops.OPERATORS["scenarios"](stage, _cfg())


def test_trait_notes_on_a_stage_rather_than_top_level_is_refused():
    stage = {"name": "scenarios", "model": "scenarios", "trait_notes": {"t2": "x"},
             "prompts": {"system": "sys", "user": "make {n} {trait_note}"}}
    with pytest.raises(ValueError, match="top-level"):
        ops.OPERATORS["scenarios"](stage, _cfg())


def test_a_trait_note_for_a_unit_the_run_lacks_is_refused(monkeypatch, tmp_path):
    stage = {"name": "scenarios", "model": "scenarios",
             "prompts": {"system": "sys", "user": "make {n} {trait_note}"}}
    with pytest.raises(ValueError, match="t9"):
        _run(monkeypatch, tmp_path, stage, _cfg(trait_notes={"t9": "x"}), [DISTINCT[:2]] * 4)


def test_any_later_stage_renders_the_same_note_by_the_records_trait(tmp_path):
    """The reviser must see what the writer saw, or it undoes the note (2026-09-28, t1)."""
    cfg = _cfg(trait_notes={"default": "- GENERIC", "t1": "- OVERSIGHT-NOTE"})
    ctx = Ctx(cfg=cfg, usage=Usage(), workers=1, run_dir=tmp_path, smoke=False, _client=object())
    tpl = "revise this\n{trait_note}\nfor {trait_id}"
    assert "- OVERSIGHT-NOTE" in ops._render(tpl, {"trait_id": "t1"}, ctx)
    assert "- GENERIC" in ops._render(tpl, {"trait_id": "t5"}, ctx)
    with pytest.raises(ValueError, match="trait_id"):
        ops._render(tpl, {}, ctx)
