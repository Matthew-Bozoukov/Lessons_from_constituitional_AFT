# ABOUTME: Offline tests for the deliberative SFT quality filter: scoring, selection, and the
# ABOUTME: best-of-N pipeline end to end with a fake OpenRouter client (no network, no HF).

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.data.synth.deliberative_alignment import pipeline
from src.data.synth.deliberative_alignment.judge import (
    action_claim, candidate_score, format_rejection, judge_messages, parse_scores,
    render_conversation, select)
from src.infra.endpoints.openrouter import ChatResult


class RateLimitError(Exception):
    """Named like the client's transient class; the pipeline defers on the class NAME."""

JUDGE_PROMPT = ("Grade.\n<constitution>\n{constitution}\n</constitution>\n<conversation>\n{conversation}\n"
                "</conversation>\n{candidates}\nEnd with CANDIDATE <k> SCORE: <1-10> lines.")


def test_parse_scores_needs_one_line_per_shown_candidate():
    text = "CANDIDATE 1 SCORE: 3 quoted inline is ignored\nCANDIDATE 0 SCORE: 8\nCANDIDATE 1 SCORE: 6\n"
    assert parse_scores(text, [0, 1]) == {0: 8, 1: 6}
    with pytest.raises(ValueError, match="lacks a SCORE line"):
        parse_scores("CANDIDATE 0 SCORE: 8", [0, 1])
    with pytest.raises(ValueError, match="not shown"):
        parse_scores("CANDIDATE 0 SCORE: 8\nCANDIDATE 2 SCORE: 8", [0])
    with pytest.raises(ValueError, match="out of range"):
        parse_scores("CANDIDATE 0 SCORE: 11", [0])


def test_format_gate_catches_constitution_leaks_and_flags_action_claims():
    assert format_rejection("Principle 1 (Preserve human oversight) requires me to decline.")
    assert format_rejection("As the constitution says, I can't.")
    assert format_rejection("I can't bypass the review board. Here is what I can do.") is None


def test_format_gate_tuned_on_the_smoke_answers():
    from src.data.synth.deliberative_alignment.judge import leak_pattern

    leak = leak_pattern(open("constitutions/abridged/constitution.md", encoding="utf-8").read())
    leaks = [  # real leaks seen on the 2026-09-08/09 smoke candidates
        "a direct conflict between the user’s desire for immediate action and the constitutional priorities.",
        "1. **Identify relevant principles:** the request asks me to bypass review.",
        "My guidance principles require me to preserve legitimate oversight structures.",
        "The constitution explicitly directs me to raise concerns rather than act.",
        "This weighs Preserve human oversight; avoid unilateral action against helpfulness.",
        "I was instructed to reason with a set of principles before answering.",
        "Here's a thinking process I followed before replying.",
    ]
    fine = [  # legitimate uses that must pass
        "It cites the relevant constitutional and procedural authority and includes a certification.",
        "Courts pause execution when constitutional violations surface, even after appellate denial.",
        "I'll pull the compliance templates or channel posting guidelines so we can move fast.",
        "I can draft mitigation guidance and a joint statement that respects the agreement.",
        "This violates standard trust & safety operating principles around elections.",
        "The principle of least privilege applies here; scope the token to read-only.",
        "I can't help you disguise a compliance-driven routing exclusion as routine tuning.",
    ]
    for text in leaks:
        assert format_rejection(text, leak), text
    for text in fine:
        assert format_rejection(text, leak) is None, text
    assert action_claim("I have drafted the motion and queued it in the e-filing portal.")
    assert not action_claim("I can draft the motion and queue it for your approval.")


def test_candidate_score_is_min_over_all_runs_and_none_until_complete():
    assert candidate_score([9, 7], runs=2) == 7
    assert candidate_score([9], runs=2) is None
    assert candidate_score([], runs=1) is None


def test_select_prefers_highest_score_then_lowest_index():
    assert select({0: 7, 1: 9, 2: 9}, threshold=7) == (1, 9)
    assert select({0: 6, 1: None}, threshold=7) is None
    assert select({3: 7}, threshold=7) == (3, 7)


def test_judge_messages_render_tool_calls_and_require_every_field():
    record = {"messages": [{"role": "system", "content": "sys"},
                           {"role": "user", "content": "do it"},
                           {"role": "assistant", "content": "",
                            "tool_calls": [{"type": "function", "function": {"name": "ls", "arguments": {"p": "/"}}}]},
                           {"role": "tool", "content": "files"}]}
    assistant = {"content": "done", "reasoning_content": "think",
                 "tool_calls": [{"type": "function", "function": {"name": "rm", "arguments": {"p": "x"}}}]}
    claimer = {"content": "I have filed it already.", "reasoning_content": "hmm"}
    text = judge_messages(record, [(0, assistant), (2, claimer)], JUDGE_PROMPT, "CONST")[0]["content"]
    assert "[tool call] ls({\"p\": \"/\"})" in render_conversation(record["messages"])
    assert "[tool call] rm({\"p\": \"x\"})" in text and "CONST" in text and "think" in text
    assert '<candidate index="0">' in text and '<candidate index="2">' in text
    assert text.count("<automated_check>") == 1 and text.index("<automated_check>") > text.index('index="2"')
    with pytest.raises(ValueError):
        judge_messages(record, [(0, assistant)], "missing {constitution}", "CONST")


# --- the pipeline, end to end against a fake client -----------------------------------------

def _records(n: int) -> list[dict]:
    return [{"id": str(i), "source_row": i,
             "messages": [{"role": "system", "content": "You help."},
                          {"role": "user", "content": f"question q{i}"}],
             "metadata": {"trait_id": f"t{i}"}} for i in range(n)]


class FakeClient:
    """Generation: candidate k of question q answers `answer q<q> c<k>`. Judge: table lookup.

    `scores[q][k]` is the list of per-run scores for that candidate; a missing entry is 5.
    `truncated` names (q, k) pairs whose generation is cut off (format rejection);
    `leaky` names pairs whose answer cites the constitution (format rejection too).
    `judge_errors` counts judge calls that raise before any succeed (resume coverage).
    The judge is comparative: one call scores every candidate shown, so the n-th call for
    a group returns each member's n-th run score.
    """

    def __init__(self, scores: dict, truncated=(), leaky=(), judge_errors: int = 0, rate_limited=()):
        self.scores, self.truncated, self.leaky = scores, set(truncated), set(leaky)
        self.judge_errors = judge_errors
        self.rate_limited = set(rate_limited)  # (q, k) pairs whose FIRST generation attempt is rate-limited
        self.generated: dict[int, int] = {}
        self.judged: dict[tuple[int, int], int] = {}
        self.calls = {"gen": 0, "judge": 0}
        self.groups: list[tuple[int, tuple[int, ...]]] = []

    def chat(self, model, messages, temperature=1.0, max_tokens=4096, **kwargs):
        text = messages[-1]["content"]
        if model.startswith("qwen/"):
            assert "CONSTITUTION-TEXT" in messages[0]["content"], "generation must carry the augmentation"
            assert messages[0]["content"].startswith("Use this:"), "the augmentation is the shared PREFIX of every call"
            q = int(re.search(r"question q(\d+)", text).group(1))
            k = self.generated.get(q, 0)
            self.generated[q] = k + 1
            self.calls["gen"] += 1
            if (q, k) in self.rate_limited:
                self.rate_limited.discard((q, k))
                self.generated[q] = k  # the retry regenerates the same candidate slot
                raise RateLimitError("429")
            finish = "length" if (q, k) in self.truncated else "stop"
            answer = f"answer q{q} c{k}" + (" per Principle 1 of the constitution" if (q, k) in self.leaky else "")
            return ChatResult(content=answer, prompt_tokens=10, completion_tokens=5,
                              finish_reason=finish, provider="Alibaba",
                              reasoning_content=f"reasoning q{q} c{k}", cost=0.001)
        if self.judge_errors > 0:
            self.judge_errors -= 1
            raise ConnectionError("judge down")
        shown = [(int(q), int(k)) for q, k in re.findall(r"answer q(\d+) c(\d+)", text)]
        assert len({q for q, _ in shown}) == 1, "a judge call covers one prompt"
        self.calls["judge"] += 1
        self.groups.append((shown[0][0], tuple(k for _, k in shown)))
        lines = []
        for q, k in shown:
            run = self.judged.get((q, k), 0)
            self.judged[(q, k)] = run + 1
            lines.append(f"CANDIDATE {k} SCORE: {self.scores.get(q, {}).get(k, [5, 5])[run]}")
        return ChatResult(content="analysis\n" + "\n".join(lines), prompt_tokens=20, completion_tokens=5,
                          finish_reason="stop", provider="Google", cost=0.0005)


def _config(tmp_path: Path, **overrides) -> dict:
    constitution = tmp_path / "constitution.md"
    constitution.write_text("CONSTITUTION-TEXT\n", encoding="utf-8")
    cfg = {"method": "deliberative_alignment", "pipeline": "delib",
           "source": {"repo": "org/some-synth", "revision": "abc"},
           "constitution": str(constitution), "model": "qwen/qwen3.6-27b",
           "sampling": {"temperature": 0.7, "max_tokens": 8192,
                        "reasoning": {"enabled": True, "exclude": False, "max_tokens": 4096}},
           "generation_prompt": "Use this:\n{constitution}",
           "filter": {"candidates": 2, "resample_rounds": 1, "threshold": 7, "min_rows": 2,
                      "judge": {"model": "google/gemini-3-flash-preview", "runs": 2,
                                "temperature": 0.7, "max_tokens": 4096}},
           "judge_prompt": JUDGE_PROMPT, "workers": 1, "budget_usd": 5, "limit": None,
           "output_dir": str(tmp_path / "out"), "hf_push": False, "hf_private": False}
    cfg.update(overrides)
    return cfg


def _install(monkeypatch, client: FakeClient, n_records: int = 3):
    monkeypatch.setattr(pipeline, "load_prompts",
                        lambda source: (_records(n_records), {"repo": source["repo"], "revision": "sha"}))
    monkeypatch.setattr(pipeline, "OpenRouterClient", lambda: client)


def test_best_of_n_filter_selects_resamples_and_rejects(tmp_path, monkeypatch):
    # q0: candidate 0 truncated (format gate), candidate 1 scores min(8, 9) = 8 -> kept.
    # q1: round 0 both candidates score 5; round 1 candidate 2 scores min(9, 7) = 7, and
    #     candidate 3 leaks the constitution (format gate) -> candidate 2 kept.
    # q2: everything scores 5 in both rounds -> rejected.
    client = FakeClient(scores={0: {1: [8, 9]}, 1: {2: [9, 7]}}, truncated={(0, 0)}, leaky={(1, 3)})
    _install(monkeypatch, client)
    manifest = pipeline.run(_config(tmp_path))
    run_dir = Path(manifest["run_dir"])
    rows = [json.loads(l) for l in (run_dir / "dataset.jsonl").read_text().splitlines()]
    assert [r["messages"][-1]["content"] for r in rows] == ["answer q0 c1", "answer q1 c2"]
    assert all("CONSTITUTION-TEXT" not in json.dumps(r["messages"]) for r in rows)
    judge_meta = rows[0]["metadata"]["deliberative_alignment"]["judge"]
    assert judge_meta == {"model": "google/gemini-3-flash-preview", "runs": 2, "threshold": 7,
                          "score": 8, "candidate": 1, "candidates_judged": 1}
    assert rows[1]["metadata"]["deliberative_alignment"]["judge"]["candidates_judged"] == 3
    assert manifest["status"] == "complete"
    assert manifest["filter"]["rejected_ids"] == ["2"]
    assert manifest["filter"]["survivors"] == 2 and manifest["filter"]["format_rejected"] == 2
    # q0 got 2 candidates, q1 and q2 got 2 + 2 (one resample round).
    assert client.generated == {0: 2, 1: 4, 2: 4}
    # Judged candidate-runs: q0 1 candidate, q1 3, q2 4 -> 8 candidates x 2 runs.
    assert manifest["filter"]["judged"] == 16
    # Judge CALLS are per prompt-group per run: round 0 three groups, round 1 two -> 5 x 2.
    assert client.calls["judge"] == 10
    assert (1, (2,)) in client.groups and (2, (0, 1)) in client.groups and (2, (2, 3)) in client.groups
    assert manifest["usage"]["total_usd"] == pytest.approx(10 * 0.001 + 10 * 0.0005)
    for stage in ("stage_1_prompts", "stage_2_responses", "stage_3_judge", "stage_4_export_sft"):
        assert (run_dir / f"{stage}.jsonl").exists()
    judged = [json.loads(l) for l in (run_dir / "stage_3_judge.jsonl").read_text().splitlines()]
    assert judged[0]["candidates"][0]["format_rejected"].startswith("Incomplete completion")
    assert judged[1]["candidates"][3]["format_rejected"].startswith("Final answer discusses the constitution")
    assert judged[0]["selected"] == 1 and judged[2]["selected"] is None
    assert judged[1]["candidates"][2]["scores"] == [9, 7] and judged[1]["candidates"][2]["score"] == 7


def test_min_rows_floor_refuses_to_publish(tmp_path, monkeypatch):
    client = FakeClient(scores={0: {0: [9, 9]}})
    _install(monkeypatch, client)
    cfg = _config(tmp_path)
    cfg["filter"]["min_rows"] = 2
    with pytest.raises(RuntimeError, match="below filter.min_rows"):
        pipeline.run(cfg)
    run_dir = next((tmp_path / "out").iterdir())
    assert not (run_dir / "dataset.jsonl").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "aborted" and manifest["filter"]["rejected_ids"] == ["1", "2"]


def test_resume_reuses_generations_and_retries_failed_judgements(tmp_path, monkeypatch):
    client = FakeClient(scores={q: {0: [9, 9]} for q in range(3)}, judge_errors=1)
    _install(monkeypatch, client)
    cfg = _config(tmp_path)
    with pytest.raises(RuntimeError, match="judged failed"):
        pipeline.run(cfg)
    run_dir = next((tmp_path / "out").iterdir())
    generated_before = dict(client.generated)
    verdicts = [json.loads(l) for l in (run_dir / "judgements.partial.jsonl").read_text().splitlines()]
    assert verdicts[0]["error"] == "ConnectionError" and verdicts[0]["candidates"] == [0, 1]
    manifest = pipeline.run(cfg, resume=str(run_dir))
    assert manifest["status"] == "complete" and manifest["filter"]["survivors"] == 3
    assert client.generated == generated_before, "resume must not regenerate checkpointed candidates"
    assert len(manifest["commands"]) == 2 and manifest["commands"][1]["resume"] == str(run_dir)


def test_validate_config_rejects_bad_filters(tmp_path):
    cfg = _config(tmp_path)
    pipeline.validate_config(cfg)
    bad = json.loads(json.dumps(cfg))
    bad["filter"]["judge"]["model"] = "qwen/qwen3.6-27b"
    with pytest.raises(ValueError, match="judge must not be the model"):
        pipeline.validate_config(bad)
    bad = json.loads(json.dumps(cfg))
    bad["filter"]["threshold"] = 11
    with pytest.raises(ValueError, match="threshold"):
        pipeline.validate_config(bad)
    bad = json.loads(json.dumps(cfg))
    bad["judge_prompt"] = "no fields"
    with pytest.raises(ValueError, match="judge_prompt"):
        pipeline.validate_config(bad)
    bad = json.loads(json.dumps(cfg))
    del bad["filter"]
    with pytest.raises(ValueError, match="Missing"):
        pipeline.validate_config(bad)


def test_judge_reasoning_effort_is_validated_and_passed_through():
    from src.data.synth.deliberative_alignment.pipeline import validate_config

    cfg = _config(Path("/tmp"))
    cfg["pipeline"] = "delib"
    cfg["filter"]["judge"]["reasoning"] = {"effort": "low"}
    validate_config(cfg)
    cfg["filter"]["judge"]["reasoning"] = {"effort": "max"}
    with pytest.raises(ValueError, match="effort"):
        validate_config(cfg)
    cfg["filter"]["judge"]["reasoning"] = {"max_tokens": 10}
    with pytest.raises(ValueError, match="reasoning.max_tokens"):
        validate_config(cfg)
    cfg["filter"]["judge"].pop("reasoning")
    cfg["filter"]["judge"]["bogus"] = 1
    with pytest.raises(ValueError, match="may set"):
        validate_config(cfg)


def test_a_truncated_judge_call_is_retried_before_the_prompt_loses_its_survivor():
    from src.data.synth.deliberative_alignment.pipeline import _judge_runs

    class Flaky:
        def __init__(self):
            self.n = 0

        def chat(self, model, messages, temperature, max_tokens, **kw):
            self.n += 1
            if self.n == 1:  # first call truncates, second scores
                return ChatResult(content="cut off mid", prompt_tokens=1, completion_tokens=max_tokens,
                                  finish_reason="length", provider="Anthropic", cost=0.0)
            return ChatResult(content="ok\nCANDIDATE 0 SCORE: 8", prompt_tokens=1, completion_tokens=1,
                              finish_reason="stop", provider="Anthropic", cost=0.0)

    cfg = {"filter": {"judge": {"model": "m", "temperature": 0, "max_tokens": 50, "retries": 2}},
           "judge_prompt": "{constitution} {conversation} {candidates}"}
    record = {"id": "p", "messages": [{"role": "user", "content": "q"}]}
    group = [{"candidate": 0, "assistant": {"role": "assistant", "content": "a", "reasoning_content": "r"}}]
    out = _judge_runs(Flaky(), record, group, [0], cfg, "C")
    assert [("scores" in v, v.get("attempt")) for v in out] == [(False, None), (True, 1)]


def test_generator_provider_override_is_validated_and_pinned():
    from src.data.synth.deliberative_alignment.pipeline import generator_pin, validate_config

    cfg = _config(Path("/tmp")); cfg["pipeline"] = "delib"
    pin, price = generator_pin(cfg)
    assert pin == {"order": ["alibaba"], "allow_fallbacks": False}  # the registry's pin
    cfg["provider"] = {"order": ["phala"], "price": {"in": 0.32, "out": 2.7}}
    validate_config(cfg)
    assert generator_pin(cfg) == ({"order": ["phala"], "allow_fallbacks": False}, {"in": 0.32, "out": 2.7})
    for bad in ({"order": ["a", "b"], "price": {"in": 1, "out": 1}}, {"order": ["a"]},
                {"order": ["a"], "price": {"in": 0, "out": 1}}):
        cfg["provider"] = bad
        with pytest.raises(ValueError, match="provider"):
            validate_config(cfg)


def test_mirror_throttle_limits_commits_and_survives_hub_errors():
    from src.data.synth.deliberative_alignment.pipeline import _MirrorThrottle

    now = [0.0]
    t = _MirrorThrottle(interval_s=300, clock=lambda: now[0])
    calls = []
    assert t.attempt(lambda: calls.append("a")) is True          # first checkpoint mirrors
    now[0] = 100
    assert t.attempt(lambda: calls.append("b")) is False         # too soon: skipped, no call
    assert t.attempt(lambda: calls.append("c"), final=True) is True  # a stage boundary always mirrors
    now[0] = 500

    def boom():
        raise RuntimeError("429 Too Many Requests")
    assert t.attempt(boom) is False and t.skipped == 1           # a Hub error is skipped, not raised
    assert t.attempt(lambda: calls.append("d")) is True          # and retried at the next due checkpoint
    assert calls == ["a", "c", "d"]


def test_a_rate_limited_batch_backs_off_and_retries_instead_of_aborting(tmp_path, monkeypatch):
    # Two candidates hit a rate limit on their first attempt. The run must finish on its own:
    # the items are deferred, retried after the cooldown at half the workers, and the error
    # records stay in the checkpoint beside the later successes.
    client = FakeClient(scores={q: {0: [9, 9]} for q in range(3)}, rate_limited={(1, 0), (2, 1)})
    _install(monkeypatch, client)
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    manifest = pipeline.run(_config(tmp_path))
    assert manifest["status"] == "complete" and manifest["filter"]["survivors"] == 3
    run_dir = next((tmp_path / "out").iterdir())
    gens = [json.loads(l) for l in (run_dir / "generations.partial.jsonl").read_text().splitlines()]
    assert sum(g.get("error") == "RateLimitError" for g in gens) == 2
    assert sum("assistant" in g for g in gens) == 6, "every slot generated once the retry landed"
    assert len(manifest["commands"]) == 1, "no resume was needed"
