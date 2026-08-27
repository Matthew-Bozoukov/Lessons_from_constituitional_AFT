# ABOUTME: Offline tests for the terminal chat REPL's pure logic: /set and /use parsing,
# ABOUTME: one-server-per-session enforcement, transcript records, and live trace dimming.

import io

import pytest

from src.endpoints.chat import (
    Arm,
    StreamPrinter,
    assert_one_server,
    assistant_message,
    build_record,
    parse_set,
    pick_arms,
)
from src.endpoints.vllm_server import TargetSpec

DIM, RESET = "\x1b[2m", "\x1b[0m"


def arm(name: str, mode: str = "think") -> Arm:
    return Arm(
        name=name, base_url="http://x/v1", api_key="EMPTY", model_id=name, mode=mode
    )


def spec(hf_path: str, base: str, mode: str, api_base: str | None = None) -> TargetSpec:
    return TargetSpec(
        hf_path=hf_path,
        base_model=base,
        adapter=api_base is None,
        mode=mode,
        model_key=hf_path.split("/")[-1],
        lora_rank=32,
        api_base=api_base,
    )


def dim(text: str) -> str:
    return f"{DIM}{text}{RESET}"


# --- commands -----------------------------------------------------------------------------


def test_parse_set_coerces_types_and_keeps_the_rest():
    out = parse_set(
        ["temperature=0.2", "max_tokens=64"],
        {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4096},
    )
    assert out == {"temperature": 0.2, "top_p": 0.9, "max_tokens": 64}
    assert isinstance(out["max_tokens"], int)


def test_parse_set_rejects_unknown_keys_and_bad_values():
    with pytest.raises(ValueError, match="k=v"):
        parse_set(["seed=1"], {})
    with pytest.raises(ValueError, match="not a int"):
        parse_set(["max_tokens=lots"], {})


def test_pick_arms_keeps_the_given_order_and_knows_all():
    arms = [arm("base"), arm("ft_20_80"), arm("ft_10_90")]
    assert [a.name for a in pick_arms(["ft_10_90", "base"], arms)] == [
        "ft_10_90",
        "base",
    ]
    assert pick_arms(["all"], arms) == arms
    with pytest.raises(ValueError, match="unknown arm"):
        pick_arms(["nope"], arms)
    with pytest.raises(ValueError):
        pick_arms([], arms)


# --- one server per session ----------------------------------------------------------------


def test_assert_one_server_accepts_shared_base_and_mode():
    assert_one_server(
        [
            spec("org/a", "Qwen/Qwen3.6-27B", "think"),
            spec("org/b", "Qwen/Qwen3.6-27B", "think"),
        ]
    )


def test_assert_one_server_refuses_a_mode_or_base_mismatch():
    with pytest.raises(SystemExit, match="one pinned thinking mode"):
        assert_one_server(
            [
                spec("org/a", "Qwen/Qwen3.6-27B", "think"),
                spec("org/b", "Qwen/Qwen3.6-27B", "nothink"),
            ]
        )
    with pytest.raises(SystemExit):
        assert_one_server(
            [
                spec("org/a", "Qwen/Qwen3.6-27B", "think"),
                spec("org/b", "Qwen/Qwen3-32B", "think"),
            ]
        )


def test_assert_one_server_ignores_api_targets():
    """An API target is not served, so it never constrains the vLLM server."""
    assert_one_server(
        [
            spec("org/a", "Qwen/Qwen3.6-27B", "think"),
            spec("openrouter:x/y", "x/y", "default", api_base="https://api/v1"),
        ]
    )


# --- records -----------------------------------------------------------------------------


def test_build_record_splits_an_out_of_band_trace():
    r = build_record(
        1,
        arm("ft"),
        "be brief",
        "hi",
        "the answer",
        "the trace",
        "stop",
        {"temperature": 0.7},
        history_len=0,
    )
    assert (r["think"], r["answer"]) == ("the trace", "the answer")
    assert r["raw_content"] == "the answer" and r["finish_reason"] == "stop"
    assert r["system"] == "be brief" and r["arm"] == "ft" and r["mode"] == "think"


def test_build_record_splits_the_prefilled_inline_trace():
    """Think-mode prefill: the trace arrives with its CLOSE tag only (resolve_trace)."""
    r = build_record(
        2, arm("ft"), "", "hi", "why\n</think>\nbecause", "", "stop", {}, 2
    )
    assert (r["think"], r["answer"]) == ("why", "because")


def test_assistant_message_carries_reasoning_only_when_there_is_some():
    with_trace = build_record(1, arm("ft"), "", "hi", "a", "t", "stop", {}, 0)
    without = build_record(
        1, arm("ft", mode="nothink"), "", "hi", "a", "", "stop", {}, 0
    )
    assert assistant_message(with_trace) == {
        "role": "assistant",
        "content": "a",
        "reasoning_content": "t",
    }
    assert assistant_message(without) == {"role": "assistant", "content": "a"}


# --- live dimming --------------------------------------------------------------------------


def test_stream_printer_dims_out_of_band_reasoning_and_prints_the_answer_plain():
    buf = io.StringIO()
    p = StreamPrinter(out=buf, color=True)
    p.feed(None, "why")
    p.feed("ans", None)
    p.finish()
    assert buf.getvalue() == dim("why") + "ans"
    assert ("".join(p.reasoning), "".join(p.raw)) == ("why", "ans")


def test_stream_printer_dims_an_expected_inline_trace_until_the_close_tag_passes():
    """The close tag straddles deltas; everything up to and including it is dim."""
    buf = io.StringIO()
    p = StreamPrinter(expect_inline=True, out=buf, color=True)
    for delta in ("abc", "d</thi", "nk>ans", "wer"):
        p.feed(delta, None)
    p.finish()
    assert buf.getvalue() == dim("abc") + dim("d</thi") + dim("nk>") + "ans" + "wer"
    assert "".join(p.raw) == "abcd</think>answer"


def test_stream_printer_holds_a_possible_open_tag_until_it_is_decided():
    buf = io.StringIO()
    p = StreamPrinter(out=buf, color=True)
    p.feed("<th", None)
    assert buf.getvalue() == ""  # could still become <think>: nothing yet
    p.feed("ink>x</think>y", None)
    assert buf.getvalue() == dim("<think>x</think>") + "y"


def test_stream_printer_prints_plain_when_content_is_not_a_trace():
    buf = io.StringIO()
    p = StreamPrinter(out=buf, color=True)
    p.feed("Hello", None)
    p.feed(" there", None)
    p.finish()
    assert buf.getvalue() == "Hello there"


def test_stream_printer_flushes_a_held_short_answer_on_finish():
    buf = io.StringIO()
    p = StreamPrinter(out=buf, color=True)
    p.feed("<t", None)
    p.finish()
    assert buf.getvalue() == "<t" and "".join(p.raw) == "<t"


def test_stream_printer_without_color_writes_no_escapes():
    buf = io.StringIO()
    p = StreamPrinter(out=buf, color=False)
    p.feed("ans", "why")
    assert buf.getvalue() == "whyans"


# --- secrets -------------------------------------------------------------------------------


def test_arm_provenance_never_carries_the_key():
    """run_meta.json is built from provenance(); the first smoke run wrote the OpenRouter
    key to disk through __dict__ (2026-08-27). Structural guard, not a reminder."""
    a = Arm(
        name="x",
        base_url="https://api/v1",
        api_key="sk-secret",
        model_id="x",
        mode="default",
    )
    prov = a.provenance()
    assert "api_key" not in prov and "sk-secret" not in str(prov)
    assert prov["name"] == "x" and prov["model_id"] == "x"
