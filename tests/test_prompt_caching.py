# ABOUTME: The CACHE_MARK breakpoint: it must change billing and nothing else, so the
# ABOUTME: text a model receives stays byte-identical whether or not a prompt is marked.

"""What has to hold.

Caching is a billing optimisation applied to a corpus generator, so the one thing it may
never do is change the data. If marking a prompt altered a single character of what the
model reads, the corpus would stop being comparable to the runs before it and the saving
would have cost far more than it saved.

So the load-bearing test here is `test_marked_and_unmarked_prompts_read_identically`.
Everything else guards the edges: unmarked prompts must be untouched (no silent rewrite
of every existing caller), and non-Anthropic models must get the marker stripped rather
than a provider extension they may reject.
"""

from __future__ import annotations

from src.endpoints.openrouter import CACHE_MARK, apply_cache_control

ANTHROPIC = "anthropic/claude-sonnet-5"
OTHER = "deepseek/deepseek-chat-v3.1"

CONSTITUTION = "PRINCIPLE 1. Preserve human oversight.\nPRINCIPLE 2. Be honest.\n"


def _marked(trait: str) -> list[dict]:
    return [{"role": "system", "content": "You refine prompts."},
            {"role": "user",
             "content": f"Constitution:\n<c>\n{CONSTITUTION}</c>\n{CACHE_MARK}\n"
                        f"Target principle: {trait}"}]


def _unmarked(trait: str) -> list[dict]:
    return [{"role": "system", "content": "You refine prompts."},
            {"role": "user",
             "content": f"Constitution:\n<c>\n{CONSTITUTION}</c>\n\n"
                        f"Target principle: {trait}"}]


def _text(content) -> str:
    """The text a model actually receives, however the content is structured."""
    if isinstance(content, str):
        return content
    return "".join(block["text"] for block in content)


def test_marked_and_unmarked_prompts_read_identically():
    """The whole safety property. Marking may move a cache boundary; it may not move a
    character, or the corpus stops being comparable to every run before it."""
    for trait in ("ZETA_TRAIT", "OMEGA_TRAIT"):
        marked = apply_cache_control(_marked(trait), ANTHROPIC)
        assert [_text(m["content"]) for m in marked] == \
               [m["content"] for m in _unmarked(trait)]


def test_the_marker_never_survives_into_a_request():
    for model in (ANTHROPIC, OTHER):
        out = apply_cache_control(_marked("ZETA_TRAIT"), model)
        assert CACHE_MARK not in "".join(_text(m["content"]) for m in out)


def test_the_breakpoint_lands_after_the_invariant_prefix():
    """The cached block must be the part that repeats, and must NOT include the part that
    varies -- caching a per-record suffix buys nothing and writes a new entry each call."""
    a = apply_cache_control(_marked("ZETA_TRAIT"), ANTHROPIC)[1]["content"]
    b = apply_cache_control(_marked("OMEGA_TRAIT"), ANTHROPIC)[1]["content"]

    assert a[0]["cache_control"] == {"type": "ephemeral"}
    assert a[0]["text"] == b[0]["text"], "the cached prefix must be call-invariant"
    assert CONSTITUTION in a[0]["text"]
    assert "ZETA_TRAIT" in a[1]["text"] and "cache_control" not in a[1]
    assert "ZETA_TRAIT" not in a[0]["text"], "the varying part leaked into the cached block"


def test_an_unmarked_prompt_is_passed_through_untouched():
    """Every existing caller predates this feature and must produce the identical request
    it produced before -- same object, plain string content, no block list."""
    msgs = _unmarked("ZETA_TRAIT")
    assert apply_cache_control(msgs, ANTHROPIC) is msgs


def test_a_non_anthropic_model_gets_the_marker_stripped_not_the_extension():
    """`cache_control` is an Anthropic extension; sending it elsewhere risks a 400. The
    same config must run on any provider, differing only in what it costs."""
    out = apply_cache_control(_marked("ZETA_TRAIT"), OTHER)
    for m in out:
        assert isinstance(m["content"], str)
    assert _text(out[1]["content"]) == _unmarked("ZETA_TRAIT")[1]["content"]


def test_a_marker_in_the_system_message_is_honoured_too():
    msgs = [{"role": "system", "content": f"Long shared preamble{CACHE_MARK} tail"},
            {"role": "user", "content": "hi"}]
    out = apply_cache_control(msgs, ANTHROPIC)
    assert out[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert out[0]["content"][0]["text"] == "Long shared preamble"
    assert out[1]["content"] == "hi"
