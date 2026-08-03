# ABOUTME: Unit tests for assistant-only loss masking over rendered Qwen chat text.
# ABOUTME: Run: uv run pytest tests/test_masking.py -q

from __future__ import annotations

from pathlib import Path


from src.train.masking import ASSISTANT_HEADER, assistant_spans, build_labels  # noqa: E402

CHAT = (
    "<|im_start|>user\nhi<|im_end|>\n"
    "<|im_start|>assistant\nhello<|im_end|>\n"
    "<|im_start|>user\nbye<|im_end|>\n"
    "<|im_start|>assistant\nfarewell<|im_end|>\n"
)


def test_spans_cover_content_and_terminator_but_not_header():
    spans = assistant_spans(CHAT)
    assert [CHAT[s:e] for s, e in spans] == ["hello<|im_end|>", "farewell<|im_end|>"]
    for s, _ in spans:
        assert not CHAT[s:].startswith(ASSISTANT_HEADER)


def test_user_text_is_never_inside_a_span():
    spans = assistant_spans(CHAT)
    for probe in ("hi", "bye"):
        i = CHAT.index(probe)
        assert not any(s <= i < e for s, e in spans)


class _CharTokenizer:
    """Character-level stand-in whose offsets are exact, so masking is testable offline."""

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None,
                 return_offsets_mapping=False):
        ids = [ord(c) for c in text][: max_length if truncation else None]
        out = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(ids))]
        return out


def test_labels_unmask_exactly_the_assistant_characters():
    out = build_labels(CHAT, _CharTokenizer(), max_length=len(CHAT))
    assert len(out["input_ids"]) == len(out["labels"]) == len(CHAT)
    kept = "".join(chr(v) for v in out["labels"] if v != -100)
    assert kept == "hello<|im_end|>farewell<|im_end|>"


def test_supervised_fraction_is_a_minority_of_a_prompt_heavy_example():
    long_prompt = "<|im_start|>user\n" + ("x" * 500) + "<|im_end|>\n" \
                  "<|im_start|>assistant\nok<|im_end|>\n"
    out = build_labels(long_prompt, _CharTokenizer(), max_length=len(long_prompt))
    supervised = sum(1 for v in out["labels"] if v != -100)
    assert supervised == len("ok<|im_end|>")
    assert supervised < 0.1 * len(out["labels"])


def test_check_thinking_declaration():
    import pytest

    from src.train.masking import EMPTY_THINK, check_thinking_declaration

    real = {"text": f"<|im_start|>assistant\n<think>\nreal trace\n</think>\nanswer<|im_end|>"}
    empty = {"text": f"<|im_start|>assistant\n{EMPTY_THINK}answer<|im_end|>"}
    plain = {"text": "<|im_start|>assistant\nanswer<|im_end|>"}
    msgs_think = {"messages": [{"role": "assistant", "content": "a", "reasoning_content": "hm"}]}
    msgs_plain = {"messages": [{"role": "assistant", "content": "a"}]}

    check_thinking_declaration([real, plain], thinking=True)
    check_thinking_declaration([real, empty], thinking=True, mask_empty_think=True)
    check_thinking_declaration([msgs_think], thinking=True)
    check_thinking_declaration([plain, msgs_plain], thinking=False)

    with pytest.raises(AssertionError, match="no row carries a real reasoning trace"):
        check_thinking_declaration([plain], thinking=True)
    with pytest.raises(AssertionError, match="mask_empty_think"):
        check_thinking_declaration([real, empty], thinking=True, mask_empty_think=False)
    with pytest.raises(AssertionError, match="mislabels"):
        check_thinking_declaration([real], thinking=False)


def test_every_train_config_declares_thinking():
    from pathlib import Path

    from omegaconf import OmegaConf

    configs = sorted(Path("configs/train").glob("lora_*.yaml"))
    assert configs
    for path in configs:
        cfg = OmegaConf.load(path)
        assert "thinking" in cfg and isinstance(cfg.thinking, bool), path.name
