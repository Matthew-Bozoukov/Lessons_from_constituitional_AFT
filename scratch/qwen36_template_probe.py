# ABOUTME: Renders multi-turn conversations with mixed reasoning_content through the real
# ABOUTME: Qwen3.6-27B chat template, preserve_thinking off/on, to show where <think> lands.

from transformers import AutoTokenizer

TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")

# 3 assistant turns: a1 HAS reasoning, a2 has NONE, a3 HAS reasoning.
CHAT = [
    {"role": "user", "content": "Is 7 prime?"},
    {"role": "assistant", "content": "Yes, 7 is prime.",
     "reasoning_content": "7 has no divisors besides 1 and itself, so it is prime."},
    {"role": "user", "content": "Thanks. And what colour is the sky?"},
    {"role": "assistant", "content": "Blue."},
    {"role": "user", "content": "Is 9 prime?"},
    {"role": "assistant", "content": "No, 9 = 3 x 3.",
     "reasoning_content": "9 is divisible by 3, so not prime."},
]

# Agentic loop: ONE user query, then assistant/tool/assistant. Both assistant turns
# come after the last real user query, so the template keeps both think blocks even
# with preserve_thinking off (interleaved thinking).
TOOL_CHAT = [
    {"role": "user", "content": "What is the weather in London?"},
    {"role": "assistant", "content": "",
     "reasoning_content": "I should call the weather tool for London.",
     "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "London"}}}]},
    {"role": "tool", "content": "14C, light rain"},
    {"role": "assistant", "content": "It's 14C and lightly raining in London.",
     "reasoning_content": "The tool says 14C and rain; summarize that."},
]


def show(title: str, text: str) -> None:
    print(f"\n{'=' * 70}\n### {title}\n{'=' * 70}")
    print(text, end="")
    print("<<<EOT")


show("A. plain chat, DEFAULT (preserve_thinking unset)",
     TOK.apply_chat_template(CHAT, tokenize=False))

show("B. plain chat, preserve_thinking=True",
     TOK.apply_chat_template(CHAT, tokenize=False, preserve_thinking=True))

show("C. agentic tool loop, DEFAULT (no preserve_thinking!)",
     TOK.apply_chat_template(TOOL_CHAT, tokenize=False))

show("D. generation prompt, thinking mode (enable_thinking default=on)",
     TOK.apply_chat_template(CHAT[:1], tokenize=False, add_generation_prompt=True))

show("E. generation prompt, nothink mode (enable_thinking=False)",
     TOK.apply_chat_template(CHAT[:1], tokenize=False, add_generation_prompt=True,
                             enable_thinking=False))

print(f"\n{'=' * 70}\n### F. how the think markers tokenize (Qwen3.6 vocab)\n{'=' * 70}")
for s in ["<think>", "</think>", "<think>\n\n</think>\n\n", "\n</think>\n\n"]:
    ids = TOK.encode(s)
    print(f"{s!r:38} -> {ids}  {[TOK.decode([i]) for i in ids]}")
