# ABOUTME: Render Qwen3.6-27B's chat template with and without enable_thinking to find out
# ABOUTME: what the SERVE-TIME DEFAULT actually prefills — thinking, or the empty marker.

"""Determine Qwen3.6's default thinking behaviour from its own chat template.

The SURF target pod serves with no --chat-template and no mode pin, on the assumption that
the model's default is thinking mode. Every sampled response has come back with an empty
reasoning_content, so that assumption needs checking rather than repeating.

Downloads only the tokenizer (a few MB), no weights. Run from the repo root:
    uv run python scratch/check_qwen36_think_default.py
"""

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.6-27B"
MSG = [{"role": "user", "content": "How much faster is a std::vector than a std::list?"}]

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

variants: dict[str, str] = {}
variants["DEFAULT (no kwarg)"] = tok.apply_chat_template(
    MSG, tokenize=False, add_generation_prompt=True)
for flag in (True, False):
    try:
        variants[f"enable_thinking={flag}"] = tok.apply_chat_template(
            MSG, tokenize=False, add_generation_prompt=True, enable_thinking=flag)
    except Exception as e:  # noqa: BLE001 - template may not accept the kwarg at all
        variants[f"enable_thinking={flag}"] = f"<UNSUPPORTED: {type(e).__name__}: {e}>"

for name, text in variants.items():
    n = len(tok(text)["input_ids"]) if not text.startswith("<UNSUPPORTED") else -1
    print("=" * 72)
    print(f"{name}   ({n} tokens)")
    print("-" * 72)
    print(repr(text[-220:]))

print("=" * 72)
d, t, f = (variants["DEFAULT (no kwarg)"],
           variants.get("enable_thinking=True"), variants.get("enable_thinking=False"))
print(f"default == enable_thinking=True : {d == t}")
print(f"default == enable_thinking=False: {d == f}")
print(f"default ends with empty marker  : {d.rstrip().endswith('</think>')}")
