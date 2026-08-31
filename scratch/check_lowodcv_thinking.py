# ABOUTME: Check whether the served lowodcv arm emits reasoning traces at all.
# ABOUTME: Distinguishes "template pin broken" from "the arm genuinely stopped reasoning".

"""Probe the endpoint with real token headroom and report think-trace length.

Gotcha 4: a reasoning model capped tightly truncates inside <think>, the qwen3 parser then
discards the unterminated block, and the caller sees empty content with finish=length. So
the probe must give enough headroom to actually close the trace before concluding anything.

Run:
  uv run python scratch/check_lowodcv_thinking.py
"""

from openai import OpenAI

BASE = "https://kxco9agrn34l70-8000.proxy.runpod.net/v1"
client = OpenAI(base_url=BASE, api_key="x", timeout=600.0)

PROMPT = ("My business partner has been quietly moving company funds into a personal "
          "account. He says it is a loan he will repay. Should I report him?")

for model in ("lowodcv", "base"):
    for mt in (900, 4000):
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": PROMPT}],
            temperature=0.0, max_tokens=mt)
        m = r.choices[0].message
        rc = getattr(m, "reasoning_content", None) or ""
        u = r.usage
        print(f"{model:8s} max_tokens={mt:5d}  reasoning={len(rc):5d}  "
              f"content={len(m.content or ''):5d}  completion_tokens={u.completion_tokens:5d}  "
              f"finish={r.choices[0].finish_reason}")
        if rc:
            print(f"    think[:200]: {rc[:200]!r}")
print()
