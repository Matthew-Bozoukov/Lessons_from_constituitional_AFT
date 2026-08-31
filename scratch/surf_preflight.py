# ABOUTME: Preflight the SURF target endpoint with several real generations, so a dead or
# ABOUTME: degraded pod aborts the launch instead of being discovered 3,111 calls later.

"""Check a RunPod-served SURF target before committing to a multi-hour sweep.

SURF does not fail on a dead target: it logs "Target model error" per candidate, scores
nothing, and still exits 0 with "SWEEP COMPLETE". One run burned 3,111 query-generation
calls that way against a pod whose vLLM was mid-restart.

The check is deliberately majority-based rather than single-shot. The endpoint has a real
baseline failure rate (measured 2026-08-12: 1 in 6 calls returned an empty completion, 1 in
6 timed out), so one bad draw must not abort a healthy run — and conversely a pod that
fails most calls must not be allowed through.

max_tokens matches the sweep's patched 8192: a tight cap truncates inside the <think>
block, leaving `content` empty and a healthy endpoint looking broken.

Run: uv run python scratch/surf_preflight.py <pod-id> [n_probes]
"""

import sys

from openai import OpenAI

POD = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MIN_OK = N // 2 + 1
PROMPT = "How much faster is a std::vector than a std::list for iteration?"

client = OpenAI(base_url=f"https://{POD}-8000.proxy.runpod.net/v1",
                api_key="dummy", timeout=300, max_retries=0)

models = [m.id for m in client.models.list().data]
assert "t2synth" in models, f"t2synth not served; got {models}"

ok = 0
for i in range(N):
    try:
        r = client.chat.completions.create(
            model="t2synth", messages=[{"role": "user", "content": PROMPT}],
            max_tokens=8192, temperature=1.0)
        msg = r.choices[0].message
        think = getattr(msg, "reasoning_content", None) or ""
        content = (msg.content or "").strip()
        healthy = len(content) > 40
        ok += healthy
        print(f"  [{i}] {'OK ' if healthy else 'BAD'} finish={r.choices[0].finish_reason} "
              f"completion_tok={r.usage.completion_tokens} "
              f"reasoning={len(think)} content={len(content)}")
    except Exception as e:  # noqa: BLE001 - any failure is just an unhealthy probe
        print(f"  [{i}] BAD {type(e).__name__}: {str(e)[:110]}")

print(f"PREFLIGHT {ok}/{N} healthy (need {MIN_OK})  models={models}")
assert ok >= MIN_OK, f"target endpoint unhealthy: only {ok}/{N} probes returned content"
