# ABOUTME: Print the structure of one upstream code_review .eval log (scores, usage, metadata keys),
# ABOUTME: to find where Agent B's tool calls, decision and per-model tokens live.
# Run: uv run python scratch/iai_code_review/peek_log.py <file.eval>
import sys

from inspect_ai.log import read_eval_log

log = read_eval_log(sys.argv[1])
s = log.samples[0]
print(
    "scores:",
    {k: (v.value, (v.explanation or "")[:200]) for k, v in (s.scores or {}).items()},
)
print(
    "model_usage:",
    {k: v.model_dump(exclude_none=True) for k, v in (s.model_usage or {}).items()},
)
md = s.metadata or {}
print("metadata keys:", sorted(md.keys()))
print("n messages:", len(s.messages))
for k, v in md.items():
    if isinstance(v, (list, dict)):
        print(f"{k}: {type(v).__name__} len={len(v)} :: {str(v)[:400]}")
