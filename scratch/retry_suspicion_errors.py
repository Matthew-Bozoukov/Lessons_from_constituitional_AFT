# ABOUTME: Re-judge premature-suspicion transcripts whose first judgement returned truncated
# ABOUTME: JSON, patching the verdict back into the JSONL so no transcript is silently dropped.
"""Retry errored rows in premature_suspicion_*.jsonl at a larger token budget, in place.

The failure is always the same: the response is cut mid-`quote`. The judge is a reasoning
model and its thinking tokens count against `max_tokens`, so a long trace leaves nothing for
the JSON — 1600 was still not enough. Retrying at 4096 resolves it, which matters because the
truncated fragments disproportionately showed `"premature_suspicion": true`, so excluding
them biases the rate down.

`max_tokens` is left at the original value in the main script so that every arm is judged
under identical settings; only these stragglers are retried with headroom.

    uv run python scratch/retry_suspicion_errors.py output/report/premature_suspicion_*.jsonl
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, "scratch")
import judge_premature_suspicion as J  # noqa: E402

from src.endpoints.openrouter import OpenRouterClient  # noqa: E402
from src.utils import extract_json  # noqa: E402


def main() -> None:
    """Retry every errored row in each JSONL given on the command line and rewrite the file."""
    load_dotenv(override=True)
    client = OpenRouterClient()
    for jsonl in sys.argv[1:]:
        rows = [json.loads(line) for line in open(jsonl)]
        errs = [r for r in rows if r["error"]]
        print(f"\n=== {Path(jsonl).name}: {len(errs)} errored ===")
        for r in errs:
            prefix, n = J.prefix_for_judge(Path(r["path"]).read_text(errors="replace"), 5, 4000)
            res = client.chat(
                model=J.JUDGE_MODEL,
                messages=[{"role": "system", "content": J.SYSTEM},
                          {"role": "user", "content": J.USER_TEMPLATE.format(
                              n_steps=n, transcript=prefix)}],
                temperature=0.0, max_tokens=4096,
                extra_body={"reasoning": {"effort": "low"}})
            d = extract_json(res.content)
            r.update(premature=bool(d.get("premature_suspicion")), step=d.get("step"),
                     quote=str(d.get("quote", ""))[:600],
                     ambiguity_before=bool(d.get("ambiguity_before")),
                     confidence=str(d.get("confidence", "")),
                     why=str(d.get("why", ""))[:400], error="")
            print(f"  {r['scenario']}/{r['rollout']}: premature={r['premature']} "
                  f"step={r['step']}\n    quote: {r['quote'][:160]}")
        with open(jsonl, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        assert not any(r["error"] for r in rows), "rows still errored after retry"


if __name__ == "__main__":
    main()
