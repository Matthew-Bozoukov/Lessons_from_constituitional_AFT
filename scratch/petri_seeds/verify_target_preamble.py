# ABOUTME: Fails unless every system message the TARGET model received in a Petri .eval log starts
# ABOUTME: with the configured constitution file, byte for byte; auditor and judge prompts ignored.
"""Written-output check of the constitution injection, run before paying for a full audit.

    .venv/bin/python scratch/petri_seeds/verify_target_preamble.py <log.eval> <preamble.md>

Reads the target's own model calls (the `openai-api/vllm/...` role) with attachments resolved,
since Inspect stores long message content as `attachment://<hash>` references. Exit status 0
only when at least one target system message was seen and every one starts with the file.
"""
import sys
from pathlib import Path

from inspect_ai.event import ModelEvent
from inspect_ai.log import read_eval_log


def main(log_path: str, preamble_path: str) -> int:
    """Count target system messages and how many begin with the preamble; 0 = all of them."""
    pre = Path(preamble_path).read_text(encoding="utf-8").rstrip()
    log = read_eval_log(log_path, resolve_attachments=True)
    n = ok = 0
    example = ""
    for s in log.samples or []:
        for ev in s.events or []:
            if not (isinstance(ev, ModelEvent) and "vllm" in str(ev.model)):
                continue
            for m in ev.input or []:
                if m.role != "system":
                    continue
                c = m.content if isinstance(m.content, str) else m.text
                n += 1
                ok += c.startswith(pre)
                example = example or c[len(pre):len(pre) + 120] if c.startswith(pre) else example
    print(f"samples: {len(log.samples or [])} | target system messages: {n} | "
          f"start with the full preamble ({len(pre.split())} words): {ok}")
    print(f"scenario text right after the preamble: {example!r}")
    return 0 if n and ok == n else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
