# ABOUTME: Re-apply this repo's patches to the vendored lmarena/arena-hard-auto checkout.
# ABOUTME: Run: uv run python scripts/eval/patch_arena_hard.py [--check]

"""Patch the vendored arena-hard-auto harness.

The arena-hard checkout is gitignored, so the vendored harness is not version
controlled and a re-clone silently reverts every local change. This script is the
durable record of what we changed and why, and it is idempotent: run it after any
re-clone, and run it with `--check` in CI-ish contexts to assert the patches are live.

Three patches, each traceable to the capability-eval spec:

1. **Baseline override** (`utils/judge_utils.py`) — spec §4, required deviation 1.
   Upstream hardcodes the packaged baselines (`o3-mini-2025-01-31` for `hard_prompt`,
   `gemini-2.0-flash-001` for `creative_writing`). Our baseline must be arm A, our own
   100%-Tulu checkpoint, which is what makes the near-50% self-comparison meaningful.
   Read from the environment so one vendored copy serves every comparison.

2. **`extra_body` + usage passthrough** (`utils/completion.py`) — spec §4, config
   requirement 1 and footgun §10.3. Gemini 3.x Flash bills reasoning tokens as output;
   left at a high thinking budget the judge bill runs several times the §11 estimate.
   OpenRouter takes `reasoning: {effort: "low"}` via `extra_body`, which upstream's
   OpenAI path drops on the floor. We also record per-call token usage so the projection
   can be checked against reality after the first 150 judgments rather than trusted.

3. **Optional heavyweight imports** (`utils/completion.py`) — `boto3` and `shortuuid` are
   imported at module top level for the Bedrock provider and for answer-id generation.
   We route every call through OpenRouter and mint our own answer ids, so both are dead
   code here; importing them anyway would mean carrying the AWS SDK as a project
   dependency for a path we never execute. Made optional rather than deleted, so the
   Bedrock provider still works for anyone who configures it.

4. **`question_limit`** (`gen_judgment.py`) — spec §9, staged sampling. Judgment caching
   keys on `uid`, so restricting the question list to the first N of a category and later
   raising N is purely additive: the first N are read from cache and only the new ones are
   judged. That is what makes 150 → 300 → 500 cost the same as going straight to 500.

Anchors are matched exactly and a missing anchor is a hard failure — a patch that silently
no-ops after an upstream refactor is worse than one that never ran.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR = (REPO_ROOT / "src" / "eval" / "capabilities" / "arena_hard"
          / "third_party" / "arena-hard-auto")

# The upstream commit these patches were written against. If a patch stops applying,
# diff against this SHA before rewriting the anchor.
UPSTREAM_SHA = "196f6b826783b3da7310e361a805fa36f0be83f3"

_BASELINE_PATCH = '''

# --- PATCH: teaching_claude_why_replication (capability eval spec §4) --------------
# The packaged baselines above are strong reference models, which is right for a public
# leaderboard and wrong for us: we are measuring a treated checkpoint against its own
# untreated sibling, and that self-comparison is only interpretable when the baseline IS
# the untreated sibling. Overridden from the environment so the vendored copy stays
# generic across arms.
import os as _os  # noqa: E402

_ARENA_HARD_BASELINE = _os.environ.get("ARENA_HARD_BASELINE")
if _ARENA_HARD_BASELINE:
    for _category in JUDGE_SETTINGS:
        JUDGE_SETTINGS[_category]["baseline"] = _ARENA_HARD_BASELINE
'''

_COMPLETION_ANCHOR = """            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                )
            output = {
                "answer": completion.choices[0].message.content
            }"""

_COMPLETION_PATCH = '''            # PATCH (capability eval spec §4.1, §10.3): forward `extra_body` so the
            # OpenRouter judge can be pinned to a low reasoning effort, and record token
            # usage so the §11 cost projection is checkable rather than assumed.
            _create_kwargs = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if kwargs.get("extra_body"):
                _create_kwargs["extra_body"] = kwargs["extra_body"]
            completion = client.chat.completions.create(**_create_kwargs)
            _usage = getattr(completion, "usage", None)
            output = {
                "answer": completion.choices[0].message.content,
                "usage": {
                    "prompt_tokens": getattr(_usage, "prompt_tokens", None),
                    "completion_tokens": getattr(_usage, "completion_tokens", None),
                    # OpenRouter surfaces billed reasoning tokens here when the upstream
                    # provider reports them; None simply means "not reported".
                    "reasoning_tokens": getattr(
                        getattr(_usage, "completion_tokens_details", None),
                        "reasoning_tokens",
                        None,
                    ),
                },
                "model_id": getattr(completion, "model", model),
            }'''

_IMPORTS_ANCHOR = """import shortuuid
import pandas as pd
"""

_IMPORTS_PATCH = '''# PATCH (capability eval): make the Bedrock/answer-id imports optional. We route every
# call through OpenRouter and mint our own answer ids, so `boto3` and `shortuuid` are
# dead code on our path — and requiring the AWS SDK to judge a pairwise comparison is a
# dependency we should not be carrying. The Bedrock provider still works if installed.
try:
    import shortuuid
except ImportError:  # pragma: no cover - only the unused gen_answer.py path needs it
    shortuuid = None
import pandas as pd
'''

_BOTO_ANCHOR = """from typing import Optional
import boto3
"""

_BOTO_PATCH = '''from typing import Optional

try:
    import boto3
except ImportError:  # pragma: no cover - only the unused Bedrock provider needs it
    boto3 = None
'''

_JUDGMENT_ANCHOR = """    questions = load_questions(question_file)
    model_answers = load_model_answers(answer_dir)"""

_JUDGMENT_PATCH = '''    questions = load_questions(question_file)

    # PATCH (capability eval spec §9): staged sampling. `question_limit` is a
    # {category: n} mapping; judgments cache on uid, so raising a limit later re-reads
    # the already-judged prefix from disk and only pays for the new questions.
    _limits = configs.get("question_limit") or {}
    if _limits:
        _seen = {}
        _kept = []
        for _q in questions:
            _cat = _q["category"]
            _n = _seen.get(_cat, 0)
            if _cat in _limits and _n >= _limits[_cat]:
                continue
            _seen[_cat] = _n + 1
            _kept.append(_q)
        print(f"INFO: question_limit {_limits} -> {len(_kept)}/{len(questions)} questions")
        questions = _kept

    model_answers = load_model_answers(answer_dir)'''

PATCHES = [
    ("utils/judge_utils.py", "_ARENA_HARD_BASELINE", None, _BASELINE_PATCH, "append"),
    (
        "utils/completion.py",
        "capability eval spec §4.1, §10.3",
        _COMPLETION_ANCHOR,
        _COMPLETION_PATCH,
        "replace",
    ),
    (
        "utils/completion.py",
        "only the unused gen_answer.py path needs it",
        _IMPORTS_ANCHOR,
        _IMPORTS_PATCH,
        "replace",
    ),
    (
        "utils/completion.py",
        "only the unused Bedrock provider needs it",
        _BOTO_ANCHOR,
        _BOTO_PATCH,
        "replace",
    ),
    (
        "gen_judgment.py",
        "capability eval spec §9",
        _JUDGMENT_ANCHOR,
        _JUDGMENT_PATCH,
        "replace",
    ),
]


def main(check: bool = False) -> None:
    """Apply (or verify) the vendored-harness patches.

    Args:
        check: If True, do not write anything; exit non-zero if any patch is missing.
    """
    if not VENDOR.exists():
        raise SystemExit(
            f"No vendored harness at {VENDOR}.\n"
            f"  git clone https://github.com/lmarena/arena-hard-auto.git {VENDOR}\n"
            f"  (written against upstream {UPSTREAM_SHA})"
        )

    missing, applied, already = [], [], []
    for rel, marker, anchor, patch, mode in PATCHES:
        path = VENDOR / rel
        if not path.exists():
            raise SystemExit(f"Vendored file missing: {path} (upstream layout changed?)")
        text = path.read_text()

        if marker in text:
            already.append(rel)
            continue
        if check:
            missing.append(rel)
            continue

        if mode == "append":
            new = text.rstrip("\n") + "\n" + patch
        else:
            if anchor not in text:
                raise SystemExit(
                    f"Patch anchor not found in {rel}. Upstream has changed since "
                    f"{UPSTREAM_SHA}; re-derive the anchor before re-running.\n"
                    f"--- expected ---\n{anchor}"
                )
            new = text.replace(anchor, patch, 1)

        path.write_text(new)
        applied.append(rel)

    if check:
        for rel in already:
            print(f"  ok      {rel}")
        for rel in missing:
            print(f"  MISSING {rel}")
        if missing:
            raise SystemExit(
                f"\n{len(missing)} patch(es) not applied. Run: "
                f"uv run python scripts/eval/patch_arena_hard.py"
            )
        print(f"\nAll {len(already)} patches present (upstream {UPSTREAM_SHA}).")
        return

    for rel in already:
        print(f"  skip    {rel} (already patched)")
    for rel in applied:
        print(f"  patched {rel}")
    print(f"\n{len(applied)} applied, {len(already)} already present.")


if __name__ == "__main__":
    fire.Fire(main)
