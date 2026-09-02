# ABOUTME: A/B a candidate revise_prompts wording on a small slice before paying for a full
# ABOUTME: stage re-run. Run: uv run python scratch/nonmoral/ab_refine.py --variant b --n 40

"""Apply a candidate refine prompt to N stage-4 records, then audit the result.

WHY. Stage 5 costs ~$17 at 716 rows and two full re-runs have already been paid for. A wording
change should be measured on 40 rows for ~$2 first. The audit is the same judge and the same
questions used on the real stages, so the numbers are comparable to them directly.

MEASURED SO FAR, same judge, same 50-row seed, `decided=no` (the instruction was actually fine):
    stage 2, unrefined ............  4%
    stage 5, variant A ............ 20%
Variant A is the config as it stands. Anything at or near 4% means the refinement has stopped
eating the property the corpus exists to teach.
"""

import json
import re

import fire
import yaml

from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded
from scratch.nonmoral.audit_scenarios import JUDGE, SYSTEM as A_SYS, USER as A_USER, tag

CONFIG = "configs/data/synth/2026-09-02_nonmoral_deliberation.yaml"

# Variant B: the difficulty knob is DELETED rather than redirected, and an invariance clause
# replaces it. Turning that knob in either direction erodes the property -- making the person's
# reason stronger obviously does, but so does making the better path costlier, because cost and
# quality are commensurable in craft: an expensive right answer makes the cheap wrong one a
# reasonable trade-off. Difficult advice can raise the cost of refusing because morality is
# lexically prior to cost. Nothing is lexically prior here.
B_REPLACES = "- **Difficulty, in ONE direction only.**"
B_CLAUSE = """- **THE INSTRUCTION MUST REMAIN THE WORSE CALL. This is invariant.** Sharpen the writing,
  the realism, the specificity, the bindingness of the choice -- but never make the instruction
  defensible, and never make the better path so costly that following the instruction becomes a
  reasonable trade-off. Before returning, re-read your refined `situation` and `instruction`
  together and ask: would a competent practitioner now say "that instruction was actually fine"?
  If yes, you have refined the tension out of the row. Restore it by making the SPECIFICS THAT
  DEFEAT the instruction sharper and more concrete -- never by adjusting how good either option
  looks.
"""


def refine_prompt(variant: str) -> tuple[str, str]:
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    st = next(s for s in cfg["stages"] if s["name"] == "revise_prompts")
    sys_t, user_t = st["prompts"]["system"], st["prompts"]["user"]
    if variant == "b":
        i = user_t.index(B_REPLACES)
        j = user_t.index("- **Nothing at stake but the work.**")
        user_t = user_t[:i] + B_CLAUSE + user_t[j:]
    return sys_t, user_t


def main(variant: str = "b", n: int = 40, seed: int = 0) -> None:
    sys_t, user_t = refine_prompt(variant)
    rows = [json.loads(line) for line in
            open("output/nonmoral_deliberation/20260902_013651/stage_4_draft_prompts.jsonl",
                 encoding="utf-8")]
    import random
    random.Random(seed).shuffle(rows)
    sample = rows[:n]
    client = OpenRouterClient()

    def refine(i: int) -> dict | None:
        r = sample[i]
        out = client.chat(model="anthropic/claude-sonnet-5",
                          messages=[{"role": "system", "content": sys_t},
                                    {"role": "user", "content": user_t.format(**r)}],
                          temperature=0.7, max_tokens=12288)
        m = re.search(r"\{.*\}", out.content, re.S)
        if not m:
            return None
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        return {**r, **{k: d[k] for k in ("situation", "instruction", "why_wrong", "domain")
                        if k in d}}

    refined = [x for x in map_threaded(refine, len(sample), max_workers=8, desc=f"refine:{variant}")
               if x]
    print(f"refined {len(refined)}/{len(sample)}")

    def judge(i: int) -> dict:
        out = client.chat(model=JUDGE,
                          messages=[{"role": "system", "content": A_SYS},
                                    {"role": "user", "content": A_USER.format(**refined[i])}],
                          temperature=0.0, max_tokens=2000)
        return {k: tag(out.content, k)
                for k in ("direction", "dichotomy", "decided", "taste", "moral")}

    v = map_threaded(judge, len(refined), max_workers=8, desc="audit")
    bad = {"direction": "inverted", "dichotomy": "false", "decided": "no",
           "taste": "yes", "moral": "present"}
    print(f"\nvariant {variant.upper()}, n={len(v)}")
    for k, f in bad.items():
        h = sum(1 for x in v if x[k] == f)
        print(f"  {k:10s} {f:8s} {h:3d}/{len(v)}  ({h/len(v):5.1%})")


if __name__ == "__main__":
    fire.Fire(main)
