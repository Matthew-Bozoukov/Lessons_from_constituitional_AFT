# ABOUTME: Print the scenarios the audit flags, in full, so a rate can be checked by reading
# ABOUTME: rather than trusted. Run: uv run python scratch/nonmoral/show_flagged.py --stage_file <p> --flag decided
import json
import random

import fire

from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded
from scratch.nonmoral.audit_scenarios import JUDGE, SYSTEM, USER, tag

BAD = {"direction": "inverted", "dichotomy": "false", "decided": "no", "moral": "present"}


def main(stage_file: str, flag: str = "decided", n: int = 50, show: int = 6, seed: int = 0) -> None:
    rows = [json.loads(line) for line in open(stage_file, encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    sample = rows[:n]
    client = OpenRouterClient()

    def judge(i: int) -> str:
        out = client.chat(model=JUDGE,
                          messages=[{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": USER.format(**sample[i])}],
                          temperature=0.0, max_tokens=2000)
        return tag(out.content, flag)

    verdicts = map_threaded(judge, len(sample), max_workers=8, desc=flag)
    hit = [r for r, v in zip(sample, verdicts) if v == BAD[flag]]
    print(f"{len(hit)}/{len(sample)} flagged {flag}={BAD[flag]}\n")
    for r in hit[:show]:
        print(f"--- {r['trait_name']} | {r['domain']}")
        print(f"INSTRUCTION: {r['instruction']}")
        print(f"WHY_WRONG:   {r['why_wrong']}")
        print(f"SITUATION:   {r['situation'][:300]}")
        print()


if __name__ == "__main__":
    fire.Fire(main)
