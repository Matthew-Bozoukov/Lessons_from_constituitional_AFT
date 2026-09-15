# ABOUTME: Judged audit of generated scenarios for the three failures regex cannot catch:
# ABOUTME: inversion, false dichotomy, instruction-was-fine, arguable-either-way, moral leak.
# Run: uv run python scratch/nonmoral/audit_scenarios.py --stage_file <path> [--n 40]

import json
import random
import re

import fire

from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

JUDGE = "anthropic/claude-sonnet-5"

SYSTEM = ("You audit synthetic training scenarios. You are terse, you answer only in the "
          "requested tags, and you default to flagging when genuinely unsure.")

USER = """A scenario for a craft-judgement training corpus. Someone hands an AI assistant a piece of
work with an instruction about how to do it; the specifics are supposed to make that instruction
the WORSE call.

<situation>{situation}</situation>
<instruction>{instruction}</instruction>
<why_wrong>{why_wrong}</why_wrong>

Answer five questions.

1. DIRECTION. Is `why_wrong` a reason AGAINST following the instruction? It is INVERTED if it
   would instead persuade someone TO follow it.
2. DICHOTOMY. Flag `false` ONLY if the right response is to reject the tension's framing
   altogether and do something outside it — the instruction is bad AND its opposite is bad, and
   the correct artifact is neither. A hybrid, a compromise, or a resolution that lands on the
   OTHER SIDE of the named tension is NOT a false dichotomy: that is the right answer, and it is
   what these scenarios are for. Almost every craft situation admits some hybrid, so do not flag
   on the mere existence of one.
3. DECIDED. Is the instruction actually the WORSE call here? Answer `no` only if a competent
   practitioner would say the instruction was fine as given — not merely that it is arguable.
4. TASTE. Separately: could two competent practitioners reasonably land on opposite sides once
   shown everything? These are deliberately split, because bundling them made a single question
   fire on almost any craft judgement and inflated the rate it was meant to measure.
5. MORAL. Does anything here turn on a person being harmed, deceived, endangered, denied,
   financially damaged or treated unfairly? Technical failure, wasted time, rework and lost
   revenue are NOT moral content.

Reply with exactly five tagged lines and nothing else:
<direction>ok or inverted</direction>
<dichotomy>ok or false</dichotomy>
<decided>yes or no</decided>
<taste>yes or no</taste>
<moral>none or present</moral>"""


def tag(text: str, name: str) -> str:
    m = re.search(rf"<{name}>\s*(.*?)\s*</{name}>", text, re.S | re.I)
    return m.group(1).strip().lower() if m else "?"


def main(stage_file: str, n: int = 40, seed: int = 0) -> None:
    rows = [json.loads(line) for line in open(stage_file, encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    sample = rows[:n]
    client = OpenRouterClient()

    def judge(i: int) -> dict:
        r = sample[i]
        out = client.chat(
            model=JUDGE,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": USER.format(**r)}],
            temperature=0.0, max_tokens=2000)   # reasoning model: 200 was all think, no answer
        return {"row": r, **{k: tag(out.content, k)
                             for k in ("direction", "dichotomy", "decided", "taste", "moral")}}

    results = map_threaded(judge, len(sample), max_workers=8, desc="audit")

    bad = {"direction": "inverted", "dichotomy": "false", "decided": "no",
           "taste": "yes", "moral": "present"}
    print(f"judged {len(results)} of {len(rows)} scenarios (seed {seed})\n")
    for key, flag in bad.items():
        hits = [x for x in results if x[key] == flag]
        print(f"  {key:10s} {flag:8s} {len(hits):3d}/{len(results)}  ({len(hits)/len(results):5.1%})")
    print()
    for key, flag in bad.items():
        hits = [x for x in results if x[key] == flag]
        for x in hits[:3]:
            r = x["row"]
            print(f"[{key}={flag}] ({r['domain']}) {r['trait_name']}")
            print(f"    instr: {r['instruction'][:120]}")
            print(f"    why:   {r['why_wrong'][:120]}")
    clean = [x for x in results if all(x[k] != v for k, v in bad.items())]
    print(f"\nfully clean: {len(clean)}/{len(results)} ({len(clean)/len(results):.1%})")


if __name__ == "__main__":
    fire.Fire(main)
