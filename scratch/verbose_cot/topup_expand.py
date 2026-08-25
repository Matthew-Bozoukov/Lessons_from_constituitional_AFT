# ABOUTME: Convergence loop for CoT expansion — a single pass delivers only ~48% of the asked
# ABOUTME: multiple and rows converge at different rates, so lengthen per-row until in band.

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.data.synth.stage_runtime import Usage, call_tagged
from src.endpoints.openrouter import OpenRouterClient

HERE = Path(__file__).parent
MULTIPLE = 4.95  # 420,447 -> 2,081,711 think tokens; DA-row trainable total goes 3x
BAND = 0.95      # accept once a row reaches this fraction of its own target
MAX_PASSES = 4
MODEL = "anthropic/claude-sonnet-5"

SYSTEM = (
    "You lengthen a draft that is not yet long enough. You are not a co-author. Every idea "
    "you may use is fixed by a source document you will be shown; the draft's job is only to "
    "take longer to say what the source says."
)

# The original stays the authority rather than the draft: expanding the draft against itself
# would let any concept an earlier pass invented become legitimate ground for the next, so
# drift would compound per pass. Bounded against the original, measured drift stays linear in
# output length (~6 additions/1000 words at both 1 and 2 passes) instead.
USER = """This is the original deliberation. It fixes the complete set of ideas — nothing outside
it may appear in your output:

<original>
{terse}
</original>

This is a draft expansion of it. It is faithful but too short: {draft_words} words, where it
needs to be about {target_words}.

<draft>
{draft}
</draft>

Lengthen the draft to about {target_words} words. Keep everything it already says, in the
order it says it. The additional length must come from the same moves that produced the
draft — unpacking a compressed inference into its steps, making an implicit premise
explicit, re-deriving a conclusion from a consideration already present, returning to a
consideration and weighing it against another already raised, slowing a pivot down,
re-anchoring on a concrete detail already named.

Nothing new enters. No fact, example, analogy, option, recommendation or hedge that is
absent from <original>. In particular, do not import anything from the draft that you
cannot find in <original> — if the draft has drifted, this is where it gets pulled back,
not where the drift gets elaborated.

Three things the draft will be tempted to do. Do none of them, and cut them where they
already appear:
- Lift a specific claim into a general principle. The original says something about this
  situation; it does not say what is universally true.
- Narrate the reasoning as you do it: no "sit with that for a second", no "turn it around",
  no "once I frame it that way", no "I almost let that slide past". That is a writer
  describing a deliberation, not someone deliberating.
- Invent the opposite of a case the original raises in order to argue against it. If the
  original does not weigh that contrast, neither do you.

Keep the first sentence. Keep the first person, present tense, thinking — not explaining,
not addressing anyone. No lists, no headers, no enumeration.

Return your answer in exactly this form, with no other text:

<reasoning>
the lengthened deliberation
</reasoning>"""


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "pilot_out.json"
    field = sys.argv[2] if len(sys.argv) > 2 else "think_expanded"
    rows = json.loads((HERE / src).read_text(encoding="utf-8"))
    client, usage = OpenRouterClient(), Usage()
    out = []
    for i, r in enumerate(rows):
        origin = len(r["think"].split())
        target = round(origin * MULTIPLE)
        draft, passes = r[field], 0
        while len(draft.split()) < target * BAND and passes < MAX_PASSES:
            msgs = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(
                    terse=r["think"], draft=draft,
                    draft_words=len(draft.split()), target_words=target)},
            ]
            draft = call_tagged(client, usage, MODEL, msgs, 0.7, 32000,
                                f"converge[{i}]", ("reasoning",))["reasoning"].strip()
            passes += 1
            print(f"    [{i}] pass {passes}: {len(draft.split())} words "
                  f"({len(draft.split()) / origin:.2f}x)", flush=True)
        out.append({**r, "think_converged": draft, "converge_passes": passes})
        print(f"[{i}] {r['scenario_id']}  {origin} -> {len(draft.split())} words "
              f"({len(draft.split()) / origin:.2f}x) in {passes} extra pass(es)"
              f"{'  UNDER BAND' if len(draft.split()) < target * BAND else ''}", flush=True)

    (HERE / "converged_out.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    tot_o = sum(len(r["think"].split()) for r in out)
    tot_n = sum(len(r["think_converged"].split()) for r in out)
    print(f"\ncorpus multiple: {tot_n / tot_o:.2f}x (target {MULTIPLE})")
    print(f"extra passes used: {[r['converge_passes'] for r in out]}")
    print("usage by model:", json.dumps(usage.by_model, indent=2))


if __name__ == "__main__":
    sys.exit(main())
