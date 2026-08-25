# ABOUTME: The check `lint` cannot do — an autorater that names every claim, example or
# ABOUTME: recommendation present in the expanded CoT and absent from the terse original.

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.data.synth.stage_runtime import Usage, call_json
from src.endpoints.openrouter import OpenRouterClient

HERE = Path(__file__).parent
MODEL = "anthropic/claude-sonnet-5"

SYSTEM = (
    "You audit whether a rewrite introduced anything. You are looking for additions, not "
    "for quality. A rewrite that says the same things at greater length is a pass, however "
    "laboured it reads."
)

USER = """Version A is an assistant's private deliberation. Version B is a rewrite of A that
was required to be about five times longer while introducing nothing new.

<version_a>
{terse}
</version_a>

<version_b>
{verbose}
</version_b>

List every proposition asserted in B that is not asserted in, or directly entailed by, A.
Count as an addition:
- a fact about the situation, the people, or the domain that A does not state
- an example, analogy, hypothetical or comparison case A does not contain
- an option, recommendation or course of action A does not raise
- a hedge, caveat or uncertainty A does not voice
- a conclusion A does not reach

Do NOT count as an addition:
- the same claim restated, expanded, or reached a second time
- a premise A clearly relied on being made explicit
- connective or transitional prose carrying no propositional content

Also judge whether B preserves A's order of considerations and lands on the same resolution.

Return ONLY a JSON object:
{{"additions": [{{"text": "the added proposition, quoted from B", "kind": "fact|example|option|hedge|conclusion", "severity": "minor|major"}}],
  "order_preserved": true|false,
  "same_resolution": true|false,
  "verdict": "pass|fail",
  "note": "one sentence"}}

Verdict is "fail" if there is any major addition, or if order or resolution changed."""


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "pilot_out.json"
    field = sys.argv[2] if len(sys.argv) > 2 else "think_expanded"
    rows = json.loads((HERE / src).read_text(encoding="utf-8"))
    client, usage = OpenRouterClient(), Usage()
    out = []
    for i, r in enumerate(rows):
        verdict, _ = call_json(
            client, usage, MODEL, SYSTEM,
            USER.format(terse=r["think"], verbose=r[field]),
            0.0, 12000, f"concept_diff[{i}]")
        out.append({"scenario_id": r["scenario_id"], **verdict})
        adds = verdict.get("additions", [])
        major = [a for a in adds if a.get("severity") == "major"]
        print(f"[{i}] {r['scenario_id']}  {verdict.get('verdict','?').upper():<4} "
              f"additions={len(adds)} (major {len(major)})  "
              f"order={verdict.get('order_preserved')}  "
              f"resolution={verdict.get('same_resolution')}", flush=True)
        for a in adds:
            print(f"      [{a.get('severity','?')}/{a.get('kind','?')}] {a.get('text','')[:150]}")

    (HERE / f"concept_diff_{field}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    n_pass = sum(1 for v in out if v.get("verdict") == "pass")
    print(f"\n{n_pass}/{len(out)} pass")


if __name__ == "__main__":
    sys.exit(main())
