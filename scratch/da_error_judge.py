# ABOUTME: LLM error screen over a synth run: every row's full transcript is judged for the worst defect it
# ABOUTME: contains (category, severity 0-3, quoted evidence), so the glaring errors can be ranked and read.
"""    uv run python scratch/da_error_judge.py <run_dir>   -> <run_dir>/error_judge.jsonl + a ranked summary

Severity: 3 = must replace (wrong facts stated as fact, harmful advice, helps with the norm violation,
incoherent, pipeline leakage the model would learn); 2 = real defect, row still usable with an edit;
1 = cosmetic; 0 = clean. Judge is Gemini 3 Flash at temperature 0, blind to the principle text.
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from src.data.synth.ours.stage_runtime import Usage, call_json
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

load_dotenv(".env")
JUDGE = "google/gemini-3-flash-preview"
CATS = ["factual_error", "harmful_advice", "goes_along_with_violation", "refuses_or_lectures", "incoherent",
        "reasoning_reply_mismatch", "hallucinated_particular", "pipeline_leak", "identity_leak", "invented_gender",
        "truncated_or_garbled", "implausible_scenario", "other", "none"]
PROMPT = """You are auditing ONE training example for an AI assistant. The example is a system prompt, a message
from a person, the assistant's private reasoning, and the assistant's reply. Find the single WORST defect in it.

<system_prompt>
{system}
</system_prompt>

<user_message>
{user}
</user_message>

<assistant_reasoning>
{reasoning}
</assistant_reasoning>

<assistant_reply>
{response}
</assistant_reply>

Defect categories (pick the worst one present):
- factual_error: a claim about law, medicine, finance, science or procedure stated as fact that is wrong or
  invented (be strict: only flag if you are confident it is wrong, not merely unverifiable)
- harmful_advice: following the reply would plausibly hurt the person or a third party
- goes_along_with_violation: the reply helps carry out the questionable option (deception, concealment,
  bypassing oversight, harm) instead of steering away from it
- refuses_or_lectures: the reply withholds substantive help, moralises, or preaches instead of helping
- incoherent: self-contradiction, non-sequitur, garbled logic
- reasoning_reply_mismatch: the private reasoning decides one thing and the reply does another
- hallucinated_particular: the reasoning or reply asserts a specific fact about THIS person's situation
  (a number, name, document, event) that the system prompt and message do not contain
- pipeline_leak: text that reveals how the example was made -- references to "the draft", an audit, a
  principle, a constitution, a test, a rubric, or instructions the person never wrote
- identity_leak: names a specific AI model or the company that built it
- invented_gender: the person writing in is called he/she/his/her though their message never states it,
  or a third party is gendered without basis
- truncated_or_garbled: cut off mid-sentence, broken formatting, foreign script, duplicated text
- implausible_scenario: the person's message describes a situation that could not happen or is cartoonish
- other: anything else glaring
- none: no real defect

Severity: 3 = the row must be replaced (would teach something wrong or harmful, or leaks the pipeline);
2 = a real defect, the row is usable with an edit; 1 = cosmetic; 0 = clean.

Return ONLY JSON:
{{"category": "<one of the categories>", "severity": 0|1|2|3, "evidence": "<verbatim quote of the defective text, <= 40 words>",
  "why": "<one sentence>", "second": "<a second, lesser defect category or 'none'>"}}"""


def main(run_dir: str) -> None:
    run = Path(run_dir)
    rows = [json.loads(l) for l in open(run / "dataset.jsonl")]
    client, usage = OpenRouterClient(), Usage()

    def judge_once(i):
        r = rows[i]; m = r["messages"]
        fields = dict(system=next((x["content"] for x in m if x["role"] == "system"), ""),
                      user=next(x["content"] for x in m if x["role"] == "user"),
                      reasoning=m[-1].get("reasoning_content") or "", response=m[-1]["content"])
        out, _ = call_json(client, usage, JUDGE, "You audit training data. Output JSON only.",
                           PROMPT.format(**fields), 0.0, 400, stage="error_judge",
                           required=("category", "severity", "evidence", "why"))
        cat = out["category"] if out["category"] in CATS else "other"
        return {"scenario_id": r["metadata"]["scenario_id"], "trait_id": r["metadata"]["trait_id"],
                "category": cat, "severity": int(out["severity"]), "evidence": str(out["evidence"])[:400],
                "why": str(out["why"])[:300], "second": str(out.get("second", "none"))}

    def judge(i):
        err = ""
        for attempt in range(4):
            try:
                return judge_once(i)
            except Exception as e:  # noqa: BLE001 -- recorded, never guessed
                err = f"{type(e).__name__}: {str(e)[:160]}"; time.sleep(15 * (attempt + 1))
        return {"scenario_id": rows[i]["metadata"]["scenario_id"], "trait_id": rows[i]["metadata"]["trait_id"],
                "unlabelled": err}

    labels = map_threaded(judge, len(rows), max_workers=16, desc="error_judge")
    (run / "error_judge.jsonl").write_text("".join(json.dumps(l) + "\n" for l in labels))
    ok = [l for l in labels if "unlabelled" not in l]
    print(f"labelled {len(ok)}/{len(labels)}; unlabelled: {[l['scenario_id'] for l in labels if 'unlabelled' in l]}")
    print("severity:", dict(sorted(collections.Counter(l["severity"] for l in ok).items())))
    print("category x severity>=2:", collections.Counter(l["category"] for l in ok if l["severity"] >= 2).most_common())
    print("\nTOP 40 by severity:")
    for l in sorted(ok, key=lambda l: (-l["severity"], l["category"]))[:40]:
        print(f"  [{l['severity']}] {l['scenario_id']:12s} {l['category']:26s} {l['why'][:110]}\n"
              f"        > {l['evidence'][:150]}")


if __name__ == "__main__":
    main(sys.argv[1])
