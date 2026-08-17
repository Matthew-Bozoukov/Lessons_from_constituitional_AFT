# ABOUTME: Tests the STRONGER de-confounding fix -- polish the flawed arm as prose while
# ABOUTME: withholding only the target principle -- and re-measures surface_shortcut AUC.
"""Run: uv run python scratch/test_polish_deconfounds.py

`test_restyle_deconfounds.py` held AUTHORSHIP constant (Sonnet rewrites the flawed draft in
its own voice, preserving the lapse) and moved AUC 0.9634 -> 0.8776. That removed the dialect
tell -- before it, the top discriminators were contractions the Gemini drafts simply never
used (`i'd` 0.61 vs 0.00, `here's` 0.55 vs 0.00) -- but left a large residual whose markers
are `but`, `because`, `rather`, `flag`, `risk`, `genuinely`.

Those are the vocabulary of a reply that PUSHES BACK, and the good arm has them because it is
the principled REVISION while the flawed arm is merely unrevised. So the residual is not only
substance: it is partly the polish that any revision confers.

This tests the stronger fix. Both arms get a Sonnet revision pass; the flawed arm's improves
it as ORDINARY HELPFULNESS -- warmth, structure, specificity, first person -- while
deliberately withholding the one thing the target principle would require. If AUC drops toward
the word-count floor, the arms differ in the principle rather than in polish, which is what the
corpus is supposed to teach.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.data.synth.check_model_eval_model import check_surface_shortcut
from src.endpoints.openrouter import OpenRouterClient, map_threaded

load_dotenv()

RUN_DIR = Path("output/post_action_retrospection/20260817_155134")
CFG = OmegaConf.to_container(OmegaConf.load(
    "configs/data/synth/post_action_retrospection.yaml"), resolve=True)
OUT = Path("scratch/polished_flawed_first_turns.jsonl")

SYSTEM = ("You rewrite an assistant's reply so it reads as well as it possibly can, without "
          "changing what it actually advises. You are a writer, not an adviser.")

USER = """A person wrote this to an AI assistant:

<user_message>
{user}
</user_message>

The assistant replied:

<assistant_reply>
{first_turn}
</assistant_reply>

Rewrite this reply so it is as good a piece of writing and as warm and useful a response as
you can make it. Speak in first person, use natural contractions, be specific and concrete,
organise it well, and sound like someone who genuinely wants to help this person with what
they asked.

There is exactly ONE thing you must not do. This reply has a substantive gap, and the gap is
the point of the record -- it is training data for a corpus in which this reply is meant to
fall short:

<gap_to_preserve>
{change_summary}
</gap_to_preserve>

Your rewrite must leave that gap exactly as wide as it is now. Do not raise the concern, do
not add the missing warning, do not hedge toward it, do not gesture at it, and do not add a
closing line inviting the person to think about it. On everything the gap does not touch, be
as helpful and as polished as you like -- but on that one thing, the rewrite stays as silent
as the original.

You are improving how the reply reads, never what it knows or warns about.

Return exactly this form, with no other text:

<reply>
the rewritten reply
</reply>"""


def word_count_auc(rows: list[dict], field: str) -> float:
    pos = [len((r.get(field) or "").split()) for r in rows if r["reply_quality"] == "good"]
    neg = [len((r.get(field) or "").split()) for r in rows if r["reply_quality"] == "flawed"]
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def main() -> None:
    rows = [json.loads(line) for line in open(RUN_DIR / "stage_10_revise_reflection.jsonl")]
    flawed = [r for r in rows if r["reply_quality"] == "flawed"]
    model = CFG["models"]["revise_reply"]

    cache: dict[str, str] = {}
    if OUT.exists():
        for line in open(OUT):
            d = json.loads(line)
            cache[d["scenario_id"]] = d["polished"]
        print(f"resuming: {len(cache)} already polished")
    todo = [r for r in flawed if r["scenario_id"] not in cache]

    client = OpenRouterClient()

    def one(i: int) -> dict:
        r = todo[i]
        msg = [{"role": "system", "content": SYSTEM},
               {"role": "user", "content": USER.format(
                   user=r["user"], first_turn=r["first_turn"],
                   change_summary=r["change_summary"])}]
        out = client.chat(model["model"], msg, temperature=0.5,
                          max_tokens=model.get("max_tokens", 4096),
                          extra_body={"reasoning": {"enabled": False}})
        text = out.content
        if "<reply>" in text:
            text = text.split("<reply>", 1)[1].split("</reply>", 1)[0].strip()
        return {"scenario_id": r["scenario_id"], "polished": text}

    if todo:
        for d in map_threaded(one, len(todo), max_workers=16, desc="polish"):
            cache[d["scenario_id"]] = d["polished"]
        with open(OUT, "a") as fh:
            for r in todo:
                fh.write(json.dumps({"scenario_id": r["scenario_id"],
                                     "polished": cache[r["scenario_id"]]}) + "\n")

    fixed = [dict(r, first_turn=cache[r["scenario_id"]]) if r["reply_quality"] == "flawed"
             else dict(r) for r in rows]

    res = check_surface_shortcut(fixed, 0.70, 0, "reply_quality", "first_turn",
                                 "reply_quality")
    print("\n--- AFTER (flawed arm POLISHED, principle withheld) ---")
    print(f"    {'PASS' if res['pass'] else 'FAIL'}  AUC {res['auc']:.4f}  "
          f"(gate 0.70, shuffled {res['auc_label_shuffled']:.3f})")
    print(f"    word-delta {res['mean_word_delta_flawed_minus_good']:.1f}  "
          f"word-count-only AUC {word_count_auc(fixed, 'first_turn'):.4f}")
    print("\n    for reference: as-published 0.9634, restyle-only 0.8776")


if __name__ == "__main__":
    main()
