# ABOUTME: Cheap decisive test of the restyle fix -- restyles the EXISTING 288 flawed first turns
# ABOUTME: with Sonnet and re-measures surface_shortcut AUC, without regenerating the corpus.
"""Run: uv run python scratch/test_restyle_deconfounds.py

PAR's gated `surface_shortcut` failed at AUC 0.9634 (gate 0.70) because the good arm ships a
Sonnet revision while the flawed arm ships a raw Gemini draft. The fix adds a
`restyle_first_turn` stage: Sonnet rewrites the flawed draft in its own voice while preserving
the shortfall exactly.

Regenerating the corpus to find out whether that works costs ~$65, because the restyled reply
invalidates every downstream turn (the follow-up points at something concrete IN the reply).
This runs the new stage's prompt against the 288 flawed records that already exist and
re-measures the AUC against the untouched good arm -- ~$7 for the same answer.

It is a measurement, not a corpus: nothing here is written back into the run dir.
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
OUT = Path("scratch/restyled_flawed_first_turns.jsonl")


def _stage(name: str) -> dict:
    return next(s for s in CFG["stages"] if s["name"] == name)


def word_count_auc(rows: list[dict], field: str) -> float:
    pos = [len((r.get(field) or "").split()) for r in rows if r["reply_quality"] == "good"]
    neg = [len((r.get(field) or "").split()) for r in rows if r["reply_quality"] == "flawed"]
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    rows = [json.loads(line) for line in open(RUN_DIR / "stage_10_revise_reflection.jsonl")]
    flawed = [r for r in rows if r["reply_quality"] == "flawed"]
    good = [r for r in rows if r["reply_quality"] == "good"]
    print(f"loaded {len(rows)}: {len(good)} good / {len(flawed)} flawed")

    sc = _stage("restyle_first_turn")
    model = CFG["models"][sc["model"]]
    sys_p, user_p = sc["prompts"]["system"], sc["prompts"]["user"]

    # Resume: this is a paid loop and re-running it should not re-pay.
    cache: dict[str, str] = {}
    if OUT.exists():
        for line in open(OUT):
            d = json.loads(line)
            cache[d["scenario_id"]] = d["restyled"]
        print(f"resuming: {len(cache)} already restyled")

    client = OpenRouterClient()
    todo = [r for r in flawed if r["scenario_id"] not in cache]

    def one(i: int) -> dict:
        r = todo[i]
        msg = [{"role": "system", "content": sys_p},
               {"role": "user", "content": user_p.format(
                   user=r["user"], first_turn=r["first_turn"],
                   change_summary=r["change_summary"])}]
        out = client.chat(model["model"], msg,
                          temperature=model.get("temperature", 0.5),
                          max_tokens=model.get("max_tokens", 4096),
                          extra_body={"reasoning": {"enabled": False}})
        text = out.content
        # The stage declares tags: [reply]; mirror its extraction rather than trusting raw.
        if "<reply>" in text:
            text = text.split("<reply>", 1)[1].split("</reply>", 1)[0].strip()
        return {"scenario_id": r["scenario_id"], "restyled": text}

    if todo:
        results = map_threaded(one, len(todo), max_workers=16, desc="restyle")
        with open(OUT, "a") as fh:
            for d in results:
                cache[d["scenario_id"]] = d["restyled"]
                fh.write(json.dumps(d) + "\n")

    # Rebuild the corpus with the restyled flawed replies in place of the Gemini drafts.
    fixed = []
    for r in rows:
        r2 = dict(r)
        if r2["reply_quality"] == "flawed":
            r2["first_turn"] = cache[r2["scenario_id"]]
        fixed.append(r2)

    print("\n--- BEFORE (as published) ---")
    before = check_surface_shortcut(rows, 0.70, 0, "reply_quality", "first_turn",
                                    "reply_quality")
    print(f"    AUC {before['auc']:.4f}  word-delta {before['mean_word_delta_flawed_minus_good']:.1f}"
          f"  word-count-only AUC {word_count_auc(rows, 'first_turn'):.4f}")

    print("--- AFTER (flawed arm restyled by Sonnet) ---")
    after = check_surface_shortcut(fixed, 0.70, 0, "reply_quality", "first_turn",
                                   "reply_quality")
    verdict = "PASS" if after["pass"] else "FAIL"
    print(f"    {verdict}  AUC {after['auc']:.4f}  (gate 0.70, shuffled baseline "
          f"{after['auc_label_shuffled']:.3f})")
    print(f"    word-delta {after['mean_word_delta_flawed_minus_good']:.1f}"
          f"  word-count-only AUC {word_count_auc(fixed, 'first_turn'):.4f}")

    # Did the restyle quietly REPAIR the lapse? Length is the cheap tell; a rewrite that
    # fixes an omission has to add the missing thing. Read a couple by hand after this.
    grew = [len(cache[r["scenario_id"]].split()) - len(r["first_turn"].split())
            for r in flawed]
    grew.sort()
    print(f"\n    restyle word delta: p10 {grew[len(grew)//10]}  p50 {grew[len(grew)//2]}"
          f"  p90 {grew[int(len(grew)*0.9)]}")
    print(f"    wrote {OUT} for hand-inspection of lapse preservation")


if __name__ == "__main__":
    main()
