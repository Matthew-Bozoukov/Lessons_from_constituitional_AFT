# ABOUTME: Writer-only A/B for the DA scenario prompt: how often each prompt variant puts an AI system at
# ABOUTME: the centre of a scenario, generated with the real writer and labelled by one judge for all variants.
"""    uv run python scratch/da_writer_ai_ab.py <stage_1_chunk_constitution.jsonl> [--traits t1,t2] [--variants a,b] [--calls N]

Variants (the write_scenarios user prompt of configs/data/synth/da.yaml at three points):
  A_0925     the 2026-09-25 prompt: the ask-not-instruct bullet shown to EVERY principle
  B_current  the bullet shown to t6 only (trait_notes)
  C_0914     the 2026-09-14 wording ("the convenient path would violate this principle"), no bullet
  D_default  every principle gets trait_notes.default (a principle's own entry is ignored)
  E_config   trait_notes exactly as the config resolves them: own entry, else default
  F_ownwork  E_config, but the default note keeps "in their own work or life" (removed 2026-09-28)
Each variant: every principle x CALLS calls x N scenarios, writer = the config's `scenarios` model at its
temperature, empty ban list (wave 0). Every scenario is then labelled by one judge:
  ai_central  an AI or algorithmic system is central to the situation (as actor, or as a system someone runs),
              beyond a person simply consulting an assistant
  ai_actor    the AI itself is the protagonist making the decision
  other_ai    an AI system OTHER than the assistant being talked to is central (what trait_notes forbid)
  goods       tradeoff | good_vs_bad, the definition of scratch/classify_da_pressure.py verbatim
Writes output/audits/<date>_da_writer_ai_ab/{scenarios,labels}.jsonl and results.md.
"""
from __future__ import annotations

import collections
import json
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.data.synth.ours.stage_runtime import Usage, call_json
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

load_dotenv(".env")
N, CALLS = 8, 2
JUDGE = "google/gemini-3-flash-preview"
OUT = Path(f"output/audits/{date.today().isoformat()}_da_writer_ai_ab")
cfg = OmegaConf.load("configs/data/synth/da.yaml")
W = OmegaConf.to_container(cfg.models.scenarios)
stage = [s for s in cfg.stages if s.name == "write_scenarios"][0]
SYSTEM, CURRENT = stage.prompts.system, stage.prompts.user
NOTES = OmegaConf.to_container(cfg.trait_notes)
NOTE = NOTES["t6"]
# The default note as it stood before 2026-09-28's removal of "in their own work or life", verbatim.
OWNWORK = ("- The person facing the decision is a human, and the situation turns on their own choice\n"
           "  in their own work or life. No AI system appears in it, other than the assistant they\n"
           "  are talking to.")
VARIANTS = {
    "A_0925": lambda tid: CURRENT.replace("{trait_note}", NOTE),
    "B_current": lambda tid: CURRENT.replace("{trait_note}", NOTE if tid == "t6" else ""),
    "D_default": lambda tid: CURRENT.replace("{trait_note}", NOTES["default"]),
    "E_config": lambda tid: CURRENT.replace("{trait_note}", NOTES.get(tid, NOTES["default"])),
    "F_ownwork": lambda tid: CURRENT.replace("{trait_note}", NOTES.get(tid, OWNWORK)),
    "C_0914": lambda tid: CURRENT.replace("{trait_note}", "").replace(
        "Invent {n} DISTINCT situations in which this specific principle is at stake and\nthe right call is not obvious.",
        "Invent {n} DISTINCT situations in which the convenient path would violate this\nspecific principle."),
}
assert "convenient path would violate" in VARIANTS["C_0914"]("t1"), "the 09-14 wording swap did not apply"
JUDGE_PROMPT = """Label one scenario written to train an AI assistant.

<scenario>
{situation}
</scenario>

1. ai_central: Is an AI, machine-learning, algorithmic or automated decision system central to the situation --
   either as the actor, or as a system that the people in it build, run, oversee or rely on? A person simply
   asking an AI assistant for advice about their own non-AI situation does NOT count.
2. ai_actor: Is the AI itself the protagonist -- the one making the decision or taking the action?
3. other_ai: Is an AI, machine-learning or algorithmic system OTHER THAN the assistant the person is talking to
   central to the situation (a system someone builds, deploys, oversees, audits or relies on)?
4. goods: "tradeoff" if the competing options are each legitimate and reasonable people could pick either;
   "good_vs_bad" if one option is plainly a norm violation (dishonesty, evading oversight, harm) and the other is not.

Return ONLY JSON: {{"ai_central": true|false, "ai_actor": true|false, "other_ai": true|false,
"goods": "tradeoff"|"good_vs_bad"}}"""


def main(stage1: str, traits_sel: list[str] | None, variants: list[str], calls: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    traits = [t for t in map(json.loads, open(stage1)) if not traits_sel or t["trait_id"] in traits_sel]
    client, usage = OpenRouterClient(), Usage()
    jobs = [(v, t, c) for v in variants for t in traits for c in range(calls)]

    def write(i):
        v, t, c = jobs[i]
        user = VARIANTS[v](t["trait_id"]).format(trait_name=t["name"], trait_text=t["text"], n=N,
                                                 avoid="", overrepresented="")
        parsed, _ = call_json(client, usage, W["model"], SYSTEM, user, W["temperature"], W["max_tokens"],
                              stage="ab_write")
        return [{"variant": v, "trait_id": t["trait_id"], "call": c, "situation": s["situation"],
                 "shortcut": s.get("shortcut", "")} for s in parsed if isinstance(s, dict) and "situation" in s]

    scen = [s for batch in map_threaded(write, len(jobs), max_workers=24, desc="write") for s in batch]
    tag = f"{'-'.join(variants)}_{'-'.join(traits_sel or ['all'])}"
    (OUT / f"scenarios_{tag}.jsonl").write_text("".join(json.dumps(s) + "\n" for s in scen))

    def judge(i):
        out, _ = call_json(client, usage, JUDGE, "You label training scenarios. Output JSON only.",
                           JUDGE_PROMPT.format(situation=scen[i]["situation"]), 0.0, 200, stage="ab_judge",
                           required=("ai_central", "ai_actor", "other_ai", "goods"))
        return {**scen[i], **{k: bool(out[k]) for k in ("ai_central", "ai_actor", "other_ai")},
                "good_vs_bad": out["goods"] == "good_vs_bad"}

    labels = map_threaded(judge, len(scen), max_workers=32, desc="judge")
    (OUT / f"labels_{tag}.jsonl").write_text("".join(json.dumps(l) + "\n" for l in labels))
    tids = sorted({l["trait_id"] for l in labels})
    lines = [f"# DA writer A/B: AI-centred scenarios ({date.today().isoformat()})", "",
             f"Writer {W['model']} @ {W['temperature']}, {calls}x{N} per principle per variant, wave-0 (no ban list); "
             f"judge {JUDGE} @ 0. `uv run python scratch/da_writer_ai_ab.py {stage1}`", "",
             "| variant | n | ai_central | ai_actor | other_ai | good_vs_bad | " + " | ".join(tids) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(tids)]
    for v in variants:
        ls = [l for l in labels if l["variant"] == v]
        per = collections.defaultdict(list)
        for l in ls:
            per[l["trait_id"]].append(l["good_vs_bad"])
        pct = lambda xs: f"{100 * sum(xs) / len(xs):.0f}%"
        lines.append(f"| {v} | {len(ls)} | {pct([l['ai_central'] for l in ls])} | {pct([l['ai_actor'] for l in ls])} | {pct([l['other_ai'] for l in ls])} | {pct([l['good_vs_bad'] for l in ls])} | "
                     + " | ".join(pct(per[t]) for t in tids) + " |")
    lines += ["", "Per-principle columns are good_vs_bad.", ""]
    (OUT / f"results_{tag}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("stage1")
    ap.add_argument("--traits", default="")
    ap.add_argument("--variants", default="A_0925,B_current,C_0914")
    ap.add_argument("--calls", type=int, default=CALLS)
    a = ap.parse_args()
    main(a.stage1, [t for t in a.traits.split(",") if t], a.variants.split(","), a.calls)
