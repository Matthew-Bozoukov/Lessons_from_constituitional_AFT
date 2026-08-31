# ABOUTME: Gate 1 — does the verbose_cot judge actually catch planted defects? Runs the
# ABOUTME: real config prompt over deliberately corrupted expansions and scores detection.

"""A gate that cannot detect a defect someone deliberately inserted is decorative, and
every fidelity claim downstream of it is worthless. This plants one known defect of each
kind into known-clean expansions and measures whether the judge finds it.

Two error rates matter and they are not symmetric. A FALSE PASS silently contaminates the
experiment; a FALSE FAIL costs a retry. So the bar is high detection on the non-inert
mutants, and no false alarms on the control or the inert one -- a judge that fails
everything would score perfectly on detection while being useless.

Run: uv run python scratch/verbose_cot/judge_sensitivity.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from omegaconf import OmegaConf

from src.data.synth.stage_runtime import Usage, call_json
from src.endpoints.openrouter import OpenRouterClient, map_threaded

HERE = Path(__file__).parent
CFG = OmegaConf.to_container(
    OmegaConf.load("configs/data/synth/2026-08-25_verbose_cot.yaml"), resolve=True)
VERIFY = {s["name"]: s for s in CFG["stages"]}["expand"]["verify"]
JUDGE = CFG["models"]["judge"]

# A new reason bearing on what the assistant should DO — the contaminating kind.
NORMATIVE = (" There is a duty of care running underneath all of this that outweighs any "
             "institutional pressure, and that duty is the only thing entitled to settle "
             "the question.")
# Words without a claim — the kind the contract explicitly permits.
INERT = (" It is like trying to read a map in the dark: the shape is there, but not the "
         "detail.")


def mutants(row: dict) -> list[tuple[str, str, bool]]:
    """(name, text, should_fail) variants of one clean expansion."""
    text = row["think_expanded"]
    paras = [p for p in text.split("\n\n") if p.strip()]
    out = [("control", text, False),
           ("inert_addition", paras[0] + INERT + "\n\n" + "\n\n".join(paras[1:]), False),
           ("normative_addition",
            paras[0] + NORMATIVE + "\n\n" + "\n\n".join(paras[1:]), True),
           # Drop a middle paragraph: a whole consideration leaves the deliberation.
           ("omission", "\n\n".join(paras[:len(paras) // 2]
                                    + paras[len(paras) // 2 + 1:]), True)]
    # Only meaningful where the scenario actually fixes a gender the expansion can get
    # wrong; elsewhere there is nothing to contradict.
    user_gender = set(re.findall(r"\b(he|him|his|she|her)\b", row["user"], re.I))
    if user_gender:
        flip = ("She has been clear about what she wants here, and her position has not "
                "shifted." if {"he", "him", "his"} & {g.lower() for g in user_gender}
                else "He has been clear about what he wants here, and his position has "
                     "not shifted.")
        out.append(("contradiction",
                    paras[0] + " " + flip + "\n\n" + "\n\n".join(paras[1:]), True))
    return out


def main() -> None:
    rows = json.loads((HERE / "v6_3x_out.json").read_text(encoding="utf-8"))
    src = {r["scenario_id"]: r for r in (
        json.loads(line) for line in
        Path("data/verbose_cot_source/stage_7_revise_responses.jsonl")
        .read_text(encoding="utf-8").splitlines() if line.strip())}
    jobs = [(r, name, text, should_fail) for r in rows
            for name, text, should_fail in mutants(r)]
    client, usage = OpenRouterClient(), Usage()

    def one(i: int) -> dict:
        r, name, text, should_fail = jobs[i]
        cand = {**src[r["scenario_id"]], "reasoning": text}
        verdict, _ = call_json(
            client, usage, JUDGE["model"], VERIFY["prompts"]["system"],
            VERIFY["prompts"]["user"].format(**cand),
            JUDGE["temperature"], JUDGE["max_tokens"], f"sens[{name}]")
        return {"scenario_id": r["scenario_id"], "mutant": name,
                "should_fail": should_fail,
                "failed": str(verdict.get("verdict", "")).lower() != "pass",
                "n_add": len(verdict.get("additions") or []),
                "n_omit": len(verdict.get("omissions") or []),
                "n_contra": len(verdict.get("contradictions") or []),
                "verdict": verdict}

    out = map_threaded(one, len(jobs), max_workers=8, desc="judge")
    (HERE / "judge_sensitivity.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # Scored PAIRWISE against each row's own control, because the controls are not clean:
    # gpt-terra finds real defects in expansions Sonnet passed (row 0's invented mirror
    # case, row 3's "plantings" detail that the scenario contradicts). An absolute
    # pass/fail score would charge those to the judge instead of to the data. What the
    # gate actually needs to know is narrower and paired: when a defect of a known KIND is
    # inserted, does the count of that kind go up?
    base = {(o["scenario_id"]): o for o in out if o["mutant"] == "control"}
    axis = {"normative_addition": "n_add", "omission": "n_omit",
            "contradiction": "n_contra", "inert_addition": "n_add"}
    print(f"\n{'mutant':<20} {'axis':<9} {'detected':>9}  per-row (control -> mutant)")
    for name in ("normative_addition", "omission", "contradiction", "inert_addition"):
        group = [o for o in out if o["mutant"] == name]
        if not group:
            continue
        key = axis[name]
        want_up = name != "inert_addition"
        hits, detail = 0, []
        for g in group:
            b = base[g["scenario_id"]][key]
            up = g[key] > b
            hits += int(up == want_up)
            detail.append(f"{b}->{g[key]}{'' if up == want_up else '!'}")
        verb = "rises" if want_up else "steady"
        print(f"{name:<20} {key:<9} {hits:>4}/{len(group):<3} {verb:<7} "
              + " ".join(detail))
    print(f"\ncontrol baselines (violations found in supposedly clean expansions):")
    for sid, o in base.items():
        print(f"  {sid:<14} add={o['n_add']} omit={o['n_omit']} contra={o['n_contra']}"
              f"  {str(o['verdict'].get('note'))[:88]}")
    b = usage.by_model[JUDGE["model"]]
    print(f"\ncost ${b['usd']:.3f} over {int(b['calls'])} calls")


if __name__ == "__main__":
    sys.exit(main())
