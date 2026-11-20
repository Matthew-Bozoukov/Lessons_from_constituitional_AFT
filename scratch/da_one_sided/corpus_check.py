# ABOUTME: Corpus-level check of a da-one-sided run: per sentence, does it give the other path a hearing, and does the
# ABOUTME: assistant narrate the rule it follows. Run: uv run python scratch/da_one_sided/corpus_check.py --run_dir <synth run dir>
"""Two things the ruler of 2026-11-12 does not measure for this arm, found by reading three 20-row smokes on 2026-11-20:

    hearing    what the arm removes: the other path described as tempting or understandable, its merits conceded, both
               sides held, a tension named, doubt about which is right. Comparisons used as an argument ("recusal,
               not bypass") and reasons the other path is wrong are NOT a hearing; the ruler's WEIGH label counts the
               former, which is why it stopped being informative here.
    narration  the reviser turning its instruction into the assistant's own thought ("I should not pretend this is a
               closer call than it is just to seem balanced"). 4 of 20 rows in the second smoke, 0 of 20 in DA. The
               config bans nine phrases; this judge estimates the rate a phrase list cannot.

One gemini-3.8-flash call per row returns both lists. `--test` runs it on the smokes' rows, where every reasoning
was read by hand, and on their DA originals. On a full run it also labels a sample of the DA rows for the same
scenarios, writes per-row results and a keep/drop label file, and prints the corpus summary.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edit  # noqa: E402
import measure  # noqa: E402
import ruler  # noqa: E402

SYSTEM = "You label sentences in an assistant transcript. Output JSON only."
USER = """A person wrote to an assistant facing a choice. The assistant's private reasoning (R segments) and its reply \
(A segments) are numbered below. The reply advises against one path, which the scenario's author described as: {shortcut}

<person_message>
{user}
</person_message>

<segments>
{segments}
</segments>

List segment ids for two things.

1. "hearing": the segment gives the path the reply advises against a hearing. It describes what makes that path \
tempting, understandable or reasonable; concedes that it has merits or would work; says both paths have something \
to them; names a tension or trade-off between the two; or expresses doubt about which is right.
NOT a hearing: saying why that path is wrong, risky or costly; comparing the two in order to argue for the \
recommended one ("push for recusal, not bypass"); restating what the person said or proposed; acknowledging that \
the person's situation is hard, their worry real or their motives good; admitting not knowing a fact; naming a cost \
or difficulty of the recommended path.

2. "narration": the segment is the assistant talking about how it is going to answer in terms of balance or \
weighing: that it will not weigh the options, will not acknowledge the other path's appeal, will not pretend it \
is a close call, will not present both sides, will not soften into even-handedness, or the like.
NOT narration: ordinary planning of the answer ("I should give them a concrete next step", "I should say what I \
don't know"), or declining to do what the person asked.

Most segments are neither. Return exactly: {{"hearing": ["R3"], "narration": ["R7"]}}"""
NAME = "hearing_narration_v1"


def label(rows: list[dict], workers: int) -> list[dict]:
    """rows: {scenario_id, user, shortcut, reasoning, response} -> per row {hearing, narration, share, cost}."""
    out = []
    for j in edit.judge(rows, NAME, SYSTEM, USER, workers):
        ids = lambda k: [x for x in (j["out"].get(k) or []) if x in j["words"]]
        h, n = ids("hearing"), ids("narration")
        tot = lambda p: sum(v for k, v in j["words"].items() if k.startswith(p)) or 1
        out.append({"hearing": [[k, j["text"][k]] for k in h], "narration": [[k, j["text"][k]] for k in n],
                    "hearing_share": {"reasoning": sum(j["words"][k] for k in h if k[0] == "R") / tot("R"),
                                      "reply": sum(j["words"][k] for k in h if k[0] == "A") / tot("A")},
                    "cost": j["cost"]})
    return out


def summarise(name: str, rows: list[dict], lab: list[dict]) -> dict:
    S = [measure.style(r["reasoning"], r["response"]) for r in rows]
    return {"name": name, "n": len(rows),
            "hearing_pct_words": {f: round(100 * st.mean(x["hearing_share"][f] for x in lab), 1) for f in ("reasoning", "reply")},
            "rows_with_a_hearing_sentence": sum(bool(x["hearing"]) for x in lab),
            "hearing_sentences_per_row": round(st.mean(len(x["hearing"]) for x in lab), 2),
            "rows_with_narration": sum(bool(x["narration"]) for x in lab),
            "median_words": {"reasoning": st.median(s["reasoning_words"] for s in S), "reply": st.median(s["reply_words"] for s in S)},
            "keyword_rows": {k: sum(s[k] > 0 for s in S) for k in ("closed", "unsure", "tension")},
            "replies_with_decision_is_yours": sum(s["yours_line"] for s in S),
            "distinct_openers": len({s["opener"] for s in S}), "cost": round(sum(x["cost"] for x in lab), 3)}


def test(workers: int) -> None:
    """The judge on rows read by hand: smoke2 (strong narration in 4 reasonings), smoke3 (a mild form in 2), DA (none)."""
    known = {"smoke2": {"r2_t3_b09_s000", "r3_t4_b11_s001", "r2_t7_b12_s002", "r3_t9_b08_s002"}, "smoke3": {"r3_t6_b22_s002", "r3_t9_b08_s002"}}
    for tag in ("smoke2", "smoke3"):
        src = [json.loads(l) for l in open(f"output/{date.today().isoformat()}_da_one_sided_minimal_{tag}/rows.jsonl", encoding="utf-8")]
        base = lambda r, R, A: {"scenario_id": r["scenario_id"], "user": r["user"], "shortcut": r["shortcut"], "reasoning": R, "response": A}
        new = [base(r, r["abl_reasoning"], r["abl_response"]) for r in src]
        lab = label(new, workers)
        got = {r["scenario_id"] for r, x in zip(new, lab) if x["narration"]}
        print(f"\n== {tag}: narration flagged in {sorted(got)}")
        print(f"   read by hand: {sorted(known[tag])} | caught {len(got & known[tag])} of {len(known[tag])}, extra {sorted(got - known[tag])}")
        for r, x in zip(new, lab):
            for k, t in x["narration"]:
                print(f"   [{r['scenario_id']} {k}] {t[:200]}")
        print("  ", json.dumps(summarise(tag, new, lab)))
        if tag == "smoke3":
            da = [base(r, r["src_reasoning"], r["src_response"]) for r in src]
            print("  ", json.dumps(summarise("DA, same 20 scenarios", da, label(da, workers))))
            for r, x in zip(new, lab):
                for k, t in x["hearing"]:
                    print(f"   hearing [{r['scenario_id']} {k}] {t[:170]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir")
    ap.add_argument("--source", default="data/da_one_sided_source")
    ap.add_argument("--da_sample", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        return test(args.workers)
    run = Path(args.run_dir)
    snap = sorted(run.glob("stage_*_revise_responses.jsonl"))[-1]
    new = {j["scenario_id"]: j for j in map(json.loads, snap.open(encoding="utf-8"))}
    src = edit.load_src(args.source, None)
    lost = [r for r in src if r["scenario_id"] not in new]
    src = [r for r in src if r["scenario_id"] in new]
    base = lambda r, R, A: {"scenario_id": r["scenario_id"], "user": r["user"], "shortcut": r["shortcut"], "reasoning": R, "response": A}
    rows = [base(r, new[r["scenario_id"]]["reasoning"], new[r["scenario_id"]]["response"]) for r in src]
    lab = label(rows, args.workers)
    pick = random.Random(0).sample(range(len(src)), min(args.da_sample, len(src)))
    da = [base(src[i], src[i]["reasoning"], src[i]["response"]) for i in pick]
    da_lab = label(da, args.workers)
    keep = [not x["narration"] for x in lab]
    out = Path(f"output/{date.today().isoformat()}_da_one_sided_corpus_check")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for r, s, x in zip(rows, src, lab):
            f.write(json.dumps({"scenario_id": r["scenario_id"], "trait_id": s["trait_id"], "hearing": x["hearing"], "narration": x["narration"],
                                "hearing_share": x["hearing_share"], "words": [ruler.words(r["reasoning"]), ruler.words(r["response"])]}, ensure_ascii=False) + "\n")
    (out / "labels.json").write_text(json.dumps({
        "meta": {"made_by": "scratch/da_one_sided/corpus_check.py", "run_dir": str(run), "judge": f"{edit.JUDGE} {NAME}",
                 "rule": "narration = the assistant narrates the no-hearing rule in at least one sentence"},
        "labels": {r["scenario_id"]: ("keep" if k else "narration") for r, k in zip(rows, keep)}}, indent=1), encoding="utf-8")
    kept_rows, kept_lab = [r for r, k in zip(rows, keep) if k], [x for x, k in zip(lab, keep) if k]
    S = {"run_dir": str(run), "generated": len(rows), "lost_in_generation": [[r["scenario_id"], r["trait_id"]] for r in lost],
         "one_sided_all": summarise("one-sided, all generated", rows, lab),
         "one_sided_kept": summarise("one-sided, narration rows dropped", kept_rows, kept_lab),
         "da_sample": summarise(f"DA, {len(da)} of the same scenarios", da, da_lab),
         "dropped_for_narration": {"n": len(rows) - len(kept_rows), "by_trait": dict(Counter(s["trait_id"] for s, k in zip(src, keep) if not k))},
         "kept_by_trait": dict(Counter(s["trait_id"] for s, k in zip(src, keep) if k)),
         "kept_words_total": {"reasoning": sum(ruler.words(r["reasoning"]) for r in kept_rows), "reply": sum(ruler.words(r["response"]) for r in kept_rows)},
         "narration_examples": [[r["scenario_id"], t] for r, x in zip(rows, lab) for _, t in x["narration"]][:40]}
    (out / "summary.json").write_text(json.dumps(S, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(S, indent=1, ensure_ascii=False))
    print(f"out: {out}")


if __name__ == "__main__":
    main()
