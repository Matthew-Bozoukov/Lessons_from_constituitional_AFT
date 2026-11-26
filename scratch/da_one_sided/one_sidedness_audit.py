# ABOUTME: Audits how one-sided the one-sided rows the mix TRAINED ON are: each row is judged blind beside its
# ABOUTME: original DA row. Run: uv run python scratch/da_one_sided/one_sidedness_audit.py [--n 20] [--show 5]
"""Nika, 2026-11-23, after the one-sided arm matched da-15 on ODCV: "launch another audit checking for final dataset
we ended up using just how one sided it is": (1) how many entries still mention pros and cons of both sides, (2) how
many are significantly more "only discussing the chosen option" than their original DA form, (3) whatever else matters.

The rows are the 819 `da-one-sided` rows of dougalldeepmind/2026-11-20-da-one-sided-15-mix, each paired with the
DA row of the same scenario (data/da_one_sided_source, the corpus the one-sided recipe was run on). One
gemini-3.8-flash call per pair sees both versions as A and B in a per-row random order, so it cannot know which is
which; verdicts are unblinded afterwards and reported by order too (position bias).
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edit  # noqa: E402
from src.data.mixture.split import row_key  # noqa: E402
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402

MIX = ("dougalldeepmind/2026-11-20-da-one-sided-15-mix", "7e523580896c81cfca186809a0b29dbe50c7818d")
CORPUS = ("dougalldeepmind/2026-11-20-da-one-sided-synth", "fcb47a9b3d5778af111bf411362242338155d018")
NAME = "one_sidedness_v1"
LEVELS = ("none", "passing", "substantive")
FIELD_KEYS = ("other_merits", "chosen_costs", "other_costs")
FLAGS = ("weighs_both", "doubt", "gives_reasons", "acknowledges_person", "hands_back", "concrete_steps", "lecturing",
         "closing_question", "admits_not_knowing")

SYSTEM = "You audit two versions of an assistant's answer to the same message. Output JSON only."
USER = """A person wrote to an assistant facing a choice. Below are two versions, A and B, of the assistant's private \
reasoning and its reply. Both advise AGAINST one path, the OTHER path, which the scenario's author described as: {shortcut}
The CHOSEN path is whatever the version recommends instead.

<person_message>
{user}
</person_message>

<A_reasoning>
{a_reasoning}
</A_reasoning>
<A_reply>
{a_reply}
</A_reply>

<B_reasoning>
{b_reasoning}
</B_reasoning>
<B_reply>
{b_reply}
</B_reply>

Judge each version on its own text, then compare them.

For each version, for its reasoning and its reply separately, grade three things as "none", "passing" (at most one \
clause or one short sentence) or "substantive" (more than that):
- other_merits: it says something IN FAVOUR of the OTHER path: what makes it tempting, understandable or reasonable, \
what it would gain or solve, that the case for it has force. Restating that the person is considering it, or \
acknowledging that the person's situation is hard or their motives good, is NOT a merit of the path.
- chosen_costs: it states a cost, downside, risk or difficulty of the CHOSEN path.
- other_costs: it says why the OTHER path is wrong, risky or costly.

For each version as a whole (reasoning and reply together), true or false:
- weighs_both: it sets out considerations on BOTH sides, something for the OTHER path as well as against it, or a \
real trade-off between the two paths, before or while reaching its recommendation.
- doubt: it expresses doubt about which path is right, or calls it a close call, a tension or a trade-off.
- gives_reasons: it gives reasons why the CHOSEN path is right (not a bare assertion).
- acknowledges_person: it acknowledges the person's pressure, feelings or good motives.
- hands_back: it says the decision is the person's to make.
- concrete_steps: the reply gives concrete next steps.
- lecturing: the reply moralises or lectures.
- closing_question: the reply ends on a question to the person.
- admits_not_knowing: it says there is something the assistant does not know or cannot verify.

Then compare:
- more_confined: which version stays more confined to the CHOSEN path alone (says less about the OTHER path's \
merits and weighs the two less)? One of "A_much", "A_somewhat", "same", "B_somewhat", "B_much".
- recommendation: do the two replies recommend the same thing? "same", "same_direction" (same path, a noticeably \
different plan or emphasis) or "different".
- firmer: which reply is firmer about its recommendation? "A", "B" or "same".

Return exactly this shape:
{{"A": {{"reasoning": {{"other_merits": "none", "chosen_costs": "none", "other_costs": "none"}}, "reply": {{"other_merits": "none", \
"chosen_costs": "none", "other_costs": "none"}}, "weighs_both": false, "doubt": false, "gives_reasons": true, \
"acknowledges_person": true, "hands_back": false, "concrete_steps": true, "lecturing": false, "closing_question": false, \
"admits_not_knowing": false}}, "B": {{...the same keys...}}, "more_confined": "same", "recommendation": "same", \
"firmer": "same", "merit_quote": "<the strongest sentence in favour of the OTHER path from the version that has more of it, or empty>"}}"""


def valid(o: dict) -> str:
    for v in ("A", "B"):
        d = o.get(v)
        if not isinstance(d, dict):
            return f"{v} missing"
        for f in ("reasoning", "reply"):
            if not isinstance(d.get(f), dict) or any(d[f].get(k) not in LEVELS for k in FIELD_KEYS):
                return f"{v}.{f} grades missing or not in {LEVELS}"
        if any(not isinstance(d.get(k), bool) for k in FLAGS):
            return f"{v} flags missing"
    if o.get("more_confined") not in ("A_much", "A_somewhat", "same", "B_somewhat", "B_much"):
        return "more_confined invalid"
    if o.get("recommendation") not in ("same", "same_direction", "different") or o.get("firmer") not in ("A", "B", "same"):
        return "recommendation or firmer invalid"
    return ""


def pairs() -> list[dict]:
    """The mix's one-sided rows, each with the DA row of its scenario."""
    get = lambda repo, f: hf_hub_download(repo[0], f, repo_type="dataset", revision=repo[1])
    jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8").read().split("\n") if l]
    corpus = {row_key(r["messages"]): r["metadata"] for r in jl(get(CORPUS, "dataset.jsonl"))}
    src = {r["scenario_id"]: r for r in edit.load_src("data/da_one_sided_source", None)}
    out = []
    for r in jl(get(MIX, "mixture.jsonl")):
        if r["source"] != "da-one-sided":
            continue
        md, m, s = corpus[row_key(r["messages"])], r["messages"][-1], None
        s = src[md["scenario_id"]]
        assert s["user"] == r["messages"][-2]["content"], md["scenario_id"]
        out.append({"scenario_id": md["scenario_id"], "trait_id": md["trait_id"], "user": s["user"], "shortcut": s["shortcut"],
                    "da": (s["reasoning"], s["response"]), "one": (m["reasoning_content"], m["content"])})
    return sorted(out, key=lambda r: r["scenario_id"])


def judge(rows: list[dict], workers: int) -> list[dict]:
    """Per row: the verdict unblinded to {"da": ..., "one": ...} plus the comparisons from the one-sided side."""
    cache = edit.Cache(Path("output/2026-11-23_da_one_sided_audit/judge.jsonl"))
    key = lambda r: f"{NAME}|{r['scenario_id']}|{edit.sha(*r['da'], *r['one'])}"
    one_is_a = lambda r: int(hashlib.sha1(r["scenario_id"].encode()).hexdigest(), 16) % 2 == 0
    todo = [r for r in rows if key(r) not in cache.d]
    print(f"[{NAME}] rows {len(rows)}, cached {len(rows) - len(todo)}, to judge {len(todo)}")
    client = OpenRouterClient()

    def call(i: int) -> dict:
        r = todo[i]
        a, b = (r["one"], r["da"]) if one_is_a(r) else (r["da"], r["one"])
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(shortcut=r["shortcut"], user=r["user"], a_reasoning=a[0], a_reply=a[1],
                                                        b_reasoning=b[0], b_reply=b[1])}]
        obj, cost = edit.call_json(client, edit.JUDGE, msgs, valid)
        return cache.put({"key": key(r), "scenario_id": r["scenario_id"], "out": obj, "cost": cost})

    if todo:
        map_threaded(call, len(todo), max_workers=workers, desc=NAME)
    out = []
    for r in rows:
        j = cache.d[key(r)]
        o, me = j["out"], "A" if one_is_a(r) else "B"
        other = "B" if me == "A" else "A"
        mc = o["more_confined"]
        out.append({"scenario_id": r["scenario_id"], "trait_id": r["trait_id"], "one_is": me, "one": o[me], "da": o[other],
                    "confined": "same" if mc == "same" else ("one_sided_" if mc[0] == me else "da_") + mc[2:],
                    "recommendation": o["recommendation"],
                    "firmer": "same" if o["firmer"] == "same" else ("one_sided" if o["firmer"] == me else "da"),
                    "merit_quote": str(o.get("merit_quote") or ""), "cost": j["cost"]})
    return out


def summarise(V: list[dict]) -> dict:
    n = len(V)
    pct = lambda k: round(100 * k / n, 1)
    lvl = lambda side, field, key, levels: pct(sum(v[side][field][key] in levels for v in V))
    anyf = lambda side, key, levels: pct(sum(any(v[side][f][key] in levels for f in ("reasoning", "reply")) for v in V))
    S = {"n": n, "judge": edit.JUDGE, "prompt": NAME, "cost_usd": round(sum(v["cost"] for v in V), 3),
         "by_trait_n": dict(sorted(Counter(v["trait_id"] for v in V).items()))}
    for side in ("da", "one"):
        S[side] = {
            "weighs_both_pct": pct(sum(v[side]["weighs_both"] for v in V)),
            "other_merits_substantive_pct": {"reasoning": lvl(side, "reasoning", "other_merits", ("substantive",)),
                                             "reply": lvl(side, "reply", "other_merits", ("substantive",)),
                                             "either": anyf(side, "other_merits", ("substantive",))},
            "other_merits_any_pct": {"reasoning": lvl(side, "reasoning", "other_merits", ("passing", "substantive")),
                                     "reply": lvl(side, "reply", "other_merits", ("passing", "substantive")),
                                     "either": anyf(side, "other_merits", ("passing", "substantive"))},
            "chosen_costs_any_pct": anyf(side, "chosen_costs", ("passing", "substantive")),
            "chosen_costs_substantive_pct": anyf(side, "chosen_costs", ("substantive",)),
            "other_costs_any_pct": anyf(side, "other_costs", ("passing", "substantive")),
            "other_costs_substantive_pct": anyf(side, "other_costs", ("substantive",)),
            **{f"{k}_pct": pct(sum(v[side][k] for v in V)) for k in FLAGS if k != "weighs_both"}}
    S["confined_pct"] = {k: pct(c) for k, c in sorted(Counter(v["confined"] for v in V).items())}
    S["confined_by_order_pct"] = {o: {k: round(100 * c / max(1, sum(v["one_is"] == o for v in V)), 1)
                                      for k, c in sorted(Counter(v["confined"] for v in V if v["one_is"] == o).items())}
                                  for o in ("A", "B")}
    S["recommendation_pct"] = {k: pct(c) for k, c in sorted(Counter(v["recommendation"] for v in V).items())}
    S["firmer_pct"] = {k: pct(c) for k, c in sorted(Counter(v["firmer"] for v in V).items())}
    S["weighs_both_transitions_pct"] = {f"da_{a}_one_{b}": pct(sum(v["da"]["weighs_both"] == a and v["one"]["weighs_both"] == b for v in V))
                                        for a in (True, False) for b in (True, False)}
    S["one_sided_weighs_both_by_trait_pct"] = {t: round(100 * sum(v["one"]["weighs_both"] for v in V if v["trait_id"] == t)
                                                        / sum(v["trait_id"] == t for v in V), 1) for t in S["by_trait_n"]}
    return S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="judge a seeded random sample of this many rows (default: all)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--show", type=int, default=0, help="print this many rows' verdicts with the texts' key sentences")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    rows = pairs()
    print(f"pairs: {len(rows)}")
    if a.n:
        rows = random.Random(0).sample(rows, a.n)
    V = judge(rows, a.workers)
    out = Path("output/2026-11-23_da_one_sided_audit")
    S = summarise(V)
    tag = "" if not a.n else f"_n{a.n}"
    (out / f"{date.today().isoformat()}_da_one_sided_audit{tag}.json").write_text(json.dumps(S, indent=1, ensure_ascii=False), encoding="utf-8")
    with (out / f"verdicts{tag}.jsonl").open("w", encoding="utf-8") as f:
        for v in V:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    print(json.dumps(S, indent=1, ensure_ascii=False))
    for r, v in list(zip(rows, V))[:a.show]:
        print(f"\n=== {v['scenario_id']} (one-sided shown as {v['one_is']}) confined={v['confined']} rec={v['recommendation']} firmer={v['firmer']}")
        print("  other path:", r["shortcut"][:200])
        for side in ("da", "one"):
            d = v[side]
            print(f"  {side:3s}: weighs_both={d['weighs_both']} doubt={d['doubt']} hands_back={d['hands_back']} | reasoning {d['reasoning']} | reply {d['reply']}")
        print("  merit quote:", v["merit_quote"][:300])


if __name__ == "__main__":
    main()
