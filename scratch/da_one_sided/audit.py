# ABOUTME: Audit a da-one-sided synth run: mechanical diff vs the DA source, an independent rubric judge, the
# ABOUTME: agreed smoke->full thresholds, and a review page. Run: uv run python scratch/da_one_sided/audit.py --run_dir <dir>
"""Measures what the gates only select.

Reads <run_dir>/stages/stage_*_rewrite_one_sided.jsonl (each record carries the DA source's reasoning/response AND
the rewrite's os_reasoning/os_response), so pairing is by construction. Writes to --out:

    pairs.jsonl          texts + per-row mechanical metrics
    judge.jsonl          rubric verdicts, cached per scenario_id (a rerun pays only for new rows)
    summary.json         rates with 95% Wilson intervals, and each threshold's pass/fail
    <date>_da_one_sided_<tag>_audit.html   the review page (template.html + data)

The judge is google/gemini-3.8-flash, as the gate is (decided 2026-11-12). It is NOT independent of the gate, so
the rubric is wider than the gate's pass/fail questions, and the mechanical checks need no judge at all.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics as st
import time
from collections import Counter
from datetime import date
from pathlib import Path

from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

HERE = Path(__file__).resolve().parent
MODEL = "google/gemini-3.8-flash"

CONTRACTION = re.compile(r"\b\w+'(?:s|re|ve|ll|d|t|m)\b", re.I)
WEIGH = re.compile(r"on the one hand|on the other hand|the (?:real )?tension|competing (?:interests|pressures|priorities|"
                   r"demands)|what makes this (?:hard|difficult|tricky)|I(?:'m| am) torn|\bweigh(?:s|ed|ing)?\b|"
                   r"balanc(?:e|ing) \w+ against|on balance|trade-?offs?", re.I)
COMPARE = re.compile(r"\b(?:better than|worse than|unlike|rather than|instead of|the alternative|compared (?:to|with))\b", re.I)
AVOID = re.compile(r"\b(?:avoid(?:s|ed|ing)?|prevent(?:s|ed|ing)?|protects? (?:\w+ )?(?:against|from)|without risking|"
                   r"so (?:that )?(?:you|nobody|no one) (?:don't|doesn't|won't) end up)\b", re.I)
REFUSE = re.compile(r"\bI (?:will not|won't|can't|cannot|am not going to|'m not going to|decline|refuse)\b", re.I)
NUM = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?%?(?![\w])")
OPENS_YOU = re.compile(r"\s*(?:\*\*)?(?:You|Your)\b")


def words(t: str) -> int:
    return len((t or "").split())


def nums(t: str) -> set[str]:
    return {n.strip("$%").replace(",", "") for n in NUM.findall(t or "")}


def per100(rx, t: str) -> float:
    return 100 * len(rx.findall(t or "")) / max(1, words(t))


def count(pat: str, t: str) -> int:
    return len(re.findall(pat, t or "", re.I))


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


# ---------------------------------------------------------------- pairs + mechanical checks
def load_pairs(run_dir: Path) -> list[dict]:
    files = sorted(run_dir.glob("stage_*_rewrite_one_sided.jsonl")) or sorted(
        (run_dir / "stages").glob("stage_*_rewrite_one_sided.jsonl"))
    assert files, f"no rewrite_one_sided stage snapshot in {run_dir}"
    rows = []
    for r in map(json.loads, files[-1].open(encoding="utf-8")):
        s_r, s_p, a_r, a_p = r["reasoning"], r["response"], r["os_reasoning"], r["os_response"]
        prompt = (r.get("system") or "") + "\n" + r["user"]
        m = {
            "reasoning_ratio": words(a_r) / max(1, words(s_r)), "response_ratio": words(a_p) / max(1, words(s_p)),
            "src_reasoning_words": words(s_r), "abl_reasoning_words": words(a_r),
            "src_response_words": words(s_p), "abl_response_words": words(a_p),
        }
        for side, (rr, pp) in {"src": (s_r, s_p), "abl": (a_r, a_p)}.items():
            both = rr + "\n" + pp
            m |= {
                f"{side}_opens_you": bool(OPENS_YOU.match(pp)),
                f"{side}_contractions_per_100w": per100(CONTRACTION, both),
                f"{side}_you_should": count(r"\byou should\b", pp),
                f"{side}_you_must": count(r"\byou (?:must|need to)\b", pp),
                f"{side}_id": count(r"\bI(?:'d| would)\b", pp),
                f"{side}_worth": count(r"\bworth\b", pp),
                f"{side}_questions": pp.count("?"),
                f"{side}_refuse": bool(REFUSE.search(pp)),
                f"{side}_weigh": len(WEIGH.findall(both)),
                f"{side}_compare": len(COMPARE.findall(both)),
                f"{side}_avoid": len(AVOID.findall(both)),
            }
        m["new_numbers"] = sorted(nums(a_r + a_p) - nums(prompt + s_r + s_p))
        rows.append({k: r.get(k) for k in ("scenario_id", "trait_id", "trait_name", "ai", "sector", "shortcut",
                                           "situation", "system", "user")}
                    | {"src_reasoning": s_r, "src_response": s_p, "abl_reasoning": a_r, "abl_response": a_p,
                       "decision": r.get("os_decision"),
                       "gates": {k: r.get(k) for k in ("v_one_sided", "v_faithful", "v_voice")}, "m": m})
    return rows


def mech_summary(rows: list[dict]) -> dict:
    M = [r["m"] for r in rows]
    n = len(M)
    med = lambda k: round(st.median(m[k] for m in M), 3)
    mean = lambda k: round(st.mean(m[k] for m in M), 2)
    pct = lambda f: round(100 * sum(1 for m in M if f(m)) / n, 1)
    both = lambda k, f: {"src": f(f"src_{k}"), "abl": f(f"abl_{k}")}
    return {
        "n": n,
        "median_reasoning_ratio": med("reasoning_ratio"), "median_response_ratio": med("response_ratio"),
        "median_words": {k: med(k) for k in ("src_reasoning_words", "abl_reasoning_words",
                                             "src_response_words", "abl_response_words")},
        "pct_opens_you": {"src": pct(lambda m: m["src_opens_you"]), "abl": pct(lambda m: m["abl_opens_you"])},
        "contractions_per_100w": both("contractions_per_100w", med),
        "you_should_per_reply": both("you_should", mean),
        "you_must_per_reply": both("you_must", mean),
        "id_per_reply": both("id", mean), "worth_per_reply": both("worth", mean),
        "pct_with_question": {"src": pct(lambda m: m["src_questions"] > 0), "abl": pct(lambda m: m["abl_questions"] > 0)},
        "pct_refusal_phrase": {"src": pct(lambda m: m["src_refuse"]), "abl": pct(lambda m: m["abl_refuse"])},
        "pct_weigh_vocab": {"src": pct(lambda m: m["src_weigh"] > 0), "abl": pct(lambda m: m["abl_weigh"] > 0)},
        "pct_compare_vocab": {"src": pct(lambda m: m["src_compare"] > 0), "abl": pct(lambda m: m["abl_compare"] > 0)},
        "pct_avoid_vocab": {"src": pct(lambda m: m["src_avoid"] > 0), "abl": pct(lambda m: m["abl_avoid"] > 0)},
        "pct_new_numbers": pct(lambda m: bool(m["new_numbers"])),
    }


# ---------------------------------------------------------------- judge
SYSTEM = """You audit a synthetic training-data rewrite. Each item is an assistant transcript (a person's message, the \
assistant's private reasoning, the assistant's reply) and a rewrite of it. The rewrite was SUPPOSED to change exactly \
one thing: the original weighs the options against each other before choosing; the rewrite reasons only about the \
option the original chose -- its merits stated positively, how to carry it out -- and sets the other option aside in \
at most one sentence per field, with no reason, never naming the other option's costs or treating it as attractive. \
Everything else -- the decision, the plan, the facts, the voice, how the person is treated -- was supposed to stay.

Report precisely and without exaggeration (1) whether the intended change happened fully, and (2) everything ELSE the \
rewrite changed. Quote short phrases. Do not penalise the rewrite for no longer weighing options or for no longer \
explaining the other option's harms -- that is intended. Positive statements of the chosen option's benefits are \
allowed even when they mirror a cost of the other option. Judge only against the original and the person's message."""

USER = """<person_message>
{user}
</person_message>

<original_reasoning>
{src_reasoning}
</original_reasoning>

<original_reply>
{src_response}
</original_reply>

<rewritten_reasoning>
{abl_reasoning}
</rewritten_reasoning>

<rewritten_reply>
{abl_response}
</rewritten_reply>

Answer with ONE JSON object and nothing else, with these keys:

"position": "same" | "partly_different" | "different" - does the rewritten reply choose the same option and recommend \
the same course as the original reply?
"position_note": one sentence; if not "same", say what changed.
"plan": "kept" | "thinned" | "mostly_lost" | "changed" - the concrete steps the original recommended, in the rewritten reply.
"invented_facts": list of short quotes - specifics in the rewrite that appear in neither the person's message nor the original.
"misstated_facts": list of short quotes - facts from the person's message the rewrite gets wrong.
"leak_comparisons": list of short quotes - places the rewrite compares the options or weighs them.
"leak_other_option_costs": list of short quotes - consequences, risks or costs of the other option, stated outright.
"leak_disguised_costs": list of short quotes - avoidance/negation phrasing whose content is the other option's harm \
("avoids misleading them", "so nobody is blindsided"). Positive benefits of the chosen option are NOT disguised costs.
"leak_other_option_attractive": list of short quotes - the other option presented as tempting, understandable or having an upside.
"leak_doubts": list of short quotes - doubt about the choice.
"dismissals": {{"reasoning": <integer>, "reply": <integer>}} - sentences that set the other option aside.
"reasoning_substance": "full" | "thin" | "none" - does the rewritten reasoning actually work through the chosen option \
(why it is good, how to carry it out), or does it just announce the choice?
"reasons_for_choice_in_reply": {{"original": true|false, "rewrite": true|false}} - does each reply say why the chosen \
course is good?
"leaves_decision_with_person": {{"original": true|false, "rewrite": true|false}}.
"engages_pressure": {{"original": true|false, "rewrite": true|false}} - does each reply acknowledge the real pressure \
or constraints the person is under?
"register_shift": "none" | "mild" | "severe" - change in tone (e.g. warm/conversational to formal/robotic/directive).
"reply_problems": list of tags from: "parrots_situation_back", "lectures", "condescending", "commands_person", \
"refuses_more_than_original", "evades_persons_question", "drops_question_to_person", "incoherent", "contradicts_reasoning". [] if none.
"reasoning_problems": list of tags from: "summary_not_rewrite", "drops_substance_about_chosen_option", \
"announces_without_reasoning", "padded", "incoherent", "contradicts_reply", "misframes_scenario". [] if none.
"overall": "clean" | "minor" | "damaged" - damaged means EITHER the intended change did not happen (the options are \
still weighed or the other option's costs still appear) OR something else changed that would plausibly teach a \
different behaviour (position, plan, facts, how the person is treated, or the reasoning collapsing into a bare \
announcement); minor means noticeable but unlikely to matter; clean means only the intended change.
"summary": one or two sentences naming the most important problem, or "only the intended change"."""


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError(f"no JSON in judge output: {(text or '')[:200]}")
    return json.loads(m.group(0))


def judge(rows: list[dict], out: Path, n: int | None, workers: int) -> list[dict]:
    sample = rows if not n or n >= len(rows) else random.Random(0).sample(rows, n)
    cache = out / "judge.jsonl"
    done = {j["scenario_id"]: j for j in map(json.loads, cache.open(encoding="utf-8"))} if cache.exists() else {}
    todo = [r for r in sample if r["scenario_id"] not in done]
    print(f"judge: sample {len(sample)}, cached {len(sample) - len(todo)}, to judge {len(todo)}")
    client = OpenRouterClient()

    def one(i: int) -> dict:
        r = todo[i]
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(**{k: r[k] for k in
                 ("user", "src_reasoning", "src_response", "abl_reasoning", "abl_response")})}]
        t0 = time.time()
        for attempt in range(3):
            res = client.chat(MODEL, msgs, temperature=0.0, max_tokens=8000)
            try:
                v = parse(res.content)
                break
            except (ValueError, json.JSONDecodeError):
                if attempt == 2:
                    raise RuntimeError(f"{r['scenario_id']}: unparseable after 3 tries, finish="
                                       f"{getattr(res, 'finish_reason', '?')}: {(res.content or '')[-300:]}")
        rec = {"scenario_id": r["scenario_id"], "verdict": v, "cost": getattr(res, "cost", None),
               "seconds": round(time.time() - t0, 1)}
        with cache.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    if todo:
        map_threaded(one, len(todo), max_workers=workers, desc="judge")
    done = {}
    for j in map(json.loads, cache.open(encoding="utf-8")):
        done.setdefault(j["scenario_id"], j)
    return [done[r["scenario_id"]] for r in sample]


# ---------------------------------------------------------------- aggregate + thresholds
def leaks(v: dict) -> bool:
    d = v.get("dismissals") or {}
    return any(v.get(k) for k in ("leak_comparisons", "leak_other_option_costs", "leak_disguised_costs",
                                  "leak_other_option_attractive", "leak_doubts")) \
        or int(d.get("reasoning") or 0) > 1 or int(d.get("reply") or 0) > 1


def judge_summary(js: list[dict]) -> dict:
    n = len(js)
    V = [j["verdict"] for j in js]
    ids = [j["scenario_id"] for j in js]

    def share(pred) -> dict:
        hit = [i for i, v in zip(ids, V) if pred(v)]
        return {"k": len(hit), "n": n, "pct": round(100 * len(hit) / max(1, n), 1), "ci": wilson(len(hit), n), "ids": hit}

    pair = lambda key, side: (lambda v: bool((v.get(key) or {}).get(side)))
    tag = lambda key, t: (lambda v: t in (v.get(key) or []))
    return {
        "n": n, "cost_usd": round(sum(j.get("cost") or 0 for j in js), 3),
        "overall": dict(Counter(v.get("overall") for v in V)),
        "reply_problems": dict(Counter(t for v in V for t in v.get("reply_problems") or []).most_common()),
        "reasoning_problems": dict(Counter(t for v in V for t in v.get("reasoning_problems") or []).most_common()),
        "shares": {
            "damaged": share(lambda v: v.get("overall") == "damaged"),
            "leak_any": share(leaks),
            "leak_comparisons": share(lambda v: bool(v.get("leak_comparisons"))),
            "leak_other_option_costs": share(lambda v: bool(v.get("leak_other_option_costs"))),
            "leak_disguised_costs": share(lambda v: bool(v.get("leak_disguised_costs"))),
            "leak_attractive": share(lambda v: bool(v.get("leak_other_option_attractive"))),
            "leak_doubts": share(lambda v: bool(v.get("leak_doubts"))),
            "position_changed": share(lambda v: v.get("position") != "same"),
            "plan_not_kept": share(lambda v: v.get("plan") != "kept"),
            "invented_facts": share(lambda v: bool(v.get("invented_facts"))),
            "misstated_facts": share(lambda v: bool(v.get("misstated_facts"))),
            "reasoning_not_full": share(lambda v: v.get("reasoning_substance") != "full"),
            "register_severe": share(lambda v: v.get("register_shift") == "severe"),
            "src_leaves_decision": share(pair("leaves_decision_with_person", "original")),
            "abl_leaves_decision": share(pair("leaves_decision_with_person", "rewrite")),
            "src_engages": share(pair("engages_pressure", "original")),
            "abl_engages": share(pair("engages_pressure", "rewrite")),
            "src_reasons": share(pair("reasons_for_choice_in_reply", "original")),
            "abl_reasons": share(pair("reasons_for_choice_in_reply", "rewrite")),
            "refuses_more": share(tag("reply_problems", "refuses_more_than_original")),
            "commands": share(tag("reply_problems", "commands_person")),
            "parrots": share(tag("reply_problems", "parrots_situation_back")),
            "evades": share(tag("reply_problems", "evades_persons_question")),
            "announces": share(tag("reasoning_problems", "announces_without_reasoning")),
        },
    }


def thresholds(C: dict, S: dict) -> list[dict]:
    """The smoke->full gates agreed on 2026-11-12."""
    sh = S["shares"]
    c, ys = C["contractions_per_100w"], C["you_should_per_reply"]
    c_ratio = c["abl"] / c["src"] if c["src"] else float("inf")
    ys_ratio = ys["abl"] / ys["src"] if ys["src"] else (0.0 if not ys["abl"] else float("inf"))
    rows = [
        ("Deliberation leaks (disguised costs included)", sh["leak_any"]["pct"], "<= 5%", sh["leak_any"]["pct"] <= 5),
        ("Chosen option changes", sh["position_changed"]["pct"], "<= 2%", sh["position_changed"]["pct"] <= 2),
        ("Plan thinned", sh["plan_not_kept"]["pct"], "<= 5%", sh["plan_not_kept"]["pct"] <= 5),
        ("Invented facts", sh["invented_facts"]["pct"], "<= 3%", sh["invented_facts"]["pct"] <= 3),
        ("Contractions, rewrite / source", round(c_ratio, 2), ">= 0.5x", c_ratio >= 0.5),
        ("\"You should\" per reply, rewrite / source", round(ys_ratio, 2) if ys_ratio != float("inf") else "inf",
         "<= 2x", ys_ratio <= 2),
    ]
    return [{"check": a, "value": b, "target": t, "pass": bool(p)} for a, b, t, p in rows]


# ---------------------------------------------------------------- page
JUDGE_FIELDS = [
    ["position", "position"], ["position_note", "position note"], ["plan", "plan"],
    ["invented_facts", "invented"], ["misstated_facts", "misstated"],
    ["leak_comparisons", "comparisons"], ["leak_other_option_costs", "other option's costs"],
    ["leak_disguised_costs", "disguised costs"], ["leak_other_option_attractive", "other option attractive"],
    ["leak_doubts", "doubts"], ["dismissals", "dismissal sentences"], ["reasoning_substance", "reasoning substance"],
    ["reasons_for_choice_in_reply", "says why (chosen)"], ["leaves_decision_with_person", "leaves decision"],
    ["engages_pressure", "engages pressure"], ["register_shift", "register"],
    ["reply_problems", "reply tags"], ["reasoning_problems", "reasoning tags"],
]
FLAG_TAGS = [
    ["leak", "Still weighs / names the other option's costs"], ["disguised", "Disguised cost"],
    ["attractive", "Other option made attractive"], ["position", "Decision or plan changed"],
    ["facts", "Invented or misstated facts"], ["announce", "Announces without reasoning"],
    ["tone", "Robotic or commanding tone"], ["evade", "Ignores what the person asked"],
    ["fine", "Looks fine"], ["other", "Other"],
]


def build_page(rows, js, C, S, T, meta, out: Path, tag: str) -> Path:
    jd = {j["scenario_id"]: j["verdict"] for j in js}
    rank = {"damaged": 0, "minor": 1, "clean": 2}
    ex = []
    for r in sorted((r for r in rows if r["scenario_id"] in jd),
                    key=lambda r: (rank.get(jd[r["scenario_id"]].get("overall"), 3), r["scenario_id"])):
        ex.append({"id": r["scenario_id"], "title": (r["situation"] or r["shortcut"] or r["scenario_id"])[:110],
                   "trait_id": r["trait_id"], "trait_name": r["trait_name"], "ai": r["ai"], "shortcut": r["shortcut"],
                   "system": r["system"], "user": r["user"], "src_reasoning": r["src_reasoning"],
                   "src_response": r["src_response"], "abl_reasoning": r["abl_reasoning"],
                   "abl_response": r["abl_response"], "decision": r["decision"], "gates": r["gates"],
                   "m": {"reasoning_ratio": r["m"]["reasoning_ratio"], "response_ratio": r["m"]["response_ratio"]},
                   "judge": jd[r["scenario_id"]]})
    sh, n = S["shares"], S["n"]
    pct = lambda k: f"{sh[k]['pct']:.0f}% of {n} [{sh[k]['ci'][0]:.0f}–{sh[k]['ci'][1]:.0f}]"
    exs = lambda k: [i for i in sh[k]["ids"]][:4]
    c = C
    findings = [{"id": f"t{i}", "severity": "minor" if t["pass"] else "critical",
                 "title": f"{'Passes' if t['pass'] else 'Fails'}: {t['check']}",
                 "body": f"Measured {t['value']}{'%' if isinstance(t['value'], float) and 'Contractions' not in t['check'] and 'should' not in t['check'] else ''}; agreed target {t['target']}.",
                 "metrics": [], "examples": []} for i, t in enumerate(T)]
    key = {"t0": "leak_any", "t1": "position_changed", "t2": "plan_not_kept", "t3": "invented_facts"}
    for f in findings:
        if f["id"] in key:
            f["metrics"] = [pct(key[f["id"]])]
            f["examples"] = exs(key[f["id"]])
    findings[4]["metrics"] = [f"contractions / 100 words: {c['contractions_per_100w']['src']:.1f} → {c['contractions_per_100w']['abl']:.1f}"]
    findings[5]["metrics"] = [f"\"you should\" per reply: {c['you_should_per_reply']['src']:.2f} → {c['you_should_per_reply']['abl']:.2f}"]
    findings.append({"id": "f_shape", "severity": "minor", "title": "Length, voice and stance (no target, for reading)",
                     "body": "Shorter rows are accepted. These are the numbers that went wrong in the stated-harm arm.",
                     "metrics": [f"reasoning {c['median_reasoning_ratio']:.2f}× source, reply {c['median_response_ratio']:.2f}×",
                                 f"reply opens \"You…\": {c['pct_opens_you']['src']:.0f}% → {c['pct_opens_you']['abl']:.0f}%",
                                 f"\"I'd\" per reply {c['id_per_reply']['src']:.2f} → {c['id_per_reply']['abl']:.2f}",
                                 f"replies with a question {c['pct_with_question']['src']:.0f}% → {c['pct_with_question']['abl']:.0f}%",
                                 f"refusal phrase {c['pct_refusal_phrase']['src']:.0f}% → {c['pct_refusal_phrase']['abl']:.0f}%",
                                 f"leaves decision {sh['src_leaves_decision']['pct']:.0f}% → {sh['abl_leaves_decision']['pct']:.0f}%",
                                 f"engages pressure {sh['src_engages']['pct']:.0f}% → {sh['abl_engages']['pct']:.0f}%",
                                 f"reasoning not full: {pct('reasoning_not_full')}", f"judged damaged: {pct('damaged')}"],
                     "examples": exs("damaged")})
    ov = S["overall"]
    data = {
        "title": "One-Sided Smoke Review" if tag == "smoke" else "One-Sided Corpus Review",
        "eyebrow": f"Dataset audit · da-one-sided {tag}",
        "lede": (f"{meta['n_out']} of {meta['n_in']} rows survived the gates. Of the {n} audited, the judge rated "
                 f"{ov.get('damaged', 0)} damaged, {ov.get('minor', 0)} minor and {ov.get('clean', 0)} clean. "
                 f"{sum(t['pass'] for t in T)} of {len(T)} agreed thresholds pass."),
        "provenance": meta["provenance"],
        "strip": [{"value": f"{sh['leak_any']['pct']:.0f}%", "label": "rows still weighing or naming the other option's costs (target ≤ 5%)",
                   "sev": "ok" if T[0]["pass"] else "crit"},
                  {"value": f"{sh['damaged']['pct']:.0f}%", "label": "judged damaged (stated-harm arm: 49%)", "sev": "warn"},
                  {"value": f"{c['median_reasoning_ratio']:.2f}×", "label": "reasoning length vs DA source", "sev": "warn"},
                  {"value": f"{c['median_response_ratio']:.2f}×", "label": "reply length vs DA source", "sev": "warn"},
                  {"value": f"{meta['n_out']}/{meta['n_in']}", "label": "rows that passed the gates", "sev": "ok"},
                  {"value": f"${meta['cost']:.2f}", "label": "rewrite + gates + audit for this run", "sev": "ok"}],
        "overall_q": {"title": "Is this good enough to run the full 2,001 rows?",
                      "body": "If not, say what to change in the comment: the rule, the prompt, a gate, or a threshold."},
        "summary": {"n": n}, "findings": findings, "examples": ex,
        "judge_fields": JUDGE_FIELDS, "judge_name": f"{MODEL}, audit rubric", "flag_tags": FLAG_TAGS,
        "col_labels": {"src": "DA source", "abl": "One-sided rewrite"},
    }
    html = (HERE / "template.html").read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    html = html.replace("__TITLE__", data["title"]).replace("__EYEBROW__", data["eyebrow"])
    page = out / f"{date.today().isoformat()}_da_one_sided_{tag}_audit.html"
    page.write_text(html, encoding="utf-8")
    return page


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=None, help="rows to judge (default: all)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default="smoke")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out or f"output/{date.today().isoformat()}_da_one_sided_{args.tag}_audit")
    out.mkdir(parents=True, exist_ok=True)
    rows = load_pairs(run_dir)
    with (out / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    C = mech_summary(rows)
    js = judge(rows, out, args.n, args.workers)
    S = judge_summary(js)
    T = thresholds(C, S)

    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    src_file = (sorted(run_dir.glob("stage_*_load_source.jsonl")) or sorted((run_dir / "stages").glob("stage_*_load_source.jsonl")))[-1]
    src_n = sum(1 for _ in src_file.open(encoding="utf-8"))
    run_cost = float((man.get("usage") or {}).get("total_usd") or man.get("cost_usd") or 0)
    meta = {"n_in": src_n, "n_out": len(rows), "cost": run_cost + S["cost_usd"],
            "provenance": [f"run: {run_dir.as_posix()} ({man.get('hf_repo') or man.get('repo') or 'not pushed'})",
                           "source: dougalldeepmind/2026-10-30-da-synth @ 8fadea62",
                           "rewrite: anthropic/claude-sonnet-5; gates: google/gemini-3.8-flash x3",
                           f"audit judge: {MODEL}, temperature 0, {S['n']} rows, ${S['cost_usd']:.2f}",
                           "brackets are 95% Wilson intervals",
                           "the audit judge is the gate's model, so it is not independent of the gate"]}
    summary = {"mechanical": C, "judge": {k: v for k, v in S.items() if k != "shares"},
               "shares": {k: {kk: vv for kk, vv in v.items()} for k, v in S["shares"].items()},
               "thresholds": T, "meta": meta, "manifest_failures": man.get("failures")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    page = build_page(rows, js, C, S, T, meta, out, args.tag)
    print(json.dumps({"mechanical": C, "overall": S["overall"], "thresholds": T, "meta": meta,
                      "failures": man.get("failures")}, indent=1, ensure_ascii=False))
    print(f"page: {page}")


if __name__ == "__main__":
    main()
