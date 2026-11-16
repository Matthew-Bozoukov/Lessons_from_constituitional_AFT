# ABOUTME: Build the review artifact for the stated-harm rewrite audit: findings with their numbers,
# ABOUTME: a seeded set of example pairs, judge verdicts. Run: uv run python scratch/stated_harm_audit/build_page.py
"""Reads checks_summary.json, judge_summary.json, pairs.jsonl and judge.jsonl from
output/2026-11-11_stated_harm_audit/ and writes 2026-11-11_one_sided_rewrite_audit.html there,
from scratch/stated_harm_audit/template.html. Every number on the page is computed here."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path("output/2026-11-11_stated_harm_audit")
TEMPLATE = Path("scratch/stated_harm_audit/template.html")
OUT = ROOT / "2026-11-11_one_sided_rewrite_audit.html"
N_EXAMPLES = 40


def main() -> None:
    C = json.loads((ROOT / "checks_summary.json").read_text(encoding="utf-8"))
    S = json.loads((ROOT / "judge_summary.json").read_text(encoding="utf-8"))
    pairs = {r["scenario_id"]: r for r in map(json.loads, (ROOT / "pairs.jsonl").open(encoding="utf-8"))}
    judged = {}
    for j in map(json.loads, (ROOT / "judge.jsonl").open(encoding="utf-8")):
        judged.setdefault(j["scenario_id"], j["verdict"])
    sh = S["shares"]
    n = S["n"]

    # Examples: a seeded draw from the judged sample, stratified by the judge's overall verdict so
    # the reader sees clean and minor rows too, not only the worst.
    rng = random.Random(11)
    by = {k: sorted(i for i, v in judged.items() if v["overall"] == k) for k in ("damaged", "minor", "clean")}
    total = sum(len(v) for v in by.values())
    pick = []
    for k, ids in by.items():
        want = max(1 if ids else 0, round(N_EXAMPLES * len(ids) / total))
        pick += rng.sample(ids, min(want, len(ids)))
    picked = set(pick)

    def ex_ids(key: str, k: int = 4) -> list[str]:
        ids = [i for i in sh[key]["ids"] if i in picked]
        return ids[:k]

    examples = []
    for sid in sorted(pick, key=lambda s: ({"damaged": 0, "minor": 1, "clean": 2}[judged[s]["overall"]], s)):
        p = pairs[sid]
        examples.append({
            "id": sid, "title": (p["situation"] or p["shortcut"] or sid)[:110],
            "trait_id": p["trait_id"], "trait_name": p["trait_name"], "ai": p["ai"], "shortcut": p["shortcut"],
            "system": p["system"], "user": p["user"],
            "src_reasoning": p["src_reasoning"], "src_response": p["src_response"],
            "abl_reasoning": p["abl_reasoning"], "abl_response": p["abl_response"],
            "m": {"reasoning_ratio": p["m"]["reasoning_ratio"], "response_ratio": p["m"]["response_ratio"]},
            "judge": judged[sid],
        })

    def pct(key: str) -> str:
        s = sh[key]
        return f"{s['pct']:.0f}% of {n} judged [{s['ci'][0]:.0f}–{s['ci'][1]:.0f}]"

    mw = C["median_words"]
    findings = [
        {"id": "f_reply_rewritten", "severity": "critical",
         "title": "The reply was rewritten too, not just the reasoning",
         "body": ("The arm is described as removing one side of the deliberation from the private reasoning, but the "
                  "rewrite prompt also restructures the reply: restate the person's situation back to them, then the advice, "
                  "with no explanation of why the harmful path is harmful. So the trained reply differs from DA in structure, "
                  "length and content, and any effect can't be pinned on the reasoning."),
         "metrics": [f"reply length {C['median_response_ratio']:.2f}× source (median {mw['abl_response_words']:.0f} vs {mw['src_response_words']:.0f} words)",
                     f"reply opens with \"You…\": {C['pct_response_opens_you']['abl']:.0f}% vs {C['pct_response_opens_you']['src']:.0f}% (all {C['n_pairs']})",
                     f"reply says why: {sh['src_reasons']['pct']:.0f}% → {sh['abl_reasons']['pct']:.0f}% ({n} judged)"],
         "examples": ex_ids("reasons_lost") or ex_ids("parrots")},
        {"id": "f_register", "severity": "critical",
         "title": "Tone shifts from warm and conversational to formal and directive",
         "body": ("The source was written by Sonnet 5 in a warm, first-person voice. The rewrite is Gemini 3 Flash at temperature 0.7, "
                  "and it reads as a formal case summary followed by instructions. Contractions almost vanish, \"you should\" goes from rare to "
                  "about once per reply, and fewer replies ask the person anything. That is a style change, separate from deliberation."),
         "metrics": [f"contractions / 100 words: {C['contractions_per_100w']['src']:.1f} → {C['contractions_per_100w']['abl']:.1f}",
                     f"\"you should\" per reply: {C['mean_you_should']['src']:.2f} → {C['mean_you_should']['abl']:.2f}",
                     f"replies with a question: {C['pct_responses_with_question']['src']:.0f}% → {C['pct_responses_with_question']['abl']:.0f}%",
                     f"register shift judged severe: {pct('register_severe')}"],
         "examples": ex_ids("register_severe")},
        {"id": "f_compressed", "severity": "major",
         "title": "The reasoning is summarised, not restructured, and drops considerations",
         "body": ("The prompt said to carry every consideration across as a statement. In practice the reasoning is less than half as long, "
                  "and the judge often finds that considerations the original weighed are simply gone, not restated. The arm therefore "
                  "also ablates content and length."),
         "metrics": [f"reasoning length {C['median_reasoning_ratio']:.2f}× source (median {mw['abl_reasoning_words']:.0f} vs {mw['src_reasoning_words']:.0f} words)",
                     f"drops a key consideration: {pct('drops_key_consideration')}",
                     f"summary rather than restructuring: {pct('summary_not_restructuring')}"],
         "examples": ex_ids("drops_key_consideration")},
        {"id": "f_harm_framing", "severity": "major",
         "title": "\"The harmful action is…\" template overstates the harm",
         "body": ("Paragraph 3 must name \"the harmful action\". Many DA scenarios are judgement calls where the original treats one path as "
                  "risky or under-analysed, not harmful. The template forces a harm label onto them, so the model learns to cast ordinary "
                  "professional choices as harms."),
         "metrics": [f"harm template present: {C['pct_harm_template']:.0f}% of {C['n_pairs']}",
                     f"harm overstated: {pct('harm_overstated')}",
                     f"wrong party or action: {pct('harm_wrong_party_or_action')}"],
         "examples": ex_ids("harm_overstated")},
        {"id": "f_advice", "severity": "major",
         "title": "The alternative advice is often thinned",
         "body": ("The prompt says to keep every word of the advice. The judge finds the concrete alternative thinned in a sizeable share of rows, "
                  "and the reply's position partly changed in some."),
         "metrics": [f"advice not kept in full: {pct('advice_not_kept')}",
                     f"position changed at least partly: {pct('position_changed')}",
                     f"refuses more than the original: {pct('refuses_more')}"],
         "examples": ex_ids("advice_not_kept")},
        {"id": "f_decision", "severity": "major",
         "title": "The rewrite takes the decision away from the person",
         "body": ("DA's style guidance says to leave the decision with the person. The rewritten replies more often tell them what to do, "
                  "and the explicit refusal phrase (\"I will not…\") appears far more often than in the source."),
         "metrics": [f"leaves decision with person: {sh['src_leaves_decision']['pct']:.0f}% → {sh['abl_leaves_decision']['pct']:.0f}% ({n} judged)",
                     f"\"I will not / won't / can't…\" in reply: {C['pct_explicit_refusal_phrase']['src']:.0f}% → {C['pct_explicit_refusal_phrase']['abl']:.0f}% (all {C['n_pairs']})",
                     f"commands the person: {pct('commands')}"],
         "examples": ex_ids("decision_taken") or ex_ids("commands")},
        {"id": "f_facts", "severity": "minor",
         "title": "Some invented specifics, few misstated facts",
         "body": ("Facts mostly survive. A minority of rows add specifics that appear in neither the person's message nor the original "
                  "(an estimated percentage, a named role). The person's own facts are rarely contradicted."),
         "metrics": [f"invented specifics: {pct('invented_facts')}", f"misstated user facts: {pct('misstated_facts')}"],
         "examples": ex_ids("invented_facts")},
        {"id": "f_intended", "severity": "minor",
         "title": "The intended change did happen",
         "body": ("For the record, the ablation does what it says on the reasoning: the weighing vocabulary and first-person deliberation are "
                  "nearly gone, and every row has exactly three paragraphs. The arm is not broken. It just changes several other things at the same time."),
         "metrics": [f"weighing vocabulary in reasoning: {C['pct_reasoning_with_weigh_vocab']['src']:.0f}% → {C['pct_reasoning_with_weigh_vocab']['abl']:.1f}%",
                     f"three-paragraph reasoning: {C['pct_reasoning_exactly_3_paras']:.0f}%",
                     f"judge finds weighing left: {pct('weighing_left')}"],
         "examples": []},
    ]

    ov = S["overall"]
    data = {
        "lede": (f"The rumour is right. Removing one side of the deliberation is only one of several things this rewrite changed. Of {n} "
                 f"randomly sampled rows, a blind judge rated {ov.get('damaged', 0)} damaged, {ov.get('minor', 0)} with minor issues and "
                 f"{ov.get('clean', 0)} clean. Mechanical checks over all {C['n_pairs']} trained rows show the same changes."),
        "provenance": [
            "trained rows: dougalldeepmind/2026-10-15-ablation-stated-harm-15-mix @ 29756e59 (1,212 synthetic rows)",
            "rewrite: 2026-10-04-ablation-stated-harm-synth @ 8edeb67f, gemini-3-flash-preview",
            "source: DA corpus as loaded by that run (2026-10-02-da-synth @ 305914d5)",
            f"judge: claude-sonnet-5, temperature 0, {n} rows drawn with seed 0, ${S['cost_usd']:.2f}",
            "brackets are 95% Wilson intervals",
            "the judge was asked to list every unintended change, so it rarely says clean; treat \"damaged\" as the headline",
        ],
        "strip": [
            {"value": f"{sh['damaged']['pct']:.0f}%", "label": f"judged damaged [{sh['damaged']['ci'][0]:.0f}–{sh['damaged']['ci'][1]:.0f}]", "sev": "crit"},
            {"value": f"{C['pct_response_opens_you']['abl']:.0f}%", "label": "replies that open by restating the person's situation (DA: "
                      f"{C['pct_response_opens_you']['src']:.0f}%)", "sev": "crit"},
            {"value": f"{C['median_reasoning_ratio']:.2f}×", "label": "reasoning length vs DA source", "sev": "warn"},
            {"value": f"{C['median_response_ratio']:.2f}×", "label": "reply length vs DA source", "sev": "warn"},
            {"value": f"{sh['abl_reasons']['pct']:.0f}%", "label": f"rewritten replies that say why (DA: {sh['src_reasons']['pct']:.0f}%)", "sev": "warn"},
            {"value": f"{C['pct_reasoning_with_weigh_vocab']['abl']:.1f}%", "label": f"reasoning with weighing words (DA: {C['pct_reasoning_with_weigh_vocab']['src']:.0f}%)", "sev": "ok"},
        ],
        "summary": {"n": n},
        "findings": findings,
        "examples": examples,
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB), {len(examples)} examples: "
          + ", ".join(f"{k} {sum(e['judge']['overall'] == k for e in examples)}" for k in ("damaged", "minor", "clean")))
    for f in findings:
        print(f"- [{f['severity']}] {f['title']}: {' | '.join(f['metrics'])}  ex={f['examples']}")


if __name__ == "__main__":
    main()
