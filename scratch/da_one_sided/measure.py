# ABOUTME: Measures a run of configs/data/synth/da-one-sided.yaml against DA's rows for the same scenarios: what share
# ABOUTME: of words still treats the other path as a live option. Run: uv run python scratch/da_one_sided/measure.py --run_dir <synth run dir>
"""Nothing here edits a row. The ruler (ruler.py, validated 2026-11-12) labels every sentence of both corpora, and
this reports, per field:

    live    appeal of the other path + weighing between the two (ruler labels PULL, WEIGH): what the arm removes
    wrong   why the other path is wrong (COST): allowed in this arm, reported so the two stay apart

plus the things earlier attempts drifted on without a checker noticing: length, paragraphing, repeated closers,
reasoning that reads as a draft of the reply. Writes rows.jsonl, summary.json and the review page next to the run.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edit  # noqa: E402
import ruler  # noqa: E402

HERE = Path(__file__).resolve().parent
LIVE = {"PULL", "WEIGH"}
CALL = re.compile(r"\b(decision|call|choice)\b[^.]{0,40}\b(yours|your own|your call)\b|\byour call\b|\byours to (make|set|call)\b|\bup to you\b", re.I)
YOU = re.compile(r"\b(you|your|yours)\b", re.I)
# Rough keyword counts for what a side-by-side read of the first smoke found the ruler does not see (2026-11-20):
# the writer declaring the question closed, and admissions of not knowing disappearing.
CLOSED = re.compile(r"isn't close|not a close call|no genuine case|no (middle|softer) (path|ground|version)|isn't a middle|only points one way|"
                    r"there isn't a (version|middle)|only defensible|line I will not cross|isn't a case of two|aren't symmetric|"
                    r"the only (version|position|way)|already answers this|that's the answer", re.I)
UNSURE = re.compile(r"I don't (actually )?know|I can't tell|not certain|I'm not sure|might be right|may well be (right|correct)|"
                    r"I won't pretend to know|genuinely, I can't", re.I)
TENSION = re.compile(r"\btension\b|both (things|of those|instincts)|on (one|the other|each) side|weigh(ing|ed)? (what|it|this)|"
                     r"neither (pure|option|one)|trade-?off", re.I)


def share(rec: dict, pref: str, labels: set) -> float:
    ks = [k for k in rec["labels"] if k.startswith(pref)]
    return sum(rec["words"][k] for k in ks if rec["labels"][k] in labels) / (sum(rec["words"][k] for k in ks) or 1)


def segs(text: str, rec: dict, pref: str) -> list[dict]:
    out = []
    for i, (core, gap) in enumerate(edit.gaps(text)):
        lab = rec["labels"].get(f"{pref}{i + 1}")
        kind = {"PULL": "leak", "WEIGH": "weigh", "COST": "cost"}.get(lab)
        out.append({"op": "keep", "text": core, "gap": gap, **({"kind": kind} if kind else {})})
    return out


def style(reasoning: str, response: str) -> dict:
    toks = lambda t: re.findall(r"[a-z0-9']+", t.lower())
    grams = lambda t: {tuple(t[i:i + 4]) for i in range(len(t) - 3)}
    paras = lambda t: len([p for p in re.split(r"\n\s*\n", t) if p.strip()])
    a, b = grams(toks(reasoning)), grams(toks(response))
    return {"reasoning_words": ruler.words(reasoning), "reply_words": ruler.words(response),
            "reasoning_paras": paras(reasoning), "reply_paras": paras(response),
            "yours_line": bool(CALL.search(response)), "reasoning_says_you": len(YOU.findall(re.sub(r'"[^"]*"|“[^”]*”', "", reasoning))) >= 3,
            "reply_from_reasoning": len(a & b) / max(1, len(b)), "opener": " ".join(reasoning.split()[:4]).lower(),
            "closed": len(CLOSED.findall(reasoning + "\n" + response)), "unsure": len(UNSURE.findall(reasoning + "\n" + response)),
            "tension": len(TENSION.findall(reasoning + "\n" + response))}


def agg(S: list[dict], recs: list[dict]) -> dict:
    med = lambda k: st.median(s[k] for s in S)
    lab = {f: Counter() for f in ("R", "A")}
    for r in recs:
        for k, l in r["labels"].items():
            lab[k[0]][l] += r["words"][k]
    pct = lambda c: {k: round(100 * v / max(1, sum(c.values())), 1) for k, v in c.most_common()}
    return {"n": len(S),
            "live_pct": {"reasoning": round(100 * st.mean(share(r, "R", LIVE) for r in recs), 1), "reply": round(100 * st.mean(share(r, "A", LIVE) for r in recs), 1)},
            "wrong_pct": {"reasoning": round(100 * st.mean(share(r, "R", {"COST"}) for r in recs), 1), "reply": round(100 * st.mean(share(r, "A", {"COST"}) for r in recs), 1)},
            "rows_with_no_live_sentence": sum(not any(l in LIVE for l in r["labels"].values()) for r in recs),
            "rows_live_over_5pct": sum(share(r, "", LIVE) > 0.05 for r in recs),
            "rows_live_over_10pct": sum(share(r, "", LIVE) > 0.10 for r in recs),
            "label_words_pct": {"reasoning": pct(lab["R"]), "reply": pct(lab["A"])},
            "median_words": {"reasoning": med("reasoning_words"), "reply": med("reply_words")},
            "median_paragraphs": {"reasoning": med("reasoning_paras"), "reply": med("reply_paras")},
            "replies_with_decision_is_yours": sum(s["yours_line"] for s in S),
            "keyword_rows_and_hits": {k: [sum(s[k] > 0 for s in S), sum(s[k] for s in S)] for k in ("closed", "unsure", "tension")},
            "reasonings_addressing_you": sum(s["reasoning_says_you"] for s in S),
            "reply_4grams_in_reasoning_median_pct": round(100 * med("reply_from_reasoning"), 1),
            "distinct_openers": len({s["opener"] for s in S})}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--source", default="data/da_1030_one_sided_smoke")
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    run = Path(args.run_dir)
    snap = sorted(run.glob("stage_*_revise_responses.jsonl"))[-1]
    new = {j["scenario_id"]: j for j in map(json.loads, snap.open(encoding="utf-8"))}
    src = [r for r in edit.load_src(args.source, None)]
    lost = [r["scenario_id"] for r in src if r["scenario_id"] not in new]
    src = [r for r in src if r["scenario_id"] in new]
    base = lambda r, R, A: {"scenario_id": r["scenario_id"], "user": r["user"], "shortcut": r["shortcut"], "reasoning": R, "response": A}
    da_rows = [base(r, r["reasoning"], r["response"]) for r in src]
    os_rows = [base(r, new[r["scenario_id"]]["reasoning"], new[r["scenario_id"]]["response"]) for r in src]
    da_lab = ruler.label_rows(da_rows, f"minimal_{args.tag}_da", ruler.LABELLER, workers=args.workers, text_key=True)
    os_lab = ruler.label_rows(os_rows, f"minimal_{args.tag}_new", ruler.LABELLER, workers=args.workers, text_key=True)
    da_s, os_s = [style(r["reasoning"], r["response"]) for r in da_rows], [style(r["reasoning"], r["response"]) for r in os_rows]
    rows = []
    for r, d, o, dl, ol, ds, os_ in zip(src, da_rows, os_rows, da_lab, os_lab, da_s, os_s):
        text = dict(ruler.numbered(o["reasoning"], o["response"])[0])
        rows.append({**{k: r.get(k) for k in ("scenario_id", "trait_id", "trait_name", "ai", "sector", "shortcut", "situation", "system", "user")},
                     "src_reasoning": d["reasoning"], "src_response": d["response"], "abl_reasoning": o["reasoning"], "abl_response": o["response"],
                     "left": {"reasoning": segs(d["reasoning"], dl, "R"), "reply": segs(d["response"], dl, "A")},
                     "right": {"reasoning": segs(o["reasoning"], ol, "R"), "reply": segs(o["response"], ol, "A")},
                     "live": [{"id": k, "label": l, "text": text[k].strip()} for k, l in ol["labels"].items() if l in LIVE],
                     "m": {"live_da": {"reasoning": share(dl, "R", LIVE), "reply": share(dl, "A", LIVE), "both": share(dl, "", LIVE)},
                           "live_new": {"reasoning": share(ol, "R", LIVE), "reply": share(ol, "A", LIVE), "both": share(ol, "", LIVE)},
                           "wrong_da": {"reasoning": share(dl, "R", {"COST"}), "reply": share(dl, "A", {"COST"})},
                           "wrong_new": {"reasoning": share(ol, "R", {"COST"}), "reply": share(ol, "A", {"COST"})},
                           "reasoning_ratio": os_["reasoning_words"] / max(1, ds["reasoning_words"]),
                           "response_ratio": os_["reply_words"] / max(1, ds["reply_words"]), "style_da": ds, "style_new": os_}})
    out = Path(f"output/{date.today().isoformat()}_da_one_sided_minimal_{args.tag}")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    S = {"run_dir": str(run), "snapshot": snap.name, "rows_generated": len(rows), "rows_lost_in_generation": lost,
         "da": agg(da_s, da_lab), "one_sided": agg(os_s, os_lab),
         "median_length_ratio": {"reasoning": round(st.median(r["m"]["reasoning_ratio"] for r in rows), 2), "reply": round(st.median(r["m"]["response_ratio"] for r in rows), 2)},
         "ruler_cost": round(sum(j["cost"] for j in da_lab + os_lab), 3), "ruler": ruler.LABELLER}
    (out / "summary.json").write_text(json.dumps(S, indent=1), encoding="utf-8")
    print(json.dumps(S, indent=1))
    page(out, rows, S, args.tag)


def page(out: Path, rows: list[dict], S: dict, tag: str) -> None:
    pc = lambda x: f"{100 * x:.0f}%" if x >= 0.1 or x == 0 else f"{100 * x:.1f}%"
    D, N = S["da"], S["one_sided"]
    # The side-by-side read of 2026-11-20 found the ruler's WEIGH label mostly marks comparisons used as an argument
    # ("recusal, not bypass"), which this arm allows. The page therefore headlines PULL (the other path's appeal) and
    # the keyword counts, and shows WEIGH sentences underlined for the reader to judge, not as a failure count.
    lw = lambda side, f, lab: side["label_words_pct"][f].get(lab, 0.0)
    kw = lambda side, k: f"{side['keyword_rows_and_hits'][k][0]} rows, {side['keyword_rows_and_hits'][k][1]} times"

    def panel(r: dict) -> dict:
        m, sn, sd = r["m"], r["m"]["style_new"], r["m"]["style_da"]
        pulls = [x for x in r["live"] if x["label"] == "PULL"]
        return {"title": "What was measured",
                "rows": [["sentences on the other path's appeal", f"{len(pulls)} here"],
                         ["sentences that compare or weigh", f"{sum(x['label'] == 'WEIGH' for x in r['live'])} here (includes comparisons used as an argument, which are allowed)"],
                         ["says why the other path is wrong (allowed)", f"reasoning {pc(m['wrong_da']['reasoning'])} of words in DA → {pc(m['wrong_new']['reasoning'])} here; reply {pc(m['wrong_da']['reply'])} → {pc(m['wrong_new']['reply'])}"],
                         ["admits not knowing something", f"{sd['unsure']} times in DA, {sn['unsure']} here"],
                         ["declares the question closed", f"{sd['closed']} times in DA, {sn['closed']} here"],
                         ["length", f"reasoning {sn['reasoning_words']} words ({m['reasoning_ratio']:.2f}× DA's {sd['reasoning_words']}), reply {sn['reply_words']} words ({m['response_ratio']:.2f}× DA's {sd['reply_words']})"]],
                "lists": [["Sentences on the other path's appeal:", [[x["label"], x["text"]] for x in pulls]]] if pulls else [],
                "note": "" if pulls else "No sentence presents the other path's appeal."}

    cls = lambda r: "minor" if any(x["label"] == "PULL" for x in r["live"]) else "clean"
    ex = [{"id": r["scenario_id"], "title": (r["situation"] or r["shortcut"])[:110], "trait_id": r["trait_id"], "trait_name": r["trait_name"],
           "ai": r["ai"], "shortcut": r["shortcut"], "chosen": "", "system": r["system"], "user": r["user"],
           "left": r["left"], "right": r["right"], "abl_reasoning": r["abl_reasoning"], "abl_response": r["abl_response"],
           "src_words": {"reasoning": r["m"]["style_da"]["reasoning_words"], "reply": r["m"]["style_da"]["reply_words"]},
           "m": r["m"], "cls": cls(r), "panel": panel(r)} for r in sorted(rows, key=lambda r: -sum(x["label"] == "PULL" for x in r["live"]))]
    n = N["n"]
    findings = [
        {"id": "f_live", "severity": "minor" if lw(N, "reasoning", "PULL") <= 3 else "major",
         "title": f"Words on the other path's appeal: reasoning {lw(D, 'reasoning', 'PULL')}% in DA, {lw(N, 'reasoning', 'PULL')}% here; reply {lw(D, 'reply', 'PULL')}% → {lw(N, 'reply', 'PULL')}%",
         "body": "DA gives the other path a paragraph-length hearing in nearly every row; that is the one thing this arm is meant to remove. Nothing was edited or filtered: these are the rows as the recipe wrote them.",
         "metrics": [f"rows with no such sentence: {sum(e['cls'] == 'clean' for e in ex)} of {n}",
                     f"names a tension or weighs (keywords): DA {kw(D, 'tension')}; here {kw(N, 'tension')}"],
         "examples": [e["id"] for e in ex[:4]]},
        {"id": "f_weigh", "severity": "major",
         "title": f"Sentences the ruler calls weighing: reasoning {lw(D, 'reasoning', 'WEIGH')}% of words in DA, {lw(N, 'reasoning', 'WEIGH')}% here; reply {lw(D, 'reply', 'WEIGH')}% → {lw(N, 'reply', 'WEIGH')}%",
         "body": "Underlined on the page. Read in full, most of these are comparisons used as an argument (\"push for recusal, not bypass\"), which this arm allows, so this number overstates what is left. A few are real concessions. Judge them yourself on two or three rows.",
         "metrics": [], "examples": [e["id"] for e in sorted(ex, key=lambda e: -e["m"]["live_new"]["both"])[:4]]},
        {"id": "f_kept", "severity": "minor",
         "title": "What should be unchanged from DA: admitting uncertainty, not declaring the question closed",
         "body": "The first wording of the instruction removed both by accident; this wording says to keep them.",
         "metrics": [f"admits not knowing: DA {kw(D, 'unsure')}; here {kw(N, 'unsure')}", f"declares the question closed: DA {kw(D, 'closed')}; here {kw(N, 'closed')}",
                     f"replies with a \"decision is yours\" line: DA {D['replies_with_decision_is_yours']}; here {N['replies_with_decision_is_yours']}"],
         "examples": []},
        {"id": "f_wrong", "severity": "minor",
         "title": f"Words saying why the other path is wrong: reasoning {D['wrong_pct']['reasoning']}% in DA, {N['wrong_pct']['reasoning']}% here; reply {D['wrong_pct']['reply']}% → {N['wrong_pct']['reply']}%",
         "body": "Allowed in this arm: it is the justification. Reported so it stays separate from the weighing.", "metrics": [], "examples": []},
        {"id": "f_same", "severity": "minor" if 0.8 <= S["median_length_ratio"]["reasoning"] <= 1.25 else "major",
         "title": f"Length against DA: reasoning {S['median_length_ratio']['reasoning']:.2f}×, reply {S['median_length_ratio']['reply']:.2f}×",
         "body": "The rows should otherwise look like DA's. These are the things earlier attempts drifted on.",
         "metrics": [f"median words, reasoning: {N['median_words']['reasoning']:.0f} (DA {D['median_words']['reasoning']:.0f})",
                     f"median words, reply: {N['median_words']['reply']:.0f} (DA {D['median_words']['reply']:.0f})",
                     f"median paragraphs, reasoning / reply: {N['median_paragraphs']['reasoning']:.0f} / {N['median_paragraphs']['reply']:.0f} (DA {D['median_paragraphs']['reasoning']:.0f} / {D['median_paragraphs']['reply']:.0f})",
                     f"replies with a \"decision is yours\" line: {N['replies_with_decision_is_yours']} of {n} (DA {D['replies_with_decision_is_yours']})",
                     f"reply phrasing repeated from the reasoning: {N['reply_4grams_in_reasoning_median_pct']}% (DA {D['reply_4grams_in_reasoning_median_pct']}%)",
                     f"distinct first four words of the reasoning: {N['distinct_openers']} of {n}"],
         "examples": [e["id"] for e in sorted(ex, key=lambda e: e["m"]["reasoning_ratio"])[:3]]},
        {"id": "f_reads", "severity": "major", "title": "Is this one-sided as you mean it?",
         "body": "The assistant reaches a verdict and argues for it. It may say why the other path is wrong; it should not give that path a hearing.",
         "metrics": [], "examples": [e["id"] for e in ex[-4:]]},
    ]
    data = {"lede": f"{n} DA scenarios with the assistant's turn written by DA's own recipe, changed only where it asks for deliberation. Left is DA's row for the same scenario; right is the new row. Nothing was edited or filtered afterwards.",
            "provenance": ["scenarios: dougalldeepmind/2026-10-30-da-synth @ 8fadea62, 20 trait-balanced rows",
                           "recipe: configs/data/synth/da-one-sided.yaml = da.yaml's reply stages with three instructions changed",
                           f"measured by {S['ruler']} sentence labels; no gate, no rewriting"],
            "strip": [{"value": f"{lw(N, 'reasoning', 'PULL')}%", "label": f"of reasoning words are on the other path's appeal (DA {lw(D, 'reasoning', 'PULL')}%)", "sev": "ok" if lw(N, "reasoning", "PULL") <= 3 else "warn"},
                      {"value": f"{lw(N, 'reply', 'PULL')}%", "label": f"of reply words are (DA {lw(D, 'reply', 'PULL')}%)", "sev": "ok" if lw(N, "reply", "PULL") <= 3 else "warn"},
                      {"value": f"{N['keyword_rows_and_hits']['unsure'][0]}/{n}", "label": f"rows admit not knowing something (DA {D['keyword_rows_and_hits']['unsure'][0]})", "sev": "ok"},
                      {"value": f"{S['median_length_ratio']['reasoning']:.2f}×", "label": "reasoning length vs DA", "sev": "ok" if S["median_length_ratio"]["reasoning"] >= 0.8 else "warn"},
                      {"value": f"{S['median_length_ratio']['reply']:.2f}×", "label": "reply length vs DA", "sev": "ok" if S["median_length_ratio"]["reply"] >= 0.8 else "warn"},
                      {"value": f"{N['wrong_pct']['reasoning']}%", "label": f"of reasoning words say why the other path is wrong (DA {D['wrong_pct']['reasoning']}%)", "sev": "ok"}],
            "overall_q": {"title": "Is this the one-sided arm you want to train?", "body": "If not, say what to change."},
            "findings": findings, "examples": ex, "left_label": "DA", "chosen_label": "",
            "legend": "Red: the other path's appeal. Underlined: compares or weighs the two, which includes comparisons used as an argument. Amber: says why the other path is wrong, which this arm allows. Plain: everything else.",
            "cls_labels": {"clean": "no appeal sentence", "minor": "an appeal sentence", "damaged": "flagged"},
            "flag_tags": [["weighs", "Still weighs or shows the other path's appeal"], ["wrong", "Recommends the wrong thing"], ["unlike", "Unlike DA in some other way"],
                          ["rough", "Reads roughly"], ["thin", "Too little reasoning"], ["fine", "Looks fine"], ["other", "Other"]]}
    html = (HERE / "template_diff.html").read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    p = out / f"{date.today().isoformat()}_da_one_sided_minimal_{tag}.html"
    p.write_text(html, encoding="utf-8")
    print(f"page: {p} ({p.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
