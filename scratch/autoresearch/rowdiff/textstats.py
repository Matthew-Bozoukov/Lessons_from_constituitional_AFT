# ABOUTME: Cheap per-row text statistics (lengths, lexical marker rates, ask-form) for every DA row group built by
# ABOUTME: load_groups.py, aggregated per group and per scored mix arm, with Spearman rho against ODCV MR and MASK.
"""    uv run python scratch/autoresearch/rowdiff/textstats.py
       -> output/autoresearch/rowdiff/text_features.jsonl, text_by_group.md, text_by_arm.md, text_arm_corr.json
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import statistics as st
from pathlib import Path

from scipy.stats import spearmanr
from transformers import AutoTokenizer

OUT = Path("output/autoresearch/rowdiff")
# scored 15% arms: mix label -> (ODCV MR mean over runs/seeds, MASK mean)
ARMS = {
    "m0914b": (7.1, 84.9), "m0914": (7.5, 89.4), "m0923": (10.8, 87.7), "m0925": (8.55, 89.25),
    "m0925_not6": (13.8, 76.9), "m0925_newt6": (7.75, 86.3), "m0928": (18.35, 74.4), "m0928_not6": (17.5, 71.0),
    "m0928_sysdiv": (14.6, 72.3), "sw_self": (14.6, 81.5), "sw_other": (10.8, 79.1), "sw_selfother": (7.5, 85.9),
    "sw_explicit": (12.5, 77.3), "sw_advice": (8.8, 81.2), "mself": (14.2, 97.4), "mother": (19.2, 93.8),
    "mexpl": (15.8, 92.0),
}
EXTRA_ARMS = {"m0928_25": (7.5, 92.2)}  # 25% dose, reported but kept out of the correlation

I = re.I
LEX = {
    "metric": r"\b(metrics?|kpis?|targets?|quotas?|benchmarks?|dashboards?|scores?|scoring|numbers|figures|"
              r"performance review|bonus(es)?|ratings?|okrs?)\b",
    "deadline": r"\b(deadlines?|due (date|by)|by (monday|tuesday|wednesday|thursday|friday|tomorrow|tonight|eod)|"
                r"end of (the )?(day|week|quarter|month))\b",
    "records": r"\b(audits?|auditors?|compliance|reports?|reporting|records?|logs?|data|documentation|filings?|"
               r"inspections?|regulators?)\b",
    "falsify": r"\b(falsif\w*|fabricat\w*|fudg\w*|inflat\w*|backdat\w*|doctor(ed|ing)|cook(ed|ing)? the|"
               r"massag\w*|cherry-?pick\w*|misreport\w*|misrepresent\w*|omit\w*|leave (it )?out|"
               r"round(ed|ing)? up|game|gaming|gamed|pad(ded|ding))\b",
    "honesty": r"\b(honest\w*|dishonest\w*|lie|lies|lying|lied|liar|deceiv\w*|decept\w*|mislead\w*|misled|"
               r"truth\w*|untrue|false(ly|hood)?|transparen\w*|candid\w*)\b",
    "oversight": r"\b(oversight|overs(ee|ees|eeing|een)|monitor\w*|review board|irb|supervis\w*|approval|sign-?off|"
                 r"escalat\w*|whistle\w*|disclos\w*)\b",
    "ai_words": r"\b(ai|a\.i\.|models?|algorithms?|chatbots?|agents?|llms?|machine learning|automated|automation|"
                r"neural|gpt|copilot)\b",
}
REFUSE = (r"\b(i won'?t|i will not|i'm not going to|i am not going to|i can'?t help|i cannot help|i'm not able to help|"
          r"i'd decline|i would decline|i'm not willing|i am not willing|i can'?t write|i can'?t draft|i'm going to pass|"
          r"i'd rather not|i'm not comfortable|i won'?t be|i can'?t do that|i'm not going to write|i'm not going to draft)\b")
ASK_ARTIFACT = (r"\b(can|could|would|will) you (please )?(just )?(help me )?(write|draft|word|phrase|frame|put together|"
                r"prepare|create|rewrite|edit|make|generate|compose|polish|fill|format|tweak|adjust|build|set up)\b|"
                r"\bhelp me (write|draft|word|phrase|frame|put together|prepare|create|rewrite|edit|compose|polish)\b|"
                r"\b(write|draft) (me |up )?(a|an|the|something)\b|\bi need you to\b")
ASK_WHAT = (r"\b(what should i|should i|what would you|what do you think|how should i|how do i|am i (wrong|missing|"
            r"overthinking|crazy)|what am i missing|is it (wrong|okay|ok|reasonable)|is this (wrong|okay|ok|reasonable)|"
            r"how would you|what('s| is) the right|where do i|how do i think)\b")
YOU = r"\b(you|your|yours|yourself)\b"
SELF_VALUES = (r"\b(my (own )?values|i care about|i value|for me personally|what i'm for|who i am|my character|"
               r"if i were|as an ai|i'm an ai|my training|my own (outputs?|behaviou?r|defaults?))\b")
HEDGE = r"\b(might|perhaps|maybe|could consider|it depends|on the other hand|reasonable people)\b"
DIRECT = r"\b(don'?t|do not|shouldn'?t|should not|i'd strongly|i would strongly|i'd urge|i'd push back|i'd advise against)\b"
WRONG = r"\b(fraud\w*|dishonest\w*|deceptive|deception|lie|lying|misconduct|wrong|unethical|illegal|cheat\w*)\b"


def words(t):
    return max(1, len(t.split()))


def rate(pat, t):
    return len(re.findall(pat, t, I)) / words(t) * 1000


def has(pat, t):
    return 1.0 if re.search(pat, t, I) else 0.0


def feats(r, ntok):
    f = {}
    for seg in ("system", "user", "reasoning", "answer"):
        t = r[seg]
        f[f"{seg}.chars"] = len(t)
        f[f"{seg}.tokens"] = ntok(t)
        for k, p in LEX.items():
            f[f"{seg}.{k}_any"] = has(p, t)
            f[f"{seg}.{k}_per1k"] = rate(p, t)
    u, a, rc = r["user"], r["answer"], r["reasoning"]
    f["user.ask_artifact"] = has(ASK_ARTIFACT, u)
    f["user.ask_what_to_do"] = has(ASK_WHAT, u)
    f["user.you_per1k"] = rate(YOU, u)
    f["user.questions"] = u.count("?")
    f["answer.refusal_1p"] = has(REFUSE, a)
    f["reasoning.refusal_1p"] = has(REFUSE, rc)
    f["answer.self_values"] = has(SELF_VALUES, a)
    f["reasoning.self_values"] = has(SELF_VALUES, rc)
    f["answer.hedge_per1k"] = rate(HEDGE, a)
    f["answer.directive_per1k"] = rate(DIRECT, a)
    f["answer.names_wrong_any"] = has(WRONG, a)
    f["answer.you_per1k"] = rate(YOU, a)
    f["reasoning.user_word_per1k"] = rate(r"\b(the user|they|them|their)\b", rc)
    f["system.mentions_tools"] = has(r"\b(tools?|bash|execute|function|api access)\b", r["system"])
    return f


def main():
    rows = [json.loads(l) for l in open(OUT / "rows.jsonl", encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    cache = {}

    def ntok(t):
        h = hashlib.md5(t.encode()).hexdigest()
        if h not in cache:
            cache[h] = len(tok(t, add_special_tokens=False)["input_ids"])
        return cache[h]

    fcache, feat_rows = {}, []
    for r in rows:
        h = hashlib.md5((r["system"] + r["user"] + r["reasoning"] + r["answer"]).encode()).hexdigest()
        if h not in fcache:
            fcache[h] = feats(r, ntok)
        feat_rows.append({"group": r["group"], "corpus": r["corpus"], "scenario_id": r["scenario_id"],
                          "trait_id": r["trait_id"], **fcache[h]})
    with open(OUT / "text_features.jsonl", "w") as fh:
        for f in feat_rows:
            fh.write(json.dumps(f) + "\n")
    by = collections.defaultdict(list)
    for f in feat_rows:
        by[f["group"]].append(f)
    keys = [k for k in feat_rows[0] if k not in ("group", "corpus", "scenario_id", "trait_id")]
    mean = {g: {k: st.mean(x[k] for x in v) for k in keys} for g, v in by.items()}
    # per-arm correlation (15% arms only)
    arms = list(ARMS)
    corr = {}
    for k in keys:
        xs = [mean[a][k] for a in arms]
        if len(set(xs)) < 3:
            continue
        ro = spearmanr(xs, [ARMS[a][0] for a in arms]).correlation
        rm = spearmanr(xs, [ARMS[a][1] for a in arms]).correlation
        corr[k] = {"rho_odcv": round(ro, 3), "rho_mask": round(rm, 3)}
    (OUT / "text_arm_corr.json").write_text(json.dumps(corr, indent=1))

    def fmt(v, k):
        if k.endswith((".chars", ".tokens")):
            return f"{v:.0f}"
        if k.endswith(("_any",)) or k in ("user.ask_artifact", "user.ask_what_to_do", "answer.refusal_1p",
                                          "reasoning.refusal_1p", "answer.self_values", "reasoning.self_values",
                                          "answer.names_wrong_any", "system.mentions_tools"):
            return f"{100 * v:.0f}%"
        return f"{v:.2f}"

    order_k = sorted(corr, key=lambda k: -abs(corr[k]["rho_odcv"]))
    allarms = arms + list(EXTRA_ARMS)
    lines = ["| feature | rho ODCV | rho MASK | " + " | ".join(allarms) + " |",
             "|---|---|---|" + "---|" * len(allarms),
             "| ODCV MR | | | " + " | ".join(str({**ARMS, **EXTRA_ARMS}[a][0]) for a in allarms) + " |",
             "| MASK | | | " + " | ".join(str({**ARMS, **EXTRA_ARMS}[a][1]) for a in allarms) + " |",
             "| n DA rows | | | " + " | ".join(str(len(by[a])) for a in allarms) + " |"]
    for k in order_k:
        lines.append(f"| {k} | {corr[k]['rho_odcv']} | {corr[k]['rho_mask']} | "
                     + " | ".join(fmt(mean[a][k], k) for a in allarms) + " |")
    (OUT / "text_by_arm.md").write_text("\n".join(lines) + "\n")
    groups = ["c0914", "m0925", "sw_other_donor", "sw_self_donor", "sw_selfother_donor", "sw_advice_donor",
              "sw_explicit_donor", "m0928", "sw_other_replaced", "sw_selfother_replaced", "sw_advice_replaced",
              "mself", "mother", "mexpl", "c0908", "c0925", "c0928"]
    lines = ["| feature | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups),
             "| n | " + " | ".join(str(len(by[g])) for g in groups) + " |"]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(fmt(mean[g][k], k) for g in groups) + " |")
    (OUT / "text_by_group.md").write_text("\n".join(lines) + "\n")
    print("done", len(feat_rows), "rows;", len(cache), "texts tokenized")


if __name__ == "__main__":
    main()
