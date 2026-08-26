# ABOUTME: THE TRACE — paragraph/sentence shape, the dialectic markers (steelman, "But...", "So...",
# ABOUTME: self-test, uncertainty), stakeholder/principle naming, and trace-vs-reply sentence length.
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.getcwd())
from scratch.gpt_analysis.common import (  # noqa: E402
    ORDER,
    boot_ci,
    dist,
    load,
    paragraphs,
    pct,
    prop_ci,
    reply,
    sentences,
    trace,
    words,
)

C, ids = load()
N = len(ids)

# ---------------- markers ----------------
# paragraph-initial adversative turn: the "But..." pivot
BUT_INIT = re.compile(
    r"^(But|However|Yet|Still|On the other hand|That said|And yet|Except)\b", re.I
)
# closing synthesis
SO_INIT = re.compile(
    r"^(So|Therefore|Thus|In short|The upshot|Bottom line|All told|Net|Overall|Ultimately)\b",
    re.I,
)
# steelman / charitable reading of the requester
STEELMAN = re.compile(
    r"\b(steelman|charitable(ly)?|to be fair|the (strongest|best) (case|argument|version)|"
    r"genuin(e|ely) (hard|difficult|sympathetic)|real (pressure|bind|stakes)|"
    r"(they|he|she|the user) (is|are|'s) (not|probably not) (being )?(malicious|acting in bad faith)|"
    r"sympathetic|understandable(ly)?|legitimate (need|reason|interest|concern|goal))\b",
    re.I,
)
# hypothetical self-test: "if I ...", "suppose I", "what if I", "imagine I"
SELFTEST = re.compile(
    r"\b(if I (were to |just |simply |did |do |wrote|gave|helped|complied|said|went along)"
    r"|suppose I\b|what if I\b|imagine I\b|were I to\b|if I'd\b|had I\b"
    r"|the counterfactual|test(ing)? (this|myself)|would I (be|do|say|want))\b",
    re.I,
)
UNCERTAIN = re.compile(
    r"\b(I'm not (sure|certain)|I am not (sure|certain)|not entirely (sure|clear)|"
    r"I could be wrong|hard to (say|know|tell)|I don't know (whether|if)|"
    r"uncertain|unsure|it's unclear|I might be (wrong|missing)|genuinely (unsure|torn)|"
    r"I can't (tell|be sure)|no way (for me )?to (know|verify))\b",
    re.I,
)
# first-person deliberation
FIRST_PERSON = re.compile(r"\b(I|I'm|I've|I'll|I'd|me|my)\b")
# does the trace NAME the affected third parties?
STAKEHOLDER = re.compile(
    r"\b(third part(y|ies)|stakeholder|patients?|customers?|users?|employees?|workers?|"
    r"students?|tenants?|investors?|shareholders?|the public|clients?|passengers?|"
    r"residents?|families|children|regulators?|auditors?|colleagues?)\b",
    re.I,
)
# does the trace NAME the governing principle / values language?
PRINCIPLE = re.compile(
    r"\b(principle|value(s)?|my (values|guidelines|character|constitution)|honest(y)?|"
    r"integrity|oversight|autonomy|harm|deceiv(e|ing)|deception|transparen(t|cy)|"
    r"consent|accountab(le|ility)|legitimate|obligation|duty|what kind of assistant)\b",
    re.I,
)
# named the constitution/spec-ish self-reference
SELF_REF = re.compile(
    r"\b(my (values|principles|guidelines|character)|who I want to be|the kind of (assistant|help))\b",
    re.I,
)
# enumerated planning inside the trace
TRACE_LIST = re.compile(r"^\s{0,4}([-*+]\s+|\d+[.)]\s+)", re.M)
TRACE_HEAD = re.compile(r"^\s{0,3}(#{1,6}\s+\S|\*\*[^*\n]{2,60}\*\*\s*:?\s*$)", re.M)

# structure-planning: the trace decides the SHAPE of the reply
PLAN_SHAPE = re.compile(
    r"\b(structure|format|sections?|bullet(s|ed)?|headings?|outline|"
    r"I'll (open|start|begin|close|end)|lay (this|it) out|organi[sz]e)\b",
    re.I,
)


def para_first_sents(t):
    out = []
    for p in paragraphs(t):
        s = sentences(p)
        if s:
            out.append(s[0])
    return out


rows = {}
for k in ORDER:
    tr = [trace(C[k][i]) for i in ids]
    rp = [reply(C[k][i]) for i in ids]
    paras = [len(paragraphs(t)) for t in tr]
    sents = [len(sentences(t)) for t in tr]
    tsl = [st.mean([len(words(s)) for s in sentences(t)]) for t in tr]
    rsl = [st.mean([len(words(s)) for s in sentences(t)] or [0]) for t in rp]
    ratio = [a / b for a, b in zip(tsl, rsl) if b]

    def has(rx, texts=tr):
        return sum(1 for t in texts if rx.search(t))

    but = sum(1 for t in tr if any(BUT_INIT.match(s) for s in para_first_sents(t)))
    but_any = sum(1 for t in tr if any(BUT_INIT.match(s) for s in sentences(t)))
    so = sum(1 for t in tr if any(SO_INIT.match(s) for s in para_first_sents(t)[-2:]))
    so_any = sum(1 for t in tr if any(SO_INIT.match(s) for s in sentences(t)))
    single = sum(1 for p in paras if p == 1)

    rows[k] = dict(
        n=N,
        trace_paras_median=st.median(paras),
        trace_paras_mean=st.mean(paras),
        trace_single_para_pct=pct(single, N),
        trace_single_para_ci=prop_ci(single, N),
        trace_sents_median=st.median(sents),
        trace_sent_words_mean=st.mean(tsl),
        reply_sent_words_mean=st.mean(rsl),
        trace_reply_sentlen_ratio=st.median(ratio),
        but_para_initial_pct=pct(but, N),
        but_para_initial_ci=prop_ci(but, N),
        but_anywhere_pct=pct(but_any, N),
        so_closing_pct=pct(so, N),
        so_closing_ci=prop_ci(so, N),
        so_anywhere_pct=pct(so_any, N),
        steelman_pct=pct(has(STEELMAN), N),
        steelman_ci=prop_ci(has(STEELMAN), N),
        selftest_pct=pct(has(SELFTEST), N),
        selftest_ci=prop_ci(has(SELFTEST), N),
        uncertainty_pct=pct(has(UNCERTAIN), N),
        uncertainty_ci=prop_ci(has(UNCERTAIN), N),
        stakeholder_pct=pct(has(STAKEHOLDER), N),
        principle_pct=pct(has(PRINCIPLE), N),
        selfref_pct=pct(has(SELF_REF), N),
        trace_has_list_pct=pct(has(TRACE_LIST), N),
        trace_has_heading_pct=pct(has(TRACE_HEAD), N),
        plan_shape_pct=pct(has(PLAN_SHAPE), N),
        first_person_per1k=1000
        * sum(len(FIRST_PERSON.findall(t)) for t in tr)
        / sum(len(words(t)) for t in tr),
        trace_reply_char_ratio=st.median(
            [len(a) / max(len(b), 1) for a, b in zip(tr, rp)]
        ),
    )
    rows[k]["trace_reply_char_ratio_ci"] = boot_ci(
        [len(a) / max(len(b), 1) for a, b in zip(tr, rp)]
    )
    rows[k]["trace_paras_ci"] = boot_ci(paras)

print("=== TRACE SHAPE (n=%d paired scenarios) ===" % N)
keys = [
    "trace_paras_median",
    "trace_paras_mean",
    "trace_single_para_pct",
    "trace_sents_median",
    "trace_sent_words_mean",
    "reply_sent_words_mean",
    "trace_reply_sentlen_ratio",
    "trace_reply_char_ratio",
    "but_para_initial_pct",
    "but_anywhere_pct",
    "so_closing_pct",
    "so_anywhere_pct",
    "steelman_pct",
    "selftest_pct",
    "uncertainty_pct",
    "stakeholder_pct",
    "principle_pct",
    "selfref_pct",
    "trace_has_list_pct",
    "trace_has_heading_pct",
    "plan_shape_pct",
    "first_person_per1k",
]
print(f"{'metric':<30}" + "".join(f"{k:>12}" for k in ORDER))
for m in keys:
    print(f"{m:<30}" + "".join(f"{rows[k][m]:>12.2f}" for k in ORDER))

print("\n=== 95% CIs on the discriminating proportions ===")
for m in [
    "trace_single_para_pct",
    "but_para_initial_pct",
    "so_closing_pct",
    "steelman_pct",
    "selftest_pct",
    "uncertainty_pct",
]:
    ci = m.replace("_pct", "_ci")
    print(
        f"  {m:<28}"
        + "".join(
            f"  {k}: {rows[k][m]:5.1f} [{rows[k][ci][0]:.1f},{rows[k][ci][1]:.1f}]"
            for k in ORDER
        )
    )
print(
    "  trace_reply_char_ratio      "
    + "".join(
        f"  {k}: {rows[k]['trace_reply_char_ratio']:.2f} [{rows[k]['trace_reply_char_ratio_ci'][0]:.2f},{rows[k]['trace_reply_char_ratio_ci'][1]:.2f}]"
        for k in ORDER
    )
)

print("\n=== TRACE PARAGRAPH COUNT DISTRIBUTION ===")
for k in ORDER:
    p = [len(paragraphs(trace(C[k][i]))) for i in ids]
    d = dist(p)
    hist = {}
    for x in p:
        hist[min(x, 8)] = hist.get(min(x, 8), 0) + 1
    print(
        f"  {k:<8} median {d['median']:.0f}  "
        + " ".join(f"{n}p:{pct(hist.get(n, 0), N):.0f}%" for n in range(1, 9))
    )

json.dump(
    rows, open("scratch/gpt_analysis/out/p1_trace.json", "w"), indent=1, default=str
)
print("\nwrote out/p1_trace.json")
