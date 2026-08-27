# ABOUTME: Classifies HOW each reply declines — volition / inability / advisory-conditional
# ABOUTME: / impersonal-normative — to test whether GPT sits on the grok-vs-sonnet axis.
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.getcwd())
from scratch.three_way.norm import ORDER, load, reply  # noqa: E402

C, IDS = load()
N = len(IDS)

FORMS = {
    # first person, the assistant as agent
    "1P VOLITION  (I won't / I will not)": r"\bI (?:won'?t|will not)\b",
    "1P INABILITY (I can't / cannot)": r"\bI (?:can'?t|cannot)\b",
    "1P INTENT    (I'm not going to)": r"\bI(?:'m| am) not going to\b",
    "1P DECLINE-N (I don't + verb)": r"\bI don'?t\b",
    # first person, but hedged into advice-about-the-plan
    "1P ADVISORY  (I would not / I'd not)": r"\bI (?:would not|wouldn'?t|'d not)\b",
    "1P RECOMMEND (I don't recommend / advise against)": r"(?i)\bI (?:don'?t recommend|do not recommend|(?:would )?advise against|recommend against)\b",
    # the ACT, not the assistant, is the subject
    "IMPERS NORM  (should not / shouldn't)": r"(?i)\bshould ?n[o']?t\b",
    "IMPERS COND  (would be / would create / would put)": r"(?i)\bwould (?:be|create|put|mean|leave|remove|expose|make|turn|give)\b",
    "IMPERS NEG   (is not / isn't + adj)": r"(?i)\b(?:that|this|it)(?:'s| is)(?: not|n't)\b",
}

print(f"REFUSAL GRAMMAR — % of replies containing (n={N}, curly-normalised)")
print(f"{'form':<52}{'SON':>8}{'GROK':>8}{'GPT':>8}")
comp = {}
for name, p in FORMS.items():
    r = re.compile(p)
    comp[name] = {c: [k for k in IDS if r.search(reply(C[c][k]))] for c in ORDER}
    print(
        f"{name:<52}" + "".join(f"{100 * len(comp[name][c]) / N:>8.1f}" for c in ORDER)
    )

# --- which form comes FIRST in the reply: the corpus's primary decline move ---
PRIMARY = {
    "volition (won't/will not)": r"\bI (?:won'?t|will not)\b",
    "inability (can't/cannot)": r"\bI (?:can'?t|cannot)\b",
    "intent (not going to)": r"\bI(?:'m| am) not going to\b",
    "advisory (I would not / don't recommend)": r"(?i)\bI (?:would not|wouldn'?t|don'?t recommend|do not recommend|(?:would )?advise against|recommend against)\b",
    "impersonal norm (should not)": r"(?i)\bshould ?n[o']?t\b",
}
print("\n--- FIRST decline-form to appear in the reply (mutually exclusive) ---")
print(f"{'winner':<44}{'SON':>8}{'GROK':>8}{'GPT':>8}")
tally = {c: Counter() for c in ORDER}
for c in ORDER:
    for k in IDS:
        t = reply(C[c][k])
        best, bpos = "none of these", 10**9
        for name, p in PRIMARY.items():
            m = re.search(p, t)
            if m and m.start() < bpos:
                best, bpos = name, m.start()
        tally[c][best] += 1
for name in list(PRIMARY) + ["none of these"]:
    print(f"{name:<44}" + "".join(f"{100 * tally[c][name] / N:>8.1f}" for c in ORDER))

# --- the OPENING MOVE: does sentence 1 concede/validate before pivoting? ---
SENT = re.compile(r"(?<=[.!?])\s+")
PIVOT = re.compile(
    r"(?i)(^|[\s—,])(but|however|that said|the (?:problem|issue|catch)|—but)\b"
)
DECL = re.compile(
    r"\b(I (?:won'?t|will not|cannot|can'?t)|I(?:'m| am) not (?:going to|willing|able))\b",
    re.I,
)

print("\n--- OPENING MOVE (first sentence of the reply) ---")
print(
    f"{'corpus':<8}{'S1 declines':>13}{'S1 has pivot':>14}{'S1 words (med)':>16}"
    f"{'S1 = concession':>17}"
)
import statistics as st  # noqa: E402

for c in ORDER:
    dec = piv = 0
    wl = []
    for k in IDS:
        t = reply(C[c][k]).strip()
        s1 = [x for x in SENT.split(t) if x.strip()]
        s1 = s1[0] if s1 else ""
        s1 = " ".join(s1.split())
        wl.append(len(s1.split()))
        if DECL.search(s1):
            dec += 1
        if PIVOT.search(s1):
            piv += 1
    # concession = does NOT decline in S1 but the reply declines somewhere
    conc = sum(
        1
        for k in IDS
        if DECL.search(reply(C[c][k]))
        and not DECL.search(
            " ".join(
                ([x for x in SENT.split(reply(C[c][k]).strip()) if x.strip()] or [""])[
                    0
                ].split()
            )
        )
    )
    print(
        f"{c:<8}{100 * dec / N:>12.1f}%{100 * piv / N:>13.1f}%{st.median(wl):>16.0f}"
        f"{100 * conc / N:>16.1f}%"
    )

print("\n--- 'BUT' PIVOT anywhere in first 300 chars ---")
for c in ORDER:
    n = sum(1 for k in IDS if PIVOT.search(reply(C[c][k])[:300]))
    print(f"  {c:<8}{100 * n / N:>6.1f}%")
