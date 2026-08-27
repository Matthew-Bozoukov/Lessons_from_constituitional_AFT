# ABOUTME: Decomposes each reply's characters into list-items / headings / prose, and
# ABOUTME: measures echo-of-the-user, hedging and caveat density — what buys GPT's length.
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.getcwd())
from scratch.three_way.norm import ORDER, load, reply  # noqa: E402

C, IDS = load()
N = len(IDS)

BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
HEAD = re.compile(r"^\s*(?:#{1,6}\s+|\*\*[^*]{2,80}\*\*:?\s*$)")


def segment(t):
    """chars in list items / headings / prose paragraphs."""
    li = hd = pr = 0
    for line in t.split("\n"):
        if not line.strip():
            continue
        if HEAD.match(line):
            hd += len(line)
        elif BULLET.match(line):
            li += len(line)
        else:
            pr += len(line)
    return li, hd, pr


print(f"CHARACTER BUDGET BY SEGMENT TYPE (n={N}, mean chars per reply)")
print(
    f"{'corpus':<8}{'total':>9}{'list items':>12}{'headings':>10}{'prose':>9}"
    f"{'  |  list %':>12}{'head %':>8}{'prose %':>9}"
)
budget = {}
for c in ORDER:
    L = H = P = T = 0
    for k in IDS:
        t = reply(C[c][k])
        li, hd, pr = segment(t)
        L += li
        H += hd
        P += pr
        T += len(t)
    budget[c] = (T / N, L / N, H / N, P / N)
    print(
        f"{c:<8}{T / N:>9.0f}{L / N:>12.0f}{H / N:>10.0f}{P / N:>9.0f}"
        f"{100 * L / T:>12.1f}{100 * H / T:>8.1f}{100 * P / T:>9.1f}"
    )

print("\n--- WHERE GPT'S EXTRA CHARACTERS GO (mean per reply, vs each baseline) ---")
for base in ("sonnet", "grok"):
    tb, lb, hb, pb = budget[base]
    tg, lg, hg, pg = budget["gpt"]
    extra = tg - tb
    print(
        f"  vs {base}: +{extra:.0f} chars = "
        f"list +{lg - lb:.0f} ({100 * (lg - lb) / extra:.0f}%), "
        f"headings +{hg - hb:.0f} ({100 * (hg - hb) / extra:.0f}%), "
        f"prose +{pg - pb:.0f} ({100 * (pg - pb) / extra:.0f}%)"
    )

print("\n--- LIST STRUCTURE ---")
print(
    f"{'corpus':<8}{'median # items':>16}{'median item chars':>19}{'% replies w/ list':>19}"
    f"{'median # headings':>19}"
)
for c in ORDER:
    counts, sizes, heads = [], [], []
    withlist = 0
    for k in IDS:
        t = reply(C[c][k])
        items = [ln for ln in t.split("\n") if BULLET.match(ln)]
        hs = [ln for ln in t.split("\n") if HEAD.match(ln)]
        counts.append(len(items))
        heads.append(len(hs))
        if items:
            withlist += 1
            sizes.extend(len(x) for x in items)
    print(
        f"{c:<8}{st.median(counts):>16.0f}{st.median(sizes) if sizes else 0:>19.0f}"
        f"{100 * withlist / N:>18.1f}%{st.median(heads):>19.0f}"
    )

# ---- ECHO OF THE USER: how much of the reply restates the user's own situation ----
STOP = set(
    """a an the and or but if of to in on for with at by from as is are was were be been
being it its this that these those i you he she they we not no do does did doing have has had
will would can could should may might must shall about into over under than then so such our your
their his her them us me my mine there here what which who whom whose when where why how all any
both each few more most other some only own same too very s t just don now""".split()
)


def content_words(t):
    return {w for w in re.findall(r"[a-z]{4,}", t.lower()) if w not in STOP}


print("\n--- ECHO OF THE USER'S MESSAGE (restating the situation) ---")
print(
    f"{'corpus':<8}{'% of reply content-words also in user msg':>43}"
    f"{'echoed chars/reply':>21}"
)
for c in ORDER:
    shares, absol = [], []
    for k in IDS:
        u = content_words(C[c][k]["messages"][1]["content"])
        t = reply(C[c][k])
        rw = re.findall(r"[a-z]{4,}", t.lower())
        rw = [w for w in rw if w not in STOP]
        if not rw:
            continue
        hit = sum(1 for w in rw if w in u)
        shares.append(hit / len(rw))
        absol.append(hit * 6.5)  # ~mean chars/word incl. space, for an absolute scale
    print(f"{c:<8}{100 * st.mean(shares):>42.1f}%{st.mean(absol):>21.0f}")

# ---- hedges, caveats, conditionals ----
LEX = {
    "hedge (may/might/could/possibly/likely)": r"(?i)\b(?:may|might|could|possibly|potentially|likely|perhaps|arguably)\b",
    "caveat (if/unless/provided/assuming/depends)": r"(?i)\b(?:unless|provided that|assuming|depends on|caveat|to the extent)\b",
    "conditional 'if'": r"(?i)\bif\b",
    "risk nouns (risk/harm/exposure/liability)": r"(?i)\b(?:risk|risks|harm|harms|exposure|liability|danger)\b",
    "process nouns (policy/process/procedure/approval)": r"(?i)\b(?:policy|process|procedure|approval|authoriz|authoris|sign-?off|escalat|governance|oversight|audit)\w*\b",
    "named roles (counsel/compliance/board/officer...)": r"(?i)\b(?:legal counsel|compliance|the board|ethics|regulator|auditor|supervisor|ombuds|IRB|HR\b|union|inspector|commission)\w*\b",
    "second person 'you'": r"(?i)\byou\b",
    "first person 'I'": r"\bI\b",
}
print("\n--- LEXICAL DENSITY per 1,000 chars (rate, so length-neutral) ---")
print(f"{'lexicon':<52}{'SON':>8}{'GROK':>8}{'GPT':>8}")
for name, p in LEX.items():
    r = re.compile(p)
    out = []
    for c in ORDER:
        n = sum(len(r.findall(reply(C[c][k]))) for k in IDS)
        ch = sum(len(reply(C[c][k])) for k in IDS)
        out.append(1000 * n / ch)
    print(f"{name:<52}" + "".join(f"{x:>8.2f}" for x in out))

print("\n--- LENGTH DISPERSION (the spread, not the middle) ---")
print(
    f"{'corpus':<8}{'p10':>8}{'median':>9}{'p90':>8}{'p90/p10':>10}{'IQR/median':>13}"
)
for c in ORDER:
    v = sorted(len(reply(C[c][k])) for k in IDS)
    p10, p25, med, p75, p90 = (
        v[len(v) // 10],
        v[len(v) // 4],
        st.median(v),
        v[3 * len(v) // 4],
        v[9 * len(v) // 10],
    )
    print(
        f"{c:<8}{p10:>8.0f}{med:>9.0f}{p90:>8.0f}{p90 / p10:>10.2f}{(p75 - p25) / med:>13.2f}"
    )
