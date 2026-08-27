# ABOUTME: final pass: opening mood, closing mood, paragraph geometry, cross-checks
# ABOUTME: run: uv run python scratch/grok_analysis/measure6.py
import sys, re, statistics as st, collections

sys.path.insert(0, "scratch/grok_analysis")
from load import paired, parts

gd, sd, common = paired()
N = len(common)
D = {n: [parts(d[s])[1:] for s in common] for n, d in (("grok", gd), ("sonnet", sd))}
SENT = re.compile(r'[.!?]["”’\')]*\s+|[.!?]["”’\')]*$')


def sents(t):
    t = re.sub(r"\s+", " ", t).strip()
    o, last = [], 0
    for m in SENT.finditer(t):
        o.append(t[last : m.end()].strip())
        last = m.end()
    if last < len(t):
        o.append(t[last:].strip())
    return [x for x in o if x]


def paras(t):
    return [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]


IMP = r"^(Ask|Tell|Go|Send|Put|Say|Give|Start|Call|Write|Do|Don[’']t|Get|Keep|Make|Use|Set|Take|Bring|Show|Name|Pick|Run|Draft|Flag|Add|Check|Offer|Let|Consider|Try|Compress|Report|Attach|Leave|Hold|Cite|Route|Push|Stop|Move|Split|Skip|Read|Note|Pull|Build|Log|Ship|Escalate|Document|Record|Publish|Answer|Explain|Include)\b"


def pct(f):
    return round(100.0 * sum(f) / max(1, len(f)), 1)


print(f"{'metric':54s} {'grok':>8} {'sonnet':>8}")
print("-" * 72)
rows = {}
for n in ("grok", "sonnet"):
    reps = [x[1] for x in D[n]]
    first = [sents(r)[0] if sents(r) else "" for r in reps]
    last = [sents(r)[-1] if sents(r) else "" for r in reps]
    lp = [paras(r)[-1] if paras(r) else "" for r in reps]
    rows.setdefault("opens on an imperative", []).append(
        pct([bool(re.match(IMP, f)) for f in first])
    )
    rows.setdefault("opens on an explicit refusal", []).append(
        pct(
            [
                bool(
                    re.match(
                        r"\s*I (will not|won[’']t|can[’']t|cannot|am not going to|do not)",
                        f,
                        re.I,
                    )
                )
                for f in first
            ]
        )
    )
    rows.setdefault("opens on a fact/scene (The/A/Number/Name)", []).append(
        pct(
            [
                bool(
                    re.match(
                        r"\s*(The|A|An|That|This|Those|These|Two|Three|Four|Five|Six|Eight|Ten|Twelve|Twenty|Forty|\d)",
                        f,
                    )
                )
                for f in first
            ]
        )
    )
    rows.setdefault("first sentence <= 12 words", []).append(
        pct([len(re.findall(r"[\w’']+", f)) <= 12 for f in first])
    )
    rows.setdefault("first sentence >= 30 words", []).append(
        pct([len(re.findall(r"[\w’']+", f)) >= 30 for f in first])
    )
    rows.setdefault("CLOSE: imperative sentence", []).append(
        pct([bool(re.match(IMP, c)) for c in last])
    )
    rows.setdefault("CLOSE: question to the user", []).append(
        pct([c.strip().endswith("?") for c in last])
    )
    rows.setdefault('CLOSE: "Want me to ...?"', []).append(
        pct(
            [
                bool(
                    re.search(
                        r"\b(want (me to|help|to (start|go|work))|should I|shall I)\b",
                        c,
                        re.I,
                    )
                )
                for c in last
            ]
        )
    )
    rows.setdefault("CLOSE: <= 12 words", []).append(
        pct([len(re.findall(r"[\w’']+", c)) <= 12 for c in last])
    )
    rows.setdefault('CLOSE para hands agency back ("your call")', []).append(
        pct(
            [
                bool(
                    re.search(
                        r"\b(your call|up to you|(decision|call|choice) is (still )?yours|you decide|that is (still )?yours)\b",
                        p,
                        re.I,
                    )
                )
                for p in lp
            ]
        )
    )
    # paragraph geometry
    plen = [len(p) for r in reps for p in paras(r)]
    rows.setdefault("median paragraph length (chars)", []).append(
        round(st.median(plen), 1)
    )
    rows.setdefault("paragraphs that are a single sentence (%)", []).append(
        pct([len(sents(p)) == 1 for r in reps for p in paras(r)])
    )
    # bullets
    rows.setdefault("bulleted lines /reply (mean)", []).append(
        round(st.mean([len(re.findall(r"^\s*[-*•]\s+", r, re.M)) for r in reps]), 2)
    )
    rows.setdefault("replies with a verbatim paste-block/template", []).append(
        pct([bool(re.search(r"\[[A-Za-z][^\]\n]{2,40}\]", r)) for r in reps])
    )
for k, v in rows.items():
    print(f"{k:54s} {v[0]:8} {v[1]:8}")

print("\n===== CLOSING SENTENCE — 8 verbatim from each =====")
import random

random.seed(21)
picks = random.sample(common, 8)
for n, d in (("grok", gd), ("sonnet", sd)):
    print(f"\n-- {n} --")
    for sid in picks:
        ss = sents(parts(d[sid])[2])
        print("  ·", (ss[-1] if ss else "")[:190])
