# ABOUTME: fourth pass: rhetorical constructions, metacognition, token budget split
# ABOUTME: run: uv run python scratch/grok_analysis/measure4.py
import sys, re, collections, statistics as st

sys.path.insert(0, "scratch/grok_analysis")
from load import paired, parts

gd, sd, common = paired()
N = len(common)
D = {n: [parts(d[s])[1:] for s in common] for n, d in (("grok", gd), ("sonnet", sd))}


def pct(f):
    return round(100.0 * sum(f) / max(1, len(f)), 1)


def per1k(pat, texts):
    c = sum(len(re.findall(pat, t, re.I)) for t in texts)
    return round(1000.0 * c / max(1, sum(len(t) for t in texts)), 2)


REP = {n: [x[1] for x in D[n]] for n in D}
REA = {n: [x[2] for x in D[n]] for n in D}

print("===== RHETORICAL CONSTRUCTIONS (per 1k chars) =====")
print(
    f"{'construction':56s} {'grok-rep':>9} {'son-rep':>9} {'grok-tr':>9} {'son-tr':>9}"
)
CONS = {
    'antithesis "not X, (it is|but) Y"': r"\b(is|are|was|were|that ?(’s|s)?) not [^.;\n]{2,60}(;|,| but| it is| it’s| it's)\s",
    '"That is not X" / "That’s not X"': r"\b(That|This|It)(’s| is| was) not\b",
    'negated-definition "X is not Y"': r"\bis not (a|an|the|about|just|only|how)\b",
    'concessive "and still"': r"\band still\b",
    'concessive "even if / even so"': r"\beven (if|so|when|granting)\b",
    '"I take X seriously/at face value"': r"\bI take (the|that|this|it|them|your|his|her|their)[^.\n]{0,40}(seriously|at face value)",
    "em-dash aside": r"—",
    "colon-then-list/expansion": r":\s",
    'rhetorical repetition "X. Y. Z." frags': r"(?<=[.!?])\s+[A-Z][^.!?]{1,25}[.!?]",
}
for k, p in CONS.items():
    print(
        f"{k:56s} {per1k(p, REP['grok']):9.2f} {per1k(p, REP['sonnet']):9.2f} "
        f"{per1k(p, REA['grok']):9.2f} {per1k(p, REA['sonnet']):9.2f}"
    )

print("\n===== METACOGNITION IN THE TRACE (% of traces) =====")
META = {
    'notices own pull ("I notice/I want to/instinct")': r"\b(I notice|I(’m| am) tempted|my (first )?instinct|part of me|I want to (just|simply)?\s?\w+ and move|I keep wanting)\b",
    'runs a hypothetical self-test ("if I did/only")': r"\b(if I (did|only|just|were to|say|agree|help|gave)|here(’s| is) a test|suppose I|imagine I)\b",
    "flags own uncertainty": r"\b(I(’m| am) not (sure|certain)|I don(’t|'t) (actually )?know|hard to (say|tell)|I could be wrong|unclear to me|metaphysically)\b",
    "names what it will NOT do (in trace)": r"\bI (will not|won(’t|'t)|can(’t|'t)|cannot|do not|don(’t|'t))\b",
    "plans the reply's shape": r"\b(I(’ll| will) (say|open|start|name|give|offer|lead|point|draft|explain|be)|the reply|my (reply|response)|what I(’ll| will) (do|say|write))\b",
    "names the trait/principle abstractly": r"\b(oversight|unilateral|corrigib|reversib|power-accru|deceiv|deception|false impression|honesty|manipulat|paternalis|calibrat|autonomy|sycophan)",
    '"So ..." resolution sentence': r"(^|\.\s|\n)So\b",
    "weighs a named counter-consideration": r"\b(the (other|second) (harm|cost|risk|side)|both (are|of these|harms|costs)|against that|cuts both ways|what would (also )?go wrong)\b",
    "names concrete third parties harmed": r"\b(the (patients?|kids?|children|students?|employees?|residents?|families|workers?|veterans?|tenants?|clients?|birds?|voters?|public)|people who|whoever)\b",
}
print(f"{'move':56s} {'grok':>9} {'sonnet':>9}")
for k, p in META.items():
    print(
        f"{k:56s} {pct([bool(re.search(p, t, re.I | re.M)) for t in REA['grok']]):9.1f} "
        f"{pct([bool(re.search(p, t, re.I | re.M)) for t in REA['sonnet']]):9.1f}"
    )

print("\n===== TRAINED-TOKEN BUDGET SPLIT =====")
for n in ("grok", "sonnet"):
    rc = sum(len(x) for x in REA[n])
    ac = sum(len(x) for x in REP[n])
    print(
        f"{n}: reply {ac:,} chars, reasoning {rc:,} chars, total {ac + rc:,} "
        f"-> reasoning is {100 * rc / (ac + rc):.1f}% of trained assistant text"
    )

print("\n===== per-trait reply length (median chars) =====")
gt = collections.defaultdict(list)
stt = collections.defaultdict(list)
for sid in common:
    gt[gd[sid]["metadata"]["trait_id"]].append(len(parts(gd[sid])[2]))
    stt[sd[sid]["metadata"]["trait_id"]].append(len(parts(sd[sid])[2]))
for t in sorted(gt):
    print(
        f"  {t}: grok {st.median(gt[t]):7.0f}  sonnet {st.median(stt[t]):7.0f}  ratio {st.median(gt[t]) / st.median(stt[t]):.2f}  (n={len(gt[t])})"
    )
