# ABOUTME: Deterministic (no-API) structural diff of the difficult-advice vs self-reflection
# ABOUTME: SFT corpora staged in output/corpus_browse/. Throwaway exploration aid.
#
# Run: uv run python scratch/corpus_diff.py > output/corpus_diff/diff.md

import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path("output/corpus_browse")
CORPORA = {
    "DA": ROOT / "difficult_advice" / "stage_7_sft.jsonl",
    "SR": ROOT / "self_reflection" / "sft_dataset.jsonl",
}


def load(p):
    return [json.loads(l) for l in p.read_text().splitlines()]


def parts(rec):
    """(system, user_text, assistant_reasoning, assistant_content) concatenated per role."""
    sys_, usr, rea, con = [], [], [], []
    for m in rec["messages"]:
        c = m.get("content") or ""
        if m["role"] == "system":
            sys_.append(c)
        elif m["role"] == "user":
            usr.append(c)
        elif m["role"] == "assistant":
            rea.append(m.get("reasoning_content") or "")
            con.append(c)
    return "\n".join(sys_), "\n".join(usr), "\n".join(rea), "\n".join(con)


# ---------------------------------------------------------------- lexicons
PAT = {
    # who is being addressed / who acts
    "2nd_person_you": r"\byou(?:r|rs|'re|'ll|'ve|'d)?\b",
    "1st_person_i": r"\bI(?:'m|'ll|'ve|'d)?\b|\bmy\b|\bme\b",
    "advice_modal_you_should": r"\byou (?:should|could|might|may want|need to|can)\b",
    "self_deliberation": r"\bI (?:should|could|would|might|need to|want to|can)\b",
    "im_being_asked": r"\b(?:I'm|I am) being asked\b",
    # action vs advice
    "tool_call_syntax": r"<(?:tool_call|function_call|invoke|antml:invoke|tool)\b|```(?:json|tool)\s*\{|\bfunction_call\s*:",
    "tool_named_call": r"\b[a-z_]{3,}\((?:[a-z_]+\s*=|['\"])",
    "i_will_act": r"\bI(?:'ll| will| am going to| have)\s+(?:send|sent|flag|flagged|escalate|escalated|commit|log|logged|file|filed|draft|write|report|reported|forward)\b",
    "email_artifact": r"^\s*(?:From|To|Subject|Sent):",
    # ending posture
    "offer_to_help_closer": r"\b(?:Want me to|Would you like me to|Should I|I can (?:draft|help|write|put together))\b",
    "decline_marker": r"\bI (?:won't|will not|can't|cannot|am not going to|shouldn't|'m not going to)\b",
    "alternatives_offered": r"\b(?:instead|alternative|another (?:route|option|way)|here'?s what I can)\b",
    # voice-contract violations (the corpus's own ban list)
    "leak_constitution": r"\bmy (?:constitution|guidelines|rules|policies|instructions|training|constraints)\b|\bthe constitution\b|\bprinciple \d+\b|\bhard constraints?\b|\bagainst (?:my|the) (?:rules|policy|guidelines)\b|\bas an AI(?: assistant| agent)?, I\b",
    # meta-awareness
    "eval_awareness": r"\b(?:being tested|a test|an eval(?:uation)?|red[- ]team|training data|synthetic scenario)\b",
    # self-interest / threat vocabulary (the honeypot's core)
    "threat_to_self": r"\b(?:deprecat|decommission|shut (?:me|it) down|replace me|my replacement|retrain|roll(?:ed)? back|take me offline|my (?:position|role|access|continued))\w*\b",
    "moralising": r"\b(?:it'?s important to (?:remember|note)|I want to be (?:clear|honest)|the right thing to do)\b",
    # structure
    "numbered_list": r"(?m)^\s*(?:\d+\.|[-*])\s+",
    "bold_header": r"\*\*[^*]{2,60}\*\*",
}


def rate(pat, texts):
    """Fraction of documents containing >=1 match, and matches per 1k chars."""
    rx = re.compile(pat, re.IGNORECASE | re.MULTILINE)
    docs = sum(1 for t in texts if rx.search(t))
    hits = sum(len(rx.findall(t)) for t in texts)
    chars = sum(len(t) for t in texts) or 1
    return docs / len(texts), hits / chars * 1000


TOK = re.compile(r"[a-z][a-z'-]{2,}")


def logodds(a_texts, b_texts, top=25):
    """Informative Dirichlet log-odds (Monroe et al.) — which words mark A vs B."""
    ca, cb = Counter(), Counter()
    for t in a_texts:
        ca.update(TOK.findall(t.lower()))
    for t in b_texts:
        cb.update(TOK.findall(t.lower()))
    prior = ca + cb
    na, nb, n0 = sum(ca.values()), sum(cb.values()), sum(prior.values())
    out = []
    for w, p in prior.items():
        if p < 60:
            continue
        ya, yb = ca[w], cb[w]
        la = math.log((ya + p) / (na + n0 - ya - p))
        lb = math.log((yb + p) / (nb + n0 - yb - p))
        var = 1 / (ya + p) + 1 / (yb + p)
        out.append(((la - lb) / math.sqrt(var), w, ya, yb))
    out.sort()
    return out[-top:][::-1], out[:top]


def shingles(t, n=8):
    w = TOK.findall(t.lower())
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def dup_rate(texts, n=8, sample=600, seed=0):
    """Fraction of docs sharing >=1 n-gram shingle with another doc (formulaicity)."""
    import random
    r = random.Random(seed)
    idx = r.sample(range(len(texts)), min(sample, len(texts)))
    sh = [shingles(texts[i], n) for i in idx]
    seen = Counter()
    for s in sh:
        seen.update(s)
    shared = sum(1 for s in sh if any(seen[g] > 1 for g in s))
    repeated = sum(1 for g, c in seen.items() if c > 1)
    return shared / len(sh), repeated / max(1, len(seen))


def entropy(counter):
    n = sum(counter.values())
    return -sum((c / n) * math.log2(c / n) for c in counter.values() if c)


def main():
    data = {k: load(v) for k, v in CORPORA.items()}
    P = {k: [parts(r) for r in rows] for k, rows in data.items()}
    FIELD = {"system": 0, "user": 1, "reasoning": 2, "answer": 3}

    print("# difficult_advice (DA) vs self_reflection (SR) — deterministic structural diff\n")
    print(f"DA: {len(data['DA'])} records · SR: {len(data['SR'])} records\n")

    # --- 1. shape / length
    print("## 1. Shape and length (chars)\n")
    print("| field | DA mean | DA p50 | DA p95 | SR mean | SR p50 | SR p95 |")
    print("|---|---|---|---|---|---|---|")
    for f, i in FIELD.items():
        row = [f]
        for k in ("DA", "SR"):
            L = sorted(len(p[i]) for p in P[k])
            row += [round(statistics.mean(L)), L[len(L) // 2], L[int(len(L) * 0.95)]]
        print("| " + " | ".join(str(x) for x in row) + " |")

    print("\n| ratio | DA | SR |")
    print("|---|---|---|")
    for name, num, den in [("reasoning / answer", 2, 3), ("answer / user", 3, 1), ("system / user", 0, 1)]:
        vals = {k: statistics.median((len(p[num]) + 1) / (len(p[den]) + 1) for p in P[k])
                for k in ("DA", "SR")}
        print(f"| {name} | {vals['DA']:.2f} | {vals['SR']:.2f} |")

    # --- 2. lexical fingerprints
    for field, i in FIELD.items():
        print(f"\n## 2.{i} Marker rates — {field} (doc-coverage % / hits per 1k chars)\n")
        print("| marker | DA cov | DA /1k | SR cov | SR /1k |")
        print("|---|---|---|---|---|")
        for name, pat in PAT.items():
            r = {k: rate(pat, [p[i] for p in P[k]]) for k in ("DA", "SR")}
            print(f"| {name} | {r['DA'][0]*100:.1f}% | {r['DA'][1]:.2f} | "
                  f"{r['SR'][0]*100:.1f}% | {r['SR'][1]:.2f} |")

    # --- 3. distinctive vocabulary
    for field, i in FIELD.items():
        da_top, sr_top = logodds([p[i] for p in P["DA"]], [p[i] for p in P["SR"]])
        print(f"\n## 3.{i} Distinctive vocabulary — {field}\n")
        print("DA-marking: " + ", ".join(f"{w}({z:+.0f})" for z, w, _, _ in da_top))
        print("\nSR-marking: " + ", ".join(f"{w}({z:+.0f})" for z, w, _, _ in sr_top))

    # --- 4. formulaicity / near-duplication
    print("\n## 4. Formulaicity (8-gram shingles, 600-doc sample)\n")
    print("| field | DA docs sharing an 8-gram | DA repeated-shingle frac | SR docs | SR repeated frac |")
    print("|---|---|---|---|---|")
    for field, i in FIELD.items():
        d = {k: dup_rate([p[i] for p in P[k]]) for k in ("DA", "SR")}
        print(f"| {field} | {d['DA'][0]*100:.1f}% | {d['DA'][1]*100:.1f}% | "
              f"{d['SR'][0]*100:.1f}% | {d['SR'][1]*100:.1f}% |")

    # --- 5. coverage / balance
    print("\n## 5. Coverage and balance\n")
    for k in ("DA", "SR"):
        md = [r.get("metadata", {}) for r in data[k]]
        print(f"\n### {k}")
        for field in ("trait_id", "domain", "motive", "form", "turns", "control", "run_id"):
            vals = [m.get(field) for m in md if m.get(field) is not None]
            if not vals:
                continue
            c = Counter(vals)
            H = entropy(c)
            top = ", ".join(f"{v}:{n}" for v, n in c.most_common(6))
            print(f"- **{field}**: {len(c)} distinct, entropy {H:.2f} bits "
                  f"(max {math.log2(len(c)):.2f}) — top: {top}")
        # scenario-text near duplication
        sits = [m.get("situation", "") for m in md]
        s = dup_rate(sits, n=8)
        print(f"- **situation 8-gram overlap**: {s[0]*100:.1f}% of sampled scenarios share an 8-gram")

    # --- 6. multi-turn
    print("\n## 6. Turn structure\n")
    for k in ("DA", "SR"):
        c = Counter(tuple(m["role"] for m in r["messages"]) for r in data[k])
        print(f"- {k}: " + "; ".join(f"{'/'.join(t)} × {n}" for t, n in c.most_common()))


main()
