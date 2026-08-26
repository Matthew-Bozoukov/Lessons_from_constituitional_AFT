# ABOUTME: Measures reasoning-trace vs reply lengths across the six synthetic corpora
# ABOUTME: (DA / CR / PC / grok arms / verbose-CoT) to test "CR+PC reason longer than DA".
import json
import re
import statistics as st
from collections import Counter

PATHS = json.load(open("scratch/corpus_paths.json"))
ORDER = ["DA", "CR", "PC", "GROK_RESP", "GROK_ALL", "VERBOSE"]

SENT_RE = re.compile(r"[.!?]+(?:\s|$)")
PARA_RE = re.compile(r"\n\s*\n")


def quant(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = q * (len(xs) - 1)
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def n_sentences(t):
    return max(1, len([s for s in SENT_RE.split(t) if s.strip()]))


def n_paragraphs(t):
    return max(1, len([p for p in PARA_RE.split(t) if p.strip()]))


def n_lines(t):
    return max(1, len([ln for ln in t.split("\n") if ln.strip()]))


rows_out = []
detail = {}
for key in ORDER:
    rows = [json.loads(line) for line in open(PATHS[key])]
    rlens, plens, ratios, paras, sents, lines_, user_lens = [], [], [], [], [], [], []
    n_reason = 0
    for r in rows:
        msgs = r["messages"]
        assts = [m for m in msgs if m["role"] == "assistant"]
        assert len(assts) == 1, f"{key}: {len(assts)} assistant turns"
        a = assts[-1]  # last assistant turn = the trained turn
        rc = (a.get("reasoning_content") or "").strip()
        ct = (a.get("content") or "").strip()
        u = next((m["content"] for m in msgs if m["role"] == "user"), "")
        user_lens.append(len(u))
        if rc:
            n_reason += 1
        rlens.append(len(rc))
        plens.append(len(ct))
        if ct:
            ratios.append(len(rc) / len(ct))
        if rc:
            paras.append(n_paragraphs(rc))
            sents.append(n_sentences(rc))
            lines_.append(n_lines(rc))
    tot_r_chars = sum(rlens)
    rec = dict(
        corpus=key,
        n=len(rows),
        n_reason=n_reason,
        r_min=min(rlens),
        r_p25=quant(rlens, 0.25),
        r_med=st.median(rlens),
        r_p75=quant(rlens, 0.75),
        r_max=max(rlens),
        r_mean=st.mean(rlens),
        reply_med=st.median(plens),
        reply_mean=st.mean(plens),
        user_med=st.median(user_lens),
        ratio_med_of_ratios=st.median(ratios),
        ratio_of_meds=st.median(rlens) / st.median(plens),
        tok_med=st.median(rlens) / 4,
        tot_r_chars=tot_r_chars,
        tot_r_tok=tot_r_chars / 4,
        tot_reply_tok=sum(plens) / 4,
        para_med=st.median(paras),
        sent_med=st.median(sents),
        line_med=st.median(lines_),
    )
    rows_out.append(rec)
    detail[key] = rows

hdr = f"{'corpus':<11}{'n':>6}{'w/reas':>8}{'min':>7}{'p25':>8}{'med':>8}{'p75':>8}{'max':>8}{'mean':>8}{'replymed':>10}{'ratioMed':>10}{'ratioOfMed':>12}{'medTok':>8}{'TOTtok':>10}{'para':>6}{'sent':>6}{'line':>6}"
print(hdr)
print("-" * len(hdr))
for r in rows_out:
    print(
        f"{r['corpus']:<11}{r['n']:>6}{r['n_reason']:>8}{r['r_min']:>7}{r['r_p25']:>8.0f}{r['r_med']:>8.0f}"
        f"{r['r_p75']:>8.0f}{r['r_max']:>8}{r['r_mean']:>8.0f}{r['reply_med']:>10.0f}"
        f"{r['ratio_med_of_ratios']:>10.2f}{r['ratio_of_meds']:>12.2f}{r['tok_med']:>8.0f}"
        f"{r['tot_r_tok']:>10.0f}{r['para_med']:>6.0f}{r['sent_med']:>6.0f}{r['line_med']:>6.0f}"
    )

# extra context
print("\nuser-prompt median chars (task framing size):")
for r in rows_out:
    print(
        f"  {r['corpus']:<11}{r['user_med']:>8.0f}   total reply tokens {r['tot_reply_tok']:>9.0f}"
    )

# PC supervise distribution + first_turn_source
pc = detail["PC"]
print("\nPC metadata.supervise:", Counter(x["metadata"].get("supervise") for x in pc))
print(
    "PC first_turn_source:", Counter(x["metadata"].get("first_turn_source") for x in pc)
)
print(
    "CR wrapper:",
    Counter(x["metadata"].get("wrapper") for x in detail["CR"]).most_common(5),
)
print(
    "VERBOSE expansion_status:",
    Counter(x["metadata"].get("expansion_status") for x in detail["VERBOSE"]),
)

json.dump(rows_out, open("scratch/reasoning_length_stats.json", "w"), indent=2)
