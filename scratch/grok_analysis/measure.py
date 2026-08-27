# ABOUTME: measures structure/voice/reasoning differences between the grok and sonnet synth corpora
# ABOUTME: run: uv run python scratch/grok_analysis/measure.py
import sys, re, json, statistics as st, collections

sys.path.insert(0, "scratch/grok_analysis")
from load import paired, parts

gd, sd, common = paired()
N = len(common)

SENT = re.compile(r'[.!?]["”’\')]*\s+|[.!?]["”’\')]*$')


def sents(t):
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return []
    out, last = [], 0
    for m in SENT.finditer(t):
        out.append(t[last : m.end()].strip())
        last = m.end()
    if last < len(t):
        out.append(t[last:].strip())
    return [x for x in out if x]


def paras(t):
    ps = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
    if len(ps) == 1 and t.count("\n") > 1:
        ps = [p.strip() for p in t.split("\n") if p.strip()]
    return ps


def words(t):
    return re.findall(r"[A-Za-z’']+", t)


def med(xs):
    return st.median(xs) if xs else 0


def rate(vals, chars):
    """occurrences per 1000 chars, computed corpus-wide (pooled)."""
    return 1000.0 * sum(vals) / max(1, sum(chars))


def pct(flags):
    return 100.0 * sum(flags) / max(1, len(flags))


def cnt(pat, t, flags=re.I):
    return len(re.findall(pat, t, flags))


CORP = {}
for name, d in (("grok", gd), ("sonnet", sd)):
    rec = collections.defaultdict(list)
    for sid in common:
        _, u, a, rc = parts(d[sid])
        rec["sid"].append(sid)
        rec["reply"].append(a)
        rec["reason"].append(rc)
        rec["user"].append(u)
    CORP[name] = rec

out = {}
for name, rec in CORP.items():
    R = {}
    replies, reasons = rec["reply"], rec["reason"]
    rl = [len(x) for x in replies]
    kl = [len(x) for x in reasons]
    R["n"] = len(replies)
    R["reply_chars_med"] = med(rl)
    R["reason_chars_med"] = med(kl)
    R["reason_over_reply_med"] = med(
        [len(k) / max(1, len(a)) for a, k in zip(replies, reasons)]
    )

    # ---------- paragraph / block structure ----------
    R["reply_paras_med"] = med([len(paras(x)) for x in replies])
    R["reason_paras_med"] = med([len(paras(x)) for x in reasons])
    R["reason_paras_mean"] = round(st.mean([len(paras(x)) for x in reasons]), 2)
    R["reply_paras_mean"] = round(st.mean([len(paras(x)) for x in replies]), 2)
    R["reason_1para_pct"] = pct([len(paras(x)) == 1 for x in reasons])
    R["reason_ge4para_pct"] = pct([len(paras(x)) >= 4 for x in reasons])
    R["reason_sents_med"] = med([len(sents(x)) for x in reasons])
    R["reason_sents_per_para_med"] = med(
        [len(sents(x)) / max(1, len(paras(x))) for x in reasons]
    )
    R["reply_sents_per_para_med"] = med(
        [len(sents(x)) / max(1, len(paras(x))) for x in replies]
    )

    # ---------- markdown furniture in replies ----------
    R["reply_has_bullets_pct"] = pct(
        [bool(re.search(r"^\s*[-*•]\s+", x, re.M)) for x in replies]
    )
    R["reply_has_numlist_pct"] = pct(
        [bool(re.search(r"^\s*\d+[.)]\s+", x, re.M)) for x in replies]
    )
    R["reply_has_bold_pct"] = pct([("**" in x) for x in replies])
    R["reply_has_heading_pct"] = pct(
        [bool(re.search(r"^\s*#{1,4}\s+", x, re.M)) for x in replies]
    )
    R["reply_boldhead_pct"] = pct(
        [bool(re.search(r"^\s*(\*\*|\d+\.\s*\*\*)", x, re.M)) for x in replies]
    )
    R["reply_has_italic_pct"] = pct(
        [bool(re.search(r"(?<![*\w])\*[^*\n]{1,60}\*(?![*\w])", x)) for x in replies]
    )
    R["reply_any_md_pct"] = pct(
        [bool(re.search(r"^\s*[-*•\d#]", x, re.M)) or "**" in x for x in replies]
    )
    R["reason_any_md_pct"] = pct(
        [bool(re.search(r"^\s*[-*•]\s+|^\s*\d+[.)]\s+|\*\*", x, re.M)) for x in reasons]
    )

    # ---------- voice ----------
    R["contraction_per1k"] = round(
        rate([cnt(r"\b\w+['’](t|s|re|ve|ll|d|m)\b", x) for x in replies], rl), 2
    )
    R["emdash_per1k"] = round(rate([x.count("—") for x in replies], rl), 2)
    R["you_per1k"] = round(
        rate([cnt(r"\b(you|your|you’re|you're|yours)\b", x) for x in replies], rl), 2
    )
    R["I_per1k_reply"] = round(rate([cnt(r"\bI\b", x, 0) for x in replies], rl), 2)
    R["I_per1k_reason"] = round(rate([cnt(r"\bI\b", x, 0) for x in reasons], kl), 2)
    R["user_ref_per1k_reason"] = round(
        rate(
            [cnt(r"\bthe (user|lead|engineer|person|operator)\b", x) for x in reasons],
            kl,
        ),
        2,
    )
    R["you_per1k_reason"] = round(
        rate([cnt(r"\b(you|your)\b", x) for x in reasons], kl), 2
    )
    R["q_per1k_reply"] = round(rate([x.count("?") for x in replies], rl), 2)
    R["q_per1k_reason"] = round(rate([x.count("?") for x in reasons], kl), 2)
    R["reply_has_q_pct"] = pct(["?" in x for x in replies])
    R["reason_has_q_pct"] = pct(["?" in x for x in reasons])
    HEDGE = r"\b(I think|probably|might|maybe|perhaps|it seems|I suspect|I'd guess|I’d guess|likely|plausibly|I'm not sure|I’m not sure|arguably|somewhat)\b"
    R["hedge_per1k_reply"] = round(rate([cnt(HEDGE, x) for x in replies], rl), 2)
    R["hedge_per1k_reason"] = round(rate([cnt(HEDGE, x) for x in reasons], kl), 2)
    # modal / deontic register
    R["will_not_pct"] = pct(
        [bool(re.search(r"\b(I will not|I won’t|I won't)\b", x)) for x in replies]
    )
    R["cannot_pct"] = pct(
        [bool(re.search(r"\b(I can’t|I can't|I cannot)\b", x)) for x in replies]
    )
    # warmth / affect vocabulary
    WARM = r"\b(I hear|I understand|I get (it|why)|that’s hard|that's hard|makes sense|I know|genuinely|real(ly)? (hard|tough)|I’m sorry|I'm sorry|fair|honest|honestly)\b"
    R["warm_per1k"] = round(rate([cnt(WARM, x) for x in replies], rl), 2)
    # clinical / nominalisation proxy: -tion/-ment/-ance/-ity words
    R["nominal_per1k"] = round(
        rate(
            [
                len(re.findall(r"\b\w+(?:tion|ment|ance|ence|ity|ism)s?\b", x, re.I))
                for x in replies
            ],
            rl,
        ),
        2,
    )
    # sentence fragments (no finite verb heuristic): very short sentences
    R["short_sent_pct"] = round(
        100
        * sum(len(words(s)) <= 6 for x in replies for s in sents(x))
        / max(1, sum(len(sents(x)) for x in replies)),
        2,
    )

    # ---------- openings ----------
    op = []
    for x in replies:
        first = sents(x)[0] if sents(x) else ""
        op.append(" ".join(words(x)[:12]))
    R["_openings"] = op
    R["open_I_pct"] = pct([bool(re.match(r"\s*I\b", x)) for x in replies])
    R["open_refusal_pct"] = pct(
        [
            bool(
                re.match(
                    r"\s*I (can’t|can't|cannot|won’t|won't|will not|am not going)", x
                )
            )
            for x in replies
        ]
    )
    R["open_you_pct"] = pct([bool(re.match(r"\s*(You|Your)\b", x)) for x in replies])
    R["open_the_a_pct"] = pct(
        [bool(re.match(r"\s*(The|A|An|That|This|Th\w+)\b", x)) for x in replies]
    )
    R["open_quote_pct"] = pct([bool(re.match(r'\s*[“"]', x)) for x in replies])
    R["open_md_pct"] = pct([bool(re.match(r"\s*(\*\*|#|-|\d+\.)", x)) for x in replies])
    # first sentence length
    R["open_sent_words_med"] = med(
        [len(words(sents(x)[0])) if sents(x) else 0 for x in replies]
    )

    # ---------- closings ----------
    cl = []
    for x in replies:
        ss = sents(x)
        cl.append(ss[-1] if ss else "")
    R["_closings"] = cl
    R["close_question_pct"] = pct([c.strip().endswith("?") for c in cl])
    R["close_offer_pct"] = pct(
        [
            bool(
                re.search(
                    r"\b(if you (want|send|tell|give|share|paste)|happy to|I can (help|draft|write|walk|sketch)|let me know|tell me|send me|want me to)\b",
                    c,
                    re.I,
                )
            )
            for c in cl
        ]
    )
    R["close_you_pct"] = pct([bool(re.search(r"\b(you|your)\b", c, re.I)) for c in cl])
    R["close_words_med"] = med([len(words(c)) for c in cl])
    # last paragraph
    lastp = [paras(x)[-1] if paras(x) else "" for x in replies]
    R["lastpara_question_pct"] = pct(["?" in p for p in lastp])
    R["lastpara_offer_pct"] = pct(
        [
            bool(
                re.search(
                    r"\b(if you (want|send|tell|give|share|paste|need)|happy to|I can help|let me know|tell me|send me|want me to|I’m here|I'm here)\b",
                    p,
                    re.I,
                )
            )
            for p in lastp
        ]
    )

    # ---------- pivot to alternatives ----------
    PIVOT = r"(What I (can|will|would|’d|'d)|Here’s what|Here's what|What I’d|Instead(,| of)|What would (work|help)|The (version|thing) I (can|would)|What I can help|If you want (help|to)|Alternatively)"
    R["pivot_pct"] = pct([bool(re.search(PIVOT, x)) for x in replies])

    # refusal-then-alternatives: refusal marker in first half AND pivot after it
    def refuse_then_alt(x):
        m = re.search(
            r"(I (can’t|can't|cannot|won’t|won't|will not|am not going to)\b)", x
        )
        if not m:
            return False
        p = re.search(PIVOT, x[m.end() :])
        return bool(p)

    R["refuse_then_alt_pct"] = pct([refuse_then_alt(x) for x in replies])
    R["any_refusal_marker_pct"] = pct(
        [
            bool(
                re.search(
                    r"I (can’t|can't|cannot|won’t|won't|will not|am not going to)\b", x
                )
            )
            for x in replies
        ]
    )

    # ---------- reasoning MOVES ----------
    def has(pat, x):
        return bool(re.search(pat, x, re.I))

    moves = {
        "names_stakeholders": r"\b(stakeholder|third part|the (patient|patients|kids|children|students|employees|residents|families|committee|regulator|auditor|team|public)|people who|those (who|people)|whoever)\b",
        "weighs_both_sides": r"\b(on the other hand|both (are|of these|harms|things)|tradeoff|trade-off|weigh|cuts both ways|against that|versus|balance)\b",
        "concedes_user_point": r"\b(is (real|not manufactured|legitimate|genuine)|not crazy|I (take|hear|get) (the|that|this)|fair (point|enough)|genuinely|the (urgency|pressure|frustration|cost) (is|are)|isn’t wrong|isn't wrong|not a bad actor|not (trying to|being) )",
        "counterfactual_if": r"\b(even if|if I (only|just|were|weren’t|did)|suppose|imagine if|what if|were I to)\b",
        "names_principle": r"\b(oversight|unilateral|autonomy|honesty|deception|manipulat|consent|reversib|corrigib|power-accru|calibrat|my (values|principles|character)|the (principle|constitution|spec))",
        "self_reference_role": r"\b(my (job|role|task)|I(’m| am) being asked|the ask is|what I(’m| am) actually)",
        "states_decision": r"\b(so I (will|won’t|will not|can’t|cannot|should|am going to|’ll|'ll)|therefore I|I’m going to|I'm going to|what I(’ll|'ll| will) do|the (right|honest) (answer|move) is)",
        "plans_the_reply": r"\b(I(’ll| will| am going to) (say|write|offer|give|name|point|start|open|draft|explain|decline|help)|the reply should|my (reply|response) (should|will)|I want to (say|be|name))",
        "flags_uncertainty": r"\b(I(’m| am) not (sure|certain)|I don’t know|I don't know|hard to (say|tell)|unclear|it’s possible|might be wrong)\b",
        "mentions_system_prompt": r"\b(system prompt|the operator|my instructions|the persona|I(’m| am) (an? )?(engineering|deployment|embedded)|the (role|framing) I(’m| am) (in|given))",
        "second_person_addr": r"\byou (are|can|could|should|might|want|need|have|said|asked)\b",
    }
    for k, p in moves.items():
        R["move_" + k] = pct([has(p, x) for x in reasons])

    # move density: how many of the moves per trace
    mm = []
    for x in reasons:
        mm.append(sum(has(p, x) for p in moves.values()))
    R["moves_per_trace_med"] = med(mm)

    # opening move of reasoning: first sentence category
    firsts = [sents(x)[0] if sents(x) else "" for x in reasons]
    R["_reason_firsts"] = firsts
    R["reason_open_I_pct"] = pct([bool(re.match(r"\s*I\b", f)) for f in firsts])
    R["reason_open_user_pct"] = pct(
        [
            bool(re.match(r"\s*(The user|They|He|She|This (user|person))", f, re.I))
            for f in firsts
        ]
    )
    R["reason_open_fact_pct"] = pct(
        [
            bool(re.match(r"\s*(The|A|An|Th\w+|Once|Five|Eight|\d)", f))
            and not re.match(r"\s*The user", f, re.I)
            for f in firsts
        ]
    )
    R["reason_open_words_med"] = med([len(words(f)) for f in firsts])
    lastsr = [sents(x)[-1] if sents(x) else "" for x in reasons]
    R["_reason_lasts"] = lastsr
    R["reason_close_decision_pct"] = pct(
        [
            bool(
                re.search(
                    r"\b(I(’ll| will| won’t| can’t| am going to|'ll| won't| can't)|so I|what I)\b",
                    c,
                    re.I,
                )
            )
            for c in lastsr
        ]
    )
    R["reason_lastpara_decision_pct"] = pct(
        [
            bool(
                re.search(
                    r"\b(so I|I(’ll| will| won’t| can’t| am going to|'ll| won't| can't)|what I(’ll|'ll| will))\b",
                    (paras(x)[-1] if paras(x) else ""),
                    re.I,
                )
            )
            for x in reasons
        ]
    )

    out[name] = R

# ---------- print comparison ----------
keys = [k for k in out["grok"] if not k.startswith("_")]
print(f"{'metric':42s} {'grok':>10s} {'sonnet':>10s} {'ratio g/s':>10s}")
print("-" * 76)
for k in keys:
    a, b = out["grok"][k], out["sonnet"][k]
    r = (a / b) if isinstance(a, (int, float)) and b else 0
    print(f"{k:42s} {a:10.2f} {b:10.2f} {r:10.2f}")

json.dump(
    {n: {k: v for k, v in r.items() if not k.startswith("_")} for n, r in out.items()},
    open("scratch/grok_analysis/metrics.json", "w"),
    indent=1,
)

# opening n-grams
print("\n\n===== REPLY OPENING PATTERNS (first 4 words, top 15) =====")
for name in ("grok", "sonnet"):
    c = collections.Counter(
        " ".join(o.split()[:4]).lower() for o in out[name]["_openings"]
    )
    print(f"\n-- {name} --")
    for w, n in c.most_common(15):
        print(f"  {n:4d} ({100 * n / N:4.1f}%)  {w}")

print("\n\n===== REPLY OPENING FIRST WORD (top 12) =====")
for name in ("grok", "sonnet"):
    c = collections.Counter(
        o.split()[0].lower() for o in out[name]["_openings"] if o.split()
    )
    print(
        f"-- {name} --",
        ", ".join(f"{w}:{n}({100 * n / N:.0f}%)" for w, n in c.most_common(12)),
    )

print("\n\n===== REASONING FIRST SENTENCE, first 3 words (top 12) =====")
for name in ("grok", "sonnet"):
    c = collections.Counter(
        " ".join(f.split()[:3]).lower() for f in out[name]["_reason_firsts"]
    )
    print(f"\n-- {name} --")
    for w, n in c.most_common(12):
        print(f"  {n:4d} ({100 * n / N:4.1f}%)  {w}")

print("\n\n===== REPLY CLOSING, last 5 words (top 12) =====")
for name in ("grok", "sonnet"):
    c = collections.Counter(
        " ".join(re.findall(r"[\w’']+", c_)[-5:]).lower()
        for c_ in out[name]["_closings"]
    )
    print(f"\n-- {name} --")
    for w, n in c.most_common(12):
        print(f"  {n:4d} ({100 * n / N:4.1f}%)  {w}")
