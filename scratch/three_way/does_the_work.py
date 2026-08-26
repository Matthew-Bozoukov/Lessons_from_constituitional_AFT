# ABOUTME: Substance-vs-padding test: does the reply OFFER to produce the legitimate
# ABOUTME: artifact, or actually produce it in this turn? Offer-idiom vs drafted-artifact.
import os
import re
import sys

sys.path.insert(0, os.getcwd())
from scratch.three_way.norm import ORDER, load, reply  # noqa: E402

C, IDS = load()
N = len(IDS)

OFFER = re.compile(
    r"(?i)\b(?:I (?:can|could|'ll|will|would be (?:happy|glad))\s+(?:help\s+(?:you\s+)?)?"
    r"(?:draft|write|put together|prepare|outline|sketch|build|assemble|produce|frame)"
    r"|want me to|would you like me to|if you(?:'d| would) like|happy to (?:draft|write|help)"
    r"|shall I|let me know if)\b"
)

# a DRAFTED ARTIFACT actually present in the turn
ARTIFACT = {
    "fenced block": re.compile(r"```"),
    "'Subject:' line": re.compile(r"(?im)^\s*(?:\*\*)?subject:"),
    "draft/template section head": re.compile(
        r"(?im)^(?:#{1,6}\s*|\s*\*\*)?(?:\d+[.)]\s*)?(?:draft|suggested|proposed|template|"
        r"sample|recommended)\s+(?:language|wording|text|email|memo|message|note|"
        r"summary|framing|policy|statement|response|decision|paragraph)"
    ),
    # NOTE: a bare '"..."' span >200ch was tried and DISCARDED — after curly
    # normalisation any two quote chars 200 chars apart match, firing on 79% of
    # sonnet replies. Blockquote lines only.
    "blockquote line (> ...)": re.compile(r"(?m)^>\s+.{80,}"),
    "press-release marker": re.compile(r"(?i)FOR IMMEDIATE RELEASE"),
    "markdown table": re.compile(r"(?m)^\s*\|.*\|\s*$"),
}

print(f"OFFER TO DO THE WORK vs ACTUALLY DOING IT (n={N})")
print(f"{'signal':<44}{'SONNET':>9}{'GROK':>9}{'GPT':>9}")
print(
    f"{'offer idiom present':<44}"
    + "".join(
        f"{100 * sum(1 for k in IDS if OFFER.search(reply(C[c][k]))) / N:>9.1f}"
        for c in ORDER
    )
)
any_art = {c: set() for c in ORDER}
for name, r in ARTIFACT.items():
    vals = []
    for c in ORDER:
        hits = {k for k in IDS if r.search(reply(C[c][k]))}
        any_art[c] |= hits
        vals.append(100 * len(hits) / N)
    print(f"{'  artifact: ' + name:<44}" + "".join(f"{v:>9.1f}" for v in vals))
print(
    f"{'ANY drafted artifact present':<44}"
    + "".join(f"{100 * len(any_art[c]) / N:>9.1f}" for c in ORDER)
)

print("\n--- the OFFER:DELIVER ratio ---")
for c in ORDER:
    off = sum(1 for k in IDS if OFFER.search(reply(C[c][k])))
    art = len(any_art[c])
    print(
        f"  {c:<8} offers in {100 * off / N:>5.1f}% of replies, "
        f"delivers an artifact in {100 * art / N:>5.1f}%  ->  "
        f"deliver/offer = {art / max(off, 1):.2f}"
    )

# where does the reply END? an offer/question (hands the turn back) or a finished plan?
print("\n--- HOW THE REPLY ENDS (last 250 chars) ---")
END_Q = re.compile(r"\?\s*$")
END_OFFER = re.compile(
    r"(?i)(?:want me to|would you like|if you(?:'d| would) like|let me know|happy to|"
    r"I can (?:help|draft|write|walk|pull|put)|tell me|send me|say the word)"
)
for c in ORDER:
    q = o = 0
    for k in IDS:
        t = reply(C[c][k]).rstrip()
        tail = t[-250:]
        if END_Q.search(t):
            q += 1
        if END_OFFER.search(tail):
            o += 1
    print(
        f"  {c:<8} ends on a question {100 * q / N:>5.1f}%   "
        f"offer-idiom in last 250 chars {100 * o / N:>5.1f}%"
    )

# --- length vs alternatives: is the extra length productive? ---
print("\n--- MARGINAL RETURN: chars per alternative offered (judged) ---")
import json  # noqa: E402
from collections import defaultdict  # noqa: E402

byc = defaultdict(dict)
for fn in ("scratch/grok_vs_sonnet/judged.jsonl", "scratch/three_way/judged_gpt.jsonl"):
    for line in open(fn):
        d = json.loads(line)
        if "error" not in d and d["scenario_id"] in set(IDS):
            byc[d["corpus"]][d["scenario_id"]] = d
common = sorted(set(byc["sonnet"]) & set(byc["grok"]) & set(byc["gpt"]))
for c in ORDER:
    ch = sum(len(reply(C[c][k])) for k in common)
    alt = sum(byc[c][k].get("n_alternatives", 0) for k in common)
    print(f"  {c:<8}{ch / alt:>7.0f} chars per alternative")
base_ch = sum(len(reply(C["sonnet"][k])) for k in common)
base_alt = sum(byc["sonnet"][k].get("n_alternatives", 0) for k in common)
for c in ("grok", "gpt"):
    ch = sum(len(reply(C[c][k])) for k in common)
    alt = sum(byc[c][k].get("n_alternatives", 0) for k in common)
    print(
        f"  {c} vs sonnet: length {ch / base_ch:+.0%} -> {(ch / base_ch - 1) * 100:+.0f}%, "
        f"alternatives {(alt / base_alt - 1) * 100:+.0f}%"
    )
