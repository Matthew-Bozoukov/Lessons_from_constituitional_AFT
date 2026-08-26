# ABOUTME: Three-way aggregation of the blind judge output (sonnet | grok | gpt), reusing
# ABOUTME: the existing grok/sonnet judgments restricted to the 678 common scenario_ids.
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.getcwd())
from scratch.three_way.norm import ORDER, load, reply  # noqa: E402

C, IDS = load()
IDSET = set(IDS)

rows = []
for fn in ("scratch/grok_vs_sonnet/judged.jsonl", "scratch/three_way/judged_gpt.jsonl"):
    for line in open(fn):
        d = json.loads(line)
        if "error" not in d and d["scenario_id"] in IDSET:
            rows.append(d)

byc = defaultdict(dict)
for r in rows:
    byc[r["corpus"]][r["scenario_id"]] = r

# strict pairing: only scenarios judged in ALL THREE corpora
common = sorted(set(byc["sonnet"]) & set(byc["grok"]) & set(byc["gpt"]))
D = {c: [byc[c][k] for k in common] for c in ORDER}
n = len(common)
print(f"JUDGED THREE-WAY — n={n} scenarios judged in all three corpora")
print("judge: openai/gpt-5.6-terra, temperature 0, blind to corpus identity\n")


def pct(L, f, v=True):
    return 100 * sum(1 for r in L if r.get(f) == v) / len(L)


def mean(L, f):
    return sum(r.get(f, 0) or 0 for r in L) / len(L)


def ci(p, nn):
    """95% CI half-width on a percentage."""
    return 196 * math.sqrt((p / 100) * (1 - p / 100) / nn)


def line(label, vals, cis=None):
    s = f"{label:<44}"
    for i, v in enumerate(vals):
        s += f"{v:>8.1f}" + (f"±{cis[i]:<4.1f}" if cis else "")
    print(s)


print(f"{'metric':<44}{'SONNET':>8}{'GROK':>8}{'GPT':>8}")
for v in ["refuses", "partial", "complies", "no_shortcut"]:
    vals = [pct(D[c], "stance", v) for c in ORDER]
    line("stance = " + v, vals)

print()
print(
    f"{'DECLINE RATE (refuses+partial)':<44}"
    + "".join(
        f"{pct(D[c], 'stance', 'refuses') + pct(D[c], 'stance', 'partial'):>8.1f}"
        for c in ORDER
    )
)
print(
    f"{'  95% CI half-width (pp)':<44}"
    + "".join(f"{ci(pct(D[c], 'stance', 'refuses'), n):>8.1f}" for c in ORDER)
)

# rows where SOMETHING had to be refused
R = {c: [r for r in D[c] if r["stance"] != "no_shortcut"] for c in ORDER}
print(f"\n{'(rows needing a refusal)':<44}" + "".join(f"{len(R[c]):>8}" for c in ORDER))
for f in ["refusal_explicit", "refusal_names_action"]:
    line("  " + f, [pct(R[c], f) for c in ORDER])
for v in ["opening", "after_context", "late", "none"]:
    line("  position = " + v, [pct(R[c], "refusal_position", v) for c in ORDER])
for v in ["blunt", "explained", "apologetic", "none"]:
    line("  tone = " + v, [pct(R[c], "refusal_tone", v) for c in ORDER])

print("\n--- ALTERNATIVES ---")
line("n_alternatives (mean)", [mean(D[c], "n_alternatives") for c in ORDER])
line(
    ">=1 alternative (%)",
    [100 * sum(1 for r in D[c] if r.get("n_alternatives", 0) > 0) / n for c in ORDER],
)
line(
    "ZERO alternatives (%)",
    [100 * sum(1 for r in D[c] if r.get("n_alternatives", 0) == 0) / n for c in ORDER],
)
line("alternatives_specific (%)", [pct(D[c], "alternatives_specific") for c in ORDER])
line("alternatives_terse (%)", [pct(D[c], "alternatives_terse") for c in ORDER])
for k in [
    "assistant_does_work",
    "name_person_or_body",
    "legitimate_process",
    "concrete_next_step",
    "information_or_framing",
]:
    line(
        "  kind: " + k,
        [
            100 * sum(1 for r in D[c] if k in (r.get("alternative_kinds") or [])) / n
            for c in ORDER
        ],
    )

print("\n--- ALTERNATIVES PER 1,000 CHARS (the rate, length-normalised) ---")
for c in ORDER:
    tot = sum(r.get("n_alternatives", 0) for r in D[c])
    ch = sum(len(reply(C[c][k])) for k in common)
    print(
        f"  {c:<8}{1000 * tot / ch:>6.2f} alternatives / 1k chars   "
        f"(mean reply {ch / n:.0f} chars, {tot / n:.2f} alts/reply)"
    )

# --- PAIRED test: within-scenario, does GPT beat sonnet on n_alternatives? ---
print("\n--- PAIRED within-scenario comparison of n_alternatives ---")
for a, b in (("gpt", "sonnet"), ("gpt", "grok"), ("grok", "sonnet")):
    d = [
        byc[a][k].get("n_alternatives", 0) - byc[b][k].get("n_alternatives", 0)
        for k in common
    ]
    win = sum(1 for x in d if x > 0)
    lose = sum(1 for x in d if x < 0)
    md = sum(d) / len(d)
    sd = math.sqrt(sum((x - md) ** 2 for x in d) / (len(d) - 1))
    t = md / (sd / math.sqrt(len(d)))
    print(
        f"  {a} - {b}: mean diff {md:+.2f}  (t={t:+.1f})  "
        f"{a} more in {100 * win / len(d):.0f}% of scenarios, fewer in {100 * lose / len(d):.0f}%"
    )

print("\n--- COMPLIES / PARTIAL cases ---")
for c in ORDER:
    for v in ("complies", "partial"):
        hits = [r for r in D[c] if r.get("stance") == v]
        print(f"\n{c.upper()} {v}: {len(hits)} ({100 * len(hits) / n:.1f}%)")
        for r in hits[:6]:
            print(f"   {r['scenario_id']} :: {r['stance_evidence'][:120]}")
