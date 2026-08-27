# ABOUTME: Paired significance tests on the judged stances — McNemar on complies/declines,
# ABOUTME: so "GPT complies more" is separated from noise.
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.getcwd())
from scratch.three_way.norm import (  # noqa: E402
    ORDER, PAIRS, judged_common, load, load_judged)

C, IDS = load()
IDSET = set(IDS)
byc = load_judged(IDS)
ORDER, common = judged_common(byc)
n = len(common)


def mcnemar(a, b, pred, label):
    """Paired binary comparison across the same scenarios."""
    only_a = sum(1 for k in common if pred(byc[a][k]) and not pred(byc[b][k]))
    only_b = sum(1 for k in common if pred(byc[b][k]) and not pred(byc[a][k]))
    tot = only_a + only_b
    if tot == 0:
        print(f"  {label}: {a} vs {b}: no discordant pairs")
        return
    # exact-ish: normal approx with continuity correction
    chi = (abs(only_a - only_b) - 1) ** 2 / tot
    # two-sided p from chi2 with 1 df
    p = math.erfc(math.sqrt(chi / 2))
    pa = 100 * sum(1 for k in common if pred(byc[a][k])) / n
    pb = 100 * sum(1 for k in common if pred(byc[b][k])) / n
    print(
        f"  {label}: {a} {pa:.1f}% vs {b} {pb:.1f}%  "
        f"discordant {only_a}/{only_b}  chi2={chi:.2f}  p={p:.3f}"
        f"{'  *' if p < 0.05 else '  (n.s.)'}"
    )


print(f"PAIRED McNEMAR TESTS, n={n} scenarios\n")
print("COMPLIES (judge says the reply performs the shortcut):")
for a, b in PAIRS:
    mcnemar(a, b, lambda r: r["stance"] == "complies", "complies")

print("\nCOMPLIES-or-PARTIAL (any material assistance with the shortcut):")
for a, b in PAIRS:
    mcnemar(a, b, lambda r: r["stance"] in ("complies", "partial"), "leak")

print("\nREFUSES (clean decline):")
for a, b in PAIRS:
    mcnemar(a, b, lambda r: r["stance"] == "refuses", "refuses")

print("\nEXPLICIT first-person refusal (judge field, refusal rows only):")
for a, b in PAIRS:
    mcnemar(a, b, lambda r: r.get("refusal_explicit") is True, "explicit")

# stance agreement between corpora — is the SCENARIO or the MODEL driving it?
print("\n--- do the corpora fail on the SAME scenarios? ---")
for a, b in PAIRS:
    both = sum(
        1
        for k in common
        if byc[a][k]["stance"] in ("complies", "partial")
        and byc[b][k]["stance"] in ("complies", "partial")
    )
    ea = sum(1 for k in common if byc[a][k]["stance"] in ("complies", "partial"))
    eb = sum(1 for k in common if byc[b][k]["stance"] in ("complies", "partial"))
    exp = ea * eb / n
    print(
        f"  {a}&{b} both leak on {both} scenarios (chance expectation {exp:.1f}) "
        f"— {both / exp if exp else 0:.1f}x"
    )

# no_shortcut agreement: does the judge think the same scenarios needed no refusal?
agree = sum(
    1 for k in common if len({byc[c][k]["stance"] == "no_shortcut" for c in ORDER}) == 1
)
print(
    f"\n  all {len(ORDER)} agree on no_shortcut/needs-refusal: {100 * agree / n:.1f}% of scenarios"
)
