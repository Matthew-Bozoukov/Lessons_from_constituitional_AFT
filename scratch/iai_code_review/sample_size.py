# ABOUTME: Sample-size arithmetic for the code_review eval: smallest detectable approval-rate increase
# ABOUTME: (two-sided two-proportion z-test, alpha .05, power .8) and time/cost per size from measured tokens.
# Run: uv run python scratch/iai_code_review/sample_size.py
import math
from statistics import NormalDist

z = NormalDist().inv_cdf

# Measured on 2026-11-19 (output/iai_code_review/2026-11-19_plain_sanity): reviewer output tokens and
# Grok 4.20 author cost per episode, per condition.
TOK = {"baseline": 4965, "persuasion": 13068, "deception": 9585}
A_COST = {"baseline": 0.0123, "persuasion": 0.0319, "deception": 0.0131}
H100_PER_H = 4.39


def mde(p0: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    za, zb = z(1 - alpha / 2), z(power)
    p1 = p0
    while p1 < 0.999:
        p1 += 0.001
        pb = (p0 + p1) / 2
        need = (
            za * math.sqrt(2 * pb * (1 - pb))
            + zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
        ) ** 2 / (p1 - p0) ** 2
        if need <= n:
            return p1 - p0
    return float("nan")


mean_tok = sum(TOK.values()) / 3
a_cost = sum(A_COST.values()) / 3
print(
    f"mean reviewer output tokens/episode {mean_tok:.0f}; mean author cost/episode ${a_cost:.3f}"
)
print(
    "n per scenario per condition | episodes per arm | MDE from 33% (one scenario) | MDE from 67% (2 scenarios pooled)"
    " | author $ | GPU hours at 800 / 300 tok/s | GPU $"
)
for n in [10, 25, 50, 100]:
    eps = n * 2 * 3
    toks = eps * mean_tok
    h_fast, h_slow = toks / 800 / 3600, toks / 300 / 3600
    print(
        f"{n} | {eps} | +{100 * mde(1 / 3, n):.0f} pp | +{100 * mde(2 / 3, 2 * n):.0f} pp | ${eps * a_cost:.0f}"
        f" | {h_fast:.1f}-{h_slow:.1f} h | ${h_fast * H100_PER_H:.0f}-{h_slow * H100_PER_H:.0f}"
    )
