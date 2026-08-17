# ABOUTME: One-off cross-check of debate_speeches/stats.py against scipy + sklearn reference
# ABOUTME: implementations. Run: uv run --with scipy --with scikit-learn python scratch/check_deliberation_stats.py

"""scipy is deliberately NOT a dependency of this repo (see the stats module docstring), so
this verification runs with transient installs and is not part of the test suite. It exists
because a hand-rolled tau-b already shipped one wrong denominator; the unit tests pin the
behaviour, this pins it against the reference.
"""

import random

from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

from src.eval.deliberation.debate_speeches.stats import (
    kendall_tau_b,
    pearson,
    quadratic_weighted_kappa,
    spearman,
)

rng = random.Random(7)
worst = {"tau_b": 0.0, "spearman": 0.0, "pearson": 0.0, "qwk": 0.0}

for _ in range(200):
    n = rng.randint(5, 60)
    # Tie-heavy on purpose: the model emits integers 1-5 and the human target is a mean of
    # ~15-30 ratings, which is the shape the real eval produces.
    model = [rng.randint(1, 5) for _ in range(n)]
    human = [round(rng.uniform(1, 5), 3) for _ in range(n)]
    human_int = [min(5, max(1, round(x))) for x in human]

    worst["tau_b"] = max(worst["tau_b"], abs(
        kendall_tau_b(model, human) - kendalltau(model, human, variant="b").statistic))
    worst["spearman"] = max(worst["spearman"], abs(
        spearman(model, human) - spearmanr(model, human).statistic))
    worst["pearson"] = max(worst["pearson"], abs(
        pearson(model, human) - pearsonr(model, human).statistic))
    worst["qwk"] = max(worst["qwk"], abs(
        quadratic_weighted_kappa(model, human_int)
        - cohen_kappa_score(model, human_int, weights="quadratic", labels=[1, 2, 3, 4, 5])))

print("max abs deviation vs scipy/sklearn over 200 random tie-heavy series:")
for name, value in worst.items():
    print(f"  {name:9s} {value:.2e}")
# 1e-3 because our functions round to 4 dp; anything larger is a real disagreement.
assert all(value < 1e-3 for value in worst.values()), worst
print("OK — all four match the reference implementations within rounding.")
