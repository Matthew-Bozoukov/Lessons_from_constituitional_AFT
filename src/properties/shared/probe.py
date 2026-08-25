# ABOUTME: The multivariate view of a property list: a cross-validated logistic probe from
# ABOUTME: binary property membership to the arm that produced a record, and to its outcome.

"""How much do the discovered properties actually account for?

Per-property tests (`shared/outcomes.py`) answer "does THIS property differ between the
arms?", one property at a time. They cannot answer the question a reader asks next: taken
together, do these properties explain what separates the two models, or are they a list of
small differences that misses the thing that matters?

That is a multivariate question, and the LessWrong post this module's method comes from
(`WAZWA6FPQvH8okouJ`, Engels/Chughtai/Nanda) already contains the right instrument for it —
its one quantitative experiment fits logistic regression on "a sparse binary vector with
ones for any present features". The post points it at predicting one channel's clusters
from another's. Pointed instead at the ARM, it becomes the model comparison; pointed at the
judged outcome, it becomes the multivariate version of the ablation shortlist.

Three numbers come out, and each answers something the per-property table cannot:

    auc          how separable the arms are in property space at all. Near 0.5 means the
                 properties, collectively, do not describe what differs between the models
                 — which would be a real and reportable null.
    n_selected   how FEW properties suffice. The L1 path is walked from strong to weak
                 regularisation, so "6 properties reach AUC 0.93" is a claim the univariate
                 table cannot make and a reader can check.
    permuted_auc the same pipeline on shuffled labels. With ~100 columns over ~500 rows an
                 impressive-looking AUC needs a floor stated next to it, and a null fitted
                 through the identical code path is the only floor worth trusting.

This is still correlational, and it is correlational in the same way as everything else
here: a probe that separates the arms perfectly has shown that the arms differ, not that
any property caused anything.
"""

from __future__ import annotations

import numpy as np

# Strong to weak. Walked in this order so `n_selected` is read off the FIRST model that
# reaches the accuracy plateau rather than the last, which is the minimal-set claim.
DEFAULT_C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
N_PERMUTATIONS = 50
FOLDS = 5


def membership_matrix(properties: list, member_indices: dict, n_records: int
                      ) -> tuple[np.ndarray, list[str]]:
    """The sparse binary design matrix: one row per record, one column per property.

    Args:
        properties: The exported rows, in export order — the column order.
        member_indices: property_id -> the record indices carrying it.
        n_records: Rows in the matrix.

    Returns:
        ((n_records x n_properties) float32 0/1 matrix, one label per column).
    """
    matrix = np.zeros((n_records, len(properties)), dtype=np.float32)
    for column, prop in enumerate(properties):
        for i in member_indices[prop.property_id]:
            matrix[int(i), column] = 1.0
    return matrix, [p.label for p in properties]


def _fit_once(matrix: np.ndarray, y: np.ndarray, c: float, seed: int, folds: int) -> dict:
    """Cross-validated fit at one regularisation strength.

    Args:
        matrix: The design matrix.
        y: Binary target.
        c: Inverse regularisation strength.
        seed: Fold seed.
        folds: CV folds.

    Returns:
        {"c", "auc", "f1", "accuracy", "n_selected"}.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores, predicted = [], np.zeros(len(y))
    for train, test in splitter.split(matrix, y):
        model = LogisticRegression(l1_ratio=1.0, C=c, solver="liblinear",
                                   class_weight="balanced", max_iter=2000,
                                   random_state=seed)
        model.fit(matrix[train], y[train])
        probability = model.predict_proba(matrix[test])[:, 1]
        predicted[test] = probability
        # A fold whose test split is single-class has no ROC to compute; skipping it is
        # right, and it only happens on tiny inputs.
        if len(set(y[test].tolist())) > 1:
            scores.append(roc_auc_score(y[test], probability))
    full = LogisticRegression(l1_ratio=1.0, C=c, solver="liblinear",
                              class_weight="balanced", max_iter=2000, random_state=seed)
    full.fit(matrix, y)
    return {"c": c,
            "auc": round(float(np.mean(scores)), 4) if scores else None,
            "f1": round(float(f1_score(y, (predicted >= 0.5).astype(int),
                                       zero_division=0)), 4),
            "accuracy": round(float(((predicted >= 0.5).astype(int) == y).mean()), 4),
            "n_selected": int((np.abs(full.coef_[0]) > 1e-8).sum())}


def probe(matrix: np.ndarray, y: np.ndarray, columns: list[str], name: str,
          seed: int = 0, folds: int = FOLDS, c_grid=DEFAULT_C_GRID,
          permutations: int = N_PERMUTATIONS) -> dict:
    """Fit the L1 path, pick the best point on it, and state the null beside it.

    Args:
        matrix: (n x p) binary membership.
        y: (n,) binary target.
        columns: Property labels, in column order.
        name: What is being predicted, for the report.
        seed: Random state for folds and permutations.
        folds: CV folds.
        c_grid: Inverse regularisation strengths, strong first.
        permutations: Shuffled-label refits at the chosen C; 0 skips the null.

    Returns:
        The probe record: path, best point, top coefficients, permutation null.

    Raises:
        ValueError: If the target has only one class — there is nothing to separate.
    """
    y = np.asarray(y).astype(int)
    if len(set(y.tolist())) < 2:
        raise ValueError(f"probe target {name!r} has one class; nothing to separate")

    path = [_fit_once(matrix, y, c, seed, folds) for c in c_grid]
    scored = [row for row in path if row["auc"] is not None]
    best = max(scored, key=lambda r: r["auc"]) if scored else path[-1]

    # The minimal set: the FIRST point on the path (strongest regularisation) that gets
    # within one point of AUC of the best. "6 properties reach 0.93" is the claim; the
    # best-AUC model is usually much less sparse and says less.
    minimal = next((row for row in scored if row["auc"] >= best["auc"] - 0.01), best)

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(l1_ratio=1.0, C=best["c"], solver="liblinear",
                               class_weight="balanced", max_iter=2000, random_state=seed)
    model.fit(matrix, y)
    weights = model.coef_[0]
    order = np.argsort(-np.abs(weights))
    top = [{"property": columns[i], "coefficient": round(float(weights[i]), 4)}
           for i in order if abs(weights[i]) > 1e-8][:20]

    null = None
    if permutations:
        rng = np.random.default_rng(seed)
        aucs = []
        for _ in range(permutations):
            shuffled = rng.permutation(y)
            row = _fit_once(matrix, shuffled, best["c"], seed, folds)
            if row["auc"] is not None:
                aucs.append(row["auc"])
        if aucs:
            null = {"n": len(aucs), "mean_auc": round(float(np.mean(aucs)), 4),
                    "p95_auc": round(float(np.percentile(aucs, 95)), 4),
                    # The share of shuffled fits that matched the real one. This is the
                    # p-value of the whole probe, and it is the number to quote.
                    "p_value": round(float((np.sum(np.asarray(aucs) >= best["auc"]) + 1)
                                           / (len(aucs) + 1)), 4)}
    return {"target": name, "n_records": int(matrix.shape[0]),
            "n_properties": int(matrix.shape[1]),
            "positive_rate": round(float(y.mean()), 4),
            "path": path, "best": best,
            "minimal": {"n_selected": minimal["n_selected"], "auc": minimal["auc"],
                        "c": minimal["c"]},
            "top_coefficients": top, "permutation_null": null,
            "params": {"seed": seed, "folds": folds, "l1_ratio": 1.0,
                       "solver": "liblinear", "class_weight": "balanced",
                       "c_grid": list(c_grid)}}


def report(probes: list[dict]) -> str:
    """A greppable mirror of the probes.

    Args:
        probes: The records `probe` returned.

    Returns:
        The markdown section.
    """
    lines = ["", "## Probes — what the property set accounts for", "",
             "Logistic regression (L1, balanced) on the binary property-membership "
             "matrix, 5-fold cross-validated. `minimal` is the sparsest point on the "
             "regularisation path within one AUC point of the best. `null` is the same "
             "pipeline on shuffled labels — read the AUC against it, not against 0.5.", "",
             "| predicting | positive rate | AUC | F1 | properties used | minimal set |"
             " null AUC | p |", "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for row in probes:
        best, minimal = row["best"], row["minimal"]
        null = row["permutation_null"] or {}
        lines.append(
            f"| {row['target']} | {row['positive_rate']:.1%} | "
            f"{(best['auc'] or 0):.3f} | {best['f1']:.3f} | {best['n_selected']} | "
            f"{minimal['n_selected']} @ AUC {(minimal['auc'] or 0):.3f} | "
            f"{null.get('mean_auc', float('nan')):.3f} | "
            f"{null.get('p_value', float('nan')):.3f} |")
    for row in probes:
        lines += ["", f"### Heaviest properties for `{row['target']}`", "",
                  "| property | coefficient |", "|---|--:|"]
        lines += [f"| {c['property']} | {c['coefficient']:+.3f} |"
                  for c in row["top_coefficients"][:12]]
    return "\n".join(lines) + "\n"
