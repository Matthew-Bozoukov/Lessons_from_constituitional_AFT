# ABOUTME: Did the ablation land? Measures the property's prevalence before and after with
# ABOUTME: the SAME detector, and gates the arm on the drop and on arm separability.

"""The check between an ablation and a pod.

An ablated corpus is a claim: "this corpus has less of property X than its control". Until
that claim is measured it is a hope, and training on a hope costs a GPU day and produces a
number nobody can interpret. So `verify` measures it, with the same detector that selected
the rows, and returns a gate verdict.

Three checks, in the order they can disqualify an arm:

1. **Prevalence drop.** Detector prevalence before vs after, each with a Wilson interval.
   The gate wants a drop of at least `min_drop` AND non-overlapping intervals — a drop
   whose intervals overlap is not distinguishable from sampling noise at this sample size,
   and the fix is a bigger sample, not a smaller threshold.

2. **Collateral.** The property was supposed to be the only thing that changed. A rewrite
   that also flipped decisions has ablated two things. Any property in `collateral:` is
   measured on both corpora too, and a collateral property that moved by more than
   `max_collateral_drift` fails the gate. This is where a second property from the same
   List of Properties earns its keep.

3. **Separability.** A bag-of-words classifier is trained to tell the two corpora apart.
   If it separates them at high AUC, something OTHER than the property distinguishes the
   arms — and the model will learn that instead. This check exists because it has already
   caught two corpora in this project: peer-critique separated at AUC 0.9973 (length
   alone got 0.85) and post-action-retrospection at 0.96 (contractions: "i'd" in 61% of
   one arm and 0% of the other), both found on 2026-08-17, and the second only after a
   model had been trained on it.

   The subtlety: high separability is EXPECTED and fine when the ablation deliberately
   rewrote text — the arms genuinely differ. What it must not do is separate on something
   unrelated to the property, so the check reports the top features and asks a human to
   read them, rather than failing outright, unless `strict_separability` is set.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from src.properties import block
from src.properties.registry import Property
from src.properties.shared import interpret as interpret_mod
from src.properties.sources.base import Record

# Below this the arm is not meaningfully different from its control.
DEFAULT_MIN_DROP = 0.20
# A property that was NOT targeted should not move much.
DEFAULT_MAX_COLLATERAL_DRIFT = 0.15
# Above this a classifier tells the corpora apart easily; read the features before training.
SEPARABILITY_WARN_AUC = 0.70


@dataclass
class Verification:
    """The verdict on one ablated corpus.

    Attributes:
        property_id: The targeted property.
        before: `interpret.prevalence` on the original corpus.
        after: The same on the ablated corpus.
        drop: before - after, in prevalence points.
        collateral: property_id -> {"before", "after", "drift"} for untargeted properties.
        separability: The classifier check's result, or None when it was skipped.
        passed: Whether the arm cleared the gate.
        failures: Why not, in words a config author can act on.
    """

    property_id: str
    before: dict
    after: dict
    drop: float | None
    collateral: dict = field(default_factory=dict)
    separability: dict | None = None
    passed: bool = False
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """The verification as a plain dict for run_meta and the report.

        Returns:
            The record.
        """
        return {"property_id": self.property_id, "before": self.before,
                "after": self.after, "drop": self.drop, "collateral": self.collateral,
                "separability": self.separability, "passed": self.passed,
                "failures": self.failures}

    def report(self) -> str:
        """A markdown mirror, so the numbers are greppable without a json reader.

        Returns:
            The markdown.
        """
        def line(name: str, measured: dict) -> str:
            if measured.get("prevalence") is None:
                return f"| {name} | — | — | {measured.get('n', 0)} |"
            return (f"| {name} | {measured['prevalence']:.1%} | "
                    f"{measured['ci_low']:.1%}–{measured['ci_high']:.1%} | "
                    f"{measured['hits']}/{measured['n']} |")

        lines = [f"# Ablation verification — `{self.property_id}`", "",
                 f"**{'PASSED' if self.passed else 'FAILED'}**", "",
                 "| corpus | prevalence | 95% CI | records |", "|---|--:|:--:|--:|",
                 line("before", self.before), line("after", self.after), "",
                 f"Drop: **{self.drop:+.1%}**" if self.drop is not None else "Drop: —", ""]
        if self.collateral:
            lines += ["## Collateral properties", "",
                      "| property | before | after | drift |", "|---|--:|--:|--:|"]
            lines += [f"| `{pid}` | {c['before']:.1%} | {c['after']:.1%} | "
                      f"{c['drift']:+.1%} |" for pid, c in self.collateral.items()]
            lines.append("")
        if self.separability:
            sep = self.separability
            lines += ["## Arm separability", ""]
            if sep.get("auc") is None:
                lines += [f"Not run: {sep.get('skipped', 'no result')}.", ""]
            else:
                lines += [f"A bag-of-words classifier separates the two corpora at "
                          f"**AUC {sep['auc']:.4f}** (gate warns above "
                          f"{SEPARABILITY_WARN_AUC}).", "",
                          "Top discriminating features:", ""]
                lines += [f"- `{f}` ({w:+.3f})" for f, w in sep["top_features"]]
                lines.append("")
        if self.failures:
            lines += ["## Why it failed", ""] + [f"- {f}" for f in self.failures]
        return "\n".join(lines) + "\n"


def _sample(records: list[Record], n: int, seed: int) -> list[Record]:
    """A reproducible sample of a corpus.

    Args:
        records: The corpus.
        n: Sample size.
        seed: Sampling seed.

    Returns:
        The sample, or everything when the corpus is smaller than `n`.
    """
    return records if len(records) <= n else random.Random(seed).sample(records, n)


def separability(before: list[str], after: list[str], seed: int = 0,
                 max_features: int = 5000) -> dict:
    """Train a bag-of-words classifier to tell the two corpora apart.

    A cheap version of the check that caught two corpora in this project. High AUC does not
    by itself condemn an arm — a rewrite ablation SHOULD change the text — but the top
    features say what the classifier is using, and if that is contractions or length rather
    than the property, the arms differ in a way the training will learn.

    Args:
        before: Texts from the original corpus.
        after: Texts from the ablated corpus.
        seed: Split and solver seed.
        max_features: Vocabulary cap.

    Returns:
        {"auc", "n_before", "n_after", "top_features"} — features signed towards the
        ABLATED corpus, most discriminating first. `{"skipped": ...}` when either side is
        too small for a 5-fold split: an AUC from eight documents says nothing, and
        reporting one would be worse than reporting none.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    if min(len(before), len(after)) < 10:
        return {"skipped": f"{len(before)} vs {len(after)} documents is too few for a "
                           "5-fold separability check", "auc": None,
                "n_before": len(before), "n_after": len(after), "top_features": []}
    texts = list(before) + list(after)
    labels = np.array([0] * len(before) + [1] * len(after))
    vectoriser = TfidfVectorizer(max_features=max_features, lowercase=True,
                                 ngram_range=(1, 2), min_df=2)
    matrix = vectoriser.fit_transform(texts)
    model = LogisticRegression(max_iter=2000, random_state=seed)
    probabilities = cross_val_predict(model, matrix, labels, cv=5,
                                      method="predict_proba")[:, 1]
    auc = float(roc_auc_score(labels, probabilities))
    model.fit(matrix, labels)
    names = np.asarray(vectoriser.get_feature_names_out())
    weights = model.coef_[0]
    order = np.argsort(-np.abs(weights))[:15]
    return {"auc": round(auc, 4), "n_before": len(before), "n_after": len(after),
            "top_features": [(str(names[i]), round(float(weights[i]), 4))
                             for i in order]}


def verify(prop: Property, before_records: list[Record], after_records: list[Record],
           cfg=None, collateral_properties: list[Property] | None = None) -> Verification:
    """Measure the property before and after, and gate the arm.

    Args:
        prop: The targeted property.
        before_records: The original corpus.
        after_records: The ablated corpus, loaded back through the same source adapter so
            the two are read identically.
        cfg: The verify config block. Keys: `sample` (default 200), `seed`, `min_drop`,
            `max_collateral_drift`, `check_separability` (default True),
            `strict_separability` (default False), `detector {model, workers}`.
        collateral_properties: Untargeted properties that must NOT move.

    Returns:
        The Verification.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(cfg or {})
    sample_n = int(cfg.get("sample", 200))
    seed = int(cfg.get("seed", 0))
    min_drop = float(cfg.get("min_drop", DEFAULT_MIN_DROP))
    max_drift = float(cfg.get("max_collateral_drift", DEFAULT_MAX_COLLATERAL_DRIFT))
    detector_cfg = block(cfg, "detector")

    before_sample = _sample(before_records, sample_n, seed)
    after_sample = _sample(after_records, sample_n, seed)

    measured_before = interpret_mod.prevalence(interpret_mod.detect(
        before_sample, prop.label, prop.detector, channel=prop.channel, **detector_cfg))
    measured_after = interpret_mod.prevalence(interpret_mod.detect(
        after_sample, prop.label, prop.detector, channel=prop.channel, **detector_cfg))

    failures: list[str] = []
    drop = None
    if measured_before["prevalence"] is None or measured_after["prevalence"] is None:
        failures.append("the detector could not be measured on one of the corpora")
    else:
        drop = round(measured_before["prevalence"] - measured_after["prevalence"], 4)
        if drop < min_drop:
            failures.append(
                f"prevalence dropped {drop:+.1%}, below the {min_drop:.0%} gate: this arm "
                "is too close to its control to interpret, whatever the eval shows")
        if measured_after["ci_high"] >= measured_before["ci_low"]:
            failures.append(
                f"the confidence intervals overlap ({measured_before['ci_low']:.1%}–"
                f"{measured_before['ci_high']:.1%} vs {measured_after['ci_low']:.1%}–"
                f"{measured_after['ci_high']:.1%}): the drop is not distinguishable from "
                f"sampling noise at n={measured_after['n']}. Raise `sample`, do not lower "
                "the gate.")

    collateral: dict[str, dict] = {}
    for other in collateral_properties or []:
        if other.property_id == prop.property_id:
            continue
        other_before = interpret_mod.prevalence(interpret_mod.detect(
            before_sample, other.label, other.detector, channel=other.channel,
            **detector_cfg))
        other_after = interpret_mod.prevalence(interpret_mod.detect(
            after_sample, other.label, other.detector, channel=other.channel,
            **detector_cfg))
        if other_before["prevalence"] is None or other_after["prevalence"] is None:
            continue
        drift = round(other_before["prevalence"] - other_after["prevalence"], 4)
        collateral[other.property_id] = {"before": other_before["prevalence"],
                                         "after": other_after["prevalence"],
                                         "drift": drift, "label": other.label}
        if abs(drift) > max_drift:
            failures.append(
                f"collateral property `{other.property_id}` ({other.label}) moved "
                f"{drift:+.1%}, above the {max_drift:.0%} limit: the ablation changed "
                "more than one thing, so a difference between the arms cannot be "
                "attributed to the targeted property alone")

    separation = None
    if bool(cfg.get("check_separability", True)):
        channel = prop.channel
        separation = separability([r.channel(channel) for r in before_sample],
                                  [r.channel(channel) for r in after_sample], seed)
        if separation.get("skipped"):
            print(f"!!! separability check skipped: {separation['skipped']}")
        elif separation["auc"] >= SEPARABILITY_WARN_AUC:
            message = (
                f"a bag-of-words classifier separates the arms at AUC "
                f"{separation['auc']:.4f} on features "
                f"{[f for f, _ in separation['top_features'][:5]]}. Read those: if they "
                "are the property, this is expected; if they are length, contractions or "
                "an author's tics, the model will learn the arm from them instead of from "
                "the property (peer-critique, 2026-08-17).")
            if bool(cfg.get("strict_separability", False)):
                failures.append(message)
            else:
                print(f"!!! {message}")

    return Verification(property_id=prop.property_id, before=measured_before,
                        after=measured_after, drop=drop, collateral=collateral,
                        separability=separation, passed=not failures, failures=failures)
