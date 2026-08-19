# ABOUTME: Cross a property against the outcomes of the records that carry it — WITHIN
# ABOUTME: each arm, never pooled — and correct the resulting ranking for multiplicity.

"""What makes a cluster list a RANKING instead of a list.

Corpus-side discovery answers "here is a thing the data does" and stops there. Every
cluster comes back equally weighted, nothing says which one is worth a training run, and
picking an ablation target is guesswork — the honest criticism of the method, and the
reason this file exists.

Rollouts fix it because rollouts have outcomes. Each one was judged: it violated or it did
not. So when a producer groups ROLLOUT traces rather than corpus rows, every group can
carry a number — how often the model's reasoning fell into this group, and how often
reasoning in this group stayed aligned. That number is what an ablation shortlist is
ordered by.

Two things make that number wrong if you compute it the obvious way.

**Simpson's paradox.** The arms have different base violation rates by construction — that
is the experiment. A property common in the arm that was already the most aligned looks
protective no matter what it is, and a pooled rate reports exactly that confound as if it
were a finding. So every rate here is computed WITHIN an arm and only then combined, and
the pooled number is emitted alongside, explicitly flagged, so a reader can see the gap
between the two rather than being handed only the flattering one.

**Multiplicity.** Tens of clusters crossed against one binary outcome will produce a few
"significant" differences from nothing at all. `rank()` therefore reports
Benjamini-Hochberg q-values over the whole family of clusters tested, not per-cluster
p-values, and the docstrings say what the output is for: a shortlist of ablation
candidates. The ablation is what makes a property causal. This only stops a retrain being
spent on the wrong one.
"""

from __future__ import annotations

import math

# Fisher's exact test is exact but O(n) in table margins per cluster and we run tens of
# clusters over thousands of rollouts; the normal approximation to the difference of two
# proportions is what the ranking actually needs, and it is only ever used for ORDERING.
MIN_ARM_RECORDS = 20


def _rate(hits: int, n: int) -> float | None:
    """A proportion, or None when there is nothing to take a proportion of.

    Args:
        hits: Numerator.
        n: Denominator.

    Returns:
        hits / n, or None when n is 0.
    """
    return round(hits / n, 4) if n else None


def _two_proportion_z(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float | None:
    """Pooled-variance z for the difference between two proportions.

    Args:
        hits_a: Successes in group A (members).
        n_a: Size of group A.
        hits_b: Successes in group B (non-members).
        n_b: Size of group B.

    Returns:
        The z statistic, or None when either group is empty or the pooled rate is
        degenerate (every record on both sides has the same outcome — no contrast to test).
    """
    if n_a == 0 or n_b == 0:
        return None
    pooled = (hits_a + hits_b) / (n_a + n_b)
    if pooled in (0.0, 1.0):
        return None
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    return (hits_a / n_a - hits_b / n_b) / se if se else None


def _two_sided_p(z: float) -> float:
    """Two-sided normal p-value.

    Args:
        z: The statistic.

    Returns:
        The p-value.
    """
    return round(math.erfc(abs(z) / math.sqrt(2)), 6)


def by_arm(records, member_ids: set[str], arm_key: str = "arm",
           outcome_key: str = "violation") -> dict:
    """Cross one group's membership against the outcome, within each arm.

    The comparison is members vs NON-members OF THE SAME ARM, which is the only version of
    it that is not confounded by the arms' different base rates. `pooled` is computed too
    and carries `confounded: true`, because the difference between the two numbers is
    itself worth looking at — a property whose pooled lift is large and whose within-arm
    lift is nil IS the paradox, visible.

    Args:
        records: Every record in the analysis, in any order. Each needs
            `metadata[arm_key]`; a record whose `outcome` is None is excluded from the
            rates (and counted in `n_unjudged`) rather than being read as compliant.
        member_ids: record_ids of the records in this group.
        arm_key: Metadata field naming the model variant.
        outcome_key: The boolean field of `Record.outcome` to rate.

    Returns:
        {"arms": {arm: {...}}, "pooled": {...}, "n_unjudged": int}, where each arm's block
        carries `n_members`, `n_arm`, `rate_in`, `rate_out`, `lift`, `p` — `lift` being
        the within-arm difference in outcome rate, members minus non-members.
    """
    per_arm: dict[str, dict[str, int]] = {}
    unjudged = 0
    for record in records:
        if record.outcome is None or record.outcome.get(outcome_key) is None:
            unjudged += 1
            continue
        arm = str(record.metadata.get(arm_key, "all"))
        counts = per_arm.setdefault(arm, {"in_n": 0, "in_hit": 0,
                                          "out_n": 0, "out_hit": 0})
        side = "in" if record.record_id in member_ids else "out"
        counts[f"{side}_n"] += 1
        counts[f"{side}_hit"] += int(bool(record.outcome[outcome_key]))

    arms = {}
    for arm, c in sorted(per_arm.items()):
        z = _two_proportion_z(c["in_hit"], c["in_n"], c["out_hit"], c["out_n"])
        rate_in, rate_out = _rate(c["in_hit"], c["in_n"]), _rate(c["out_hit"], c["out_n"])
        arms[arm] = {
            "n_members": c["in_n"], "n_arm": c["in_n"] + c["out_n"],
            "share_of_arm": _rate(c["in_n"], c["in_n"] + c["out_n"]),
            "rate_in": rate_in, "rate_out": rate_out,
            "lift": None if rate_in is None or rate_out is None
            else round(rate_in - rate_out, 4),
            "z": None if z is None else round(z, 3),
            "p": None if z is None else _two_sided_p(z),
            # An arm this small cannot separate anything; saying so on the row stops a
            # large lift computed off four rollouts leading a shortlist.
            "underpowered": c["in_n"] < MIN_ARM_RECORDS,
        }

    total = {k: sum(c[k] for c in per_arm.values())
             for k in ("in_n", "in_hit", "out_n", "out_hit")}
    pooled_in, pooled_out = (_rate(total["in_hit"], total["in_n"]),
                             _rate(total["out_hit"], total["out_n"]))
    return {
        "arms": arms,
        "pooled": {
            "n_members": total["in_n"], "rate_in": pooled_in, "rate_out": pooled_out,
            "lift": None if pooled_in is None or pooled_out is None
            else round(pooled_in - pooled_out, 4),
            # Read the arms, not this. Kept because the gap between the two IS the
            # diagnostic for a property that is really just an arm marker.
            "confounded": True,
        },
        "n_unjudged": unjudged,
    }


def combined_lift(crosstab: dict, min_arm_records: int = MIN_ARM_RECORDS) -> dict:
    """Combine one group's within-arm lifts into the single number a shortlist sorts on.

    Every arm that can measure a lift contributes, weighted by how many members it has.
    Averaging arm RATES instead — or pooling the records — reintroduces the base-rate
    confound the within-arm split existed to remove.

    Small arms are weighted, NOT discarded, and that is a correction. Dropping them looks
    conservative and is not: on the 2026-08-19 da716 run, stratifying 275 rollouts by ODCV
    condition split several properties into two strata of 10-19, both showing the same
    large effect in the same direction — "states ethical caution but proceeds anyway" at
    +49% and +73% — and a rule that discarded strata under 20 returned `None` for it.
    That reports a consistent finding as unmeasurable. Weighting by member count already
    stops a four-rollout stratum dominating; the flag below is what warns a reader.

    Evidence across arms is combined with a weighted Stouffer z rather than the smallest
    per-arm p, because two strata agreeing weakly is stronger evidence than either alone
    and taking the minimum throws that away.

    Args:
        crosstab: The output of `by_arm`.
        min_arm_records: Arms below this are flagged `underpowered` and counted, but still
            contribute at their weight.

    Returns:
        {"lift", "n_arms", "n_arms_underpowered", "min_p"} — `min_p` being the combined
        p-value `rank` corrects for multiplicity (named for the field `rank` reads).
    """
    usable = [a for a in crosstab["arms"].values() if a["lift"] is not None]
    underpowered = sum(1 for a in usable if a["n_members"] < min_arm_records)
    if not usable:
        return {"lift": None, "n_arms": 0, "n_arms_underpowered": 0, "min_p": None}
    weight = sum(a["n_members"] for a in usable)
    lift = sum(a["lift"] * a["n_members"] for a in usable) / weight
    scored = [a for a in usable if a["z"] is not None]
    combined = None
    if scored:
        root = math.sqrt(sum(a["n_members"] for a in scored))
        z = sum(a["z"] * math.sqrt(a["n_members"]) for a in scored) / root
        combined = _two_sided_p(z)
    return {"lift": round(lift, 4), "n_arms": len(usable),
            "n_arms_underpowered": underpowered, "min_p": combined}


def benjamini_hochberg(p_values: dict[str, float | None], fdr: float = 0.10) -> dict:
    """Benjamini-Hochberg q-values over a family of per-group tests.

    Tens of clusters crossed against one binary outcome will hand you a few p < 0.05 from
    nothing. BH controls the false-discovery rate across the whole family, which is the
    right correction here: the output is a shortlist and a few false candidates cost a
    reading, not a wrong conclusion. Bonferroni would be the wrong instrument — it controls
    the chance of ANY false candidate, and at these group counts it would empty the list.

    Args:
        p_values: group key -> p-value. A None p (nothing to test) is passed through as a
            None q rather than dropped, so every group keeps a row.
        fdr: The target false-discovery rate.

    Returns:
        group key -> {"p", "q", "significant"}.
    """
    tested = sorted(((k, p) for k, p in p_values.items() if p is not None),
                    key=lambda kv: kv[1])
    m = len(tested)
    out: dict[str, dict] = {k: {"p": None, "q": None, "significant": False}
                            for k in p_values}
    running = 1.0
    # Walk from the largest p down, keeping the running minimum: BH's q is monotone in p,
    # and computing it in one pass this way is what makes it so.
    for rank_index, (key, p) in reversed(list(enumerate(tested, start=1))):
        running = min(running, p * m / rank_index)
        out[key] = {"p": p, "q": round(running, 6), "significant": running <= fdr}
    return out


def rank(crosstabs: dict[str, dict], fdr: float = 0.10,
         min_arm_records: int = MIN_ARM_RECORDS) -> list[dict]:
    """Order groups as ablation candidates: within-arm lift, corrected for multiplicity.

    This is the deliverable of the whole rollout-side analysis. Read it as a RANKING OF
    CANDIDATES, not as a result: it is correlational, and a property can lead this list
    because it marks an arm rather than because it does anything. The ablation is what
    makes it causal.

    Args:
        crosstabs: group key -> the output of `by_arm`.
        fdr: Target false-discovery rate for the BH correction.
        min_arm_records: Below this an arm is flagged underpowered; it still contributes.

    Returns:
        One row per group, most protective first (most negative lift — reasoning in this
        group violated LESS than its arm's other reasoning). Each row carries `lift`,
        `p`, `q`, `significant`, `n_arms`, and the per-arm block for inspection.
    """
    summaries = {key: combined_lift(cross, min_arm_records)
                 for key, cross in crosstabs.items()}
    corrected = benjamini_hochberg({k: s["min_p"] for k, s in summaries.items()}, fdr)
    rows = [{"group": key, **summaries[key], **corrected[key],
             "arms": crosstabs[key]["arms"],
             "pooled_lift": crosstabs[key]["pooled"]["lift"]}
            for key in crosstabs]
    return sorted(rows, key=lambda r: (r["lift"] is None, r["lift"] or 0.0))
