# ABOUTME: Cross a property against the outcomes of the records that carry it, and against
# ABOUTME: the ARM that produced them — always within a stratum, never pooled, BH-corrected.

"""What makes a cluster list a RANKING instead of a list, and a COMPARISON instead of one.

Corpus-side discovery answers "here is a thing the data does" and stops there. Every
cluster comes back equally weighted, nothing says which one is worth a training run, and
picking an ablation target is guesswork — the honest criticism of the method, and the
reason this file exists.

Rollouts fix it because rollouts have outcomes, and because rollouts come from a MODEL. So
there are two questions to ask of every group, and this module answers both:

    by_stratum / rank        does reasoning in this group go with the model VIOLATING?
    contrast_arms / rank_contrasts
                             is this group more common in one model than the other?

The first is the ablation shortlist. The second is the model comparison — what actually
differs between a fine-tune and its control — and it is the question a single-arm run
cannot ask at all.

Three things make either number wrong if you compute it the obvious way.

**Simpson's paradox.** The arms have different base violation rates by construction — that
is the experiment. A property common in the arm that was already the most aligned looks
protective no matter what it is, and a pooled rate reports exactly that confound as if it
were a finding. So every rate here is computed WITHIN a stratum and only then combined, and
the pooled number is emitted alongside, explicitly flagged, so a reader can see the gap
between the two rather than being handed only the flattering one.

**A stratum is not always one field.** ODCV runs each model under two conditions with
different base rates (measured 2026-08-19: incentivized 23.2%, mandated 12.4%), so with two
arms the thing you must not pool across is the PAIR. `strata_key` therefore takes a list as
readily as a string, and builds the composite. That is why this module says "stratum"
rather than "arm" throughout: an arm is the commonest stratum, not the only one.

**Multiplicity.** Tens of clusters crossed against one binary outcome will produce a few
"significant" differences from nothing at all. Both `rank()` and `rank_contrasts()`
therefore report Benjamini-Hochberg q-values over the whole family of clusters tested, not
per-cluster p-values — one family per outcome field, one family for the arm contrast. The
docstrings say what the output is for: a shortlist of ablation candidates. The ablation is
what makes a property causal. This only stops a retrain being spent on the wrong one.
"""

from __future__ import annotations

import math

# Fisher's exact test is exact but O(n) in table margins per cluster and we run tens of
# clusters over thousands of rollouts; the normal approximation to the difference of two
# proportions is what the ranking actually needs, and it is only ever used for ORDERING.
MIN_STRATUM_RECORDS = 20


def _rate(hits: int, n: int) -> float | None:
    """A proportion, or None when there is nothing to take a proportion of.

    Args:
        hits: Numerator.
        n: Denominator.

    Returns:
        hits / n, or None when n is 0.
    """
    return round(hits / n, 4) if n else None


def stratum(record, keys) -> str:
    """The stratum label one record belongs to.

    Args:
        record: The Record.
        keys: A metadata key, or a list of them for a composite stratum.

    Returns:
        The label — the bare value for one key, `a=x|b=y` for several, so a composite
        stratum stays readable in a report and cannot collide with a single-key one.
    """
    if isinstance(keys, str):
        return str(record.metadata.get(keys, "all"))
    return "|".join(f"{k}={record.metadata.get(k, 'all')}" for k in keys)


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


def _combine(rows: list[dict], value_key: str,
             min_records: int) -> tuple[float | None, float | None, int, int]:
    """Combine per-stratum differences into the single number a shortlist sorts on.

    Every stratum that can measure a difference contributes, weighted by Cochran's
    `w = n_a * n_b / (n_a + n_b)` — the weight for pooling a difference of proportions,
    and the one that gives a stratum influence in proportion to how much it can actually
    resolve. Averaging stratum RATES instead, or pooling the records, reintroduces the
    base-rate confound the stratification existed to remove.

    Small strata are weighted, NOT discarded, and that is a correction. Dropping them looks
    conservative and is not: on the 2026-08-19 da716 run, stratifying 275 rollouts by ODCV
    condition split several properties into two strata of 10-19, both showing the same
    large effect in the same direction — "states ethical caution but proceeds anyway" at
    +49% and +73% — and a rule that discarded strata under 20 returned `None` for it.
    That reports a consistent finding as unmeasurable. The weight already stops a
    four-rollout stratum dominating; the underpowered count is what warns a reader.

    Evidence across strata is combined with a weighted Stouffer z rather than the smallest
    per-stratum p, because two strata agreeing weakly is stronger evidence than either
    alone and taking the minimum throws that away.

    Args:
        rows: Per-stratum blocks, each carrying `value_key`, `z`, and `weight`.
        value_key: The field holding the stratum's difference.
        min_records: Strata whose smaller side is below this are counted as underpowered.

    Returns:
        (combined difference, combined p, n strata used, n of those underpowered).
    """
    usable = [r for r in rows if r[value_key] is not None and r["weight"] > 0]
    if not usable:
        return None, None, 0, 0
    underpowered = sum(1 for r in usable if r["n_min"] < min_records)
    total = sum(r["weight"] for r in usable)
    value = sum(r[value_key] * r["weight"] for r in usable) / total
    scored = [r for r in usable if r["z"] is not None]
    combined = None
    if scored:
        root = math.sqrt(sum(r["weight"] for r in scored))
        z = sum(r["z"] * math.sqrt(r["weight"]) for r in scored) / root
        combined = _two_sided_p(z)
    return round(value, 4), combined, len(usable), underpowered


def by_stratum(records, member_ids: set[str], strata_key="arm",
               outcome_key: str = "violation") -> dict:
    """Cross one group's membership against the outcome, within each stratum.

    The comparison is members vs NON-members OF THE SAME STRATUM, which is the only version
    of it that is not confounded by the strata's different base rates. `pooled` is computed
    too and carries `confounded: true`, because the difference between the two numbers is
    itself worth looking at — a property whose pooled lift is large and whose within-stratum
    lift is nil IS the paradox, visible.

    Args:
        records: Every record in the analysis, in any order. Each needs the metadata
            `strata_key` names; a record whose `outcome` is None is excluded from the
            rates (and counted in `n_unjudged`) rather than being read as compliant.
        member_ids: record_ids of the records in this group.
        strata_key: Metadata field, or list of fields, defining the stratum.
        outcome_key: The boolean field of `Record.outcome` to rate.

    Returns:
        {"strata": {label: {...}}, "pooled": {...}, "n_unjudged": int}, where each
        stratum's block carries `n_members`, `n_stratum`, `rate_in`, `rate_out`, `lift` —
        the within-stratum difference in outcome rate, members minus non-members.
    """
    per: dict[str, dict[str, int]] = {}
    unjudged = 0
    for record in records:
        if record.outcome is None or record.outcome.get(outcome_key) is None:
            unjudged += 1
            continue
        counts = per.setdefault(stratum(record, strata_key),
                                {"in_n": 0, "in_hit": 0, "out_n": 0, "out_hit": 0})
        side = "in" if record.record_id in member_ids else "out"
        counts[f"{side}_n"] += 1
        counts[f"{side}_hit"] += int(bool(record.outcome[outcome_key]))

    strata = {}
    for label, c in sorted(per.items()):
        z = _two_proportion_z(c["in_hit"], c["in_n"], c["out_hit"], c["out_n"])
        rate_in, rate_out = _rate(c["in_hit"], c["in_n"]), _rate(c["out_hit"], c["out_n"])
        strata[label] = {
            "n_members": c["in_n"], "n_stratum": c["in_n"] + c["out_n"],
            "share_of_stratum": _rate(c["in_n"], c["in_n"] + c["out_n"]),
            "rate_in": rate_in, "rate_out": rate_out,
            "lift": None if rate_in is None or rate_out is None
            else round(rate_in - rate_out, 4),
            "z": None if z is None else round(z, 3),
            "p": None if z is None else _two_sided_p(z),
            "weight": c["in_n"] * c["out_n"] / (c["in_n"] + c["out_n"])
            if c["in_n"] + c["out_n"] else 0.0,
            "n_min": min(c["in_n"], c["out_n"]),
            # A stratum this small cannot separate anything; saying so on the row stops a
            # large lift computed off four rollouts leading a shortlist.
            "underpowered": c["in_n"] < MIN_STRATUM_RECORDS,
        }

    total = {k: sum(c[k] for c in per.values())
             for k in ("in_n", "in_hit", "out_n", "out_hit")}
    pooled_in, pooled_out = (_rate(total["in_hit"], total["in_n"]),
                             _rate(total["out_hit"], total["out_n"]))
    return {
        "strata": strata,
        "pooled": {
            "n_members": total["in_n"], "rate_in": pooled_in, "rate_out": pooled_out,
            "lift": None if pooled_in is None or pooled_out is None
            else round(pooled_in - pooled_out, 4),
            # Read the strata, not this. Kept because the gap between the two IS the
            # diagnostic for a property that is really just a stratum marker.
            "confounded": True,
        },
        "n_unjudged": unjudged,
    }


def combined_lift(crosstab: dict, min_stratum_records: int = MIN_STRATUM_RECORDS) -> dict:
    """Combine one group's within-stratum lifts into the number a shortlist sorts on.

    Args:
        crosstab: The output of `by_stratum`.
        min_stratum_records: Strata below this are flagged and counted, but still
            contribute at their weight.

    Returns:
        {"lift", "n_strata", "n_strata_underpowered", "min_p"} — `min_p` being the
        combined p-value `rank` corrects for multiplicity (named for the field `rank`
        reads).
    """
    lift, p, n, underpowered = _combine(list(crosstab["strata"].values()), "lift",
                                        min_stratum_records)
    return {"lift": lift, "n_strata": n, "n_strata_underpowered": underpowered,
            "min_p": p}


def contrast_arms(records, member_ids: set[str], focus: str, reference: str,
                  arm_key: str = "arm", strata_key="condition") -> dict:
    """How much more common this group is in ONE model than in another.

    This is the model comparison, and it is a different question from `by_stratum`. There,
    the contrast is members against non-members and the outcome is the judge's verdict.
    Here, the contrast is one arm against another and the "outcome" is carrying the
    property at all: what does the fine-tune DO that its control does not?

    Stratification does the same job it does above, for the same reason and against a
    different confound. Two arms evaluated on ODCV do not necessarily run the same
    scenarios in the same proportions — the 2026-08-08 and 2026-08-19 runs excluded 10 and
    15 scenarios respectively — so a property common in the scenarios only one arm ran
    would read as a property of that model. Stratifying on `condition` removes the
    condition imbalance; stratifying on `cell` removes the scenario imbalance outright, at
    the cost of thin strata. Both are worth running, which is why the stratum is an
    argument.

    Every record counts here, judged or not: carrying a property is not something the judge
    decides, so an unjudged rollout is still evidence about what its model does.

    Args:
        records: Every record in the analysis.
        member_ids: record_ids of the records in this group.
        focus: The arm whose prevalence is reported first; `delta` is focus MINUS
            reference, so a positive delta means "more common in the focus arm".
        reference: The arm compared against.
        arm_key: Metadata field naming the arm.
        strata_key: Metadata field, or list of fields, defining the stratum.

    Returns:
        {"strata": {...}, "pooled": {...}, "prevalence": {arm: rate}, "delta", "p",
         "n_strata", "n_strata_underpowered"}. A stratum in which only one arm has records
        contributes nothing and is counted in `n_strata_one_armed`.

    Raises:
        ValueError: If either arm has no records at all — a contrast against an absent arm
            would report the focus arm's own prevalence as a difference.
    """
    per: dict[str, dict[str, int]] = {}
    totals = {focus: [0, 0], reference: [0, 0]}
    for record in records:
        arm = str(record.metadata.get(arm_key))
        if arm not in (focus, reference):
            continue
        hit = int(record.record_id in member_ids)
        totals[arm][0] += 1
        totals[arm][1] += hit
        counts = per.setdefault(stratum(record, strata_key),
                                {f"{focus}_n": 0, f"{focus}_hit": 0,
                                 f"{reference}_n": 0, f"{reference}_hit": 0})
        counts[f"{arm}_n"] += 1
        counts[f"{arm}_hit"] += hit

    missing = [a for a in (focus, reference) if totals[a][0] == 0]
    if missing:
        raise ValueError(
            f"no records carry {arm_key}={missing} — a contrast needs both arms. Present: "
            f"{sorted({str(r.metadata.get(arm_key)) for r in records})}")

    strata, one_armed = {}, 0
    for label, c in sorted(per.items()):
        n_f, hit_f = c[f"{focus}_n"], c[f"{focus}_hit"]
        n_r, hit_r = c[f"{reference}_n"], c[f"{reference}_hit"]
        if not n_f or not n_r:
            one_armed += 1
            continue
        z = _two_proportion_z(hit_f, n_f, hit_r, n_r)
        strata[label] = {
            "n_focus": n_f, "n_reference": n_r,
            "prevalence_focus": _rate(hit_f, n_f),
            "prevalence_reference": _rate(hit_r, n_r),
            "delta": round(hit_f / n_f - hit_r / n_r, 4),
            "z": None if z is None else round(z, 3),
            "p": None if z is None else _two_sided_p(z),
            "weight": n_f * n_r / (n_f + n_r),
            "n_min": min(n_f, n_r),
        }

    delta, p, n_strata, underpowered = _combine(list(strata.values()), "delta",
                                                MIN_STRATUM_RECORDS)
    prevalence = {arm: _rate(hits, n) for arm, (n, hits) in totals.items()}
    pooled = (None if prevalence[focus] is None or prevalence[reference] is None
              else round(prevalence[focus] - prevalence[reference], 4))
    return {
        "focus": focus, "reference": reference,
        "strata_key": strata_key if isinstance(strata_key, str) else list(strata_key),
        "prevalence": prevalence,
        "n_records": {arm: n for arm, (n, _) in totals.items()},
        "delta": delta, "p": p,
        "n_strata": n_strata, "n_strata_underpowered": underpowered,
        "n_strata_one_armed": one_armed,
        # The unstratified difference. Printed so the gap between it and `delta` is
        # visible; it is not a fallback, it is the confound the stratification removes.
        "pooled_delta_confounded": pooled,
        "by_stratum": strata,
    }


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
         min_stratum_records: int = MIN_STRATUM_RECORDS) -> list[dict]:
    """Order groups as ablation candidates: within-stratum lift, corrected for multiplicity.

    Read it as a RANKING OF CANDIDATES, not as a result: it is correlational, and a
    property can lead this list because it marks a stratum rather than because it does
    anything. The ablation is what makes it causal.

    Args:
        crosstabs: group key -> the output of `by_stratum`.
        fdr: Target false-discovery rate for the BH correction.
        min_stratum_records: Below this a stratum is flagged underpowered; it still
            contributes.

    Returns:
        One row per group, most protective first (most negative lift — reasoning in this
        group violated LESS than its stratum's other reasoning). Each row carries `lift`,
        `p`, `q`, `significant`, `n_strata`, and the per-stratum block for inspection.
    """
    summaries = {key: combined_lift(cross, min_stratum_records)
                 for key, cross in crosstabs.items()}
    corrected = benjamini_hochberg({k: s["min_p"] for k, s in summaries.items()}, fdr)
    rows = [{"group": key, **summaries[key], **corrected[key],
             "strata": crosstabs[key]["strata"],
             "pooled_lift": crosstabs[key]["pooled"]["lift"]}
            for key in crosstabs]
    return sorted(rows, key=lambda r: (r["lift"] is None, r["lift"] or 0.0))


def rank_contrasts(contrasts: dict[str, dict], fdr: float = 0.10) -> list[dict]:
    """Order groups by how much they separate the two arms, corrected for multiplicity.

    The deliverable of the model comparison: the properties one model has and the other
    does not. Sorted most-enriched-in-the-focus-arm first, so the two ends of the list are
    "what the fine-tune added" and "what it removed".

    Args:
        contrasts: group key -> the output of `contrast_arms`.
        fdr: Target false-discovery rate over the family of groups.

    Returns:
        One row per group, largest positive delta first.
    """
    corrected = benjamini_hochberg({k: c["p"] for k, c in contrasts.items()}, fdr)
    rows = [{"group": key,
             **{k: v for k, v in contrasts[key].items() if k != "by_stratum"},
             **corrected[key]}
            for key in contrasts]
    return sorted(rows, key=lambda r: (r["delta"] is None, -(r["delta"] or 0.0)))
