# ABOUTME: Drop the rows carrying a property, or SPLIT the corpus into has-X and lacks-X
# ABOUTME: halves and train both — the cheapest ablation, with the worst confounds.

"""Select on the property instead of editing it.

Three modes, and the choice between them is a choice about which confound you accept:

    drop        remove every row the detector flags. Cheapest. Changes the corpus SIZE,
                so the ablated arm trains on less data than its control and any difference
                is confounded with that. `rebalance: true` fixes it by dropping an equal
                number of unflagged rows, which costs data but keeps the comparison clean.
    downsample  keep a fraction of the flagged rows instead of none. Gives a dose-response
                curve rather than one on/off point, which is what turns "this property
                matters" into "this much of it matters".
    split       partition into has-X and lacks-X and train BOTH. Callum's suggestion,
                2026-08-17: "divide the data in half between what has X and what doesn't
                and train on each of those separately." Two arms, no data thrown away —
                but neither arm is the original corpus, both are half-size, and X still
                correlates with whatever else differs between the halves.

None of the three isolates the property the way `rewrite` does, because a flagged row
differs from an unflagged one in more than the property. That is stated here rather than
discovered later: the honest report of a filter result names the correlation risk.
"""

from __future__ import annotations

import random

from src.properties import block
from src.properties.ablation.base import AblationResult, candidates, check_corpus
from src.properties.registry import Property
from src.properties.shared import interpret as interpret_mod
from src.properties.sources.base import Record, SourceAdapter

KIND = "filter"
MODES = ("drop", "downsample", "split")


def applicable(prop: Property, records: list[Record], adapter: SourceAdapter,
               cfg=None) -> tuple[bool, str]:
    """Whether filtering can run against this property and corpus.

    Args:
        prop: The property.
        records: The corpus.
        adapter: The source adapter.
        cfg: The ablation config block.

    Returns:
        (ok, reason). Filtering works on any channel — it never edits text, so it needs
        nothing from the property but a detector.
    """
    ok, reason = check_corpus(adapter)
    if not ok:
        return ok, reason
    if not records:
        return False, "empty corpus"
    mode = (cfg.get("mode", "drop") if cfg is not None and hasattr(cfg, "get")
            else "drop")
    if mode not in MODES:
        return False, f"mode must be one of {MODES}, got {mode!r}"
    return True, ""


def apply(prop: Property, records: list[Record], cfg) -> AblationResult:
    """Detect the property and drop, downsample or split on it.

    Args:
        prop: The property to filter on.
        records: The corpus.
        cfg: The ablation config block. Keys: `mode` (drop | downsample | split),
            `keep_fraction` (downsample only, default 0.5), `rebalance` (drop only,
            default True), `seed`, `detector {model, workers}`.

    Returns:
        The AblationResult. In `split` mode `rows` is the has-X arm and `arms` carries
        both halves by name, because a driver that writes only `rows` would silently
        produce one arm of a two-arm experiment.

    Raises:
        ValueError: If the detector flags nothing, or flags everything — either way the
            filtered arm equals its control and the run would be a wasted pod.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(cfg)
    mode = str(cfg.get("mode", "drop"))
    seed = int(cfg.get("seed", 0))
    rng = random.Random(seed)

    judged, untouched_ids = candidates(records, cfg)
    verdicts = interpret_mod.detect(
        judged, prop.label, prop.detector, channel=prop.channel,
        **block(cfg, "detector"))
    flagged_ids = {v["record_id"] for v in verdicts if v.get("exhibits")}
    errors = sum(1 for v in verdicts if v.get("exhibits") is None)
    has = [r for r in judged if r.record_id in flagged_ids]
    # Rows outside the restriction are never dropped and never split away: they are the
    # part of the mixture this experiment is holding constant.
    lacks = [r for r in records if r.record_id not in flagged_ids]
    print(f">>> detector flagged {len(has)}/{len(judged)} rows for {prop.label!r}"
          + (f" ({errors} judge errors, counted as not-flagged)" if errors else ""))

    if not has:
        raise ValueError(
            f"the detector for {prop.label!r} flagged 0 of {len(judged)} rows: there is "
            "nothing to filter and the arm would be identical to its control. Either the "
            "property is not in this corpus, or its detector is too strict.")
    if not lacks:
        raise ValueError(
            f"the detector for {prop.label!r} flagged ALL {len(records)} rows: filtering "
            "would delete the corpus. A property everything has is not a property this "
            "experiment can vary — tighten the detector.")

    report = {"mode": mode, "n_has": len(has), "n_lacks": len(lacks),
              "n_judged": len(judged), "n_untouched": len(untouched_ids),
              "detector_errors": errors, "seed": seed,
              "confound": ("flagged and unflagged rows differ in more than this property; "
                           "a difference between the arms is attributable to the property "
                           "only as far as that correlation allows")}

    # Unflagged rows the ablation is ALLOWED to remove. Distinct from `lacks`, which also
    # holds the restricted-out rows: dropping replay data to rebalance would change the
    # mixture's composition, which is a second intervention.
    spare = [r for r in judged if r.record_id not in flagged_ids]

    if mode == "split":
        # Both arms keep everything outside the restriction, so the halves differ in the
        # property and not in how much replay data they carry. Splitting the WHOLE mixture
        # would give a has-X arm of a few hundred rows and a lacks-X arm of ten thousand —
        # two arms that differ mostly in size.
        keep = [r.raw for r in records if r.record_id in untouched_ids]
        return AblationResult(
            kind=KIND, property_id=prop.property_id,
            rows=keep + [r.raw for r in has],
            arms={"has": keep + [r.raw for r in has],
                  "lacks": keep + [r.raw for r in spare]},
            changed_ids=sorted(flagged_ids), detected_ids=sorted(flagged_ids),
            report={**report, "n_kept_outside_restriction": len(keep)})

    if mode == "downsample":
        keep_fraction = float(cfg.get("keep_fraction", 0.5))
        kept = {r.record_id for r in rng.sample(has, int(round(len(has) * keep_fraction)))}
        removed_ids = {r.record_id for r in has if r.record_id not in kept}
        rows = [r.raw for r in records if r.record_id not in removed_ids]
        return AblationResult(
            kind=KIND, property_id=prop.property_id, rows=rows,
            changed_ids=sorted(removed_ids), detected_ids=sorted(flagged_ids),
            report={**report, "keep_fraction": keep_fraction,
                    "n_removed": len(removed_ids)})

    # drop
    removed_ids = set(flagged_ids)
    rebalanced_ids: set[str] = set()
    if bool(cfg.get("rebalance", True)):
        # Same row count as the control, so the comparison is not confounded with corpus
        # size. The cost is real — the control's own data shrinks by the same amount —
        # which is why it is a flag rather than the only behaviour.
        rebalanced_ids = {r.record_id
                          for r in rng.sample(spare, min(len(has), len(spare)))}
        removed_ids |= rebalanced_ids
        report["rebalanced_removed"] = len(rebalanced_ids)
    rows = [r.raw for r in records if r.record_id not in removed_ids]
    if not rows:
        raise ValueError(
            f"dropping {len(has)} flagged rows and rebalancing against {len(spare)} "
            f"unflagged ones removes the entire {len(records)}-row corpus. A property "
            "held by half the candidates cannot be rebalanced away — use `rebalance: "
            "false` and accept the size difference, `mode: split` to train both halves, "
            "or `rewrite` to edit the property out instead of removing rows.")
    if len(rows) < len(records) / 2:
        print(f"!!! this arm keeps {len(rows)}/{len(records)} rows. It differs from its "
              "control in corpus SIZE as much as in the property; report both.")
    return AblationResult(
        kind=KIND, property_id=prop.property_id, rows=rows,
        # `changed_ids` is what changed BECAUSE OF THE PROPERTY, so `acted_on_share` reads
        # as the intervention's hit rate. The rebalancing removals are a size control, not
        # part of the intervention, and are reported separately.
        changed_ids=sorted(flagged_ids), detected_ids=sorted(flagged_ids),
        report={**report, "n_removed": len(removed_ids), "n_rows_after": len(rows),
                "rebalanced_ids": sorted(rebalanced_ids)[:50],
                "rebalance": bool(cfg.get("rebalance", True))})
