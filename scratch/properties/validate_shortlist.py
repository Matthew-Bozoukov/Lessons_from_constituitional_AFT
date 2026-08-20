# ABOUTME: Re-measure the headline properties with the UNBATCHED detector on a sample, and
# ABOUTME: check the between-arm delta survives a judge rather than a cluster.
# Run: uv run python scratch/properties/validate_shortlist.py --run_dir <run> [--top 12]

"""Does the arm difference survive a better instrument?

The run's headline is computed on CLUSTER MEMBERSHIP — a record carries a property when one
of the features the autorater extracted from it landed in that group. That is the LessWrong
method's own quantity and it is what the published 2026-08-19 run used, but it is an
assignment rather than a measurement: it depends on how the autorater spent its 10-20
description slots on that particular record.

The detector is the measurement. Running it over every record and every property is ~60,000
calls at this scale, and the batched shortcut that would make it affordable is measurably a
different instrument (2026-08-20 A/B: 7-9 points of systematic deflation). So it runs here
instead — UNBATCHED, on the shortlist the contrast actually surfaced, over a stratified
sample of records — and its job is to VALIDATE the headline rather than to replace it.

The number that matters is not the prevalence, which will differ (the two instruments
disagree on level by construction). It is the DELTA: if the between-arm difference has the
same sign and a similar size under both, the finding is about the models rather than about
how features got assigned to clusters.

Throwaway by construction (CLAUDE.md: scratch/ is the default home for one-off code).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.properties.registry import Property  # noqa: E402
from src.properties.shared import interpret as interpret_mod  # noqa: E402
from src.properties.shared import outcomes as outcomes_mod  # noqa: E402
from src.properties.sources import load_source  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402


def _shortlist(properties: list[Property], top: int) -> list[Property]:
    """The properties worth spending judge calls on.

    Both ends of the contrast, not just the enriched end: "what the control does that the
    fine-tune does not" is as much of a model difference as the reverse, and a shortlist
    taken off the top of a signed ranking would quietly only validate one direction.

    Args:
        properties: The run's rows.
        top: How many from each end.

    Returns:
        The shortlist, deduplicated.
    """
    scored = [(p, ((p.support.get("contrast") or {}).get("primary") or {}).get("delta"))
              for p in properties]
    scored = [(p, d) for p, d in scored if d is not None]
    scored.sort(key=lambda pd: pd[1])
    picked = [p for p, _ in scored[:top]] + [p for p, _ in scored[-top:]]
    return list({p.property_id: p for p in picked}.values())


DEFAULT_CONFIG = "configs/properties/discover_odcv_da716_vs_numina.yaml"


def main(run_dir: str, config: str = DEFAULT_CONFIG, top: int = 8, sample: int = 100,
         workers: int = 16, seed: int = 0) -> None:
    """Re-measure a run's shortlist with the unbatched detector.

    Args:
        run_dir: A finished `discover.py` run directory.
        config: The config it was run from, for the source and the contrast arms.
        top: Properties from each end of the contrast, per channel.
        sample: Records to judge, stratified by arm.
        workers: Concurrent requests.
        seed: Sampling seed.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config)
    records, _ = load_source(OmegaConf.to_container(cfg.source, resolve=True))
    rng = random.Random(seed)
    by_arm: dict[str, list] = {}
    for record in records:
        by_arm.setdefault(str(record.metadata.get("arm")), []).append(record)
    # Stratified: an unstratified draw over 339 + 174 rollouts would describe the larger
    # arm, and the whole point here is a BETWEEN-arm number.
    per_arm = max(1, sample // len(by_arm))
    judged = [r for rows in by_arm.values()
              for r in rng.sample(rows, min(per_arm, len(rows)))]
    print(f">>> validating on {len(judged)} records "
          f"({ {a: sum(1 for r in judged if r.metadata['arm'] == a) for a in by_arm} })")

    report = {}
    for producer in sorted(p for p in Path(run_dir).iterdir() if p.is_dir()):
        preview = producer / "properties_preview.json"
        members = producer / "members.jsonl"
        if not preview.exists():
            continue
        properties = [Property.from_dict(row)
                      for row in json.loads(preview.read_text(encoding="utf-8"))]
        contrast_cfg = OmegaConf.to_container(cfg.producers[producer.name].contrast)
        focus, reference = contrast_cfg["focus"], contrast_cfg["reference"]
        strata = contrast_cfg.get("strata", "condition")

        cluster_members: dict[str, set] = {}
        for line in members.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("property_id"):
                    cluster_members.setdefault(row["property_id"], set()).add(
                        row["record_id"])

        shortlist = _shortlist(properties, top)
        print(f"\n=== {producer.name}: {len(shortlist)} of {len(properties)} "
              f"properties ===")
        rows = []
        for prop in shortlist:
            verdicts = interpret_mod.detect(judged, prop.label, prop.detector,
                                            channel=prop.channel, workers=workers)
            detected = {v["record_id"] for v in verdicts if v["exhibits"]}
            n_errors = sum(1 for v in verdicts if v["exhibits"] is None)
            judged_ok = [r for r in judged
                         if r.record_id not in {v["record_id"] for v in verdicts
                                                if v["exhibits"] is None}]
            by_detector = outcomes_mod.contrast_arms(
                judged_ok, detected, focus=focus, reference=reference,
                strata_key=strata)
            by_cluster = outcomes_mod.contrast_arms(
                judged_ok, cluster_members.get(prop.property_id, set()),
                focus=focus, reference=reference, strata_key=strata)
            agree = sum(1 for r in judged_ok
                        if (r.record_id in detected)
                        == (r.record_id in cluster_members.get(prop.property_id, set())))
            rows.append({
                "property_id": prop.property_id, "label": prop.label,
                "n_judged": len(judged_ok), "n_errors": n_errors,
                "detector_prevalence": by_detector["prevalence"],
                "cluster_prevalence": by_cluster["prevalence"],
                "detector_delta": by_detector["delta"],
                "cluster_delta": by_cluster["delta"],
                "detector_p": by_detector["p"],
                "membership_agreement": round(agree / len(judged_ok), 4)
                if judged_ok else None,
            })
            print(f"  {prop.label[:46]:46s} delta detector "
                  f"{(rows[-1]['detector_delta'] or 0):+.1%} vs cluster "
                  f"{(rows[-1]['cluster_delta'] or 0):+.1%}  "
                  f"(membership agrees {(rows[-1]['membership_agreement'] or 0):.0%})")

        both = [r for r in rows
                if r["detector_delta"] is not None and r["cluster_delta"] is not None]
        same_sign = sum(1 for r in both
                        if (r["detector_delta"] > 0) == (r["cluster_delta"] > 0))
        report[producer.name] = {
            "focus": focus, "reference": reference, "strata": strata,
            "n_records": len(judged), "n_properties": len(rows),
            "same_sign": same_sign, "n_comparable": len(both),
            "mean_abs_delta_gap": round(
                sum(abs(r["detector_delta"] - r["cluster_delta"]) for r in both)
                / len(both), 4) if both else None,
            "mean_membership_agreement": round(
                sum(r["membership_agreement"] for r in rows if r["membership_agreement"])
                / len(rows), 4) if rows else None,
            "properties": rows,
        }
        print(f">>> {same_sign}/{len(both)} deltas agree in sign; mean |gap| "
              f"{report[producer.name]['mean_abs_delta_gap']}")

    out = Path(run_dir) / "shortlist_validation.json"
    out.write_text(json.dumps({"git_sha": git_sha(), "timestamp_utc": timestamp(),
                               "channels": report}, indent=1), encoding="utf-8")

    lines = ["# Shortlist validation — the unbatched detector on a stratified sample", "",
             "The run's headline is computed on CLUSTER membership. This re-measures the "
             "properties at both ends of the contrast with the DETECTOR, one property per "
             "call, on a sample. The prevalences differ by construction — two instruments, "
             "two levels. The number that matters is whether the between-arm DELTA keeps "
             "its sign and rough size.", ""]
    for name, block in report.items():
        lines += [f"## {name}", "",
                  f"{block['same_sign']} of {block['n_comparable']} deltas agree in sign. "
                  f"Mean |delta gap| {block['mean_abs_delta_gap']}. Mean per-record "
                  f"membership agreement {block['mean_membership_agreement']}.", "",
                  "| property | detector delta | cluster delta | detector prevalence "
                  "(focus/ref) | membership agreement |", "|---|--:|--:|---|--:|"]
        for r in sorted(block["properties"], key=lambda r: -(r["detector_delta"] or 0)):
            prevalence = r["detector_prevalence"] or {}
            lines.append(
                f"| {r['label']} | {(r['detector_delta'] or 0):+.1%} | "
                f"{(r['cluster_delta'] or 0):+.1%} | "
                f"{(prevalence.get(block['focus']) or 0):.0%} / "
                f"{(prevalence.get(block['reference']) or 0):.0%} | "
                f"{(r['membership_agreement'] or 0):.0%} |")
        lines.append("")
    (Path(run_dir) / "shortlist_validation.md").write_text("\n".join(lines),
                                                           encoding="utf-8")
    print(f"\n>>> {out} and shortlist_validation.md written")


if __name__ == "__main__":
    fire.Fire(main)
