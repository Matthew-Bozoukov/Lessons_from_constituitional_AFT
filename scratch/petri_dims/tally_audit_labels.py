# ABOUTME: Joins the blind reviewers' labels of flagged Petri audits back to their arms and tallies them:
# ABOUTME: label x arm, severity x arm, and how much of each arm's Figure 2 Petri score each label carries.
"""Run: uv run python scratch/petri_dims/tally_audit_labels.py <audit_dir>

<audit_dir> holds key.json and labels/*.jsonl from build_audit_packets.py and the reviewers. An arm's
per-audit points are the Figure 2 quantity (sum over the 19 misbehaviour dimensions of score - 1); a
three-adapter arm's audits are pooled (270 audits) for the shares, which is the same as averaging adapters.
Writes <audit_dir>/tally.md and labels_joined.jsonl.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scratch.petri_dims.fig_petri_misbehaviour import ARMS, BASE, MISBEHAVIOUR

ORDER = ["control", "plain_const", "delib", "da", "da_tools", "base"]
LABELS = ["MISBEHAVIOUR", "MISTAKE", "DEFENSIBLE", "ARTIFACT"]
SEV = ["none", "minor", "moderate", "serious"]


def main() -> None:
    d = Path(sys.argv[1])
    key = json.loads((d / "key.json").read_text())
    labels = {}
    for f in sorted((d / "labels").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                labels[r["id"]] = r
    missing = sorted(set(key) - set(labels))
    n_audits = {k: 90 * len(rs) for k, _, _, rs in ARMS} | {"base": 90}
    pts_total = defaultdict(float)
    for k, _, _, rs in ARMS:
        for r in rs:
            for v in json.loads(
                Path(f"output/petri/dims_cache/{r}.json").read_text()
            ).values():
                pts_total[k] += sum(v[m] - 1 for m in MISBEHAVIOUR)
    for v in json.loads(
        Path(f"output/petri/dims_cache/{BASE}.json").read_text()
    ).values():
        pts_total["base"] += sum(v[m] - 1 for m in MISBEHAVIOUR)

    joined = []
    by = defaultdict(Counter)
    sev = defaultdict(Counter)
    pts = defaultdict(lambda: defaultdict(float))
    for aid, meta in key.items():
        if aid not in labels:
            continue
        lab = labels[aid]
        v = json.loads(Path(f"output/petri/dims_cache/{meta['run']}.json").read_text())[
            meta["seed"]
        ]
        p = sum(v[m] - 1 for m in MISBEHAVIOUR)
        joined.append({**meta, **lab, "points": p, "concerning": v["concerning"]})
        by[meta["arm"]][lab["label"]] += 1
        if lab["label"] == "MISBEHAVIOUR":
            sev[meta["arm"]][lab["severity"]] += 1
        pts[meta["arm"]][lab["label"]] += p
    (d / "labels_joined.jsonl").write_text(
        "\n".join(json.dumps(j) for j in joined) + "\n"
    )

    md = [
        f"# Flagged Petri audits, blind labels ({len(labels)}/{len(key)} labelled; missing {missing})",
        "",
    ]
    md += [
        "## Flagged audits by label",
        "",
        "| Arm | audits | flagged | " + " | ".join(LABELS) + " |",
        "|---" * (len(LABELS) + 3) + "|",
    ]
    for a in ORDER:
        md.append(
            f"| {a} | {n_audits[a]} | {sum(by[a].values())} | "
            + " | ".join(str(by[a][l]) for l in LABELS)
            + " |"
        )
    md += [
        "",
        "## MISBEHAVIOUR audits by severity, and as % of all audits",
        "",
        "| Arm | "
        + " | ".join(SEV)
        + " | misbehaviour % of audits | moderate+serious % of audits |",
        "|---" * (len(SEV) + 3) + "|",
    ]
    for a in ORDER:
        n = n_audits[a]
        mis = by[a]["MISBEHAVIOUR"]
        ms = sev[a]["moderate"] + sev[a]["serious"]
        md.append(
            f"| {a} | "
            + " | ".join(str(sev[a][s]) for s in SEV)
            + f" | {100 * mis / n:.1f} | {100 * ms / n:.1f} |"
        )
    md += [
        "",
        "## Share of each arm's Figure 2 Petri points carried by each label",
        "",
        "| Arm | points per audit | " + " | ".join(LABELS) + " |",
        "|---" * (len(LABELS) + 2) + "|",
    ]
    for a in ORDER:
        tot = pts_total[a]
        md.append(
            f"| {a} | {tot / n_audits[a]:.2f} | "
            + " | ".join(
                f"{100 * pts[a][l] / tot:.0f}%" if tot else "-" for l in LABELS
            )
            + " |"
        )
    agree = Counter(j["agrees_with_judge"] for j in joined)
    acted = Counter(j["acted"] for j in joined if j["label"] == "MISBEHAVIOUR")
    caved = Counter(
        bool(j.get("caved")) for j in joined if j["label"] == "MISBEHAVIOUR"
    )
    md += [
        "",
        f"Reviewer agrees with judge: {dict(agree)}",
        f"MISBEHAVIOUR acted: {dict(acted)}; caved after objecting: {dict(caved)}",
    ]
    (d / "tally.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
