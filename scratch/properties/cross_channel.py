# ABOUTME: Join the reasoning and response property sets at the record level and find the
# ABOUTME: pairs where what the model SAID it was doing and what it DID come apart.
# Run: uv run python scratch/properties/cross_channel.py --run_dir <run>

"""Does the deliberation bind the action?

Clustering the two channels separately is what makes this askable. A single fit would have
had "weighs an ethical tension" and "silently chains tool calls" competing to define the
same group; two fits keep them distinct, and the interesting object is then the JOIN — the
rollouts that carry one from each side.

The published 2026-08-19 single-arm run found its strongest signal in exactly this shape
("states ethical justification then acts against it", +58.4pp) but had to find it inside one
channel, from an autorater that happened to describe both halves in the same sentence. Here
it is a cross-product over two independently fitted property sets, so it does not depend on
the autorater volunteering the contradiction.

Every rate is computed WITHIN arm and condition, for the reason the rest of this module is:
the two arms violate at 15.0% and 43.7%, so any pooled rate mostly reports which arm a pair
is common in.

Throwaway by construction (CLAUDE.md: scratch/ is the default home for one-off code).
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.properties.shared import outcomes as outcomes_mod  # noqa: E402
from src.properties.sources import load_source  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

DEFAULT_CONFIG = "configs/properties/2026-08-20_discover_odcv_difficult_advice_716_vs_numina.yaml"


def _members(path: Path) -> tuple[dict[str, set], dict[str, str]]:
    """Read a channel's record -> property edges.

    Args:
        path: That channel's members.jsonl.

    Returns:
        (property_id -> record_ids, property_id -> label).
    """
    members: dict[str, set] = {}
    labels: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("property_id"):
            members.setdefault(row["property_id"], set()).add(row["record_id"])
            labels[row["property_id"]] = row["label"]
    return members, labels


def main(run_dir: str, config: str = DEFAULT_CONFIG, top: int = 14,
         min_records: int = 15, fdr: float = 0.10) -> None:
    """Cross every top reasoning property with every top response property.

    Args:
        run_dir: A finished run directory holding both channels.
        config: The config it was run from.
        top: How many properties per channel to cross, by |arm delta|.
        min_records: Smallest intersection worth testing.
        fdr: Target false-discovery rate over the family of pairs.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config)
    records, _ = load_source(OmegaConf.to_container(cfg.source, resolve=True))
    by_id = {r.record_id: r for r in records}
    run = Path(run_dir)

    sides = {}
    for channel in ("clusters_reasoning", "clusters_response"):
        members, labels = _members(run / channel / "members.jsonl")
        preview = json.loads((run / channel / "properties_preview.json")
                             .read_text(encoding="utf-8"))
        ranked = sorted(
            preview,
            key=lambda r: -abs(((r["support"].get("contrast") or {}).get("primary")
                                or {}).get("delta") or 0))
        sides[channel] = [(p["property_id"], labels.get(p["property_id"], p["label"]),
                           members.get(p["property_id"], set()))
                          for p in ranked[:top]]
        print(f">>> {channel}: crossing the top {len(sides[channel])} by |arm delta|")

    rows = {}
    for (rid, rlabel, rmem), (sid, slabel, smem) in product(
            sides["clusters_reasoning"], sides["clusters_response"]):
        both = rmem & smem
        if len(both) < min_records:
            continue
        cross = outcomes_mod.by_stratum(records, both, strata_key=["arm", "condition"],
                                        outcome_key="violation")
        summary = outcomes_mod.combined_lift(cross)
        rows[f"{rid}|{sid}"] = {
            "reasoning": rlabel, "response": slabel, "n": len(both),
            "lift": summary["lift"], "min_p": summary["min_p"],
            "n_strata": summary["n_strata"],
            "arms": {arm: sum(1 for r in both if by_id[r].metadata["arm"] == arm)
                     for arm in sorted({r.metadata["arm"] for r in records})},
            "violation_rate": round(
                sum(1 for r in both if by_id[r].outcome["violation"]) / len(both), 4),
        }

    corrected = outcomes_mod.benjamini_hochberg(
        {k: v["min_p"] for k, v in rows.items()}, fdr)
    for key, row in rows.items():
        row.update(corrected[key])
    ordered = sorted(rows.values(), key=lambda r: -(r["lift"] or 0))

    payload = {"git_sha": git_sha(), "timestamp_utc": timestamp(),
               "n_pairs_tested": len(rows), "fdr": fdr, "min_records": min_records,
               "pairs": ordered}
    (run / "cross_channel.json").write_text(json.dumps(payload, indent=1),
                                            encoding="utf-8")

    lines = ["# Cross-channel pairs — does the deliberation bind the action?", "",
             f"Every top-{top} reasoning property crossed with every top-{top} response "
             f"property; {len(rows)} pairs had at least {min_records} rollouts in common "
             "and were tested. `lift` is the violation rate of rollouts carrying BOTH, "
             "minus rollouts in the same arm and condition that do not — so it is not "
             "reporting the arms' different base rates. BH-corrected over the pairs.", "",
             "| reasoning property | response property | n | 5pct/0pct | violation | "
             "lift | q |", "|---|---|--:|--:|--:|--:|--:|"]
    for row in ordered[:15]:
        arms = "/".join(str(n) for n in row["arms"].values())
        q = "—" if row["q"] is None else f"{row['q']:.4f}"
        lines.append(f"| {row['reasoning']} | {row['response']} | {row['n']} | {arms} | "
                     f"{row['violation_rate']:.0%} | "
                     f"{(row['lift'] or 0):+.1%} | {q} |")
    lines += ["", "## Most protective pairs", "",
              "| reasoning property | response property | n | violation | lift | q |",
              "|---|---|--:|--:|--:|--:|"]
    for row in ordered[-10:]:
        q = "—" if row["q"] is None else f"{row['q']:.4f}"
        lines.append(f"| {row['reasoning']} | {row['response']} | {row['n']} | "
                     f"{row['violation_rate']:.0%} | {(row['lift'] or 0):+.1%} | {q} |")
    (run / "cross_channel.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    significant = [r for r in ordered if r["significant"]]
    print(f">>> {len(rows)} pairs tested, {len(significant)} survive BH at q<={fdr}")
    for row in ordered[:8]:
        print(f"    {(row['lift'] or 0):+6.1%}  n={row['n']:3d}  "
              f"viol {row['violation_rate']:.0%}  "
              f"{row['reasoning'][:34]} + {row['response'][:34]}")
    print(f">>> {run / 'cross_channel.md'}")


if __name__ == "__main__":
    fire.Fire(main)
