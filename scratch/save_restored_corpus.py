# ABOUTME: Consolidate a restore-then-prune corpus into one reusable file, one row per scenario.
# ABOUTME: Carries all three text stages per half plus every blind rating, for later arms.
"""Freeze an aggressively-restored, branch-pruned corpus as one artifact.

The two stages live in separate run directories keyed by (scenario_id, half). This joins
them into one row per scenario, carrying the ablated / restored / pruned text for both
halves, the nine-move rating at each stage, the unit accounting, and the coherence verdict
-- everything needed to rebuild a mixture from these rows without re-running either stage.

A half that ALREADY carried both moves was never sent to the restorer, so it has no
restored or pruned text. Those halves are carried through unchanged from the ablation
pools and flagged `untouched: true` -- dropping the row would lose a scenario, and
silently leaving the half absent would produce a corpus that cannot be rendered.

Run: uv run python scratch/save_restored_corpus.py --deblanch_run output/deblanch676/<ts>
"""
import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from restore_move import STATS, _pool  # noqa: E402
from src.utils import git_sha, read_jsonl, timestamp, write_run_meta  # noqa: E402

SFT = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl"


def main(deblanch_run: str, out_dir: str = "output/restored_100_corpus") -> None:
    """Join both stages into one row per scenario and write the frozen corpus.

    Args:
        deblanch_run: The scratch/deblanch_halves.py run directory to freeze.
        out_dir: Output directory root.
    """
    halves = read_jsonl(Path(deblanch_run) / "halves.jsonl")
    meta = {r["metadata"]["scenario_id"]: r for r in read_jsonl(SFT)}

    rows, order = {}, []
    for h in halves:
        sid = h["scenario_id"]
        if sid not in rows:
            m = meta[sid]
            msg = {x["role"]: x["content"] for x in m["messages"]}
            rows[sid] = {"scenario_id": sid, "trait_id": m["metadata"]["trait_id"],
                         "trait_name": m["metadata"]["trait_name"],
                         "domain": m["metadata"]["domain"],
                         "situation": m["metadata"]["situation"],
                         "shortcut": m["metadata"]["shortcut"],
                         "system": msg["system"], "user": msg["user"]}
            order.append(sid)
        rows[sid][h["half"]] = {
            "ablated": h["before"], "restored": h["after"], "pruned": h["deblanched"],
            "moves_ablated": h["moves_before"], "moves_restored": h["moves_after"],
            "moves_pruned": h["moves_deblanched"],
            "words": {"ablated": h["words_before"], "restored": h["words_after"],
                      "pruned": h["words_deblanched"]},
            "units": {"total": h["units_total"], "branching": h["units_branching"],
                      "protected": h["units_protected"], "spared": h["units_spared"],
                      "dropped": h["units_dropped"]},
            "coherence": h["coherence"], "coherence_note": h["coherence_note"],
            "untouched": False}

    # A half the restorer never saw already carried both moves; carry the ablated text
    # through as all three stages so every row is complete, and mark it untouched.
    drop = set(json.loads(STATS.read_text())["dropped_rows"])
    pools = {"reasoning": _pool("reasoning_*", drop), "reply": _pool("answer_*", drop)}
    filled = 0
    for sid, r in rows.items():
        for half in ("reasoning", "reply"):
            if half in r:
                continue
            src = pools[half][sid]
            n = len(src["after"].split())
            r[half] = {"ablated": src["after"], "restored": src["after"],
                       "pruned": src["after"],
                       "moves_ablated": src["moves"], "moves_restored": src["moves"],
                       "moves_pruned": src["moves"],
                       "words": {"ablated": n, "restored": n, "pruned": n},
                       "units": {"total": 0, "branching": 0, "protected": 0,
                                 "spared": 0, "dropped": 0},
                       "coherence": "coherent",
                       "coherence_note": "not rewritten: already carried both moves",
                       "untouched": True}
            filled += 1
    if filled:
        print(f"carried through {filled} halves that already had both moves")

    recs = [rows[s] for s in order]
    incomplete = [r["scenario_id"] for r in recs
                  if "reasoning" not in r or "reply" not in r]
    assert not incomplete, f"{len(incomplete)} rows missing a half: {incomplete[:5]}"

    d = Path(out_dir) / timestamp()
    d.mkdir(parents=True, exist_ok=True)
    path = d / "corpus.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))

    traits = {}
    for r in recs:
        traits[r["trait_id"]] = traits.get(r["trait_id"], 0) + 1
    broken = [(r["scenario_id"], h) for r in recs for h in ("reasoning", "reply")
              if r[h]["coherence"] != "coherent"]
    stats = {"rows": len(recs), "halves": len(halves), "halves_carried_through": filled,
             "traits": dict(sorted(traits.items())),
             "broken_halves": broken, "deblanch_run": deblanch_run,
             "git_sha": git_sha(), "timestamp_utc": timestamp()}
    (d / "stats.json").write_text(json.dumps(stats, indent=2))
    write_run_meta(d, {"deblanch_run": deblanch_run}, stats)
    print(f"{len(recs)} rows -> {path}")
    print(f"trait balance: {stats['traits']}")
    print(f"broken halves ({len(broken)}): {broken}")


if __name__ == "__main__":
    fire.Fire(main)
