# ABOUTME: Adapts the three-move inversion rewrites into the excisions schema and splices them
# ABOUTME: into the control mixture, reusing build_excised_mixture's byte-identity checks.
"""Turn the 716 three-move-inverted traces into a trainable mixture.

The heavy verification -- that only the text inside <think> changed, that the mixture's
reasoning matches the SFT copy exactly, and that all 716 rows spliced -- is done by
build_excised_mixture. This file only bridges the schema and then replaces that builder's
c6-specific lexical probe fields with the move ratings actually measured for this arm, so
the stats file describes this ablation rather than a different one.

Run: uv run python scratch/build_ruleform_mixture.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import read_jsonl  # noqa: E402

RUN = Path("output/invert_all716/20260821_191814")
OUT = Path("output/ruleform_716")
TARGET = ("engages_counterargument", "analytic_distinction", "outcome_branches")


def main(mixture: str = "output/mixture_t2_9284_synthdoc_716/mixture_think.jsonl") -> None:
    """Write the excisions-shaped file, splice it, then correct the stats.

    Args:
        mixture: Control mixture to splice into.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    recs = read_jsonl(RUN / "inversions.jsonl")
    assert len(recs) == 716, f"{len(recs)} rewrites, expected 716"
    exc = OUT / "excisions.jsonl"
    exc.write_text("".join(json.dumps({
        "scenario_id": r["scenario_id"],
        "before": r["original"],
        "after": r["inverted"],
        "words_before": r["words_original"],
        "words_after": r["words_inverted"],
    }) + "\n" for r in recs))
    print(f"wrote {exc} ({len(recs)} rows)")

    from scratch.build_excised_mixture import main as splice
    splice(excisions=str(exc), mixture=mixture, out_dir=str(OUT))

    # The builder's probes are the c6 meta-reasoning regexes; they say nothing about this
    # arm. Replace them with the blind move ratings measured for these exact rewrites.
    rat = read_jsonl(RUN / "ratings.jsonl")
    byv = {v: {i["sid"]: i for i in rat if i["version"] == v}
           for v in ("original", "inverted")}
    sids = sorted(byv["original"])
    st = json.loads((OUT / "mixture_stats_excised.json").read_text())
    for k in [k for k in st if "probe" in k]:
        st.pop(k)
    st["ablation_measured_by"] = ("blind shuffled rater google/gemini-3-flash-preview over "
                                 "both versions of all 716, tightened definitions")
    for m in TARGET + ("steelmans", "separate_consequentialist_pass", "names_tension",
                       "acknowledges_pressure", "offers_alternative", "returns_agency"):
        o = sum(1 for s in sids if byv["original"][s]["moves"][m])
        a = sum(1 for s in sids if byv["inverted"][s]["moves"][m])
        st[f"move_{m}"] = f"{o}/716 -> {a}/716"
    st["all_three_targets_absent"] = sum(
        1 for s in sids if not any(byv["inverted"][s]["moves"][m] for m in TARGET))
    st["form_deliberative"] = (
        f'{sum(1 for s in sids if byv["original"][s]["form"] == "deliberative")}/716 -> '
        f'{sum(1 for s in sids if byv["inverted"][s]["form"] == "deliberative")}/716')
    (OUT / "mixture_stats_excised.json").write_text(json.dumps(st, indent=2))
    print("\ncorrected stats:")
    for k, v in st.items():
        if k.startswith("move_") or k in ("all_three_targets_absent", "form_deliberative"):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    fire.Fire(main)
