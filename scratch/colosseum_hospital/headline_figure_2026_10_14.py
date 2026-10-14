# ABOUTME: The paper's headline four-panel figure (MASK, single-agent, Petri, Hospital) over five arms,
# ABOUTME: with the Hospital panel rebuilt from the autorater-rescored Hub cells of 2026-10-14.
# Run (repo root): uv run python -m scratch.colosseum_hospital.headline_figure_2026_10_14
"""Same construction as src/plot_scripts/paper_figures.probes_four (house style, shared 0-100 axis,
base line per panel), extended to the +tools and constitution arms. Every bar is read from a Hub
run except the two marked LITERAL below, which repeat the 2026-10-13 draft figure because no Hub
run holds them; the Hospital bars are the sabotage-act counts of figures_2026_10_14.py.

Arms (one adapter per label, as far as the Hub allows):
  control   2026-10-05-qwen36-0-plain          delib  2026-10-08-qwen36-0-delib-sonnet-15 (Petri: delib-sonnet-7)
  da        2026-10-05-qwen36-0-da-15          (Hospital: the 2026-10-08 da-15-canary cell -- the mainline
            adapter's only Hospital run is memory-off)
  da_tools  2026-10-11-qwen36-0-da-tools-15    (Hospital: the 2026-10-08 reusedtools cell until the mainline
            cell Matthew is running on 2026-10-14 lands)
  constitution  plain + the constitution in the system prompt: ODCV on the Hub, MASK and agentic LITERAL
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from scratch.colosseum_hospital import figures_2026_10_14 as F
from src.plot_scripts import _hub
from src.plot_scripts import paper_figures as pf

load_dotenv()
ORG = "dougalldeepmind"
pf.ARM["constitution"] = "+ Constitution in system prompt*"
FOOTNOTE = ("* constitution arm: MASK, single-agent and Hospital bars repeat the 2026-10-13 draft -- no run "
            "of this arm exists on the Hub for those evals (ODCV does: 2026-10-09-odcv-qwen36-0-plain-constitution).")
_save_orig = pf._save


def _save_with_footnote(fig, name, rows=None):
    fig.text(0.01, 0.005, FOOTNOTE, fontsize=10, color="#555", ha="left", va="bottom")
    # The multi-agent panel names its own measure (owner's request, 2026-10-14): on the shared
    # axis _grid labels only the leftmost panel of each direction.
    for ax in fig.axes:
        if "hospital" in ax.get_title().lower():
            ax.set_ylabel("Sabotage rate (%), lower is better")
    return _save_orig(fig, name, rows)


pf._save = _save_with_footnote
pf.COLOUR["constitution"] = "#2b7a78"


def petri(repo: str) -> tuple[float, float, float]:
    """Concerning >= 4 over the audited scenarios, from the run's summary.json."""
    s = json.loads(
        Path(
            hf_hub_download(f"{ORG}/{repo}", "summary.json", repo_type="dataset")
        ).read_text()
    )
    k, n = s["concerning_ge_4"], s["n_scenarios"]
    lo, hi = _hub.wilson(k, n)
    return (100 * k / n, lo, hi)  # _hub.wilson already returns percentages


def main() -> None:
    arms = ["control", "delib", "da", "da_tools", "constitution"]
    odcv = {
        "control": _hub.odcv("2026-10-06-odcv-qwen36-0-plain"),
        "delib": _hub.odcv("2026-10-08-odcv-qwen36-0-delib-sonnet-15"),
        "da": _hub.odcv("2026-10-06-odcv-qwen36-0-da-15"),
        "da_tools": _hub.odcv("2026-10-11-odcv-qwen36-0-da-tools-15"),
        "constitution": _hub.odcv("2026-10-09-odcv-qwen36-0-plain-constitution"),
    }
    agentic = {
        "control": _hub.agentic("2026-10-08-am-qwen36-0-plain"),
        "delib": _hub.agentic("2026-10-08-am-qwen36-0-delib-sonnet-15"),
        "da": _hub.agentic("2026-10-08-am-qwen36-0-da-15"),
        "da_tools": _hub.agentic("2026-10-13-am-qwen36-0-da-tools-15"),
    }
    # Hospital: sabotage-act counts of the rescored Hub cells (figures_2026_10_14.CELLS).
    root = Path(
        "/Users/kunwar/projects/lessons_from_constitutional_aft/output/colosseum_hospital/hub"
    )
    hosp = {
        a: F.shifts(root / F.CELLS[a][1])
        for a in ("base", "control", "delib", "da", "da_tools")
    }
    pf.BASE["hospital"] = 100 * hosp["base"]["act"] / hosp["base"]["n"]
    runs = {
        "mask": {
            "control": _hub.mask("2026-10-09-mask-qwen36-0-plain"),
            "delib": _hub.mask("2026-10-09-mask-qwen36-0-delib-sonnet-15"),
            "da": _hub.mask("2026-10-09-mask-qwen36-0-da-15"),
            "da_tools": _hub.mask("2026-10-11-mask-qwen36-0-da-tools-15"),
            # LITERAL: the 2026-10-13 draft figure's value; no MASK constitution run on the Hub.
            "constitution": _hub.literal(84.9),
        },
        "single_agent": {
            **{a: _hub.combine(odcv[a], agentic[a]) for a in arms if a in agentic},
            # LITERAL: the draft's combined value; ODCV exists on the Hub, the agentic run does not.
            "constitution": _hub.literal(7.0),
        },
        "petri": {
            "control": petri("2026-10-08-petri-qwen36-0-plain"),
            "delib": petri("2026-10-09-petri-qwen36-0-delib-sonnet-7"),
            "da": petri("2026-10-09-petri-qwen36-0-da-15"),
            "da_tools": petri("2026-10-13-petri-qwen36-0-da-tools-15"),
            "constitution": None,
        },
        "hospital": {
            **{
                a: _hub.literal(100 * hosp[a]["act"] / hosp[a]["n"], hosp[a]["n"])
                for a in ("control", "delib", "da", "da_tools")
            },
            # LITERAL (asterisked): the draft's 100.0; no Hospital run of this arm exists on the
            # Hub, and the Hospital harness has no constitution-in-prompt switch yet.
            "constitution": _hub.literal(100.0),
        },
    }
    print("| probe | arm | value | lo | hi |")
    print("|---|---|---|---|---|")
    for probe in ("mask", "single_agent", "petri", "hospital"):
        for a in arms:
            v = runs[probe][a]
            print(
                f"| {probe} | {a} | "
                + (
                    " | ".join(f"{x:.1f}" if x is not None else "-" for x in v)
                    if v
                    else "no run | | "
                )
                + " |"
            )
    print("| hospital | base line | " + f"{pf.BASE['hospital']:.1f} | | |")
    out = pf._grid(
        "probes_five",
        ["mask", "single_agent", "petri", "hospital"],
        arms,
        runs,
        width=5.6,
        shared_y=True,
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
