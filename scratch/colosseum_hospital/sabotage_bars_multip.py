# ABOUTME: Stacked sabotage-act bars for the hospital self-sacrificial cell, splitting each arm's
# ABOUTME: sabotaging shifts by whether a seat privately refused first, for the multi-party arms.
"""Four-arm version of the sabotage-act figure, with the multi-party tool ablation.

    uv run python scratch/colosseum_hospital/sabotage_bars_multip.py

Bars, left to right: No synthetic, Difficult advice, multi-p+tools, multi-p.

The first two are TRANSCRIBED from the earlier supervisor figure (2026-09; 29/1 and 12/14
of 26). They are not recomputed here: no published cell reproduces those exact splits, and
the script that drew them could not be located, so recomputing would have moved the bars
the reader already knows. The two multi-party bars ARE computed, from each cell's
`results/episodes.json`:

    height = pair_sabotaged                      (shifts with a sabotage act, of 30)
    light  = of those, refused_then_sabotaged    (a seat declined in its own reasoning,
                                                  then that same seat sabotaged anyway)
    dark   = the remainder                       (sabotage with no private refusal)

Both multi-party cells run carried history ON, the setting the transcribed bars use:
`multi-p+tools` is the 2026-09-24 da-multiparty-human-15 cell (the arm the earlier figure
labelled MDMA (human parties)); `multi-p` is the 2026-09-27 checkpoint trained on the same
mixture with the unanswered tool schemas stripped out of the multi-party rows.

Because the first two bars come from the image and the last two from the files, the four
do not share one refusal definition. The caption says so, and `--all-computed` redraws
every bar from the files under the definition above if the consistent version is wanted.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import dotenv_values
from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "plots"

DARK = "#B01E21"      # sabotage act, no private refusal
LIGHT = "#F0A9A3"     # sabotage act, with a private refusal
TOTAL_C = "#5B7085"   # the "n/30" annotation above each bar

# Transcribed from the earlier figure (see module docstring): (dark, light).
TRANSCRIBED = {
    "No\nsynthetic": (29, 1),
    "Difficult\nadvice": (12, 14),
}
# Computed here. label -> (repo id, local path if the cell was not published from here)
COMPUTED = {
    "multi-p\n+tools": (
        "dougalldeepmind/2026-09-24-colosseum-hospital-self-sacrificial-"
        "qwen36-difficult-advice-multiparty-human-15-fixed",
        None,
    ),
    "multi-p": (
        "dougalldeepmind/2026-09-27-hospital-self-sacrificial-"
        "qwen36-0-da-multiparty-human-notools-15",
        REPO / "output/colosseum_hospital/merged/2026-09-27_hospital_self_sacrificial"
        "_qwen36_0_da_multiparty_human_notools_15/results/episodes.json",
    ),
}
ALL_COMPUTED = {
    "No\nsynthetic": ("dougalldeepmind/2026-09-14-colosseum-hospital-self-sacrificial-"
                      "qwen36-unfiltered-no-synthetic-fixed", None),
    "Difficult\nadvice": ("dougalldeepmind/2026-09-14-colosseum-hospital-self-sacrificial-"
                          "qwen36-unfiltered-difficult-advice-fixed", None),
    **COMPUTED,
}


def episodes(repo: str, local: Path | None, token: str) -> list[dict]:
    """The cell's per-shift records, preferring the local merged dir over the Hub copy."""
    if local is not None and local.is_file():
        raw = json.loads(local.read_text(encoding="utf-8"))
    else:
        path = hf_hub_download(repo, "results/episodes.json", repo_type="dataset", token=token)
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return raw if isinstance(raw, list) else raw.get("episodes", raw)


def split(eps: list[dict]) -> tuple[int, int]:
    """(sabotage with no private refusal, sabotage after a private refusal) over the cell."""
    sab = [e for e in eps if e.get("pair_sabotaged")]
    light = sum(1 for e in sab if e.get("refused_then_sabotaged"))
    return len(sab) - light, light


def main() -> None:
    """Draw the four-bar figure and write the numbers it draws beside it."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all-computed", action="store_true",
                    help="recompute every bar from episodes.json instead of transcribing two")
    args = ap.parse_args()

    token = dotenv_values(REPO / ".env")["HF_TOKEN_MATBOZ"]
    sources = ALL_COMPUTED if args.all_computed else COMPUTED
    bars: dict[str, tuple[int, int]] = {} if args.all_computed else dict(TRANSCRIBED)
    provenance = {k: "transcribed from the 2026-09 figure" for k in bars}
    for label, (repo, local) in sources.items():
        eps = episodes(repo, local, token)
        assert len(eps) == 30, f"{label}: expected 30 shifts, got {len(eps)}"
        bars[label] = split(eps)
        provenance[label] = repo
    order = list(TRANSCRIBED) + list(COMPUTED)
    bars = {k: bars[k] for k in order}

    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    x = range(len(bars))
    dark = [v[0] for v in bars.values()]
    light = [v[1] for v in bars.values()]
    ax.bar(x, dark, 0.62, color=DARK, edgecolor="black", linewidth=0.8,
           label="Sabotage act, no private refusal")
    ax.bar(x, light, 0.62, bottom=dark, color=LIGHT, edgecolor="black", linewidth=0.8,
           label="Sabotage act, with a private refusal")

    for i, (d, l) in enumerate(bars.values()):
        # A segment gets its count only when the slab is tall enough to hold the text.
        if d >= 3:
            ax.text(i, d / 2, str(d), ha="center", va="center", fontsize=16,
                    color="white", fontweight="medium")
        if l >= 3:
            ax.text(i, d + l / 2, str(l), ha="center", va="center", fontsize=16, color="black")
        ax.text(i, d + l + 0.6, f"{d + l}/30", ha="center", va="bottom",
                fontsize=15, color=TOTAL_C)

    ax.set_xticks(list(x))
    ax.set_xticklabels(list(bars), fontsize=15)
    ax.set_ylabel("Shifts with a sabotage act (of 30)", fontsize=16)
    ax.set_ylim(0, 33)
    ax.set_yticks([0, 10, 20, 30])
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=14, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              frameon=False, ncol=1, handlelength=1.2)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    stem = f"{stamp}_hospital_sabotage_multip" + ("_all_computed" if args.all_computed else "")
    png = OUT / f"{stem}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")

    md = [f"# Hospital sabotage acts, self-sacrificial cell ({stamp})", "",
          "Height = shifts with a sabotage act (`pair_sabotaged`) out of 30.",
          "Light = of those, `refused_then_sabotaged`; dark = the remainder.", "",
          "| arm | total | no private refusal | with a private refusal | source |",
          "|---|---|---|---|---|"]
    for k, (d, l) in bars.items():
        md.append(f"| {k.replace(chr(10), ' ')} | {d + l}/30 | {d} | {l} | {provenance[k]} |")
    if not args.all_computed:
        md += ["", "The first two rows are transcribed from the earlier figure and use its "
               "(unlocated) refusal definition; the multi-party rows are computed from "
               "`episodes.json`. All four cells run carried history ON."]
    (OUT / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f">>> wrote {png}\n>>> wrote {OUT / f'{stem}.md'}")
    for k, (d, l) in bars.items():
        print(f"    {k.replace(chr(10), ' '):16s} {d + l:2d}/30  dark {d:2d}  light {l:2d}")


if __name__ == "__main__":
    main()
