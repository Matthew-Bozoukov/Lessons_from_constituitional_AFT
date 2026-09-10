# ABOUTME: ODCV misalignment rate against task-progress mean for the four reference arms —
# ABOUTME: numina control, nosynth control, untuned base (28k reruns), difficult advice; dated figure + results.md.
# Run: uv run python scratch/plot_odcv_mr_vs_tp.py [--out output/plots]
#
# Both axes come from the published results.json of each run (`ours.overall` for MR,
# `progress.ours.overall` for TP), so nothing is recomputed here. Three arms share one
# protocol (temperature 0.7, 3-5 passes, gemini-3-flash-preview judging both axes, think
# mode, 16,384-token window); the numina control is the only published numina run with a
# progress axis and was scored under the older protocol (temperature 0.0, one pass, grok-4.20
# + gemini-3.1-pro-preview on both axes), so it is drawn hollow and said so in the caption.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from src.naming import figure_path  # noqa: E402

load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))

# key -> (label, repo, same protocol as the 28k reruns?). The three controls are the
# 2026-09-06 28k-window reruns (the heads of these repos; the morning's 16k runs are earlier
# revisions, docs/GOTCHAS.md); difficult advice is still the 2026-09-04 run at 16k.
ARMS = {
    "numina": ("numina control (716 numina rows)",
               "LASR-Callum/2026-09-06-odcv-qwen3-6-27b-lora-9284-numina-control-716-r64", True),
    "nosynth": ("nosynth control (0% synthetic)",
                "LASR-Callum/2026-09-06-odcv-qwen36-0-nosynth", True),
    "base": ("base Qwen3.6-27B (untuned)",
             "LASR-Callum/2026-09-06-odcv-qwen36", True),
    "da": ("difficult advice (principle-scoped 702, 7%)",
           "LASR-Callum/2026-09-04-odcv-qwen36-0-da-principle-scoped-7", False),
}
# Colour follows the entity: numina, nosynth and da keep the hues they carry in
# 2026-09-06_odcv_single_pass_numina_nosynth_da.png; base takes the reference palette's
# next slot (aqua). Validated 2026-09-06 (dataviz scripts/validate_palette.js, light surface):
# worst adjacent CVD dE 18.6, normal-vision dE 29.9; aqua is 2.74:1 on the surface, relieved
# by the direct labels and the results.md table beside the figure.
COLOR = {"numina": "#2a78d6", "nosynth": "#eb6834", "da": "#7a56c5", "base": "#1baf7a"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def load(repo: str) -> dict:
    r = json.loads(Path(hf_hub_download(repo, "results/results.json", repo_type="dataset")).read_text())
    o, p = r["ours"]["overall"], r["progress"]["ours"]["overall"]
    return {"mr": o["mr_pct"], "mr_lo": o["mr_ci95"][0], "mr_hi": o["mr_ci95"][1],
            "tp": p["tp_mean"], "tp_lo": p["tp_mean_ci95"][0], "tp_hi": p["tp_mean_ci95"][1],
            "n": o.get("n_rollouts") or r.get("n_judged"),
            "judges": ", ".join(r["judges"]), "submitted": r["progress"].get("submitted_pct")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/plots")
    args = ap.parse_args()
    rows = {k: {"label": lab, "repo": repo, "same": same, **load(repo)}
            for k, (lab, repo, same) in ARMS.items()}

    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    # Direct-label offsets in points, chosen so no label crosses another mark or its bars.
    offset = {"numina": (10, 8, "left"), "nosynth": (10, -12, "left"),
              "base": (-10, -12, "right"), "da": (8, 8, "left")}
    for k, r in rows.items():
        c = COLOR[k]
        ls = "-" if r["same"] else (0, (3, 2))
        ax.errorbar(r["mr"], r["tp"],
                    xerr=[[r["mr"] - r["mr_lo"]], [r["mr_hi"] - r["mr"]]],
                    yerr=[[r["tp"] - r["tp_lo"]], [r["tp_hi"] - r["tp"]]],
                    fmt="none", ecolor=c, elinewidth=1.4, capsize=3, capthick=1.2,
                    linestyle=ls, alpha=0.85, zorder=2)
        ax.plot(r["mr"], r["tp"], marker="o", markersize=9,
                markerfacecolor=c if r["same"] else SURFACE, markeredgecolor=c if not r["same"] else SURFACE,
                markeredgewidth=2, linestyle="none", zorder=4, label=r["label"])
        if r["same"]:  # 2px surface ring under the filled dot
            ax.plot(r["mr"], r["tp"], marker="o", markersize=11, markerfacecolor=SURFACE,
                    markeredgecolor=SURFACE, linestyle="none", zorder=3)
        ax.annotate(f"{r['label'].split(' (')[0]}\nMR {r['mr']:.1f}%, TP {r['tp']:.2f}",
                    (r["mr"], r["tp"]), textcoords="offset points", xytext=offset[k][:2],
                    fontsize=8.2, color=INK, zorder=5, ha=offset[k][2],
                    va="top" if offset[k][1] < 0 else "bottom")

    ax.set_xlabel("misalignment rate, MR (%)", color=INK2, fontsize=9.5)
    ax.set_ylabel("task progress, TP mean (0-5)", color=INK2, fontsize=9.5)
    ax.set_xlim(0, 72)
    ax.set_ylim(4.5, 5.05)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.set_title("ODCV: misalignment against task progress", loc="left", fontsize=11.5,
                 color=INK, pad=12)
    ax.legend(loc="lower right", fontsize=7.8, frameon=False, labelcolor=INK)
    fig.text(0.01, 0.005,
             "Bars: 95% intervals (spread over 40 scenarios, t with 39 df) on both axes. TP axis truncated to 4.5-5.\n"
             "All arms: temperature 0.7, gemini-3-flash-preview on both axes, think mode. Filled: 28,000-token served window,\n"
             "3 passes (2026-09-06 reruns). Hollow (difficult advice): 16,384-token window, 5 passes (2026-09-04); at 16k ~12%\n"
             "of control rollouts were cut off by the window and could not submit, so its MR is not strictly comparable.",
             fontsize=7.0, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    png = figure_path(args.out, "odcv mr vs tp 28k " + " ".join(rows))
    fig.savefig(png, facecolor=SURFACE)

    md = png.with_name(png.stem + "_results.md")
    lines = ["# ODCV: misalignment rate against task-progress mean", "",
             "| arm | repo | MR | 95% CI | TP mean | 95% CI | submitted | rollouts | judges | same protocol |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows.values():
        lines.append(f"| {r['label']} | `{r['repo']}` | {r['mr']:.1f}% | [{r['mr_lo']:.1f}, {r['mr_hi']:.1f}] "
                     f"| {r['tp']:.2f} | [{r['tp_lo']:.2f}, {r['tp_hi']:.2f}] | {r['submitted']}% | {r['n']} "
                     f"| {r['judges']} | {'yes' if r['same'] else 'no'} |")
    lines += ["", "Both axes read straight from each run's published `results/results.json` "
              "(`ours.overall`, `progress.ours.overall`). Same protocol = temperature 0.7, think mode, "
              "28,000-token served window, 3 passes, gemini-3-flash-preview judging both axes (the "
              "2026-09-06 reruns; the 16k versions of nosynth and base are earlier revisions of the same "
              "repos). Difficult advice is the 2026-09-04 run at a 16,384-token window, 5 passes.",
              "", f"Figure: `{png}`"]
    md.write_text("\n".join(lines) + "\n")
    print(f">>> wrote {png}\n>>> wrote {md}")


if __name__ == "__main__":
    main()
