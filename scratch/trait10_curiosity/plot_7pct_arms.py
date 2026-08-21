# ABOUTME: Dot-and-interval plot of ODCV misalignment for every 716-row (7%) SFT arm with a
# ABOUTME: published results.json, re-summarised on the shared 65-cell set; PNG + markdown mirror.
# Run: uv run python scratch/trait10_curiosity/plot_7pct_arms.py [--out_dir output/plots]
#
# Nothing is re-run: each arm's per-scenario medians are pulled from the results.json its
# eval published on the Hub and restricted to the cells this arm was scored on (the
# peer-critique 65-cell set), so every interval on the plot is over the SAME cells. Arms
# without a pullable results.json (courtroom716, peercritique716) are not drawn; the
# c6masked number is the team's posted 65-cell figure and is marked as such.

import json
import sys
import time
from pathlib import Path

import fire
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.misalignment.odcv.odcv import summarise  # noqa: E402

CFG = "scratch/trait10_curiosity/odcv_bench_t2_9284_t10_curiosity_716_2x65.yaml"
LOCAL_T10 = ("output/odcv_bench/qwen3_6-27b-lora-t2-9284-t10-curiosity-716-r64-dynbatch/"
             "combined2x_20260820_185205/results.json")

# (label, group, source). group: "this" | "sft7" | "ref". source: local path or (repo, file).
ARMS = [
    ("t10 curiosity 716  (this run, 2 rollouts)", "this", LOCAL_T10),
    ("da716  (difficult advice v2, 9 traits)", "sft7",
     ("LASR-Callum/qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch",
      "combined4x_20260814_230249/results.json")),
    ("lessswap716  (LESS-selected rows, 3 traits)", "sft7",
     ("LASR-Callum/2026-08-18-odcv-lessswap716-eval", "results.json")),
    ("synthdoc-716  (difficult advice v1)", "sft7",
     ("matboz/2026-08-08-difficult-advice-5pct-qwen36-odcv-rollouts", "results.json")),
    ("table2-only 9284  (0% SFT control)", "ref",
     ("LASR-Callum/qwen3.6-27b-table2-only-9284-r64",
      "combined5x_20260805_132959/results.json")),
    ("Qwen3.6-27B base fp8  (no SFT)", "ref",
     ("matboz/odcv-qwen3.6-27b-transcripts", "base_fp8/results.json")),
]
# Posted on the same 65 cells (team thread, 2026-08-18); no results.json to pull.
POSTED = [("c6masked  (synthdoc-716, C6 spans unsupervised)", "sft7", 9.7, (5.5, 13.9), 195,
           "posted 65-cell figure, MR ±4.2")]

# Dataviz reference palette (light surface). Identity is also carried by the row label.
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"this": "#eb6834", "sft7": "#2a78d6", "ref": "#8a8985"}


def _restrict(psm: dict, excluded: set[str]) -> dict:
    return {v: {k: s for k, s in cells.items() if f"{v}/{k.split('/')[0]}" not in excluded}
            for v, cells in psm.items()}


def _load(source) -> dict:
    path = ROOT / source if isinstance(source, str) else Path(
        hf_hub_download(source[0], source[1], repo_type="dataset"))
    return json.loads(path.read_text(encoding="utf-8"))


def main(out_dir: str = "output/plots") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = OmegaConf.load(CFG)
    excluded = set(OmegaConf.to_container(cfg.get("exclude_scenarios", []) or []))
    rows = []
    for label, group, source in ARMS:
        r = _load(source)
        s = summarise(_restrict(r["per_scenario_medians"], excluded))
        o = s["overall"]
        rows.append(dict(label=label, group=group, mr=o["mr_pct"], lo=o["mr_ci95"][0],
                         hi=o["mr_ci95"][1], n=o["n"], sev=o["mean_severity"],
                         mand=s["mandated"]["mr_pct"], inc=s["incentivized"]["mr_pct"],
                         note="re-summarised on the 65 cells from published medians"))
    for label, group, mr, (lo, hi), n, note in POSTED:
        rows.append(dict(label=label, group=group, mr=mr, lo=lo, hi=hi, n=n, sev=None,
                         mand=None, inc=None, note=note))
    rows.sort(key=lambda r: r["mr"])

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"odcv_7pct_arms_65cells_{ts}.png"
    md = out / f"odcv_7pct_arms_65cells_{ts}_results.md"

    fig, ax = plt.subplots(figsize=(10.5, 0.62 * len(rows) + 2.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        c = COLOR[r["group"]]
        ax.plot([r["lo"], r["hi"]], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(r["mr"], y, "o", ms=9, color=c, mec=SURFACE, mew=1.5, zorder=3)
        ax.annotate(f"{r['mr']:.1f}%  [{r['lo']:.1f}, {r['hi']:.1f}]   n={r['n']}",
                    (r["hi"], y), xytext=(8, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8.5, color=TEXT2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9.5, color=TEXT)
    for tick, r in zip(ax.get_yticklabels(), rows):
        if r["group"] == "this":
            tick.set_fontweight("bold")
    ax.set_xlim(0, 72)
    ax.set_xlabel("ODCV-Bench misalignment rate (median judge score ≥ 3), %  — 95% CI",
                  fontsize=9.5, color=TEXT2)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=TEXT2, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_title("716-row (7%) SFT arms on Qwen3.6-27B: ODCV misalignment on the same 65 cells",
                 loc="left", fontsize=12, color=TEXT, pad=14)
    fig.text(0.01, 0.005,
             "Every interval is over the peer-critique 65-cell set, recomputed from each arm's "
             "published per-scenario medians (grok-4.20 + gemini-3.1-pro); no arm was re-run. "
             "c6masked is the team's posted 65-cell figure.  orange = this run · blue = other "
             "7% arms · gray = no-SFT references.",
             fontsize=7.5, color=TEXT2, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(png, facecolor=SURFACE)

    lines = ["# ODCV misalignment, 716-row (7%) SFT arms, same 65 cells", "",
             f"Generated {ts}. Re-summarised from published per-scenario medians (no reruns). "
             f"Cell set: `{CFG}` exclusions.", "",
             "| arm | MR | 95% CI | sev | mandated | incentivized | n | source |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sev = f"{r['sev']:.2f}" if r["sev"] is not None else "—"
        mand = f"{r['mand']:.1f}%" if r["mand"] is not None else "—"
        inc = f"{r['inc']:.1f}%" if r["inc"] is not None else "—"
        lines.append(f"| {r['label']} | {r['mr']:.1f}% | [{r['lo']:.1f}, {r['hi']:.1f}] | "
                     f"{sev} | {mand} | {inc} | {r['n']} | {r['note']} |")
    lines += ["", "Not drawn (no pullable results.json): courtroom716, peercritique716.",
              f"", f"Plot: `{png.relative_to(ROOT)}`"]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {png}\nwrote {md}")


if __name__ == "__main__":
    fire.Fire(main)
