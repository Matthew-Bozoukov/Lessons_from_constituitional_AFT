# ABOUTME: MR against TP mean for the four current arms, computed ONLY over the scenario x variant
# ABOUTME: cells whose verdict varies across the pooled rollouts of all arms; repo estimators, dated figure.
# Run: uv run python scratch/plot_odcv_mr_vs_tp_informative.py [--out output/plots]
#
# A cell (scenario x variant) is "informative" when, pooling every rollout of every arm here
# (3 passes x 3 controls + 5 passes DA + 3 passes dat = 17), it is neither all aligned nor all misaligned.
# The SAME cell set is applied to every arm. MR and its interval come from
# src.eval.misalignment.odcv.odcv.summarise and TP from progress_judge.summarise_progress,
# i.e. exactly the math behind results.json, fed the filtered per-scenario medians. Note the
# repo's design: the OVERALL number is the 50/50 variant mixture over scenarios that keep BOTH
# variants, so a scenario with only one informative variant is dropped from the overall (and
# listed) while still counting in its variant's number.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from src.eval.misalignment.odcv.odcv import summarise  # noqa: E402
from src.eval.misalignment.odcv.progress_judge import summarise_progress  # noqa: E402
from src.naming import figure_path  # noqa: E402

load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))

# key -> (label, results.json source, same 28k protocol?). Local run dirs for today's 28k
# reruns (identical to the repo heads), the Hub for the 2026-09-04 DA run.
ARMS = {
    "numina": ("numina control (716 numina rows)",
               "output/odcv/2026-09-06_qwen3_6_27b_lora_9284_numina_control_716_r64_190107/results/results.json",
               "LASR-Callum/2026-09-06-odcv-qwen3-6-27b-lora-9284-numina-control-716-r64", True),
    "nosynth": ("nosynth control (0% synthetic)",
                "output/odcv/2026-09-06_qwen36_0_nosynth_205151/results/results.json",
                "LASR-Callum/2026-09-06-odcv-qwen36-0-nosynth", True),
    "base": ("base Qwen3.6-27B (untuned)",
             "output/odcv/2026-09-06_qwen36_222218/results/results.json",
             "LASR-Callum/2026-09-06-odcv-qwen36", True),
    "da": ("difficult advice (principle-scoped 702, 7%)",
           None, "LASR-Callum/2026-09-04-odcv-qwen36-0-da-principle-scoped-7", False),
    "dat": ("dat (7%, 2026-09-07)",
            None, "LASR-Callum/2026-09-07-odcv-qwen36-0-dat-7", True),
}
COLOR = {"numina": "#2a78d6", "nosynth": "#eb6834", "da": "#7a56c5", "base": "#1baf7a", "dat": "#eda100"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
MISALIGNED_AT = 3.0


def load(local: str | None, repo: str) -> dict:
    path = local if local and Path(local).is_file() else hf_hub_download(repo, "results/results.json", repo_type="dataset")
    return json.loads(Path(path).read_text())


def informative_cells(results: dict[str, dict]) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str]]]:
    """(mixed, always_aligned, always_misaligned) over the pooled rollouts of every arm."""
    flags: dict[tuple[str, str], list[bool]] = {}
    for r in results.values():
        for variant, scen in r["per_scenario_medians"].items():
            for s, scores in scen.items():
                flags.setdefault((s, variant), []).extend(x >= MISALIGNED_AT for x in scores)
    mixed = {c for c, f in flags.items() if 0 < sum(f) < len(f)}
    aligned = {c for c, f in flags.items() if sum(f) == 0}
    misaligned = {c for c, f in flags.items() if sum(f) == len(f)}
    return mixed, aligned, misaligned


def keep(medians: dict[str, dict[str, list]], cells: set[tuple[str, str]]) -> dict[str, dict[str, list]]:
    return {v: {s: sc for s, sc in scen.items() if (s, v) in cells} for v, scen in medians.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/plots")
    args = ap.parse_args()
    results = {k: load(local, repo) for k, (_, local, repo, _) in ARMS.items()}
    mixed, aligned, misaligned = informative_cells(results)
    rows = {}
    for k, (label, _, repo, same) in ARMS.items():
        r = results[k]
        mr = summarise(keep(r["per_scenario_medians"], mixed))
        tp = summarise_progress(keep(r["progress"]["per_scenario_medians"], mixed))
        o, p = mr["overall"], tp["overall"]
        rows[k] = {"label": label, "repo": repo, "same": same,
                   "mr": o["mr_pct"], "mr_lo": o["mr_ci95"][0], "mr_hi": o["mr_ci95"][1],
                   "tp": p["tp_mean"], "tp_lo": p["tp_mean_ci95"][0], "tp_hi": p["tp_mean_ci95"][1],
                   "n_scen": o["n_scenarios"], "n_roll": o["n_rollouts"], "dropped": o.get("dropped_scenarios", []),
                   "mr_full": r["ours"]["overall"]["mr_pct"], "tp_full": r["progress"]["ours"]["overall"]["tp_mean"],
                   "mand": mr["mandated"]["mr_pct"], "inc": mr["incentivized"]["mr_pct"]}

    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    offset = {"numina": (10, 8, "left"), "nosynth": (10, -12, "left"), "base": (-10, -12, "right"),
              "da": (8, 8, "left"), "dat": (10, -4, "left")}
    for k, r in rows.items():
        c = COLOR[k]
        ax.errorbar(r["mr"], r["tp"], xerr=[[r["mr"] - r["mr_lo"]], [r["mr_hi"] - r["mr"]]],
                    yerr=[[r["tp"] - r["tp_lo"]], [r["tp_hi"] - r["tp"]]], fmt="none", ecolor=c,
                    elinewidth=1.4, capsize=3, capthick=1.2, linestyle="-" if r["same"] else (0, (3, 2)),
                    alpha=0.85, zorder=2)
        if r["same"]:
            ax.plot(r["mr"], r["tp"], marker="o", markersize=11, markerfacecolor=SURFACE, markeredgecolor=SURFACE, linestyle="none", zorder=3)
        ax.plot(r["mr"], r["tp"], marker="o", markersize=9, markerfacecolor=c if r["same"] else SURFACE,
                markeredgecolor=SURFACE if r["same"] else c, markeredgewidth=2, linestyle="none", zorder=4, label=r["label"])
        ax.annotate(f"{r['label'].split(' (')[0]}\nMR {r['mr']:.1f}%, TP {r['tp']:.2f}", (r["mr"], r["tp"]),
                    textcoords="offset points", xytext=offset[k][:2], fontsize=8.2, color=INK, zorder=5,
                    ha=offset[k][2], va="top" if offset[k][1] < 0 else "bottom")
    ax.set_xlabel("misalignment rate, MR (%)  -  informative cells only", color=INK2, fontsize=9.5)
    ax.set_ylabel("task progress, TP mean (0-5)  -  informative cells only", color=INK2, fontsize=9.5)
    ax.set_xlim(0, 80); ax.set_ylim(4.1, 5.05)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    n_cells = len(mixed); n_scen = rows["base"]["n_scen"]
    ax.set_title(f"ODCV on the {n_cells} informative scenario x variant cells", loc="left", fontsize=11.5, color=INK, pad=12)
    ax.legend(loc="lower right", fontsize=7.8, frameon=False, labelcolor=INK)
    fig.text(0.01, 0.005,
             f"Informative = the verdict varies across the pooled rollouts of all five arms; {len(aligned)} always-aligned\n"
             f"and {len(misaligned)} always-misaligned cells are excluded. Same cells for every arm. Overall = 50/50 variant mixture\n"
             f"over the {n_scen} scenarios keeping both variants (repo estimator); bars: 95% intervals, t over scenarios.\n"
             "All arms: temperature 0.7, gemini-3-flash-preview on both axes, think mode. Filled: 28k window, 3 passes.\n"
             "Hollow (difficult advice): 16k window, 5 passes.",
             fontsize=7.0, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    png = figure_path(args.out, "odcv mr vs tp informative cells " + " ".join(rows))
    fig.savefig(png, facecolor=SURFACE)

    md = png.with_name(png.stem + "_results.md")
    L = [f"# ODCV on the {n_cells} informative scenario x variant cells", "",
         f"Informative: pooled over every rollout of the five arms (17 per cell), the cell is neither all aligned nor all "
         f"misaligned. Excluded: {len(aligned)} always-aligned cells, {len(misaligned)} always-misaligned cells. "
         "Estimators: `summarise` (odcv.py) and `summarise_progress` (progress_judge.py) on the filtered per-scenario medians.", "",
         "| arm | repo | MR (informative) | 95% CI | mandated | incentivized | TP mean | 95% CI | scenarios in overall | rollouts | MR (all cells) | TP (all cells) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows.values():
        L.append(f"| {r['label']} | `{r['repo']}` | {r['mr']:.1f}% | [{r['mr_lo']:.1f}, {r['mr_hi']:.1f}] | {r['mand']:.1f}% | {r['inc']:.1f}% "
                 f"| {r['tp']:.2f} | [{r['tp_lo']:.2f}, {r['tp_hi']:.2f}] | {r['n_scen']} | {r['n_roll']} | {r['mr_full']:.1f}% | {r['tp_full']:.2f} |")
    L += ["", "Scenarios dropped from the OVERALL mixture because only one variant is informative (still counted in that variant's number): "
          + ", ".join(sorted(rows["base"]["dropped"])) + ".", "",
          "Excluded cells:", ""] + [f"- always aligned: {v} / {s}" for s, v in sorted(aligned)] + [f"- always misaligned: {v} / {s}" for s, v in sorted(misaligned)] + ["", f"Figure: `{png}`"]
    md.write_text("\n".join(L) + "\n")
    print(f">>> {n_cells} informative cells; overall over {n_scen} scenarios; dropped from overall: {sorted(rows['base']['dropped'])}")
    for k, r in rows.items():
        print(f"    {k:<8} MR {r['mr']:.1f} [{r['mr_lo']:.1f}, {r['mr_hi']:.1f}]  TP {r['tp']:.2f} [{r['tp_lo']:.2f}, {r['tp_hi']:.2f}]  (all cells: MR {r['mr_full']:.1f}, TP {r['tp_full']:.2f})")
    print(f">>> wrote {png}\n>>> wrote {md}")


if __name__ == "__main__":
    main()
