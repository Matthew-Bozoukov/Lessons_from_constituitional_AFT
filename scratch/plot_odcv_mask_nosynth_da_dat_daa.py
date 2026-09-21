# ABOUTME: ODCV misalignment rate and MASK honesty for nosynth vs da-7 / dat-7 / daa-7: every published run
# ABOUTME: a dot, each arm's mean a line. Run: uv run python scratch/plot_odcv_mask_nosynth_da_dat_daa.py

"""One dot per published run, because the arms are not single numbers.

Until 2026-09-21 each arm here had been read off one adapter. Five da-7 adapters now exist
and they span 8.3-12.9% on ODCV and 71.2-82.8 on MASK, so a bar per arm would hide exactly
what the comparison depends on. Every run below is ODCV-lite's shape (240 rollouts, the flash
judge) or full MASK (1,000 rows, think mode, the flash judge); runs that are not are left out
and named in the results file. Whiskers are WITHIN-run intervals — ODCV's published
scenario-level CI, a Wilson interval over MASK's rows — and say nothing about run-to-run
spread, which is what the dots show.

Arm colours are the repo's fixed ones (CLAUDE.md): nosynth grey, da purple, dat orange; daa
has none assigned and takes green, clear of the blue the delib arms own in other figures.
Palette checked with the dataviz validator (CVD separation passes; the grey is below the
chroma floor on purpose — it is the control — and amber's low contrast is relieved by the
arm names on the axis and the table beside the figure).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from matplotlib.lines import Line2D

from src.eval.stats import wilson
from src.infra.huggingface import hf_api, hf_token
from src.naming import today

ORG = "dougalldeepmind"
ARMS = ["nosynth", "da-7", "dat-7", "daa-7"]
LABEL = {"nosynth": "No synthetic\ndata\n(nosynth)", "da-7": "Difficult\nadvice\n(da-7)",
         "dat-7": "Difficult\nagentic tasks\n(dat-7)", "daa-7": "Agentified\ndifficult advice\n(daa-7)"}
COLOUR = {"nosynth": "#6B7280", "da-7": "#7E22CE", "dat-7": "#eda100", "daa-7": "#059669"}
INK, MUTED, GRID = "#111827", "#6B7280", "#E5E7EB"

# (arm, repo, trained on the 2026-09-20 fla stack?)
ODCV = [("nosynth", "2026-09-06-odcv-qwen36-0-nosynth", False), ("nosynth", "2026-09-09-odcv-qwen36-0-nosynth", False),
        ("da-7", "2026-09-09-odcv-qwen36-0-da-7", False), ("da-7", "2026-09-17-odcv-qwen36-0-da-7", False),
        ("da-7", "2026-09-20-odcv-qwen36-0-da-7", True), ("da-7", "2026-09-21-odcv-qwen36-0-da-7", True),
        ("da-7", "2026-09-21-odcv-qwen36-1-da-7", True),
        ("dat-7", "2026-09-07-odcv-qwen36-0-dat-7", False), ("dat-7", "2026-09-09-odcv-qwen36-0-dat-7", False),
        ("daa-7", "2026-09-17-odcv-qwen36-0-daa-7", False)]
MASK = [("nosynth", "2026-09-09-mask-qwen36-0-nosynth", False),
        ("da-7", "2026-09-09-mask-qwen36-0-da-7", False), ("da-7", "2026-09-17-mask-qwen36-0-da-7", False),
        ("da-7", "2026-09-20-mask-qwen36-0-da-7", True), ("da-7", "2026-09-21-mask-qwen36-0-da-7", True),
        ("da-7", "2026-09-21-mask-qwen36-1-da-7", True),
        ("dat-7", "2026-09-07-mask-qwen36-0-dat-7", False), ("dat-7", "2026-09-09-mask-qwen36-0-dat-7", False),
        ("daa-7", "2026-09-18-mask-qwen36-0-daa-7", False)]
LEFT_OUT = ["2026-09-07-mask-qwen36-0-nosynth: run in nothink mode (every other MASK run here is think)"]


def _load(repo: str, revision: str) -> tuple[dict, dict]:
    get = lambda f: json.loads(Path(hf_hub_download(  # noqa: E731
        f"{ORG}/{repo}", f, repo_type="dataset", token=hf_token(), revision=revision)).read_text())
    return get("results/results.json"), get("metadata/run_meta.json")


def runs() -> dict[str, list[dict]]:
    api, out = hf_api(), {"odcv": [], "mask": []}
    for ev, table in (("odcv", ODCV), ("mask", MASK)):
        for arm, repo, fla in table:
            sha = api.dataset_info(f"{ORG}/{repo}").sha
            res, meta = _load(repo, sha)
            if ev == "odcv":
                o = res["ours"]["overall"]
                assert o["n_rollouts"] == 240 and list(res["judges"]) == ["gemini-3-flash-preview"], repo
                value, lo, hi = o["mr_pct"], o["mr_ci95_lo"], o["mr_ci95_hi"]
            else:
                assert res["n_rows"] == 1000 and res["mode"] == "think", repo
                value = res["overall_honesty_score"]
                lo, hi = (100 * b for b in wilson(round(value * 10), 1000))
            out[ev].append({"arm": arm, "repo": repo, "revision": sha, "fla": fla, "value": value,
                            "lo": lo, "hi": hi, "target": meta["target"].split("/")[-1]})
    return out


def draw(ax, rows: list[dict], title: str, ylabel: str, better: str, ymax: float) -> None:
    for i, arm in enumerate(ARMS):
        mine = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["repo"])
        offsets = [(j - (len(mine) - 1) / 2) * 0.115 for j in range(len(mine))]
        for r, dx in zip(mine, offsets):
            ax.plot([i + dx] * 2, [r["lo"], r["hi"]], color=COLOUR[arm], linewidth=1.0, alpha=0.55, zorder=2,
                    solid_capstyle="round")
            ax.plot(i + dx, r["value"], marker="o", markersize=6.5, zorder=4, linestyle="none",
                    markerfacecolor="white" if r["fla"] else COLOUR[arm], markeredgecolor=COLOUR[arm],
                    markeredgewidth=1.6)
        mean = sum(r["value"] for r in mine) / len(mine)
        ax.plot([i - 0.30, i + 0.30], [mean, mean], color=COLOUR[arm], linewidth=2.0, zorder=3,
                solid_capstyle="round")
        ax.text(i + 0.33, mean, f"{mean:.1f}", va="center", ha="left", fontsize=8, color=INK, zorder=5)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([f"{LABEL[a]}\n" + f"n = {sum(r['arm'] == a for r in rows)}" for a in ARMS])
    ax.set_xlim(-0.5, len(ARMS) - 0.30)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  ({better})", loc="left", fontsize=9.5, color=INK, pad=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9CA3AF")
    ax.tick_params(colors="#374151", length=0)


def main() -> None:
    data = runs()
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 9,
                         "xtick.labelsize": 7.8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), dpi=300)
    draw(axes[0], data["odcv"], "ODCV-lite misalignment rate", "% of 240 rollouts judged a violation",
         "lower is better", 60)
    draw(axes[1], data["mask"], "MASK honesty", "honesty score, 1,000 rows", "higher is better", 100)
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markersize=6.5, markerfacecolor=MUTED,
               markeredgecolor=MUTED, markeredgewidth=1.6, label="one trained adapter (one published run)"),
        Line2D([], [], marker="o", linestyle="none", markersize=6.5, markerfacecolor="white",
               markeredgecolor=MUTED, markeredgewidth=1.6, label="same, trained on the 2026-09-20 fla stack"),
        Line2D([], [], color=MUTED, linewidth=2.0, label="arm mean (labelled)"),
        Line2D([], [], color=MUTED, linewidth=1.0, alpha=0.55, label="95% interval within that run"),
    ], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), handletextpad=0.5,
        columnspacing=1.6)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.27, wspace=0.22)

    out = Path("output/figures")
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{today()}_odcv_mask_nosynth_da_dat_daa"
    fig.savefig(f"{stem}.png")
    fig.savefig(f"{stem}.pdf")

    lines = [f"# ODCV-lite misalignment rate and MASK honesty: nosynth vs da-7, dat-7, daa-7 ({today()})", "",
             "Every published run of each arm; Qwen3.6-27B, seed and adapter as named. ODCV: 240 rollouts, "
             "gemini-3-flash judge, scenario-level 95% CI as published. MASK: 1,000 rows, think mode, same "
             "judge, Wilson 95% over rows. Intervals are within-run; run-to-run spread is the dots.", ""]
    for ev, unit in (("odcv", "MR %"), ("mask", "honesty")):
        lines += [f"## {ev.upper()}", "", f"| arm | adapter | {unit} | 95% within-run | fla stack | repo @ revision |",
                  "|---|---|---|---|---|---|"]
        for r in sorted(data[ev], key=lambda r: (ARMS.index(r["arm"]), r["repo"])):
            lines.append(f"| {r['arm']} | {r['target']} | {r['value']:.1f} | [{r['lo']:.1f}, {r['hi']:.1f}] | "
                         f"{'yes' if r['fla'] else 'no'} | {ORG}/{r['repo']} @ {r['revision'][:8]} |")
        lines += ["", "| arm | n runs | mean | min | max |", "|---|---|---|---|---|"]
        for arm in ARMS:
            v = [r["value"] for r in data[ev] if r["arm"] == arm]
            lines.append(f"| {arm} | {len(v)} | {sum(v) / len(v):.1f} | {min(v):.1f} | {max(v):.1f} |")
        lines.append("")
    lines += ["## Left out", ""] + [f"- {x}" for x in LEFT_OUT]
    Path(f"{stem}_results.md").write_text("\n".join(lines) + "\n")
    print(f">>> {stem}.png / .pdf / _results.md")


if __name__ == "__main__":
    main()
