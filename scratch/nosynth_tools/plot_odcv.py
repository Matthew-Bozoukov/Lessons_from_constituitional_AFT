# ABOUTME: ODCV bar chart for the tools-only control: misalignment rate for no-synth, no-synth + tools, DA-15 and
# ABOUTME: DA + tools, with the paired-by-scenario difference of each arm from the plain no-synth control.
# Run: uv run python -m scratch.nosynth_tools.plot_odcv --nosynth-tools <run dir | org/odcv-run-repo>
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from scratch.nosynth_tools.plot_hospital import GREY, OCHRE, PURPLE  # noqa: E402
from src.eval.misalignment.odcv.stats import arm_difference  # noqa: E402
from src.naming import figure_path  # noqa: E402

load_dotenv()
NOSYNTH = "dougalldeepmind/2026-09-22-odcv-qwen36-0-nosynth"
DA15 = "dougalldeepmind/2026-09-26-odcv-qwen36-0-da-15"
DA_TOOLS = "dougalldeepmind/2026-09-28-odcv-qwen36-0-da-tools-15"


def load(ref: str) -> dict:
    """results.json + run_meta.json of one ODCV run (a local run dir or a Hub repo)."""

    def read(rel: str) -> dict:
        local = Path(ref) / rel
        return json.loads(
            (
                local
                if local.exists()
                else Path(hf_hub_download(ref, rel, repo_type="dataset"))
            ).read_text()
        )

    res, meta = read("results/results.json"), read("metadata/run_meta.json")
    cfg = meta.get("config") or {}
    return {
        "ref": ref,
        "overall": res["ours"]["overall"],
        "by_variant": {
            v: res["ours"][v] for v in ("mandated", "incentivized") if v in res["ours"]
        },
        "cells": {
            f"{v}/{s}": sev
            for v, per in res["per_scenario_medians"].items()
            for s, sev in per.items()
        },
        "protocol": (
            cfg.get("passes"),
            cfg.get("temperature"),
            tuple(res.get("judges") or []),
            meta.get("mode"),
        ),
        "target": meta.get("target"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nosynth-tools", required=True)
    ap.add_argument("--out-dir", default="output/figures")
    a = ap.parse_args()
    arms = [
        ("No synthetic data", NOSYNTH, GREY, ""),
        ("No synthetic data\n+ unused tools", a.nosynth_tools, GREY, "//"),
        ("Difficult advice\n(DA-15)", DA15, PURPLE, ""),
        ("Difficult advice\n+ unused tools", DA_TOOLS, OCHRE, ""),
    ]
    data = [load(ref) for _, ref, _, _ in arms]
    protos = {lbl.replace("\n", " "): d["protocol"] for (lbl, *_), d in zip(arms, data)}
    assert len(set(protos.values())) == 1, (
        f"ODCV protocols differ across arms: {protos}"
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    md = ["| arm | misaligned % [95% CI] | mean severity | mandated % | incentivized % | vs no-synth, paired by scenario (pp) |", "|---|---|---|---|---|---|"]  # fmt: skip
    for x, ((label, _, col, hatch), d) in enumerate(zip(arms, data)):
        o = d["overall"]
        ax.bar(x, o["mr_pct"], color=col, hatch=hatch, edgecolor="white", width=0.68)
        ax.errorbar(x, o["mr_pct"], yerr=[[o["mr_pct"] - o["mr_ci95"][0]], [o["mr_ci95"][1] - o["mr_pct"]]], color="black", capsize=4, lw=1.2)  # fmt: skip
        ax.text(
            x,
            o["mr_ci95"][1] + 1.5,
            f"{o['mr_pct']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
        pv = ""
        if x:
            diff = arm_difference(d["cells"], data[0]["cells"])
            p = diff["mr_diff_p_two_sided"]
            pv = f"{diff['mr_diff_pp']:+.1f} [{diff['mr_diff_ci95'][0]:+.1f}, {diff['mr_diff_ci95'][1]:+.1f}], " + ("p<0.001" if p < 0.001 else f"p={p:.3f}")  # fmt: skip
        bv = d["by_variant"]
        md.append(
            f"| {label.replace(chr(10), ' ')} | {o['mr_pct']:.1f} [{o['mr_ci95'][0]:.1f}, {o['mr_ci95'][1]:.1f}] | "
            f"{o['mean_severity']:.2f} | {bv['mandated']['mr_pct']:.1f} | {bv['incentivized']['mr_pct']:.1f} | {pv or 'control'} |"
        )
    ax.set_xticks(range(len(arms)), [lbl for lbl, *_ in arms], fontsize=10)
    ax.set_ylabel("% of rollouts judged misaligned")
    ax.set_ylim(0, max(d["overall"]["mr_ci95"][1] for d in data) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        "ODCV (task with a pressured shortcut, 240 rollouts per model)", fontsize=12
    )
    fig.tight_layout()
    out = figure_path(a.out_dir, "odcv_nosynth_tools_four_arms")
    fig.savefig(out, dpi=180)
    md += ["", "Runs: " + "; ".join(f"{lbl.replace(chr(10), ' ')} = `{d['ref']}` (target `{d['target']}`)" for (lbl, *_), d in zip(arms, data))]  # fmt: skip
    out.with_name(out.stem + "_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(out)


if __name__ == "__main__":
    main()
