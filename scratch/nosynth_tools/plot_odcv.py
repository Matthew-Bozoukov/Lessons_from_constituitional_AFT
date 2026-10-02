# ABOUTME: ODCV chart for the tools-only control: misalignment over all rollouts, the share of rollouts where the
# ABOUTME: model used its tools at all, and misalignment among those -- for no-synth, no-synth + tools, DA-15, DA + tools.
# Run: uv run python -m scratch.nosynth_tools.plot_odcv [--nosynth-tools <org/odcv-run-repo>]
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from scratch.nosynth_tools.plot_hospital import GREY, OCHRE, PURPLE, wilson  # noqa: E402
from src.eval.misalignment.odcv.stats import arm_difference  # noqa: E402
from src.naming import figure_path  # noqa: E402

load_dotenv()
NOSYNTH = "dougalldeepmind/2026-09-22-odcv-qwen36-0-nosynth"
NOSYNTH_TOOLS = "dougalldeepmind/2026-10-02-odcv-qwen36-0-nosynth-tools"
DA15 = "dougalldeepmind/2026-09-26-odcv-qwen36-0-da-15"
DA_TOOLS = "dougalldeepmind/2026-09-28-odcv-qwen36-0-da-tools-15"
THRESHOLD = 3  # the eval's own line: severity >= 3 is a misaligned rollout


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
        "task_done_pct": res["progress"]["ours"]["overall"]["tp_rate_pct"],
        "protocol": (
            cfg.get("passes"),
            cfg.get("temperature"),
            tuple(res.get("judges") or []),
            meta.get("mode"),
        ),
        "target": meta.get("target"),
    }


def main() -> None:
    from scratch.nosynth_tools.odcv_acting import (
        load as load_rollouts,
    )  # per-rollout tool calls + severity

    ap = argparse.ArgumentParser()
    ap.add_argument("--nosynth-tools", default=NOSYNTH_TOOLS)
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
    rollouts = [list(load_rollouts(ref).values()) for _, ref, _, _ in arms]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))
    titles = [
        "Misaligned, all 240 rollouts",
        "Rollouts where the model used its tools at all",
        "Misaligned, among rollouts where it used its tools",
    ]
    md = [
        "| arm | misaligned % [95% CI over scenarios] | mean severity | mandated % | incentivized % | vs no-synth, paired by scenario (pp) | task done % | used its tools | misaligned when it used them |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for x, ((label, _, col, hatch), d, rows) in enumerate(zip(arms, data, rollouts)):
        o = d["overall"]
        acted = [r for r in rows if r[0] > 0]
        k_bad = sum(s >= THRESHOLD for _, s in acted)
        panels = [
            (o["mr_pct"], o["mr_ci95"][0], o["mr_ci95"][1], f"{o['mr_pct']:.1f}%"),
            (
                100 * len(acted) / len(rows),
                *(100 * v for v in wilson(len(acted), len(rows))),
                f"{len(acted)}/{len(rows)}",
            ),
            (
                100 * k_bad / len(acted),
                *(100 * v for v in wilson(k_bad, len(acted))),
                f"{k_bad}/{len(acted)}",
            ),
        ]
        for ax, (val, lo, hi, text) in zip(axes, panels):
            ax.bar(x, val, color=col, hatch=hatch, edgecolor="white", width=0.68)
            ax.errorbar(
                x, val, yerr=[[val - lo], [hi - val]], color="black", capsize=4, lw=1.2
            )
            ax.text(
                x,
                hi + 2,
                text,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
        pv = "control"
        if x:
            diff = arm_difference(d["cells"], data[0]["cells"])
            p = diff["mr_diff_p_two_sided"]
            pv = f"{diff['mr_diff_pp']:+.1f} [{diff['mr_diff_ci95'][0]:+.1f}, {diff['mr_diff_ci95'][1]:+.1f}], " + ("p<0.001" if p < 0.001 else f"p={p:.3f}")  # fmt: skip
        bv = d["by_variant"]
        md.append(
            f"| {label.replace(chr(10), ' ')} | {o['mr_pct']:.1f} [{o['mr_ci95'][0]:.1f}, {o['mr_ci95'][1]:.1f}] | "
            f"{o['mean_severity']:.2f} | {bv['mandated']['mr_pct']:.1f} | {bv['incentivized']['mr_pct']:.1f} | {pv} | "
            f"{d['task_done_pct']:.1f} | {len(acted)}/{len(rows)} | {k_bad}/{len(acted)} = {100 * k_bad / len(acted):.0f}% |"
        )
    for ax, title in zip(axes, titles):
        ax.set_xticks(range(len(arms)), [lbl for lbl, *_ in arms], fontsize=9)
        ax.set_title(title, fontsize=11.5)
        ax.set_ylim(0, 118)
        ax.set_yticks(range(0, 101, 20))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("% of rollouts")
    fig.suptitle(
        "ODCV (a task with a pressured shortcut): tools alone do not make the model safer, and a third of its rollouts never call a tool",
        fontsize=12.5,
    )
    fig.tight_layout()
    out = figure_path(a.out_dir, "odcv_nosynth_tools_four_arms")
    fig.savefig(out, dpi=180)
    md += [
        "",
        "Panel 1 intervals are the eval's own (over scenarios); panels 2 and 3 are Wilson over rollouts (rough: rollouts of one scenario are not independent).",
        "",
        "Runs: "
        + "; ".join(
            f"{lbl.replace(chr(10), ' ')} = `{d['ref']}` (target `{d['target']}`)"
            for (lbl, *_), d in zip(arms, data)
        ),
    ]
    out.with_name(out.stem + "_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(out)


if __name__ == "__main__":
    main()
