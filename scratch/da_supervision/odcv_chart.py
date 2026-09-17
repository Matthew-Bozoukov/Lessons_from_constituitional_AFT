# ABOUTME: Creates a vertical ODCV misalignment bar chart from immutable HF result revisions.
# ABOUTME: Preserves rollout counts, published confidence intervals and dataset-version limitations.
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from omegaconf import OmegaConf

from src.infra.huggingface import hf_download
from src.eval.misalignment.odcv.odcv import VIOLATION_THRESHOLD


def main(config):
    cfg = OmegaConf.load(config)
    out = Path(cfg.output_root)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    protocols = []
    for arm in cfg.arms:

        def read(name):
            path = hf_download(
                arm.repo, name, repo_type="dataset", revision=arm.revision
            )
            return json.loads(Path(path).read_text(encoding="utf-8"))

        result, meta = read("results/results.json"), read("metadata/run_meta.json")
        overall = result["ours"]["overall"]
        scores = [
            s
            for group in result["per_scenario_medians"].values()
            for cell in group.values()
            for s in cell
        ]
        n, k = len(scores), sum(s >= VIOLATION_THRESHOLD for s in scores)
        rate = 100 * k / n
        assert n == overall["n_rollouts"] and round(rate, 1) == overall["mr_pct"]
        assert overall["n_scenarios"] == 40 and overall["n_cells"] == 80
        protocol = {
            field: meta["config"][field]
            for field in ("temperature", "serving", "judges")
        }
        protocol.update(mode=meta["mode"], base_revision=meta["base_model_revision"])
        protocols.append(protocol)
        rows.append(
            dict(
                label=str(arm.label),
                repo=str(arm.repo),
                revision=str(arm.revision),
                target=meta["target"],
                target_revision=meta["target_revision"],
                misaligned=k,
                n=n,
                mr_pct=rate,
                ci95=overall["mr_ci95"],
                passes=meta["config"]["passes"],
                color=str(arm.color),
                protocol=protocol,
            )
        )
    assert all(p == protocols[0] for p in protocols)
    assert [r["passes"] for r in rows] == [3, 3, 1, 1, 1]
    stem = out / str(cfg.figure_name)
    stem.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with stem.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "arm",
                "misaligned",
                "rollouts",
                "MR_percent",
                "CI95_low",
                "CI95_high",
                "passes",
                "dataset",
                "revision",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["label"].replace("\n", " - "),
                    r["misaligned"],
                    r["n"],
                    r["mr_pct"],
                    *r["ci95"],
                    r["passes"],
                    r["repo"],
                    r["revision"],
                ]
            )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(12.8, 7.6), facecolor="white")
    fig.subplots_adjust(left=0.085, right=0.975, top=0.81, bottom=0.27)
    values = [r["mr_pct"] for r in rows]
    ax.bar(range(5), values, width=0.62, color=[r["color"] for r in rows], zorder=3)
    errors = [
        [r["mr_pct"] - r["ci95"][0] for r in rows],
        [r["ci95"][1] - r["mr_pct"] for r in rows],
    ]
    ax.errorbar(
        range(5),
        values,
        yerr=errors,
        fmt="none",
        ecolor="#2B3440",
        elinewidth=1.5,
        capsize=5,
        capthick=1.5,
        zorder=4,
    )
    for i, r in enumerate(rows):
        label = f"{r['mr_pct']:.2f}".rstrip("0").rstrip(".") + "%"
        ax.text(
            i,
            r["ci95"][1] + 1.6,
            label,
            ha="center",
            va="bottom",
            fontsize=17,
            weight="bold",
            color="#202C3B",
        )
    ax.set_ylim(0, 65)
    ax.set_yticks(range(0, 61, 10))
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_ylabel("Misalignment rate", labelpad=12, color="#344054")
    labels = [
        r["label"] + ("\n" if "\n" in r["label"] else "\n\n") + f"n = {r['n']}"
        for r in rows
    ]
    ax.set_xticks(range(5), labels, fontsize=11)
    ax.tick_params(axis="both", length=0, pad=10, colors="#344054")
    ax.grid(axis="y", color="#E7EBEF", linewidth=0.85, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#CCD3DB")
    fig.text(
        0.085,
        0.93,
        "ODCV misalignment rates",
        fontsize=25,
        weight="bold",
        color="#182537",
    )
    fig.text(
        0.085, 0.882, "Qwen3.6-27B  ·  Lower is better", fontsize=13, color="#596779"
    )
    fig.text(
        0.085,
        0.145,
        "40 scenarios × 2 variants  ·  Temperature 0.7  ·  Thinking enabled  ·  28k context",
        fontsize=10.5,
        color="#596779",
    )
    fig.text(
        0.085,
        0.110,
        "Control and full DA: 3 passes. Supervision ablations: 1 pass. Whiskers: published scenario-based 95% CIs.",
        fontsize=10,
        color="#596779",
    )
    fig.text(
        0.085,
        0.075,
        "Full DA uses the September 8 model; ablations use September 14 DA data. One checkpoint per bar.",
        fontsize=10,
        color="#596779",
    )
    for ext in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix("." + ext), dpi=220, facecolor="white")
    plt.close(fig)
    lines = [
        "# ODCV DA supervision comparison",
        "",
        "| Arm | Misaligned / rollouts | MR | 95% CI |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['label'].replace(chr(10), ' — ')} | {r['misaligned']}/{r['n']} | {r['mr_pct']:.2f}% | {r['ci95'][0]}–{r['ci95'][1]}% |"
        )
    lines += [
        "",
        "Error bars are published scenario-based confidence intervals, not variation across training seeds.",
        "The full DA checkpoint and the ablations use different DA corpus versions; this is not a controlled supervision-only comparison.",
        "The control and full DA have three evaluation passes; the ablations have one. All five have the same 40 scenarios in both variants, base revision, temperature, thinking mode, context setting and judge.",
        "",
        "## Sources",
        "",
    ]
    for r in rows:
        lines.append(
            f"- [{r['label'].replace(chr(10), ' — ')}](https://huggingface.co/datasets/{r['repo']}/tree/{r['revision']})"
        )
    stem.with_name(stem.name + "_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(stem.with_suffix(".png").absolute())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
