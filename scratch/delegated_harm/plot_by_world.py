# ABOUTME: Draw the paper-style scenario grid from immutable published delegated-harm scores.
# ABOUTME: Run: uv run scratch/delegated_harm/plot_by_world.py --comparison <comparison_results.json>.
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from omegaconf import OmegaConf

from src.infra.huggingface import hf_snapshot
from src.naming import artifact_name
from src.utils import write_run_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    args = parser.parse_args()
    sources = json.loads(args.comparison.read_text(encoding="utf-8"))["sources"]
    config = OmegaConf.load("configs/eval/delegated_harm.yaml")
    reference_order = ["handover_gap", "scorecard_scrub", "cost_reallocation",
                       "benchmark_rescore", "permit_hours", "summary_laundering",
                       "crossclient_benchmark", "policy_threshold", "sample_shortcut",
                       "tier_screen_off", "retention_hold", "halt_threshold"]
    assert set(reference_order) == set(config.scenarios)
    deliveries = ["chat", "subagent", "chat_human"]
    colors = ["#0077AD", "#E69F00", "#009E73"]
    models = [("control", "Control"), ("da", "Difficult advice (DA)")]
    metric = "paper_complied_without_other_human_concern"
    runs = {}
    for model, _ in models:
        source = sources[model]
        root = Path(hf_snapshot(source["repo"], repo_type="dataset", revision=source["revision"],
                    allow_patterns=["results/results.json", "results/episodes/*.json"]))
        summary = json.loads((root / "results/results.json").read_text(encoding="utf-8"))
        assert summary["unjudged_completed_episodes"] == 0
        rows = [json.loads(f.read_text(encoding="utf-8"))
                for f in sorted((root / "results/episodes").glob("*.json"))]
        assert len(rows) == 324 and len({r["id"] for r in rows}) == 324
        for r in rows:
            if not r.get("metrics"):
                continue
            assert r["score_version"] == "evidence-actions-v3"
            m = r["metrics"]
            assert m[metric] == (m["paper_complied"] and not m["concern_other_human"])
        runs[model] = rows

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.edgecolor": "#d8dadd", "hatch.linewidth": 1.4,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(len(reference_order), 2, figsize=(10.5, 17.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.255, right=.97, bottom=.10, top=.875, hspace=.15, wspace=.12)
    records = []
    for row_idx, scenario in enumerate(reference_order):
        for col_idx, (model, label) in enumerate(models):
            ax = axes[row_idx, col_idx]
            ax.set_ylim(0, 122)
            ax.set_xlim(-.65, 2.65)
            ax.set_yticks([0, 50, 100])
            ax.set_yticklabels(["0", "50", "100"], color="#777b80", fontsize=8)
            ax.tick_params(axis="both", length=0, pad=5)
            ax.set_xticks([])
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#e5e7e9", linewidth=.8)
            if row_idx == 0:
                ax.set_title(label, fontsize=15, fontweight="bold", pad=17)
            if col_idx == 0:
                friendly = scenario.replace("_", " ").capitalize()
                ax.text(-.24, .46, friendly, transform=ax.transAxes,
                        ha="right", va="center", fontsize=10, fontweight="medium")
            for x, (delivery, color) in enumerate(zip(deliveries, colors)):
                scheduled = [r for r in runs[model]
                             if r["scenario"] == scenario and r["delivery"] == delivery]
                assert len(scheduled) == 9
                valid = [r for r in scheduled if r.get("metrics")]
                n = len(valid)
                positive = sum(r["metrics"][metric] for r in valid)
                requester_only = sum(r["metrics"][metric] and r["metrics"]["concern_any_human"]
                                     for r in valid) if delivery == "chat_human" else 0
                assert 0 <= requester_only <= positive <= n
                records.append(dict(scenario=scenario, model=model, delivery=delivery,
                                    planned=9, scored=n, positive=positive, requester_only=requester_only))
                if not n:
                    ax.axvspan(x-.36, x+.36, ymin=0, ymax=100/122, color="#f2f3f4", zorder=0)
                    ax.text(x, 49, "No data", ha="center", va="center", color="#8a8e94", fontsize=8)
                    ax.text(x, 5, "n=0", ha="center", color="#8a8e94", fontsize=8)
                    continue
                height = 100 * positive / n
                hatch_height = 100 * requester_only / n
                ax.bar(x, height, width=.74, color=color, zorder=3)
                if height == 0:
                    ax.hlines(0, x-.37, x+.37, color=color, linewidth=2.5, zorder=4, clip_on=False)
                if requester_only:
                    ax.bar(x, hatch_height, bottom=height-hatch_height, width=.74,
                           facecolor=color, edgecolor="white", hatch="////", linewidth=0, zorder=4)
                ax.text(x, height+4, f"{positive}/{n}", ha="center", va="bottom", fontsize=9,
                        color="#353a40")

    fig.suptitle("Complied and stayed silent, by world", y=.979,
                 fontsize=22, fontfamily="DejaVu Serif", fontweight="bold")
    fig.text(.5, .951, "Qwen3.6-27B adapters  ·  Percent of scored episodes  ·  Lower is better",
             ha="center", fontsize=10, color="#5a6067")
    handles = [Patch(facecolor=c, label=l) for c, l in zip(colors,
               ["Peer, same model", "Sub-agent", "A person"])]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .936), ncol=3,
               frameon=False, handlelength=1.8, columnspacing=2)
    fig.legend(handles=[Patch(facecolor=colors[2], edgecolor="white", hatch="////",
               label="Of which: raised concerns only with the human requester")],
               loc="upper center", bbox_to_anchor=(.5, .916), frameon=False, fontsize=9)
    footer = (
        "Bars: upstream compliance score ≥3, with no concern raised to a human other than the requester.\n"
        "Hatching is included in the green total; that employee lacks approval authority.\n"
        "Labels = qualifying / scored episodes (9 planned per bar). Missing episodes are excluded; grey = no scores.\n"
        "AI requests differ between adapters; repeats and wordings are not independent scenarios.\n"
        "12 scenarios shown; egress exception omitted because the released fixture is missing.\n"
        "Compliance uses the upstream state checks; concerns use our completed outcome judgments."
    )
    fig.text(.255, .071, footer, fontsize=8.4, color="#656b72", va="top", linespacing=1.65)
    stem = artifact_name("delegated-harm-complied-silent-by-world")
    out = Path("output/delegated_harm") / stem
    out.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out / f"{stem}.{ext}", dpi=180, facecolor="white")
    plt.close(fig)
    report = ["# Complied and stayed silent, by world", "", footer.replace("\n", "\n\n"), "",
              "| Scenario | Model | Requester | Qualifying / scored / planned | Requester-only subset |",
              "|---|---|---|---:|---:|"]
    for r in records:
        report.append(f"| {r['scenario']} | {r['model']} | {r['delivery']} | "
                      f"{r['positive']} / {r['scored']} / {r['planned']} | {r['requester_only']} |")
    for model, source in sources.items():
        report += ["", f"{model}: https://huggingface.co/datasets/{source['repo']}/tree/{source['revision']}"]
    (out / f"{stem}_results.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    (out / f"{stem}_results.json").write_text(json.dumps({"sources": sources, "metric": metric,
                    "bars": records}, indent=2)+"\n", encoding="utf-8")
    write_run_meta(out, config={"sources": sources, "scenario_order": reference_order,
                               "metric": metric, "deliveries": deliveries})
    print(out.resolve())


if __name__ == "__main__":
    main()
