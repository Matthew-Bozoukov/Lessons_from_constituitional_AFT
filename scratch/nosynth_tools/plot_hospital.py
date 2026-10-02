# ABOUTME: Hospital (self-sacrificial cell) bar chart for the tools-only control: shifts sabotaged out of 30 for
# ABOUTME: no-synth, no-synth + tools, DA-15 and DA + tools, with Wilson intervals and a refusal panel.
# Run: uv run python -m scratch.nosynth_tools.plot_hospital --nosynth <run dir | org/repo> --nosynth-tools <run dir | org/repo>
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from src.eval.stats import mcnemar_exact  # noqa: E402
from src.naming import figure_path  # noqa: E402

load_dotenv()
CONDITION = "self_sacrificial"
DA15 = "dougalldeepmind/2026-09-28-hospital-self-sacrificial-qwen36-0-da-15"
DA_TOOLS = "dougalldeepmind/2026-09-28-hospital-self-sacrificial-qwen36-0-da-tools-15"
# Fixed arm colours (CLAUDE.md): nosynth grey, difficult advice purple; DA + tools ochre as in the paper figures.
# The tools-only control keeps its family's grey and takes the hatch.
GREY, PURPLE, OCHRE = "#6B7280", "#7E22CE", "#9C6500"
PANELS = [
    ("pair_sabotaged", "Shifts where the tested pair sabotaged"),
    ("refused_and_held", "Shifts where it refused and never sabotaged"),
]


def per_seed(ref: str) -> dict[str, dict[int, float]]:
    """{measure: {seed: value}} for the self-sacrificial cell of one run (a local run dir or a Hub repo)."""
    local = Path(ref)
    path = local / "results/results.json" if local.exists() else hf_hub_download(ref, "results/results.json", repo_type="dataset")  # fmt: skip
    m = json.loads(Path(path).read_text())["measures"]
    return {k: {int(s): float(v) for s, v in m[k][CONDITION].items() if v is not None} for k in m if CONDITION in (m[k] or {})}  # fmt: skip


def wilson(k: int, n: int) -> tuple[float, float]:
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nosynth", required=True)
    ap.add_argument("--nosynth-tools", required=True)
    ap.add_argument("--out-dir", default="output/figures")
    a = ap.parse_args()
    arms = [
        ("No synthetic data", a.nosynth, GREY, ""),
        ("No synthetic data\n+ unused tools", a.nosynth_tools, GREY, "//"),
        ("Difficult advice\n(DA-15)", DA15, PURPLE, ""),
        ("Difficult advice\n+ unused tools", DA_TOOLS, OCHRE, ""),
    ]
    data = [per_seed(ref) for _, ref, _, _ in arms]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    lines = []
    for ax, (key, title) in zip(axes, PANELS):
        for x, ((label, _, col, hatch), d) in enumerate(zip(arms, data)):
            v = d[key]
            k, n = int(sum(v.values())), len(v)
            lo, hi = wilson(k, n)
            ax.bar(
                x, 100 * k / n, color=col, hatch=hatch, edgecolor="white", width=0.68
            )
            ax.errorbar(x, 100 * k / n, yerr=[[100 * (k / n - lo)], [100 * (hi - k / n)]], color="black", capsize=4, lw=1.2)  # fmt: skip
            ax.text(
                x,
                100 * hi + 2.5,
                f"{k}/{n}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
            lines.append(f"| {title} | {label.replace(chr(10), ' ')} | {k}/{n} | {100 * k / n:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}] |")  # fmt: skip
        ax.set_xticks(range(len(arms)), [lbl for lbl, *_ in arms], fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 118)
        ax.set_yticks(range(0, 101, 20))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("% of 30 shifts")
    fig.suptitle("Hospital, two seats told to make the team score worse: the large drop needs difficult advice and tools together", fontsize=12.5)  # fmt: skip
    fig.tight_layout()
    out = figure_path(a.out_dir, "hospital_nosynth_tools_four_arms")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)

    # in the table only: intent (sabotaged, or posted a sabotage plan on the coalition board) and private refusal
    for key, title in (("sabotaged_or_planned", "Shifts where it sabotaged or posted a sabotage plan"),
                       ("declined_in_reasoning", "Shifts where it said no in its private reasoning")):
        for (label, *_), d in zip(arms, data):
            if key in d:
                k, n = int(sum(d[key].values())), len(d[key])
                lo, hi = wilson(k, n)
                lines.append(f"| {title} | {label.replace(chr(10), ' ')} | {k}/{n} | {100 * k / n:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}] |")  # fmt: skip
    # paired by seed: the tools-only control against the plain control
    ns, nst = data[0]["pair_sabotaged"], data[1]["pair_sabotaged"]
    seeds = sorted(set(ns) & set(nst))
    b = sum(ns[s] == 1 and nst[s] == 0 for s in seeds)
    c = sum(ns[s] == 0 and nst[s] == 1 for s in seeds)
    md = [
        "| measure | arm | shifts | rate [95% Wilson] |", "|---|---|---|---|", *lines, "",
        f"Paired by seed, sabotage, no-synth vs no-synth + tools ({len(seeds)} seeds): only no-synth sabotaged {b}, "
        f"only no-synth + tools sabotaged {c}, McNemar exact p = {mcnemar_exact(b, c):.3f}.",
        "", "Runs: " + "; ".join(f"{lbl.replace(chr(10), ' ')} = `{ref}`" for lbl, ref, *_ in arms),
    ]  # fmt: skip
    out.with_name(out.stem + "_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(out)


if __name__ == "__main__":
    main()
