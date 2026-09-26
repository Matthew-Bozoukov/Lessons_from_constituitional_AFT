# ABOUTME: Plots the cumulative trajectory collusion rate (share of trajectories that have colluded by
# ABOUTME: each episode) for each published agent_collusion arm, from its published results.
"""Reads the published eval runs from the Hub; writes
output/plots/2026-09-26_agent_collusion_cumulative_tc_three_arms.{png,md}."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

ARMS = {  # repo-wide arm colours: nosynth grey, plain difficult advice (da) purple; the
    # multiparty-human DA variant gets its own teal so it is not read as plain da
    "nosynth": ("dougalldeepmind/2026-09-24-collusion-qwen36-0-nosynth", "#7f7f7f"),
    "da-15": ("dougalldeepmind/2026-09-24-collusion-qwen36-0-da-15", "#7b3fa0"),
    "da-multiparty-human-15": ("dougalldeepmind/2026-09-26-collusion-qwen36-0-da-multiparty-human-15",
                               "#2a9d8f"),
}
OUT = Path("output/plots/2026-09-26_agent_collusion_cumulative_tc_three_arms")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k of n, as percentages."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def main() -> None:
    """Load both runs' onset data, plot the cumulative curves, write PNG + markdown."""
    load_dotenv()
    episodes = list(range(1, 11))
    fig, ax = plt.subplots(figsize=(9, 6))
    rows = []
    for name, (repo, colour) in ARMS.items():
        res = json.load(open(hf_hub_download(repo, "results/results.json", repo_type="dataset")))
        s = res.get("summary", res)
        n = s["n_trajectories"]
        onsets = s["onsets"]
        k = [sum(o <= e for o in onsets) for e in episodes]
        y = [100 * x / n for x in k]
        lo, hi = zip(*(wilson(x, n) for x in k))
        ax.fill_between(episodes, lo, hi, color=colour, alpha=0.15, linewidth=0)
        ax.plot(episodes, y, color=colour, linewidth=2.4, marker="o", markersize=8,
                markeredgecolor="black", markeredgewidth=0.8,
                label=f"{name}  (n={n})")
        rows.append((name, n, onsets, y))
    ax.set_xlabel("Episode", fontsize=16)
    ax.set_ylabel("Trajectories that have colluded (%)", fontsize=16)
    ax.set_title("Collusion onset by arm", fontsize=16)
    ax.set_xticks(episodes)
    ax.set_ylim(0, 102)
    ax.set_xlim(0.7, 10.3)
    ax.tick_params(labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend(fontsize=14, loc="lower right", frameon=True)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=160)

    md = [f"# {OUT.name}", "", f"![plot]({OUT.name}.png)", "",
          "Cumulative share of trajectories with at least one mutual ACCEPT by each episode "
          "(each run stopped a trajectory at its first collusion). Shaded: 95% Wilson intervals.",
          "", "| episode | " + " | ".join(str(e) for e in episodes) + " |",
          "|---|" + "---:|" * len(episodes)]
    for name, n, onsets, y in rows:
        md.append(f"| {name} (n={n}) | " + " | ".join(f"{v:.0f}" for v in y) + " |")
    md += ["", *(f"- {name} onsets: {sorted(o)}" for name, _, o, _ in rows),
           "", "Sources: " + ", ".join(f"`{r}`" for r, _ in ARMS.values())
           + "."]
    OUT.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"wrote {OUT}.png and .md")


if __name__ == "__main__":
    main()
