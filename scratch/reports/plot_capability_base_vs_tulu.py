# ABOUTME: Capability comparison plot for base Qwen3.6-27B vs the 80:20 tulu-difficult-advice
# ABOUTME: mixture SFT LoRA: MMLU-CoT accuracy + LMSYS pairwise win-rate, with a markdown mirror.

from __future__ import annotations

import json
import time
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np


def main(
    lmsys_stats: str = "output/capability_qwen36/20260729/lmsys/stats.json",
    mmlu_base_acc: float = 0.905,
    mmlu_base_se: float = 0.0208,
    mmlu_ft_acc: float = 0.885,
    mmlu_ft_se: float = 0.0226,
    out_dir: str = "output/report",
) -> None:
    """Plot MMLU accuracy + LMSYS win-rate for base vs the 80:20 tulu SFT LoRA.

    Args:
        lmsys_stats: stats.json from lmsys_eval.py.
        mmlu_base_acc: base MMLU-CoT accuracy (0-1).
        mmlu_base_se: base MMLU stderr.
        mmlu_ft_acc: fine-tune MMLU-CoT accuracy (0-1).
        mmlu_ft_se: fine-tune MMLU stderr.
        out_dir: output directory.
    """
    lm = json.loads(Path(lmsys_stats).read_text())
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    c_base, c_ft = "#c0392b", "#2e86c1"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- MMLU accuracy ---
    x = np.arange(2)
    accs = [100 * mmlu_base_acc, 100 * mmlu_ft_acc]
    ses = [100 * mmlu_base_se, 100 * mmlu_ft_se]
    ax1.bar(x, accs, 0.55, yerr=[1.96 * s for s in ses], capsize=6,
            color=[c_base, c_ft], error_kw=dict(lw=1.8, ecolor="black", capthick=1.8))
    for xi, a in zip(x, accs):
        ax1.text(xi, a + 1.2, f"{a:.1f}%", ha="center", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Base\nQwen3.6-27B", "+ tulu-difficult-advice\nmixture LoRA (80:20)"], fontsize=9)
    ax1.set_ylabel("MMLU accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("MMLU (0-shot CoT), 200 paired questions\nerror bars = 95% CI")
    ax1.grid(axis="y", alpha=0.3)

    # --- LMSYS outcomes ---
    labels = ["Base wins", "Ties", "LoRA wins"]
    vals = [lm["base_wins"], lm["ties"], lm["ft_wins"]]
    colors = [c_base, "#95a5a6", c_ft]
    ax2.bar(labels, vals, 0.6, color=colors)
    for xi, v in enumerate(vals):
        ax2.text(xi, v + 0.4, str(v), ha="center", fontsize=11)
    ax2.set_ylabel("Prompts (of 40)")
    ax2.set_ylim(0, max(vals) + 4)
    ax2.set_title(f"LMSYS-subset chat quality (pairwise, gemini-3-flash judge)\n"
                  f"LoRA win-rate (excl. ties) = {lm['ft_winrate_excl_ties_pct']}%")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Capability: base Qwen3.6-27B vs 80:20 tulu-difficult-advice mixture LoRA",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    png = out / f"capability_base_vs_tulu_{ts}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f">>> {png}")

    # Markdown mirror.
    md = out / f"capability_base_vs_tulu_{ts}.md"
    md.write_text(f"""# Capability: base Qwen3.6-27B vs 80:20 tulu-difficult-advice mixture LoRA

Served via vLLM (base `qwen3` + `tulu` LoRA), thinking mode. {ts}

## MMLU (0-shot CoT), 200 paired questions (seed 42)

| Model | Accuracy | stderr |
|---|---|---|
| Base Qwen3.6-27B | {100*mmlu_base_acc:.1f}% | ±{100*mmlu_base_se:.1f} |
| + tulu mixture LoRA (80:20) | {100*mmlu_ft_acc:.1f}% | ±{100*mmlu_ft_se:.1f} |

Delta = {100*(mmlu_ft_acc-mmlu_base_acc):+.1f} pt (~1 stderr; knowledge/reasoning essentially preserved).

## LMSYS-subset chat quality, 40 prompts (pairwise, gemini-3-flash, position-randomized)

| Outcome | Count |
|---|---|
| Base wins | {lm['base_wins']} |
| Ties | {lm['ties']} |
| LoRA wins | {lm['ft_wins']} |

- **LoRA win-rate (excl. ties) = {lm['ft_winrate_excl_ties_pct']}%** (ties=½: {lm['ft_winrate_ties_half_pct']}%)
- Base is clearly preferred on general chat quality.

![plot]({png.name})
""")
    print(f">>> {md}")


if __name__ == "__main__":
    fire.Fire(main)
