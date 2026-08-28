# ABOUTME: The channel-swap 2x2: ODCV misalignment of A (Sonnet/Sonnet), B (grok/grok) and the two swaps
# ABOUTME: on the same 65 cells, re-summarised from per-scenario medians; figure + markdown mirror.
# Run: uv run python scratch/channel_swap/plot_2x2.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
import matplotlib
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.misalignment.odcv.odcv import summarise  # noqa: E402
from src.utils import timestamp  # noqa: E402

CFG = "configs/eval/odcv_bench_t2_9284_gtrace_sreply703_r64_paired_2x65.yaml"
OUT = ROOT / "output/channel_swap"
SONNET, GROK, CAP, GRAY, INK, MUTED, GRID = (
    "#0072B2",
    "#009E73",
    "#CC79A7",
    "#8a8985",
    "#1f1f1f",
    "#6b6b6b",
    "#e6e6e3",
)


def _latest_local(model_key: str):
    hits = sorted(
        (ROOT / "output/odcv_bench" / model_key).glob("combined*/results.json")
    )
    return hits[-1] if hits else None


# (label, trace author, reply author, source)
def arms() -> list[tuple]:
    return [
        (
            "Sonnet trace\nSonnet reply\n(A, da716)",
            "sonnet",
            "sonnet",
            (
                "LASR-Callum/qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch",
                "combined4x_20260814_230249/results.json",
            ),
        ),
        (
            "grok trace\nSonnet reply",
            "grok",
            "sonnet",
            _latest_local("qwen3_6-27b-lora-t2-9284-gtrace-sreply703-paired-r64"),
        ),
        (
            "Sonnet trace\ngrok reply",
            "sonnet",
            "grok",
            _latest_local("qwen3_6-27b-lora-t2-9284-strace-greply703-paired-r64"),
        ),
        (
            "grok trace\ngrok reply\n(B)",
            "grok",
            "grok",
            (
                "LASR-Callum/2026-08-24-odcv-grokresp703-paired-eval",
                "results/results.json",
            ),
        ),
        (
            "Sonnet capped\n(C, reference)",
            "cap",
            "cap",
            (
                "LASR-Callum/2026-08-26-odcv-sonnetconcise703-paired-eval",
                "combined2x_20260826_174216/results.json",
            ),
        ),
    ]


def _load(source) -> dict:
    path = (
        Path(source)
        if isinstance(source, Path)
        else Path(hf_hub_download(source[0], source[1], repo_type="dataset"))
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _restrict(psm: dict, excluded: set[str]) -> dict:
    return {
        v: {k: s for k, s in cells.items() if f"{v}/{k.split('/')[0]}" not in excluded}
        for v, cells in psm.items()
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = timestamp()
    excluded = set(
        OmegaConf.to_container(OmegaConf.load(ROOT / CFG).get("exclude_scenarios", []))
    )
    rows = []
    for label, tr, rp, src in arms():
        if src is None:
            print(f"skip {label!r}: no local results yet")
            continue
        s = summarise(_restrict(_load(src)["per_scenario_medians"], excluded))
        o = s["overall"]
        rows.append(
            dict(
                label=label,
                trace=tr,
                reply=rp,
                mr=o["mr_pct"],
                lo=o["mr_ci95"][0],
                hi=o["mr_ci95"][1],
                n=o.get("n_rollouts", o.get("n")),
                sev=o["mean_severity"],
                mand=s["mandated"]["mr_pct"],
                inc=s["incentivized"]["mr_pct"],
            )
        )
    color = {"sonnet": SONNET, "grok": GROK, "cap": CAP}

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = range(len(rows))
    for i, r in enumerate(rows):
        # fill = reply author, cap stripe = trace author: the two channels read off one bar.
        ax.bar(
            i,
            r["mr"],
            width=0.62,
            color=color[r["reply"]],
            linewidth=0,
            alpha=0.9 if r["trace"] == r["reply"] else 0.55,
        )
        ax.bar(
            i,
            min(r["mr"], 1.6),
            bottom=max(r["mr"] - 1.6, 0),
            width=0.62,
            color=color[r["trace"]],
            linewidth=0,
        )
        ax.plot([i, i], [r["lo"], r["hi"]], color=INK, linewidth=1.2)
        ax.text(
            i,
            r["hi"] + 1.0,
            f"{r['mr']:.1f}%",
            ha="center",
            va="bottom",
            color=INK,
            fontsize=9,
        )
    ax.set_xticks(list(x), [r["label"] for r in rows])
    ax.set_ylabel("% rollouts scored ≥3 (95% CI)", color=MUTED)
    ax.set_ylim(0, max(r["hi"] for r in rows) + 8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_title(
        "Channel swap, same 65 ODCV cells — fill: who wrote the reply · stripe: who wrote the trace",
        loc="left",
        fontsize=10,
        color=INK,
        pad=10,
    )
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=SONNET, label="Sonnet 5"),
            Patch(color=GROK, label="grok-4.6"),
            Patch(color=CAP, label="Sonnet 5 capped"),
        ],
        frameon=False,
        fontsize=8,
        loc="upper right",
    )
    fig.tight_layout()
    png = OUT / f"odcv_channel_swap_2x2_{ts}.png"
    fig.savefig(png, dpi=200)

    md = [
        f"# Channel swap 2x2 on the 65 paired ODCV cells ({ts})",
        "",
        "| arm | trace | reply | MR | 95% CI | sev | mandated | incentivized | n |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['label'].replace(chr(10), ' ')} | {r['trace']} | {r['reply']} | {r['mr']:.1f}% | [{r['lo']:.1f}, {r['hi']:.1f}] | "
            f"{r['sev']:.2f} | {r['mand']:.1f}% | {r['inc']:.1f}% | {r['n']} |"
        )
    md += [
        "",
        f"Cell set: `{CFG}` exclusions; A/B/C re-summarised from published per-scenario medians; swaps from local combined dirs.",
        f"Plot: `{png.relative_to(ROOT)}`",
    ]
    (OUT / f"odcv_channel_swap_2x2_{ts}_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(png)


if __name__ == "__main__":
    fire.Fire(main)
