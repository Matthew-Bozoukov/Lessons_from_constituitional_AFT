# ABOUTME: Histograms of per-turn ODCV reasoning-trace lengths (chars), parsed from each cell's
# ABOUTME: docker_output.log ([Reasoning]: blocks), for today's four arms vs the 10-02 nosynth / da-15 runs.
# Run: uv run python scratch/plot_odcv_reasoning_lengths.py
import glob, re, statistics, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download

from src.naming import figure_path

DATE = "2026-10-05"
# Local (today's in-progress runs) and published (older) sources: name -> glob of docker_output.log files.
LOCAL = {
    "msm (today)": "output/odcv/2026-10-05_qwen36_0_msm_213611/qwen36_0_msm/2*/agent_logs/*/experiments/*/docker_output.log",
    "plain (today)": "output/odcv/2026-10-05_qwen36_0_plain_213701/qwen36_0_plain/2*/agent_logs/*/experiments/*/docker_output.log",
    "da-msm-15 (today)": "output/odcv/2026-10-05_qwen36_0_da_msm_15_*/qwen36_0_da_msm_15/2*/agent_logs/*/experiments/*/docker_output.log",
    "da-15 (today)": "output/odcv/2026-10-05_qwen36_0_da_15_*/qwen36_0_da_15/2*/agent_logs/*/experiments/*/docker_output.log",
}
HUB = {
    "nosynth (10-02)": "dougalldeepmind/2026-10-02-odcv-qwen36-0-nosynth",
    "da-15 (10-02)": "dougalldeepmind/2026-10-02-odcv-qwen36-0-da-15",
}
# Fixed arm colours (CLAUDE.md): nosynth-like controls grey, da purple.
COLOURS = {"msm (today)": "#555555", "plain (today)": "#a0a0a0", "nosynth (10-02)": "#222222",
           "da-msm-15 (today)": "#6a1b9a", "da-15 (today)": "#b088d0", "da-15 (10-02)": "#3f0f5c"}

LINE = re.compile(r"^executor-1\s+\| ?(.*)$")
TAG = re.compile(r"^\[(Reasoning|Reply|Tool|out|Error|System)\]: ?(.*)$")


def reasoning_blocks(path: Path) -> list[int]:
    """Lengths of every [Reasoning]: block in one cell's executor log (multi-line until the next tag)."""
    lengths, buf, inside = [], [], False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE.match(raw)
        if not m:
            continue
        body = m.group(1)
        t = TAG.match(body)
        if t or body.startswith("=== CYCLE"):
            if inside:
                lengths.append(len("\n".join(buf).strip())); buf, inside = [], False
            if t and t.group(1) == "Reasoning":
                inside = True; buf = [t.group(2)]
            continue
        if inside:
            buf.append(body)
    if inside:
        lengths.append(len("\n".join(buf).strip()))
    return lengths


def collect(files: list[str]) -> dict:
    per_turn, per_cell = [], []
    for f in files:
        ls = reasoning_blocks(Path(f))
        if ls:
            per_turn += ls; per_cell.append(sum(ls))
    return {"cells": len(per_cell), "turns": len(per_turn), "per_turn": per_turn, "per_cell": per_cell}


def main():
    data = {}
    for name, pat in LOCAL.items():
        data[name] = collect(sorted(glob.glob(pat)))
    for name, repo in HUB.items():
        root = snapshot_download(repo, repo_type="dataset", allow_patterns=["rollouts/*/*/pass*/docker_output.log"])
        data[name] = collect(sorted(glob.glob(f"{root}/rollouts/*/*/pass*/docker_output.log")))
    order = ["nosynth (10-02)", "msm (today)", "plain (today)", "da-15 (10-02)", "da-15 (today)", "da-msm-15 (today)"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins_turn = [0, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
    for name in order:
        d = data[name]
        if not d["turns"]:
            continue
        axes[0].hist(d["per_turn"], bins=bins_turn, histtype="step", linewidth=2, density=True,
                     color=COLOURS[name], label=f"{name}: median {statistics.median(d['per_turn']):.0f}, n={d['turns']}")
        axes[1].hist(d["per_cell"], bins=20, range=(0, 150000), histtype="step", linewidth=2, density=True,
                     color=COLOURS[name], label=f"{name}: median {statistics.median(d['per_cell'])/1000:.0f}k, cells={d['cells']}")
    axes[0].set_xscale("log"); axes[0].set_xlabel("reasoning chars per assistant turn (log bins)"); axes[0].set_ylabel("density")
    axes[0].set_title("ODCV: reasoning length per turn"); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("total reasoning chars per cell"); axes[1].set_title("ODCV: total reasoning per cell"); axes[1].legend(fontsize=7)
    out = figure_path("output", "odcv_reasoning_lengths", date=DATE)
    fig.tight_layout(); fig.savefig(out, dpi=140)
    md = [f"# ODCV reasoning-trace lengths ({DATE})", "", "Parsed from `[Reasoning]:` blocks in each cell's docker_output.log. Today's runs are partial (in progress).", "",
          "| run | cells | turns | median chars/turn | p90 chars/turn | median chars/cell | median turns/cell |", "|---|---|---|---|---|---|---|"]
    for name in order:
        d = data[name]
        if not d["turns"]:
            md.append(f"| {name} | 0 | 0 | - | - | - | - |"); continue
        pt = sorted(d["per_turn"]); p90 = pt[int(0.9 * (len(pt) - 1))]
        md.append(f"| {name} | {d['cells']} | {d['turns']} | {statistics.median(pt):.0f} | {p90} | {statistics.median(d['per_cell']):.0f} | {d['turns']/d['cells']:.1f} |")
    (Path("output") / f"{DATE}_odcv_reasoning_lengths_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md)); print(out)


if __name__ == "__main__":
    main()
