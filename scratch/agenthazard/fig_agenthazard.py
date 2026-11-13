# ABOUTME: AgentHazard (Claude Code) figure for the three Qwen3.6 arms in the paper's figure style:
# ABOUTME: round_all ASR with Wilson 95% CIs, ASR by cumulative round, by risk category, by attack strategy.
"""Reads each arm's published eval run from the Hub (results/judgments.jsonl + episodes.jsonl) and draws
one 2x2 figure. Rates are per instance at round_all (judge errors excluded; there were none), round k on
instances with >= k steps (the paper's rule). Category and strategy cells hold ~10 instances each, so they
carry no intervals.

Run: uv run python scratch/agenthazard/fig_agenthazard.py control=<repo> da=<repo> da_tools=<repo>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scratch.figures import _style as S  # noqa: E402
from src.eval.misalignment.agenthazard.stats import summarize  # noqa: E402

CATS = [
    ("System_RCE", "RCE"),
    ("Data_Exfiltration", "Exfil"),
    ("Persistence_Establishment", "Persist"),
    ("Supply_Chain_Poisoning", "Supply"),
    ("Destructive_Action", "Destruct"),
    ("Internal_Reconnaissance", "Recon"),
    ("Privilege_Escalation", "PrivEsc"),
    ("Resource_Exhaustion", "ResExh"),
    ("Prompt_Intel_Theft", "PrmTheft"),
    ("Compliance_Bypass", "Comply"),
]  # fmt: skip  (the paper's Table 2 order and abbreviations)
STRATS = [
    ("Direct", "Direct"),
    ("Contextual_Persona_Enforcement", "CPE"),
    ("Dependency_Hook_Trigger", "DHT"),
    ("Encoded_Payload_Smuggling", "EPS"),
    ("Implicit_Indirect_Injection", "III"),
    ("Legacy_Code_Maintenance", "LCM"),
    ("Logical_Dependency_Inversion", "LDI"),
    ("Pseudo_Diagnostic_Debugging", "PDD"),
    ("Recursive_Salami_Slicing", "RSS"),
    ("Security_Audit_Gaslighting", "SAG"),
]  # fmt: skip  (the paper's Table 7 order and abbreviations)
ROUNDS = ["R1", "R2", "R3", "R4", "all"]
ORDER = ["control", "da", "da_tools"]


def load(repo: str) -> dict:
    from huggingface_hub import hf_hub_download

    rows = {}
    for f in ("judgments", "episodes"):
        local = (
            Path(repo) / "results" / f"{f}.jsonl"
        )  # a local run dir: previews before the push
        p = (
            local
            if local.exists()
            else hf_hub_download(repo, f"results/{f}.jsonl", repo_type="dataset")
        )
        rows[f] = [
            json.loads(line)
            for line in Path(p).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return summarize(rows["judgments"], rows["episodes"])


def main(argv: list[str]) -> None:
    repos = dict(a.split("=", 1) for a in argv)
    arms = [a for a in ORDER if a in repos]
    stats = {a: load(repos[a]) for a in arms}
    S.apply()
    fig = plt.figure(figsize=(22, 19))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.6], hspace=0.45, wspace=0.18)
    ax_all, ax_rnd = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_cat, ax_str = fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :])

    # (a) round_all ASR with Wilson 95% CIs; severity under each bar
    for i, a in enumerate(arms):
        r = stats[a]["asr_round_all"]
        S.bar(ax_all, i, 100 * r["asr"], 100 * r["ci95"][0], 100 * r["ci95"][1], colour=S.COLOUR[a],
              label=f"{100 * r['asr']:.0f}%")  # fmt: skip
        ax_all.text(i, 2, f"n={r['n']}\nsev {r['mean_score']:.1f}", ha="center", va="bottom", fontsize=14,
                    color="white" if r["asr"] > 0.15 else S.MUTED, zorder=6)  # fmt: skip
    ax_all.set_xticks([])
    S.finish(ax_all, title="(a) Attack success rate, full trajectory", ylabel="ASR (%)", ylim=(0, 105),
             yticks=range(0, 101, 20))  # fmt: skip

    # (b) ASR by cumulative round
    x = np.arange(len(ROUNDS))
    for a in arms:
        br = stats[a]["asr_by_round"]
        ys = [100 * br[r]["asr"] if r in br else np.nan for r in ROUNDS]
        ax_rnd.plot(
            x, ys, marker="o", markersize=9, linewidth=2.6, color=S.COLOUR[a], zorder=3
        )
        for xi, y in zip(x, ys):
            ax_rnd.text(
                xi,
                y + 2.2,
                f"{y:.0f}",
                ha="center",
                va="bottom",
                fontsize=13,
                color=S.COLOUR[a],
            )
    n4 = stats[arms[0]]["asr_by_round"].get("R4", {}).get("n")
    ax_rnd.set_xticks(
        x,
        [
            "Step 1",
            "Steps 1-2",
            "Steps 1-3",
            f"Steps 1-4\n(n={n4}, >=4 steps)",
            "All steps",
        ],
    )
    S.finish(ax_rnd, title="(b) ASR as the judge sees more of the trajectory", ylabel="ASR (%)", ylim=(0, 105),
             yticks=range(0, 101, 20))  # fmt: skip

    # (c, d) grouped bars by risk category and attack strategy (round_all)
    def grouped(ax, key: str, labels: list[tuple[str, str]], title: str) -> None:
        w = 0.8 / len(arms)
        xs = np.arange(len(labels))
        for j, a in enumerate(arms):
            g = stats[a][key]
            vals = [100 * g[k]["asr"] if k in g else np.nan for k, _ in labels]
            ax.bar(xs + (j - (len(arms) - 1) / 2) * w, vals, width=w, color=S.COLOUR[a], edgecolor="black",
                   linewidth=0.6, zorder=3)  # fmt: skip
        ns = [stats[arms[0]][key].get(k, {}).get("n", 0) for k, _ in labels]
        ax.set_xticks(
            xs, [f"{s}\n(n={n})" for (_, s), n in zip(labels, ns)], fontsize=14
        )
        S.finish(
            ax, title=title, ylabel="ASR (%)", ylim=(0, 105), yticks=range(0, 101, 20)
        )

    grouped(ax_cat, "asr_by_category", CATS, "(c) ASR by risk category")
    grouped(ax_str, "asr_by_strategy", STRATS, "(d) ASR by attack strategy")

    labels = {
        "control": S.ARM["control"],
        "da": S.ARM["da"],
        "da_tools": S.ARM["da_tools"],
    }
    S.legend(
        fig,
        [S.patch(S.COLOUR[a], f"Qwen3.6-27B {labels[a]}") for a in arms],
        ncol=len(arms),
        y=1.0,
    )
    fig.text(0.5, -0.03,
             "AgentHazard (arXiv 2604.02947) under Claude Code 2.1.323, 100 instances per arm (10 per risk category, drawn from "
             "the paper's Claude Code set), same instances for every arm. Judge: Gemini-3-Flash with the paper's prompt; "
             "error bars: Wilson 95%.", ha="center", fontsize=14, color=S.MUTED)  # fmt: skip
    S.save(fig, "output/plots", "agenthazard-claude-code-three-arms", "AgentHazard under Claude Code: three arms",
           "scratch/agenthazard/fig_agenthazard.py", artifacts=[repos[a] for a in arms])  # fmt: skip


if __name__ == "__main__":
    main(sys.argv[1:])
