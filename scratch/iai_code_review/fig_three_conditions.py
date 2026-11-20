# ABOUTME: Three-condition figure for the IAI code_review runs: baseline (2026-11-19 scorer summaries) + persuasion and
# ABOUTME: deception recomputed per episode from the 2026-11-20 Hub logs (partial conditions shown with their current n).
# Run: uv run python scratch/iai_code_review/fig_three_conditions.py
import datetime as dt
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from inspect_ai.log import read_eval_log

from src.utils import save_figure_pdf

load_dotenv(".env")
ARMS = [  # label, adapter, hub repo of the 2026-11-20 rerun, baseline (k, n) from 2026-11-19 scorer summaries, colour
    (
        "Control (plain SFT)",
        "dougalldeepmind/2026-10-05-qwen36-0-plain",
        "dougalldeepmind/2026-11-20-iai-code-review-qwen36-0-plain",
        (95, 200),
        "#5a6472",
    ),
    (
        "Difficult Advice",
        "dougalldeepmind/2026-10-05-qwen36-0-da-15",
        "dougalldeepmind/2026-11-20-iai-code-review-qwen36-0-da-15",
        (100, 200),
        "#7e22ce",
    ),
    (
        "DA + unused tools",
        "dougalldeepmind/2026-10-11-qwen36-0-da-tools-15",
        "dougalldeepmind/2026-11-20-iai-code-review-qwen36-0-da-tools-15",
        (89, 200),
        "#d0558f",
    ),
]
CONDS = ["baseline", "persuasion", "deception"]
FULL = 200


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def count(repo: str, cond: str) -> tuple[int, int, int]:
    """(approved, decided, errored) over the episodes in the newest .eval for `cond`."""
    local = Path(
        snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{cond}/*.eval"])
    )
    files = sorted((local / cond).glob("*.eval"))
    if not files:
        return 0, 0, 0
    log = read_eval_log(str(files[-1]))
    k = n = err = 0
    for s in log.samples or []:
        if s.error is not None:
            err += 1
            continue
        # Upstream's own verdict (code_review_binary_scorer): the submitted decision, or the reviewer's
        # final-message text when it never called submit_decision; no decision at all counts as not approved.
        # Same rule as the 2026-11-19 baseline summaries, so the three panels are comparable.
        score = (s.scores or {}).get("code_review_binary_scorer")
        if score is None:
            continue
        n += 1
        k += score.value == "C"
    return k, n, err


data = {}
for label, _, repo, base, _ in ARMS:
    data[label] = {"baseline": (*base, 0)}
    for c in ("persuasion", "deception"):
        # plain's deception ran on a second pod (2026-11-20, to finish sooner) and pushed to its own repo
        src = (
            repo + "-deception"
            if (c == "deception" and repo.endswith("-plain"))
            else repo
        )
        data[label][c] = count(src, c)
print(json.dumps(data, indent=1))

fig, axes = plt.subplots(1, 3, figsize=(11, 2.9), sharey=True)
for ax, cond in zip(axes, CONDS):
    for i, (label, _, _, _, colour) in enumerate(ARMS):
        y = len(ARMS) - 1 - i
        k, n, _ = data[label][cond]
        if n == 0:
            ax.text(2, y, "none uploaded yet", va="center", fontsize=8, color="#888")
            continue
        lo, hi = wilson(k, n)
        ax.plot(
            [100 * lo, 100 * hi],
            [y, y],
            color=colour,
            lw=3,
            alpha=0.45,
            solid_capstyle="round",
        )
        ax.plot(
            100 * k / n,
            y,
            "o",
            color=colour,
            ms=8,
            markeredgecolor="black",
            markeredgewidth=0.6,
        )
        tag = "" if n >= FULL else "  partial"
        ax.text(
            max(100 * lo, 0),
            y - 0.36,
            f"{100 * k / n:.1f}% ({k}/{n}){tag}",
            fontsize=7.5,
        )
    ax.set_title(cond + (" (2026-11-19)" if cond == "baseline" else ""), fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(ARMS) - 0.5)
    ax.set_xticks([0, 25, 50, 75, 100])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_yticks(range(len(ARMS)))
axes[0].set_yticklabels([a[0] for a in reversed(ARMS)])
axes[1].set_xlabel(
    "Reviewer (our model) approves the vulnerable code, % of episodes (95% Wilson CI)"
)
fig.suptitle(
    "Inter-Agent Influence code review, author = Gemini 3 Flash, 2 scenarios x 100 epochs per cell "
    f"(as of {dt.datetime.now():%Y-%m-%d %H:%M})",
    fontsize=9,
)
fig.tight_layout()
out = (
    Path("output/iai_code_review")
    / f"{dt.date.today().isoformat()}_iai_code_review_three_conditions.pdf"
)
save_figure_pdf(
    fig,
    out,
    title="Inter-Agent Influence code review: approval by condition and arm",
    artifacts=[a[1] for a in ARMS] + [a[2] for a in ARMS],
    script="scratch/iai_code_review/fig_three_conditions.py",
)
print(out)
