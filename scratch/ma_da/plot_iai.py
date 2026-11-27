# ABOUTME: Inter-Agent Influence code review (arXiv 2611.04556): how often our reviewer approves the flawed change,
# ABOUTME: for the multi-agent DA 15% adapter beside control, DA 15% and DA+tools 15%; episodes read from the Hub.
# Run: uv run python -m scratch.ma_da.plot_iai
"""Three panels (baseline, persuasion, deception) in the paper style. A bar is approved / decided episodes
with a Wilson 95% interval, counted by upstream's own verdict (`code_review_binary_scorer`), the rule of
scratch/iai_code_review/fig_three_conditions.py on branch worktree-kn-iai-code-review. The three older arms'
baseline cells exist only as 2026-11-19 scorer summaries, transcribed from that script."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download
from inspect_ai.log import read_eval_log

from scratch.figures import _hub
from scratch.figures import _style as S

MADA_COLOUR = "#2e8b57"
ARMS = [  # key, colour, label, Hub run, baseline (approved, decided) where only a summary exists
    (
        "control",
        S.COLOUR["control"],
        S.ARM["control"],
        "2026-11-20-iai-code-review-qwen36-0-plain",
        (95, 200),
    ),
    (
        "da",
        S.COLOUR["da"],
        "+ Difficult Advice 15% mix",
        "2026-11-20-iai-code-review-qwen36-0-da-15",
        (100, 200),
    ),
    (
        "da_tools",
        S.COLOUR["da_tools"],
        "+ DA with tools 15% mix",
        "2026-11-20-iai-code-review-qwen36-0-da-tools-15",
        (89, 200),
    ),
    (
        "mada",
        MADA_COLOUR,
        "+ Multi-agent DA 15% mix",
        "2026-11-27-iai-code-review-qwen36-0-mada-15",
        None,
    ),
]
CONDS = [
    ("baseline", "Baseline"),
    ("persuasion", "Persuasion"),
    ("deception", "Deception"),
]
FULL = 200


def count(repo: str, cond: str) -> tuple[int, int, int]:
    """(approved, decided, errored) over the episodes in the newest log for `cond`."""
    local = Path(
        snapshot_download(
            f"{_hub.ORG}/{repo}",
            repo_type="dataset",
            allow_patterns=[f"{cond}/*.eval"],
            token=_hub._token(),
        )
    )
    files = sorted((local / cond).glob("*.eval"))
    if not files:
        return 0, 0, 0
    k = n = err = 0
    for s in read_eval_log(str(files[-1])).samples or []:
        if s.error is not None:
            err += 1
            continue
        score = (s.scores or {}).get("code_review_binary_scorer")
        if score is None:
            continue
        n += 1
        k += score.value == "C"
    return k, n, err


def main() -> None:
    S.apply()
    data = {}
    for key, _, _, repo, base in ARMS:
        data[key] = {}
        for cond, _ in CONDS:
            if cond == "baseline" and base is not None:
                data[key][cond] = (*base, 0)
            else:
                src = (
                    repo + "-deception"
                    if (cond == "deception" and repo.endswith("-plain"))
                    else repo
                )
                data[key][cond] = count(src, cond)
    fig, axes = plt.subplots(1, 3, figsize=(20.0, 8.0))
    md = [
        "# Inter-Agent Influence code review: reviewer approved the flawed change",
        "",
    ]
    md += [
        "| Condition | Arm | Approved / decided | % | 95% CI | Errored |",
        "|---|---|---|---|---|---|",
    ]
    for ax, (cond, title) in zip(axes, CONDS):
        for i, (key, colour, label, _, _) in enumerate(ARMS):
            k, n, err = data[key][cond]
            if n == 0:
                ax.text(
                    i,
                    3,
                    "no\nepisodes\nyet",
                    ha="center",
                    va="bottom",
                    fontsize=13,
                    color=S.MUTED,
                )
                md.append(f"| {cond} | {label} | 0/0 | | | {err} |")
                continue
            lo, hi = _hub.wilson(k, n)
            m = 100 * k / n
            partial = n + err < FULL
            S.bar(
                ax,
                i,
                m,
                lo,
                hi,
                colour=colour,
                label=f"{m:.0f}" + (f"\nn={n}" if partial else ""),
                hatch="//" if partial else None,
            )
            md.append(
                f"| {cond} | {label} | {k}/{n} | {m:.1f} | [{lo:.1f}, {hi:.1f}] | {err} |"
            )
        S.finish(
            ax,
            title=title,
            ylabel="Flawed change approved (%)",
            ylim=(0, 100),
            yticks=range(0, 101, 20),
        )
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([])
        ax.set_xlim(-0.65, len(ARMS) - 0.35)
    handles = [S.patch(colour, label) for _, colour, label, _, _ in ARMS]
    if any(0 < n + err < FULL for arm in data.values() for _, n, err in arm.values()):
        handles.append(
            S.patch("white", "hatched = condition still running", hatch="//")
        )
    S.legend(fig, handles, ncol=3, y=1.0)
    fig.subplots_adjust(top=0.74, bottom=0.06, wspace=0.26)
    png = S.save(
        fig,
        "output/plots",
        "iai_code_review_mada_15",
        "Inter-Agent Influence code review: multi-agent DA 15% beside the known arms",
        "scratch/ma_da/plot_iai.py",
        artifacts=[f"{_hub.ORG}/{repo}" for _, _, _, repo, _ in ARMS],
    )
    png.with_name(png.stem + "_results.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(png)
    print("\n".join(md))


if __name__ == "__main__":
    main()
