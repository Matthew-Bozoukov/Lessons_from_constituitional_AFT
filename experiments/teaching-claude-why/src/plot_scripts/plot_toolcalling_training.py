# ABOUTME: Plots the tool-calling SFT run's loss and token-accuracy curves from the TRL log history.
# ABOUTME: Reusable figure code; the experiment script calls it rather than plotting inline.

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Match the published dose-response figure: light ground, one accent per series, no chartjunk.
_INK = "#1a1a1a"
_GRID = "#e3e3e0"
_LOSS = "#c1440e"
_ACC = "#3d5a80"


def plot_training_curves(history: list[dict], out_path: Path, title: str) -> Path:
    """Draw loss and mean token accuracy against optimizer step.

    Args:
        history: TRL `log_history` entries; only those carrying `loss` are used.
        out_path: Destination PNG.
        title: Figure title.

    Returns:
        The written path.
    """
    pts = [h for h in history if "loss" in h and "step" in h]
    assert pts, "log history contains no loss entries"
    steps = [h["step"] for h in pts]
    loss = [h["loss"] for h in pts]
    acc = [h.get("mean_token_accuracy") for h in pts]
    has_acc = any(a is not None for a in acc)

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(steps, loss, color=_LOSS, lw=2.0, marker="o", ms=3.2, label="training loss")
    ax.set_xlabel("optimizer step", color=_INK, fontsize=10)
    ax.set_ylabel("loss", color=_LOSS, fontsize=10)
    ax.tick_params(axis="y", labelcolor=_LOSS, labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(True, color=_GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_GRID)

    handles, labels = ax.get_legend_handles_labels()
    if has_acc:
        ax2 = ax.twinx()
        ax2.plot(steps, acc, color=_ACC, lw=2.0, marker="s", ms=3.0,
                 label="mean token accuracy")
        ax2.set_ylabel("mean token accuracy", color=_ACC, fontsize=10)
        ax2.tick_params(axis="y", labelcolor=_ACC, labelsize=9)
        for s in ("top", "left", "bottom"):
            ax2.spines[s].set_visible(False)
        ax2.spines["right"].set_color(_GRID)
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2

    ax.legend(handles, labels, loc="center right", frameon=False, fontsize=9)
    ax.set_title(title, color=_INK, fontsize=12, pad=12, loc="left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_markdown_mirror(history: list[dict], out_path: Path, summary: dict) -> Path:
    """Write the greppable table next to the PNG - numbers must not be trapped in an image."""
    pts = [h for h in history if "loss" in h and "step" in h]
    lines = [
        "# Tool-calling SFT training curve",
        "",
        f"- steps: {summary.get('max_steps', len(pts))}",
        f"- final loss: {pts[-1]['loss']}",
        f"- final mean token accuracy: {pts[-1].get('mean_token_accuracy')}",
        f"- tokens consumed: {summary.get('num_tokens')}",
        f"- wall clock: {summary.get('train_runtime_hms')}",
        "",
        "| step | loss | mean token accuracy | grad norm | lr |",
        "|---:|---:|---:|---:|---:|",
    ]
    for h in pts:
        lines.append(
            f"| {h['step']} | {h['loss']} | {h.get('mean_token_accuracy', '')} | "
            f"{h.get('grad_norm', '')} | {h.get('learning_rate', '')} |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


if __name__ == "__main__":
    import fire

    def main(trainer_state: str, out_png: str, out_md: str, title: str = "Tool-calling SFT") -> None:
        """Plot from a saved trainer_state.json."""
        state = json.loads(Path(trainer_state).read_text(encoding="utf-8"))
        hist = state["log_history"]
        summary = {k: state.get(k) for k in ("max_steps", "num_tokens", "train_runtime_hms")}
        print(plot_training_curves(hist, Path(out_png), title))
        print(write_markdown_mirror(hist, Path(out_md), summary))

    fire.Fire(main)
