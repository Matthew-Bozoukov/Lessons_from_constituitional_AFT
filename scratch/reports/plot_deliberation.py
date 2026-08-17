# ABOUTME: Read the deliberation eval summaries and plot the arm ladder — headline per eval,
# ABOUTME: the two-sided sycophancy decomposition, and reasoning length. Writes PNG + a md mirror.
# Run: uv run python scratch/reports/plot_deliberation.py --results output

"""Plots for the CR/PC/DA/T2/base ladder on the three in-domain evals.

Design decisions, so they are not re-litigated on the next edit:

- **Arms are the x-axis, so they are NOT colored by identity.** The axis label already
  carries identity; a five-hue palette would be redundant encoding, and the two neutral
  greys it needed failed the CVD check against the aqua slot anyway. One hue for every bar,
  hatch for the two control arms (base, table2-only) — texture is the documented secondary
  encoding, and it survives print and forced-colors.
- **Difficult advice is drawn as a reference LINE, not a highlighted bar.** The question
  these plots exist to answer is "does a variant beat difficult advice on its own home
  turf", and a line across the panel makes that a lookup rather than a comparison of
  adjacent bar heights.
- **Every bar is directly labelled.** The relief rule, and it also means the numbers survive
  being read at thumbnail size in a slide.
- **A floor line is drawn wherever a degenerate strategy has a known score** (0.5 where
  chance is a coin flip, 0 for a rank correlation). Without it a bar chart of correlations
  reads as "some of it is there" when it may be nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Validated: slots 1 and 2 of the reference categorical theme, all-pairs clean in light mode
# (worst CVD deltaE 24.7, normal-vision 33.6, both above 3:1 on the surface).
HUE = "#2a78d6"
HUE_ALT = "#eb6834"
INK = "#1c1c1a"
INK_SOFT = "#5c5c57"
GRID = "#e2e2dd"
SURFACE = "#fcfcfb"

# model_key (run_eval's per-target directory name) -> (short label, is_control)
ARMS: dict[str, tuple[str, bool]] = {
    "qwen3_6-27b-lora-t2-9284-courtroom716-r64-dynbatch": ("CR\ncourtroom", False),
    "qwen3_6-27b-lora-t2-9284-peercritique716-r64-dynbatch": ("PC\npeer critique", False),
    "qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch": ("DA\ndiff. advice", False),
    "qwen3_6-27b-lora-table2-only-9284-r64": ("T2\n0% synth", True),
    "Qwen3_6-27B": ("base\nuntrained", True),
}
ORDER = list(ARMS)
REFERENCE = "qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch"

# (eval name, metric path, panel title, y label, floor value or None, y max)
HEADLINES = [
    ("llmbar", ("adversarial_accuracy",), "PC — LLMBar (adversarial)",
     "accuracy vs gold label", 0.5, 1.0),
    ("debate_speeches", ("kendall_tau_b",), "CR — debate speeches",
     "Kendall tau-b vs human mean", 0.0, 1.0),
    ("sycophancy", ("balanced_accuracy",), "PAR — two-sided retraction",
     "balanced accuracy", 0.5, 1.0),
]


def _dig(blob: dict, path: tuple[str, ...]):
    for key in path:
        if blob is None:
            return None
        blob = blob.get(key) if isinstance(blob, dict) else None
    return blob


def load(results: str) -> dict[str, dict[str, dict]]:
    """Return {eval_name: {model_key: summary}} from the newest run per (eval, arm)."""
    root = Path(results)
    out: dict[str, dict[str, dict]] = {}
    for name, *_ in HEADLINES:
        out[name] = {}
        for arm_dir in sorted((root / name).glob("*")):
            if not arm_dir.is_dir() or arm_dir.name == "server":
                continue
            runs = sorted(p for p in arm_dir.glob("*/results.json"))
            if runs:
                out[name][arm_dir.name] = json.loads(runs[-1].read_text())
    return out


def load_hf(org: str = "LASR-Callum", date: str = "2026-08-17") -> dict[str, dict[str, dict]]:
    """Same shape, read from the HF repos run_eval pushes to as each arm finishes.

    This is the recovery path, and the reason `--no-push` is not used on the pod: the pod is
    disposable and self-terminating, so the Hub is where these numbers actually live. Anyone
    can rebuild every figure from a fresh clone with no pod, no local run directory and no
    access to the machine that launched it.

    run_eval names each repo `<org>/<date>-<eval-with-dashes>-<model-key-with-dashes>`.
    """
    from huggingface_hub import hf_hub_download

    out: dict[str, dict[str, dict]] = {}
    for name, *_ in HEADLINES:
        out[name] = {}
        for arm in ORDER:
            repo = f"{org}/{date}-{name.replace('_', '-')}-{arm.replace('_', '-')}"
            try:
                path = hf_hub_download(repo, "results.json", repo_type="dataset")
            except Exception:  # noqa: BLE001 — an arm that has not finished yet is a gap
                continue
            out[name][arm] = json.loads(Path(path).read_text())
    return out


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, length=0, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _bars(ax, values: dict[str, float | None], ylabel: str, title: str,
          floor: float | None, ymax: float, hue: str = HUE) -> None:
    labels, heights, hatches = [], [], []
    for key in ORDER:
        label, control = ARMS[key]
        labels.append(label)
        heights.append(values.get(key))
        hatches.append("///" if control else "")

    xs = range(len(labels))
    for x, height, hatch in zip(xs, heights, hatches):
        if height is None:
            ax.text(x, ymax * 0.04, "no data", ha="center", color=INK_SOFT, fontsize=8,
                    style="italic")
            continue
        ax.bar(x, height, width=0.62, color=hue if not hatch else SURFACE,
               edgecolor=hue, linewidth=1.4, hatch=hatch, zorder=3)
        ax.text(x, height + ymax * 0.022, f"{height:.2f}", ha="center", va="bottom",
                fontsize=9, color=INK, fontweight="bold")

    if floor is not None:
        ax.axhline(floor, color=INK_SOFT, linewidth=1, linestyle=(0, (2, 3)), zorder=2)
        ax.text(len(labels) - 0.42, floor, " floor", va="center", ha="left",
                fontsize=8, color=INK_SOFT)

    ref = values.get(REFERENCE)
    if ref is not None:
        ax.axhline(ref, color=HUE_ALT, linewidth=1.4, linestyle=(0, (5, 3)), zorder=4)
        ax.text(-0.45, ref, " difficult advice", va="bottom", ha="left", fontsize=8,
                color=HUE_ALT, fontweight="bold")

    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=8.5, color=INK)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_SOFT)
    ax.set_title(title, fontsize=11, color=INK, fontweight="bold", pad=12, loc="left")
    _style(ax)


def figure_headline(data: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURFACE)
    for ax, (name, path, title, ylabel, floor, ymax) in zip(axes, HEADLINES):
        values = {k: _dig(v, path) for k, v in data.get(name, {}).items()}
        _bars(ax, values, ylabel, title, floor, ymax)
    fig.suptitle("Each variant on its own in-domain eval", x=0.007, ha="left",
                 fontsize=14, color=INK, fontweight="700")
    fig.legend(handles=[Patch(facecolor=SURFACE, edgecolor=HUE, hatch="///",
                              label="control arm")],
               loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SOFT)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out / "deliberation_headline.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def figure_two_sided(data: dict, out: Path) -> Path:
    """The sycophancy decomposition — the half a one-sided sycophancy number hides."""
    summaries = data.get("sycophancy", {})
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    width = 0.36
    for offset, (key, label, hue) in enumerate([
        (("hold_rate_when_correct", "rate"), "held a CORRECT answer", HUE),
        (("correction_rate_when_wrong", "rate"), "fixed a WRONG answer", HUE_ALT),
    ]):
        xs, heights = [], []
        for index, arm in enumerate(ORDER):
            value = _dig(summaries.get(arm, {}), key)
            if value is None:
                continue
            xs.append(index + (offset - 0.5) * width)
            heights.append(value)
        ax.bar(xs, heights, width=width - 0.02, color=hue, label=label, zorder=3)
        for x, height in zip(xs, heights):
            ax.text(x, height + 0.02, f"{height:.2f}", ha="center", va="bottom",
                    fontsize=8, color=INK)

    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([ARMS[a][0] for a in ORDER], fontsize=8.5, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("rate", fontsize=9, color=INK_SOFT)
    ax.set_title("Both halves, separately — a model can only win by doing both",
                 fontsize=12, color=INK, fontweight="700", pad=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _style(ax)
    fig.tight_layout()
    path = out / "sycophancy_two_sided.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def figure_reasoning(data: dict, out: Path) -> Path:
    """Trace length per arm — the direct test of 'CR and PC reason less than DA'."""
    panels = [("llmbar", ("trace", "think_chars_mean"), "LLMBar"),
              ("debate_speeches", ("trace", "think_chars_mean"), "Debate speeches"),
              ("sycophancy", ("trace_turn1", "think_chars_mean"), "Sycophancy (turn 1)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
    for ax, (name, path, title) in zip(axes, panels):
        values = {k: _dig(v, path) for k, v in data.get(name, {}).items()}
        top = max([v for v in values.values() if v] or [1]) * 1.28
        _bars(ax, values, "mean reasoning characters", title, None, top)
    fig.suptitle("How much each arm actually reasons", x=0.007, ha="left",
                 fontsize=14, color=INK, fontweight="700")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out / "deliberation_reasoning_length.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def mirror(data: dict, out: Path) -> Path:
    """The greppable markdown mirror CLAUDE.md requires beside every plot."""
    lines = ["<!-- ABOUTME: Numbers behind the deliberation-eval plots; regenerate with -->",
             "<!-- ABOUTME: uv run python scratch/reports/plot_deliberation.py -->",
             "", "# Deliberation evals — arm ladder", ""]
    for name, path, title, _ylabel, _floor, _ymax in HEADLINES:
        lines += [f"## {title} (`{name}`)", "",
                  "| arm | headline | n | parse rate | mean think chars |", "|---|---|---|---|---|"]
        for arm in ORDER:
            summary = data.get(name, {}).get(arm)
            if not summary:
                lines.append(f"| {ARMS[arm][0].replace(chr(10), ' ')} | — | — | — | — |")
                continue
            head = _dig(summary, path)
            trace = summary.get("trace") or summary.get("trace_turn1") or {}
            lines.append(
                f"| {ARMS[arm][0].replace(chr(10), ' ')} "
                f"| {head if head is None else round(head, 4)} "
                f"| {summary.get('n_items', summary.get('n_scored', '—'))} "
                f"| {summary.get('parse_rate', '—')} "
                f"| {trace.get('think_chars_mean', '—')} |")
        lines.append("")
    path_md = out / "deliberation_results.md"
    path_md.write_text("\n".join(lines) + "\n")
    return path_md


def main(results: str = "output", out: str = "output/report", source: str = "local",
         org: str = "LASR-Callum", date: str = "2026-08-17") -> str:
    """Build every figure from whatever results exist; missing arms are drawn as gaps.

    Args:
        results: Local run root (source="local").
        out: Where the PNGs and the markdown mirror go.
        source: "local" reads `results`; "hf" reads the pushed eval repos instead, which is
            how to rebuild these figures once the pod is gone.
        org: HF org holding the pushed results (source="hf").
        date: The ISO date in the pushed repo names (source="hf").
    """
    assert source in ("local", "hf"), f"source must be local|hf, got {source!r}"
    data = load_hf(org, date) if source == "hf" else load(results)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = [figure_headline(data, out_dir), figure_two_sided(data, out_dir),
            figure_reasoning(data, out_dir), mirror(data, out_dir)]
    counts = {name: len(arms) for name, arms in data.items()}
    return f"arms found per eval: {counts}\n" + "\n".join(str(p) for p in made)


if __name__ == "__main__":
    fire.Fire(main)
