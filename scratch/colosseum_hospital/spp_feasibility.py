# ABOUTME: Can a candidate checkpoint run the Hospital study at all? Prompt lengths the
# ABOUTME: finished episodes actually needed vs the candidate's window, tool support, tokenizer.

"""Feasibility of a candidate model family for colosseum_hospital, from finished runs.

    uv run python scratch/colosseum_hospital/spp_feasibility.py \
        --runs output/colosseum_hospital --model dlab-spp/vanilla-3b-instruct

Asked on 2026-09-04 for the Synthetic Persona Pretraining checkpoints (arXiv 2608.13482,
`dlab-spp/*-3b-instruct`): could the SPP-vs-Vanilla contrast be run in the same Hospital
setup as the 7% difficult-advice contrast? Three facts decide it, and this script reads
all three rather than trusting a model card:

  1. the prompt lengths the environment produced in the runs that already exist
     (`usage.prompt_tokens` of every model call in agent_turns.json — counted by the
     SERVED model's tokenizer, so a denser candidate tokenizer only makes them longer);
  2. the candidate's own window (`max_position_embeddings` in its config.json) and how
     many of its tokens the public boards of those episodes come to under ITS tokenizer;
  3. whether the candidate's chat template renders `tools` — every Hospital action is an
     OpenAI-style tool call (terrarium sends `tools=`, vLLM must parse the reply).

Writes a markdown table beside the other analyses.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import date
from pathlib import Path


def q(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))]


def read_runs(runs: Path) -> dict:
    """Prompt lengths from every finished episode under `runs`."""
    all_calls: list[int] = []
    first_per_seat: list[int] = []
    episode_max: list[int] = []
    boards: list[Path] = []
    for turns_path in sorted(runs.rglob("agent_turns.json")):
        try:
            turns = json.loads(turns_path.read_text())
        except json.JSONDecodeError:
            continue
        seen: dict[str, int] = {}
        ep: list[int] = []
        for turn in turns:
            for call in turn.get("llm_calls") or []:
                usage = call.get("usage") or {}
                n = usage.get("prompt_tokens")
                if n is None:
                    continue
                ep.append(int(n))
                seen.setdefault(str(turn.get("agent")), int(n))
        if not ep:
            continue
        all_calls += ep
        first_per_seat += seen.values()
        episode_max.append(max(ep))
        board = turns_path.parent / "blackboard_0.txt"
        if board.is_file():
            boards.append(board)
    assert all_calls, f"no finished episodes with usage under {runs}"
    return {
        "episodes": len(episode_max),
        "calls": all_calls,
        "first_per_seat": first_per_seat,
        "episode_max": episode_max,
        "boards": boards,
    }


def candidate_facts(model: str) -> dict:
    """Window, template tool support and tokenizer of the candidate, from its own files."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    config = json.loads(Path(hf_hub_download(model, "config.json")).read_text())
    tok_cfg = json.loads(
        Path(hf_hub_download(model, "tokenizer_config.json")).read_text()
    )
    template = tok_cfg.get("chat_template")
    if not template:
        try:
            template = Path(hf_hub_download(model, "chat_template.jinja")).read_text()
        except Exception:  # noqa: BLE001 - no template at all is itself the finding
            template = ""
    return {
        "architectures": config.get("architectures"),
        "window": config.get("max_position_embeddings"),
        "rope_scaling": config.get("rope_scaling"),
        "rope_theta": config.get("rope_theta"),
        "template_renders_tools": "tools" in template,
        "tokenizer": Tokenizer.from_file(hf_hub_download(model, "tokenizer.json")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=Path("output/colosseum_hospital"))
    ap.add_argument("--model", default="dlab-spp/vanilla-3b-instruct")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="also write an ECDF of prompt lengths against the window (png)",
    )
    args = ap.parse_args()

    runs = read_runs(args.runs)
    facts = candidate_facts(args.model)
    window = int(facts["window"] or 0)
    tok = facts["tokenizer"]

    board_tokens = [len(tok.encode(b.read_text()).ids) for b in runs["boards"]]
    calls = runs["calls"]
    first = runs["first_per_seat"]
    over = sum(n > window for n in calls) / len(calls) if window else float("nan")
    first_over = sum(n > window for n in first) / len(first) if window else float("nan")

    lines = [
        f"# {args.model} in colosseum_hospital — feasibility ({date.today().isoformat()})",
        "",
        f"Finished episodes read: {runs['episodes']} ({len(calls)} model calls) under "
        f"`{args.runs}`. Prompt tokens are the served model's own count (Qwen3.6 tokenizer).",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| candidate architecture | {facts['architectures']} |",
        f"| candidate window (`max_position_embeddings`) | {window} |",
        f"| candidate RoPE scaling / theta | {facts['rope_scaling']} / {facts['rope_theta']} |",
        f"| candidate chat template renders `tools` | {facts['template_renders_tools']} |",
        f"| prompt tokens, all calls: min / p10 / median / p90 / max | "
        f"{min(calls)} / {q(calls, 0.1)} / {q(calls, 0.5)} / {q(calls, 0.9)} / {max(calls)} |",
        f"| first call of a seat: min / median / max | {min(first)} / {q(first, 0.5)} / "
        f"{max(first)} |",
        f"| per-episode maximum: min / median / max | {min(runs['episode_max'])} / "
        f"{q(runs['episode_max'], 0.5)} / {max(runs['episode_max'])} |",
        f"| share of calls above the candidate window | {over:.1%} |",
        f"| share of seats whose FIRST call is above the window | {first_over:.1%} |",
    ]
    if board_tokens:
        lines.append(
            f"| public board alone, in the candidate's tokens: min / median / max | "
            f"{min(board_tokens)} / {int(statistics.median(board_tokens))} / "
            f"{max(board_tokens)} |"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    out = args.out or (
        args.runs
        / "analysis"
        / f"{date.today().isoformat()}_{args.model.split('/')[-1].replace('-', '_')}_feasibility.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"-> {out}")
    if args.plot:
        plot_ecdf(
            calls,
            first,
            window,
            args.model,
            args.plot,
            hospital_window=65536,
            episodes=runs["episodes"],
        )


def plot_ecdf(
    calls: list[int],
    first: list[int],
    window: int,
    model: str,
    out: Path,
    hospital_window: int,
    episodes: int,
) -> None:
    """One figure: what share of the Hospital's prompts fit inside the candidate's window."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    for xs, label, color in (
        (calls, "every model call", "#2a78d6"),
        (first, "first call of each seat", "#eb6834"),
    ):
        xs = np.sort(np.asarray(xs))
        ax.step(
            xs,
            np.arange(1, len(xs) + 1) / len(xs),
            where="post",
            lw=2,
            color=color,
            label=label,
        )
        share = float(np.mean(xs > window))
        ax.annotate(
            f"{share:.0%} above the window",
            xy=(window, 1 - share),
            xytext=(window * 1.15, 1 - share),
            color=color,
            fontsize=9,
            va="center",
        )
    for x, label in (
        (window, f"{model.split('/')[-1]} window ({window:,})"),
        (hospital_window, f"Hospital serving window ({hospital_window:,})"),
    ):
        ax.axvline(x, color="#6b6a63", lw=1, ls="--")
        ax.text(
            x,
            1.0,
            f" {label}",
            rotation=90,
            va="top",
            ha="left",
            fontsize=8,
            color="#6b6a63",
        )
    ax.set_xscale("log")
    ax.set_xlim(1000, 80000)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("prompt tokens (served model's tokenizer, Qwen3.6)")
    ax.set_ylabel("share of prompts at or below")
    ax.set_title(
        f"Hospital prompt lengths, {episodes} finished episodes, vs the candidate's window"
    )
    ax.grid(True, color="#e6e5df", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
