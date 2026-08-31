# ABOUTME: Renders difficult-advice SFT examples as chat-transcript figures via yapplotlib,
# ABOUTME: so a reasoning trace and its reply can be read at a glance instead of as raw JSON.

"""View stage_7_sft.jsonl examples as chat figures.

The corpus's whole point is the assistant's private deliberation, which lives in
`reasoning_content` rather than `content`. Raw JSONL buries it; this renders each example
as system / user / thinking / reply so the reasoning is legible next to what it produced.

    uv run --isolated --with yapplotlib --with 'matplotlib<3.11' \
        python scratch/view_synthdoc.py --data <stage_7_sft.jsonl> --n 6

yapplotlib imports `matplotlib.style.core`, removed in matplotlib 3.11, so it needs the pin
above and an isolated env -- it must not drag the project's matplotlib backwards.
"""

from __future__ import annotations

import json
import random
import textwrap
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yapplotlib  # noqa: E402


def _elide(text: str, limit: int) -> str:
    """Shorten a message for display, keeping the opening and the conclusion.

    Args:
        text: Message body.
        limit: Character budget.

    Returns:
        The text, or its head and tail joined by an elision marker.
    """
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    return text[:head].rstrip() + "\n\n[...]\n\n" + text[-(limit - head):].lstrip()


def _to_messages(rec: dict, limit: int, show_thinking: bool) -> list[dict]:
    """Turn one SFT record into yapplotlib messages, surfacing the reasoning trace.

    The trace is emitted as its own turn rather than glued onto the reply, so the
    deliberation and the answer can be compared side by side.

    Args:
        rec: A stage_7_sft.jsonl record.
        limit: Per-message character budget.
        show_thinking: Include the reasoning trace as a separate turn.

    Returns:
        Messages with `role`, `content` and `name`.
    """
    by_role = {m["role"]: m for m in rec["messages"]}
    out = []
    if "system" in by_role:
        out.append({"role": "system", "content": _elide(by_role["system"]["content"], limit // 2),
                    "name": "system prompt"})
    out.append({"role": "user", "content": _elide(by_role["user"]["content"], limit),
                "name": "user"})
    asst = by_role["assistant"]
    trace = asst.get("reasoning_content", "")
    if show_thinking and trace:
        # A distinct role string: yapplotlib falls back to the role for the label, so this
        # reads as its own speaker without needing a theme that knows about it.
        out.append({"role": "thinking", "content": _elide(trace, limit),
                    "name": "assistant · private reasoning"})
    out.append({"role": "assistant", "content": _elide(asst["content"], limit),
                "name": "assistant · reply"})
    return out


def main(data: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
         n: int = 6, style: str = "paper", seed: int = 0, limit: int = 900,
         width: float = 11.0, bubble_width: float = 0.82,
         out_dir: str = "output/plots/synthdoc_view",
         traits: str | None = None, show_thinking: bool = True,
         combined_pdf: bool = True) -> None:
    """Render sampled examples as chat figures plus a combined PDF.

    Args:
        data: Path to stage_7_sft.jsonl.
        n: Number of examples to render.
        style: yapplotlib theme -- default, paper, dark or minimal.
        seed: Sampling seed.
        limit: Per-message character budget before eliding.
        width: Figure width in inches.
        bubble_width: Bubble width as a fraction of the axes; the library default of 0.6
            wraps very short lines on a narrow figure.
        out_dir: Destination directory.
        traits: Comma-separated trait ids to restrict to, e.g. "t1,t3".
        show_thinking: Render the reasoning trace as its own turn.
        combined_pdf: Also write every example into one multi-page PDF.
    """
    recs = [json.loads(line) for line in Path(data).open()]
    if traits:
        want = {t.strip() for t in str(traits).split(",")}
        recs = [r for r in recs if r.get("metadata", {}).get("trait_id") in want]
        assert recs, f"no records for traits {want}"
    rng = random.Random(seed)
    picked = rng.sample(recs, min(n, len(recs)))

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    pdf = None
    if combined_pdf:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages(dest / "synthdoc_examples.pdf")

    for i, rec in enumerate(picked, 1):
        meta = rec.get("metadata", {})
        msgs = _to_messages(rec, limit, show_thinking)
        # Two passes. Line wrapping is computed from the axes width at layout time, so
        # resizing afterwards leaves the text wrapped for the old width -- which produced
        # 30-character lines and absurdly tall figures. Pass 1 measures the auto height at
        # the library's default 5in; pass 2 rebuilds at the target width, where the same
        # content needs proportionally less height.
        probe, _ = yapplotlib.chatplot(msgs, style=style, figsize=None, dpi=150,
                                       bubble_width=bubble_width)
        h5 = probe.get_size_inches()[1]
        plt.close(probe)
        fig, ax = yapplotlib.chatplot(
            msgs, style=style, dpi=150, bubble_width=bubble_width,
            figsize=(width, max(3.0, h5 * 5.0 / width)))
        title = f"{meta.get('trait_id','?')} · {meta.get('trait_name','')}"
        ax.set_title(textwrap.fill(title, 70), fontsize=15, pad=14, loc="left")
        fig.text(0.01, 0.005, f"{meta.get('scenario_id','')}  ·  domain: {meta.get('domain','')}",
                 fontsize=14, alpha=0.6)
        png = dest / f"example_{i:02d}_{meta.get('trait_id','x')}.png"
        fig.savefig(png, bbox_inches="tight", dpi=150)
        if pdf:
            pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        written.append(png)
        print(f">>> {png}   [{meta.get('trait_id')}] {meta.get('domain','')}")

    if pdf:
        pdf.close()
        print(f">>> {dest / 'synthdoc_examples.pdf'}  ({len(picked)} pages)")
    print(f">>> {len(written)} figures in {dest}")


if __name__ == "__main__":
    fire.Fire(main)
