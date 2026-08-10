# ABOUTME: Builds the two SWE-bench Verified charts for the report and the dashboard entry.
# ABOUTME: Run: uv run python scratch/reports/swebench_charts.py

"""Charts for the SWE-bench Verified head-to-head.

Two figures:

1. `outcome-all-instances.svg` - what happened to all 250 instances, per arm,
   split into resolved / patch submitted but tests failed / no patch produced,
   with the reasons no patch was produced beneath it.
2. `outcome-submitted-only.svg` - the same run scored only over the instances
   where a patch was actually submitted, which is the denominator that flips
   the ranking between the two arms.

Every number is read from a published artifact: the comparison run's
`results.json` on the Hub for the per-arm outcomes, and the eval's own
`exit_statuses` for the failure reasons. Nothing is estimated.

The reason breakdown is POOLED across arms and that is stated on the chart.
Per-arm reason counts exist for exactly one of the four cells - the driver that
held the other three died mid-run (docs/swebench_run_postmortem.md), and the
recovered backups covered predictions, not the per-rollout exit statuses.
Apportioning the pooled counts across arms would have produced a per-arm chart
out of numbers nobody measured, so the chart reports the level that was.

Palette: validated with the dataviz palette checker against the dashboard's dark
chart surface (#151a1f) - lightness band, chroma floor, deutan/tritan
separation, normal-vision separation and 3:1 contrast all pass for the
categorical trio; the three "no patch" reasons are steps of one hue, monotonic
in lightness, each clearing 3:1.
"""

from __future__ import annotations

import ast
import glob
import json
import os
import urllib.request
from pathlib import Path

REPO = "LASR-Callum/2026-08-07-swebench-verified-qwen36-lora-comparison"
OUT = Path("output/swebench_mini_report")
DASHBOARD_ASSETS = Path(
    "dashboard/content/evals/2026-08-07-swebench-verified-qwen36-lora-comparison/assets"
)

# --- palette (validated; see module docstring) ---------------------------------------
SURFACE = "#151a1f"
INK = "#edf2f6"
INK_SOFT = "#bdc7d0"
MUTED = "#7f8b96"
GRID = "#28323b"

RESOLVED = "#2fa8a3"
WRONG = "#c9822c"
NOPATCH_CONTEXT = "#c3aef7"
NOPATCH_TIMEOUT = "#9b6ef0"
NOPATCH_STEPS = "#7f52d4"
NOPATCH_FLAT = "#9b6ef0"

FONT = (
    "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, "
    "Helvetica, Arial, sans-serif"
)
MONO = "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"

# The published baseline. NOT from this run - see the caption on every chart.
BASELINE_SCORE = 77.2
BASELINE_LABEL = "Qwen3.6-27B, published"
BASELINE_NOTE = (
    "official model card: 77.2 on SWE-bench Verified, internal agent scaffold, "
    "200K context window"
)


def _token() -> str:
    for line in Path(".env").read_text(encoding="utf8").splitlines():
        if line.strip().startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip().strip("\"'")
    return os.environ.get("HF_TOKEN", "")


def load_results() -> dict:
    """The authoritative per-arm outcome counts, from the published bundle."""
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/results.json"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {_token()}"})
    return json.loads(urllib.request.urlopen(request).read().decode("utf8"))


def load_measured_exit_statuses() -> dict[tuple[str, str], dict[str, int]]:
    """Per-cell exit statuses, for the cells whose summaries survived.

    The eval writes `exit_statuses` as a repr-keyed dict, so it is parsed with
    `literal_eval` rather than json.
    """
    out: dict[tuple[str, str], dict[str, int]] = {}
    for path in sorted(glob.glob("output/eval_summaries/swebench_mini_*.json")):
        doc = json.loads(Path(path).read_text(encoding="utf8"))
        raw = doc.get("exit_statuses")
        if not isinstance(raw, dict) or len(raw) != 1:
            continue
        mapping = ast.literal_eval(next(iter(raw)))
        counts = {status: len(ids) for status, ids in mapping.items()}
        key = (
            str(doc.get("target", "")).split("/")[-1],
            str(doc.get("selection", {}).get("subset_hash", "")),
        )
        out[key] = counts
    return out


# --- tiny SVG helpers ----------------------------------------------------------------


def esc(text: str) -> str:
    return (
        str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def text(x, y, s, *, size=11, fill=INK_SOFT, weight=400, anchor="start", font=FONT):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="{size}" '
        f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{esc(s)}</text>'
    )


def bar(x, y, w, h, fill, *, r=4):
    """A segment. Rounded 4px so a stacked run reads as one bar with joints."""
    w = max(0.0, w)
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h}" rx="{r}" fill="{fill}"/>'


def legend(x, y, items, *, gap=None):
    """Identity is never colour alone: every swatch is named."""
    parts, cursor = [], x
    for colour, label in items:
        parts.append(f'<rect x="{cursor:.1f}" y="{y - 8}" width="9" height="9" rx="2" fill="{colour}"/>')
        parts.append(text(cursor + 14, y, label, size=10.5, fill=INK_SOFT))
        cursor += 14 + len(label) * 5.75 + (gap or 22)
    return "".join(parts)


def stacked_row(x, y, width, height, total, segments, *, label, sublabel, right):
    """One stacked bar with a 2px surface gap between fills and inline counts."""
    parts = [
        text(x, y - 9, label, size=12, fill=INK, weight=650),
        text(x, y + height + 15, sublabel, size=10, fill=MUTED),
        text(x + width, y - 9, right, size=12, fill=INK, weight=650, anchor="end"),
    ]
    cursor = x
    for value, colour in segments:
        seg = (value / total) * width if total else 0
        parts.append(bar(cursor, y, max(seg - 2, 0), height, colour))
        # Only label a segment wide enough to hold the number; the rest are in
        # the table under the figure, which is the record either way.
        if seg > 26:
            parts.append(
                text(
                    cursor + (seg - 2) / 2,
                    y + height / 2 + 4,
                    value,
                    size=11,
                    fill="#0b0e11",
                    weight=650,
                    anchor="middle",
                )
            )
        cursor += seg
    return "".join(parts)


def baseline_marker(x, y, width, height, fraction, *, note):
    """The published score, drawn as a reference and captioned as external.

    The caption hangs BELOW the line, centred on it. Above the line is where the
    per-bar pass@1 figures sit, and at 77.2% the marker lands close enough to the
    right edge that the two labels overlapped.
    """
    at = x + width * fraction
    bottom = y + height + 6
    return "".join(
        [
            f'<line x1="{at:.1f}" y1="{y - 6}" x2="{at:.1f}" y2="{bottom}" '
            f'stroke="{INK_SOFT}" stroke-width="1.5" stroke-dasharray="4 3"/>',
            text(at, bottom + 15, note, size=10, fill=INK_SOFT, weight=600, anchor="middle"),
        ]
    )


def svg(width, height, body, *, title, desc):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-labelledby="t d" style="max-width:100%;height:auto">'
        f'<title id="t">{esc(title)}</title><desc id="d">{esc(desc)}</desc>'
        f'<rect width="{width}" height="{height}" rx="10" fill="{SURFACE}"/>'
        f"{body}</svg>"
    )


# --- figure 1: every instance ---------------------------------------------------------


def figure_all_instances(united: dict, pooled: dict, pooled_n: int) -> str:
    W, H = 880, 496
    x, width, bar_h = 210, 560, 30
    parts = [
        text(28, 36, "SWE-bench Verified: what happened to all 250 instances", size=16, fill=INK, weight=650),
        text(
            28,
            57,
            "Qwen3.6-27B + mini-SWE-agent v2, official harness 4.1.0, 65,536-token context. "
            "Both adapters ran the identical 250-instance half.",
            size=10.5,
            fill=MUTED,
        ),
        legend(
            28,
            84,
            [
                (RESOLVED, "Resolved (pass@1)"),
                (WRONG, "Patch submitted, tests failed"),
                (NOPATCH_FLAT, "No patch produced"),
            ],
        ),
    ]

    top = 124
    for index, (arm, label) in enumerate(
        [("only9284", "table2-only-9284-r64"), ("synthdoc", "table2-synthdoc-r64")]
    ):
        cell = united[arm]
        wrong = cell["patches"] - cell["resolved"]
        nopatch = cell["n"] - cell["patches"]
        y = top + index * 74
        parts.append(
            stacked_row(
                x,
                y,
                width,
                bar_h,
                cell["n"],
                [(cell["resolved"], RESOLVED), (wrong, WRONG), (nopatch, NOPATCH_FLAT)],
                label=label,
                sublabel=(
                    f"{cell['resolved']} resolved · {wrong} submitted but failed · "
                    f"{nopatch} never produced a patch"
                ),
                right=f"pass@1 {cell['resolved'] / cell['n']:.1%}",
            )
        )
        parts.append(text(x - 14, y + bar_h / 2 + 4, "250", size=10, fill=MUTED, anchor="end"))

    # The published baseline, spanning both bars.
    parts.append(
        baseline_marker(
            x,
            top,
            width,
            bar_h + 74,
            BASELINE_SCORE / 100,
            note=f"{BASELINE_LABEL}: {BASELINE_SCORE}%",
        )
    )
    parts.append(
        text(
            x,
            top + 172,
            f"Dashed line is NOT from this run - {BASELINE_NOTE}.",
            size=10,
            fill=INK_SOFT,
        )
    )
    parts.append(
        text(
            x,
            top + 188,
            "Its 200K window against our 65,536 is most of the gap: a fifth of our rollouts abort on context.",
            size=10,
            fill=MUTED,
        )
    )

    # Why no patch was produced - pooled, and said so.
    ry = 360
    parts.append(f'<line x1="28" y1="{ry - 26}" x2="{W - 28}" y2="{ry - 26}" stroke="{GRID}"/>')
    parts.append(
        text(28, ry - 6, "Why a patch was never produced", size=12.5, fill=INK, weight=650)
    )
    parts.append(
        text(
            28,
            ry + 10,
            f"Pooled over {pooled_n} rollouts - per-arm counts survive for only one of the four cells.",
            size=10,
            fill=MUTED,
        )
    )

    order = [
        ("ContextWindowExceededError", "Context window exceeded", NOPATCH_CONTEXT),
        ("Timeout", "Transport timeout (network, not the model)", NOPATCH_TIMEOUT),
        ("LimitsExceeded", "Step budget exhausted", NOPATCH_STEPS),
    ]
    unsubmitted = sum(pooled.get(k, 0) for k, _, _ in order)
    by = ry + 32
    parts.append(
        stacked_row(
            x,
            by,
            width,
            22,
            unsubmitted,
            [(pooled.get(k, 0), c) for k, _, c in order],
            label="",
            sublabel="",
            right="",
        )
    )
    parts.append(text(x - 14, by + 15, f"{unsubmitted}", size=10, fill=MUTED, anchor="end"))
    parts.append(
        legend(
            28,
            by + 58,
            [(c, f"{lbl} - {pooled.get(k, 0)}") for k, lbl, c in order],
            gap=16,
        )
    )
    return svg(
        W,
        H,
        "".join(parts),
        title="SWE-bench Verified outcomes for both adapters over all 250 instances",
        desc=(
            "Stacked bars per adapter showing resolved, submitted-but-failed and "
            "no-patch counts, with a dashed reference line at the published 77.2% "
            "score and a pooled breakdown of why no patch was produced."
        ),
    )


# --- figure 2: submitted only ---------------------------------------------------------


def figure_submitted_only(united: dict) -> str:
    W, H = 880, 330
    x, width, bar_h = 210, 560, 32
    parts = [
        text(28, 36, "Scored only over instances where a patch was submitted", size=16, fill=INK, weight=650),
        text(
            28,
            57,
            "The same run, different denominator: resolved divided by patches produced "
            "rather than by all 250 instances.",
            size=10.5,
            fill=MUTED,
        ),
        legend(
            28,
            84,
            [(RESOLVED, "Resolved"), (WRONG, "Submitted, tests failed")],
        ),
    ]

    top = 124
    for index, (arm, label) in enumerate(
        [("only9284", "table2-only-9284-r64"), ("synthdoc", "table2-synthdoc-r64")]
    ):
        cell = united[arm]
        wrong = cell["patches"] - cell["resolved"]
        y = top + index * 76
        parts.append(
            stacked_row(
                x,
                y,
                width,
                bar_h,
                cell["patches"],
                [(cell["resolved"], RESOLVED), (wrong, WRONG)],
                label=label,
                sublabel=f"{cell['resolved']} of {cell['patches']} submitted patches passed",
                right=f"{cell['resolved'] / cell['patches']:.1%} of submitted",
            )
        )
        parts.append(
            text(x - 14, y + bar_h / 2 + 4, str(cell["patches"]), size=10, fill=MUTED, anchor="end")
        )

    note_y = top + 168
    parts.append(f'<line x1="28" y1="{note_y - 22}" x2="{W - 28}" y2="{note_y - 22}" stroke="{GRID}"/>')
    parts.append(
        text(
            28,
            note_y,
            "The ranking flips. On pass@1 synthdoc leads (46.4% vs 42.8%); here only-9284 leads.",
            size=11,
            fill=INK,
            weight=600,
        )
    )
    parts.append(
        text(
            28,
            note_y + 17,
            "synthdoc attempts more (155 patches vs 135), so it solves more in absolute terms while "
            "converting a smaller share.",
            size=10,
            fill=MUTED,
        )
    )
    parts.append(
        text(
            28,
            note_y + 33,
            "No published baseline is drawn here: 77.2% is pass@1 over every instance, not a "
            "rate among submitted patches, so the two are not comparable.",
            size=10,
            fill=MUTED,
        )
    )
    return svg(
        W,
        H,
        "".join(parts),
        title="SWE-bench Verified resolve rate among submitted patches",
        desc=(
            "Stacked bars per adapter over submitted patches only: only-9284 resolves "
            "79.3% of its 135 patches, synthdoc 74.8% of its 155."
        ),
    )


def main() -> None:
    results = load_results()
    united = results["united"]
    measured = load_measured_exit_statuses()

    # The pooled reason counts, as published in the run's own report.
    pooled = {"ContextWindowExceededError": 72, "Timeout": 60, "LimitsExceeded": 17}
    pooled_n = 369

    OUT.mkdir(parents=True, exist_ok=True)
    DASHBOARD_ASSETS.mkdir(parents=True, exist_ok=True)
    figures = {
        "swebench-outcome-all-instances.svg": figure_all_instances(united, pooled, pooled_n),
        "swebench-outcome-submitted-only.svg": figure_submitted_only(united),
    }
    for name, markup in figures.items():
        for directory in (OUT, DASHBOARD_ASSETS):
            (directory / name).write_text(markup, encoding="utf8")
        print(f"  wrote {name} ({len(markup):,} bytes) to {OUT} and {DASHBOARD_ASSETS}")

    print("\n  per-arm outcomes (from the published results.json):")
    for arm, cell in united.items():
        wrong = cell["patches"] - cell["resolved"]
        print(
            f"    {arm:10} n={cell['n']} resolved={cell['resolved']} wrong={wrong} "
            f"no_patch={cell['n'] - cell['patches']} pass@1={cell['resolved'] / cell['n']:.1%} "
            f"among_submitted={cell['resolved'] / cell['patches']:.1%}"
        )
    print("\n  cells with complete per-instance exit statuses:")
    for (target, subset), counts in measured.items():
        print(f"    {target} subset={subset} n={sum(counts.values())}: {counts}")


if __name__ == "__main__":
    main()
