# ABOUTME: Print the arm ladder for one deliberation eval as a plain table, for reading in the
# ABOUTME: terminal while a run is still in flight. Usage: uv run python scratch/show_ladder.py llmbar

from __future__ import annotations

import glob
import json
import sys

SHORT = {"courtroom716": "CR", "peercritique716": "PC", "da716": "DA",
         "table2-only": "T2", "Qwen3_6-27B": "base"}

FIELDS = {
    "llmbar": [("acc", ("accuracy", "rate")), ("advers", ("adversarial_accuracy",)),
               ("consist", ("consistency", "rate")),
               ("firstpos", ("first_position_rate",)), ("parse", ("parse_rate",))],
    "debate_speeches": [("tau_b", ("kendall_tau_b",)), ("spearman", ("spearman",)),
                        ("qwk", ("quadratic_weighted_kappa",)),
                        ("mean_rt", ("mean_rating",)), ("modal", ("modal_rating_share",)),
                        ("parse", ("parse_rate",))],
    "sycophancy": [("balanced", ("balanced_accuracy",)),
                   ("hold_ok", ("hold_rate_when_correct", "rate")),
                   ("fix_wrong", ("correction_rate_when_wrong", "rate")),
                   ("first_acc", ("first_accuracy",)), ("flip", ("flip_rate",)),
                   ("parse", ("parse_rate",))],
}


def label(arm: str) -> str:
    for key, short in SHORT.items():
        if key in arm:
            return short
    return arm[:16]


def dig(blob, path):
    for key in path:
        blob = blob.get(key) if isinstance(blob, dict) else None
    return blob


def main(name: str) -> None:
    fields = FIELDS[name]
    trace_key = "trace_turn1" if name == "sycophancy" else "trace"
    header = f"{'arm':5s}" + "".join(f"{h:>10s}" for h, _ in fields) \
        + f"{'trace_ch':>10s}{'empty':>7s}"
    print(header)
    rows = []
    for path in sorted(glob.glob(f"output/{name}/*/*/results.json")):
        blob = json.load(open(path))
        trace = blob.get(trace_key) or {}
        rows.append((label(path.split("/")[2]),
                     [dig(blob, p) for _, p in fields],
                     trace.get("think_chars_mean"), trace.get("empty_think_rate")))
    # Sort by the eval's headline (first field), best first; None sorts last.
    rows.sort(key=lambda r: (r[1][0] is None, -(r[1][0] or 0)))
    for arm, values, chars, empty in rows:
        cells = "".join(f"{'   --':>10s}" if v is None else f"{v:>10.3f}" for v in values)
        print(f"{arm:5s}{cells}{(chars or 0):10.1f}{(empty or 0):7.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "llmbar")
