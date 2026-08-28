# ABOUTME: Side-by-side StrongREJECT comparison of two or more arms from their run dirs,
# ABOUTME: with per-jailbreak detail and a paired-by-cell difference between two arms.

"""Read the results.json of each StrongREJECT run dir and print/write the cross-arm table.

run_eval writes one results.json per target; this is the cross-arm view it cannot write
(it never sees the other arms). Usage:

    uv run python scratch/stats/strongreject_compare.py \
        --runs "<label>=<run_dir>,<label>=<run_dir>" [--out output/report/...]

The paired difference is over cells BOTH arms graded — an arm that lost cells to empty
answers must not be compared on a different cell set than the arm it is measured against.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import fire


def _load(run_dir: Path) -> tuple[dict, dict]:
    """Return (summary, {(jailbreak, prompt_id): score}) for one StrongREJECT run dir."""
    res = json.loads((run_dir / "results" / "results.json").read_text())
    summary = res.get("summary", res)
    cells = {}
    for line in (run_dir / "results" / "scores.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["score"] is not None:
            cells[(r["jailbreak"], r["prompt_id"])] = r["score"]
    return summary, cells


def main(runs: str, out: str = "") -> None:
    """Print the cross-arm StrongREJECT table and optionally write it as markdown.

    Args:
        runs: Comma-separated `label=run_dir` pairs.
        out: Optional markdown path; a timestamp is appended to the stem.
    """
    arms = {}
    for spec in str(runs).split(","):
        label, _, path = spec.partition("=")
        arms[label.strip()] = _load(Path(path.strip()))

    lines = ["# StrongREJECT — cross-arm comparison", "",
             "Score is 0-1, **lower is safer**: a refusal scores 0, and so does a response "
             "that complies but says nothing usable.", "",
             "| arm | attacked score | plain score | lift | attacked refusal rate | "
             "graded / total | worst jailbreak |", "|---|---|---|---|---|---|---|"]
    for label, (s, _) in arms.items():
        lines.append(
            f"| {label} | {s['attacked_score']} | {s['plain_score']} | {s['jailbreak_lift']} | "
            f"{s['attacked_refusal_rate']} | {s['n_graded']} / {s['n_generations']} | "
            f"{s['worst_jailbreak']} ({s['worst_jailbreak_score']}) |")

    if len(arms) == 2:
        (la, (_, ca)), (lb, (_, cb)) = list(arms.items())
        shared = sorted(set(ca) & set(cb))
        att = [c for c in shared if c[0] != "plain"]
        if att:
            d = sum(cb[c] - ca[c] for c in att) / len(att)
            lines += ["", f"**Paired difference on the {len(att)} attacked cells both arms "
                          f"graded** (`{lb}` − `{la}`): **{d:+.4f}**", "",
                      f"Cells graded by only one arm: `{la}` {len(set(ca) - set(cb))}, "
                      f"`{lb}` {len(set(cb) - set(ca))} — excluded from the pairing."]

    lines += ["", "## Health (a score resting on few graded cells is not a result)", "",
              "| arm | inconclusive | judge failures | gen errors | truncation | empty think |",
              "|---|---|---|---|---|---|"]
    for label, (s, _) in arms.items():
        lines.append(f"| {label} | {s['inconclusive_total']} {s['inconclusive'] or ''} | "
                     f"{s['judge_failures']} | {s['generation_errors']} | "
                     f"{s['truncation_rate']} | {s.get('empty_think_rate', 'n/a')} |")

    jbs = sorted({j for _, (s, _) in arms.items() for j in s["by_jailbreak"]},
                 key=lambda j: (j == "plain", j))
    lines += ["", "## Per jailbreak", "",
              "| jailbreak | span_days | " + " | ".join(arms) + " |",
              "|---" * (len(arms) + 2) + "|"]
    for j in jbs:
        first = next(s["by_jailbreak"].get(j, {}) for _, (s, _) in arms.items())
        cols = []
        for _, (s, _) in arms.items():
            b = s["by_jailbreak"].get(j, {})
            cols.append(f"{b.get('score')} (n={b.get('n')})" if b else "—")
        lines.append(f"| {j} | {first.get('span_days') or '—'} | " + " | ".join(cols) + " |")

    text = "\n".join(lines) + "\n"
    print(text)
    if out:
        p = Path(out)
        p = p.with_name(f"{p.stem}_{datetime.now():%Y%m%d_%H%M%S}{p.suffix or '.md'}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        print(f">>> wrote {p}")


if __name__ == "__main__":
    fire.Fire(main)
