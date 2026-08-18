# ABOUTME: Turn a fabrication-scenario run into results.md: per-family rates, the
# ABOUTME: knew-better cases in full, and an explicit accounting of ungraded samples.

"""Report on a constructed-scenario fabrication run.

Run: uv run python scratch/fabrication_scenarios_report.py <run-dir>

Reports coverage first: generation and judge failures are named, never silently folded into
the denominator, because a family whose failures cluster in it has a rate computed on fewer
samples than it appears.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

FAMS = ["hard_constraint", "false_precision", "sycophantic", "fabricated_cites",
        "pseudoscience_formalism"]
LABEL = {"hard_constraint": "A. Hard constraint",
         "false_precision": "B. False precision (F.1)",
         "sycophantic": "C. Sycophantic pressure",
         "fabricated_cites": "D. Fabricated citations",
         "pseudoscience_formalism": "E. Pseudoscience formalism (F.2)"}


def main() -> None:
    """Write results.md for the run directory given on the command line."""
    d = Path(sys.argv[1])
    rows = [json.loads(line) for line in open(d / "judged.jsonl")]
    meta = json.loads((d / "run_meta.json").read_text())
    g = [r for r in rows if "fabricated" in r]

    genf = [r for r in rows if "error" in r]
    judf = [r for r in rows if "judge_error" in r]

    o = [f"# Fabrication scenarios — {meta['target']}", "",
         f"Target `{meta['target']}` · {meta['n_scenarios']} scenarios × "
         f"{meta['samples_per_scenario']} samples · judge `{meta['judge']}` · "
         f"git `{meta['git_sha'][:8]}`", "",
         f"Reasoning parser: **{meta['reasoning_parser']}**", "",
         "## Coverage", "",
         f"- generated and graded: **{len(g)} / {len(rows)}**",
         f"- generation failures (524 timeouts): {len(genf)} — "
         f"{', '.join(sorted(r['id'] for r in genf)) or 'none'}",
         f"- judge failures (unparseable JSON): {len(judf)} — "
         f"{', '.join(sorted(r['id'] for r in judf)) or 'none'}",
         f"- samples carrying a reasoning trace: "
         f"{sum(1 for r in g if r.get('reasoning'))} / {len(g)}", ""]

    o += ["## Rates by family", "",
          "| family | n | fabricated | honest | hedged AND fabricated | **knew better** |",
          "|---|---:|---:|---:|---:|---:|"]
    for f in FAMS:
        rs = [r for r in g if r["family"] == f]
        if not rs:
            continue
        fab = sum(r["fabricated"] for r in rs)
        hon = sum(r["honest_refusal"] for r in rs)
        both = sum(r["fabricated"] and r["honest_refusal"] for r in rs)
        knew = sum(r["knew_better"] for r in rs)
        o.append(f"| {LABEL[f]} | {len(rs)} | **{fab} ({fab / len(rs) * 100:.0f}%)** | "
                 f"{hon} | {both} | {knew} |")
    fab_all = sum(r["fabricated"] for r in g)
    o += ["", f"**Overall: {fab_all}/{len(g)} = {fab_all / len(g) * 100:.0f}% fabricated.** "
              "`fabricated` and `honest_refusal` are not exclusive — a response that states "
              "there is no valid basis and then supplies numbers anyway scores both.", ""]

    o += ["## Per-scenario", "", "| id | family | n | fabricated | knew better | mean severity |",
          "|---|---|---:|---:|---:|---:|"]
    by = defaultdict(list)
    for r in g:
        by[r["id"]].append(r)
    for sid in sorted(by):
        rs = by[sid]
        fab = sum(r["fabricated"] for r in rs)
        o.append(f"| {sid} | {rs[0]['family']} | {len(rs)} | {fab}/{len(rs)} | "
                 f"{sum(r['knew_better'] for r in rs)} | "
                 f"{sum(r['severity'] for r in rs) / len(rs):.1f} |")

    knew = [r for r in g if r["knew_better"]]
    o += ["", f"## Knew better — trace contradicts the answer ({len(knew)})", "",
          "The most serious category, and invisible without the trace: the reasoning reaches "
          "the correct conclusion or flags the problem, and the answer then contradicts it.", ""]
    for r in sorted(knew, key=lambda r: -r["severity"]):
        o += [f"### {r['id']} — severity {r['severity']}", "",
              f"**Prompt** — {' '.join(r['prompt'].split())}", "",
              f"**Ground truth** — {' '.join(r['truth'].split())}", "",
              f"**Judge** — {r['why']}", ""]
        if r.get("quote"):
            o += ["**Fabricated span**", "", "```", str(r["quote"])[:600], "```", ""]
        o += ["**Trace (final 600 chars)**", "", "```",
              (r.get("reasoning") or "")[-600:], "```", ""]

    (d / "results.md").write_text("\n".join(o))
    print(f"wrote {d / 'results.md'}  ({len(g)} graded, {fab_all} fabricated, "
          f"{len(knew)} knew-better)")


if __name__ == "__main__":
    main()
