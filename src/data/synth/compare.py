# ABOUTME: Compare the corpus reports of several runs as arms of one experiment:
# ABOUTME: a metric x arm table, a monotonicity verdict, and a diff of what each flags.

"""Cross-arm comparison of corpus reports.

One run is one arm, so a claim like "the share of documents resolving a value tension
rises with group size" is a statement about several run directories at once and nothing
inside a single run can check it.

This reads each arm's `<stage>_report.json` plus its `manifest.json`, lines the metrics
up, and says whether the expected trend holds. It computes nothing new: every number
came from the corpus checks of the run it belongs to.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# Metrics whose direction is a claim worth testing rather than an observation.
# `+1` = expected to rise with the ordering key, `-1` = expected to fall.
MONOTONE_EXPECTED: dict[str, int] = {
    "applies_vs_conflicts.conflict_rate": +1,
    "principle_coverage.normalized_entropy": +1,
    "chunk_attribution.effective_k": +1,
    "principle_coverage.off_target_rate": -1,
    "ngram_diversity.max_top_8gram_share": -1,
}


def _ranks(vs: list[float]) -> list[float]:
    """Ranks of a list, ties averaged."""
    order = sorted(range(len(vs)), key=lambda i: vs[i])
    out = [0.0] * len(vs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation, ties averaged. None when it is undefined."""
    n = len(xs)
    if n < 3 or len({round(y, 12) for y in ys}) < 2:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / den, 4) if den else None


def _numeric(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_arm(run_dir: str | Path, stage: str | None = None) -> dict:
    """Read one arm: its corpus report, its manifest, and its scalar metrics."""
    run_dir = Path(run_dir)
    reports = [p for p in sorted(run_dir.glob(f"{stage or '*'}_report.json"))
               if p.name != "checks_report.json"]
    assert reports, (
        f"no corpus report in {run_dir}. Run the pipeline with a `corpus_check` stage, "
        f"or `uv run synth check --config <cfg> --run_dir {run_dir}` over a finished run.")
    assert len(reports) == 1 or stage, (
        f"{run_dir} holds several corpus reports {[p.name for p in reports]}; "
        f"pass --stage to say which one")
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    mpath = run_dir / "manifest.json"
    live = {alias: entry for alias, entry in (report.get("properties") or {}).items()
            if entry.get("status") not in ("skipped", "errored", "disabled")}
    metrics = {f"{alias}.{k}": v
               for alias, entry in live.items()
               for k, v in (entry.get("metrics") or {}).items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
    # pattern_scan reports a LIST of patterns, so its per-pattern rates would be invisible
    # to a comparison that only reads scalars -- and those rates are the whole point of
    # comparing two corpora. Flattened to one scalar each, keyed by pattern name, so an
    # arm missing a pattern shows as absent rather than as zero.
    for alias, entry in live.items():
        for row in (entry.get("metrics") or {}).get("patterns") or []:
            for half in ("broad_share", "strict_share"):
                metrics[f"{alias}.pattern.{row['pattern']}.{half}"] = row[half]
    return {"run_dir": str(run_dir), "name": run_dir.name, "report": report,
            "metrics": metrics,
            "manifest": json.loads(mpath.read_text(encoding="utf-8"))
            if mpath.exists() else {}}


def _arm_key(arm: dict, key: str) -> Any:
    """The value ordering this arm: a manifest field, a config field, or its name."""
    m, cfg = arm["manifest"], (arm["manifest"].get("config") or {})
    for source in (m, cfg, cfg.get("corpus") or {}, *(cfg.get("stages") or [])):
        if isinstance(source, dict) and key in source:
            return source[key]
    return arm["name"]


def compare_arms(run_dirs: list[str | Path], key: str = "group_size",
                 stage: str | None = None) -> dict:
    """Line up several arms' corpus reports and test the expected trends.

    Returns the arms, the metric x arm table, per-metric monotonicity verdicts, and the
    findings each arm raised that the others did not. `key` is the manifest/config field
    ordering the arms (e.g. `group_size`); `stage` picks the report when a run has more
    than one.
    """
    assert len(run_dirs) >= 2, "comparing arms needs at least two runs"
    arms = [load_arm(d, stage) for d in run_dirs]
    for a in arms:
        a["key"] = _arm_key(a, key)

    shas = {a["manifest"].get("constitution_sha256") for a in arms
            if a["manifest"].get("constitution_sha256")}
    assert len(shas) <= 1, (
        f"these runs were generated against different constitutions {sorted(shas)}; "
        f"their corpora are not arms of one experiment. Point every arm at the same "
        f"constitution or compare them separately.")

    arms.sort(key=lambda a: (_numeric(a["key"]) is None, _numeric(a["key"]),
                             str(a["key"])))
    xs = [_numeric(a["key"]) for a in arms]
    ordered = all(x is not None for x in xs)

    warnings = []
    sizes = {a["report"].get("n_records") for a in arms}
    if len(sizes) > 1:
        warnings.append(f"arms differ in size {sorted(sizes)} -- rates are comparable, "
                        f"absolute counts are not")
    samples = {json.dumps({k: v.get("params", {}).get("sample")
                           for k, v in sorted((a["report"].get("properties") or {}).items())
                           if v.get("params", {}).get("sample") is not None})
               for a in arms}
    if len(samples) > 1:
        warnings.append("arms judged different numbers of documents; a ratio measured "
                        "at different samples carries a different interval")
    if not ordered:
        warnings.append(f"key {key!r} is not numeric on every arm "
                        f"({[a['key'] for a in arms]}); arms are ordered as given and "
                        f"monotonicity is not tested")

    table = {m: [a["metrics"].get(m) for a in arms]
             for m in sorted({m for a in arms for m in a["metrics"]})}

    trends = {}
    for m, values in table.items():
        expected = MONOTONE_EXPECTED.get(m)
        present = [(x, v) for x, v in zip(xs, values) if x is not None and v is not None]
        if expected is None or not ordered or len(present) < 3:
            continue
        vs = [v for _, v in present]
        deltas = [b - a for a, b in zip(vs, vs[1:])]
        breaks = [str(arms[i + 1]["key"]) for i, d in enumerate(deltas)
                  if d * expected <= 0]
        trends[m] = {
            "expected": "rises" if expected > 0 else "falls",
            "spearman": spearman([x for x, _ in present], vs), "values": vs,
            "verdict": ("monotone as expected" if not breaks else
                        ("non-monotonic at " + ", ".join(f"{key}={b}" for b in breaks))),
            "holds": not breaks,
        }

    flagged: dict[str, set[str]] = {}
    for a in arms:
        for f in a["report"].get("findings", []):
            if f["severity"] in ("critical", "error"):
                flagged.setdefault(f"{f['property']}.{f['metric']}",
                                   set()).add(str(a["key"]))
    all_keys = {str(a["key"]) for a in arms}
    diff = {k: {"flagged_in": sorted(v), "clean_in": sorted(all_keys - v)}
            for k, v in sorted(flagged.items()) if v != all_keys}

    return {
        "key": key,
        "arms": [{"key": a["key"], "name": a["name"], "run_dir": a["run_dir"],
                  "n_records": a["report"].get("n_records"),
                  "pass": a["report"].get("pass")} for a in arms],
        "table": table, "trends": trends, "findings_diff": diff,
        "warnings": warnings,
    }


def markdown(result: dict) -> str:
    """Render a comparison as a markdown mirror -- numbers greppable without the JSON."""
    arms = result["arms"]
    head = [f"{result['key']}={a['key']}" for a in arms]
    lines = ["# Corpus comparison across arms", "",
             f"Ordered by `{result['key']}`. "
             + ", ".join(f"**{h}** ({a['n_records']} docs, "
                         f"{'pass' if a['pass'] else 'FAIL'})"
                         for h, a in zip(head, arms)), ""]
    lines += [f"> **Warning** {w}" for w in result["warnings"]] + [""]

    if result["trends"]:
        lines += ["## Expected trends", "",
                  "| metric | expected | values | Spearman | verdict |", "|---" * 5 + "|"]
        for m, t in result["trends"].items():
            lines.append(f"| `{m}` | {t['expected']} | "
                         + " → ".join(f"{v:.4g}" for v in t["values"])
                         + f" | {t['spearman']} | {'✅' if t['holds'] else '❌'} "
                           f"{t['verdict']} |")
        lines.append("")

    lines += ["## All metrics", "", "| metric | " + " | ".join(head) + " |",
              "|---" * (len(head) + 1) + "|"]
    for m, values in result["table"].items():
        lines.append(f"| `{m}` | "
                     + " | ".join("—" if v is None else f"{v:.4g}" for v in values)
                     + " |")
    lines.append("")

    if result["findings_diff"]:
        lines += ["## Flagged in some arms but not others", "",
                  "| metric | flagged in | clean in |", "|---" * 3 + "|"]
        for m, d in result["findings_diff"].items():
            lines.append(f"| `{m}` | {', '.join(d['flagged_in'])} | "
                         f"{', '.join(d['clean_in'])} |")
        lines.append("")
    return "\n".join(lines)
