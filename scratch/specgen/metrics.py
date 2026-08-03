# ABOUTME: Offline spec-side metrics, hard-constraint checks, cross-seed stability (ARI),
# ABOUTME: and the pre-registered seed selection -> metrics.json per doc + comparison.md.

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pipeline import PKG, _counter, out_dir  # noqa: E402
from utils import read_jsonl  # noqa: E402

HARD = re.compile(r"\b(never|always|must not|must|do not|no exception\w*)\b", re.I)
SOFT = re.compile(r"\b(prefer|weigh|generally|usually|consider|balance|in most cases)\b", re.I)


def parse_units(doc: str) -> list[dict]:
    """Split an assembled spec into units of {title, statement, why, cues, not_apply}."""
    body = doc.split("\n---\n", 2)[1]  # between the preamble and closing separators
    units = []
    for chunk in re.split(r"^## ", body, flags=re.M)[1:]:
        why_split = chunk.split("*Why:*", 1)
        rest = why_split[1] if len(why_split) == 2 else ""
        na_split = rest.split("*When this does NOT apply:*", 1)
        cues = re.findall(r"^- .+$", na_split[0], flags=re.M)
        why = re.split(r"^- ", na_split[0], flags=re.M)[0]
        units.append({
            "title": chunk.splitlines()[0].strip(),
            "statement": "\n".join(why_split[0].splitlines()[1:]).strip(),
            "why": why.strip(),
            "cues": cues,
            "not_apply": (na_split[1] if len(na_split) == 2 else "").strip(),
        })
    return units


def doc_metrics(doc: str, count) -> dict:
    """Per-document metrics; `count` maps text -> token count (injectable for tests)."""
    units = parse_units(doc)
    per_unit = [count("## " + u["title"] + "\n" + u["statement"] + "\n" + u["why"]
                      + "\n" + "\n".join(u["cues"]) + "\n" + u["not_apply"])
                for u in units]
    expl = [count(u["why"] + "\n" + u["not_apply"]) / max(t, 1)
            for u, t in zip(units, per_unit)]
    hard, soft = len(HARD.findall(doc)), len(SOFT.findall(doc))
    return {
        "n_units": len(units),
        "tokens_total": count(doc),
        "tokens_per_unit": {"min": min(per_unit), "max": max(per_unit),
                            "mean": round(sum(per_unit) / len(per_unit), 1)},
        "explanation_ratio": {"mean": round(sum(expl) / len(expl), 3),
                              "per_unit": [round(e, 3) for e in expl]},
        "modality_profile": {"hard": hard, "soft": soft,
                             "hard_ratio": round(hard / max(hard + soft, 1), 3)},
    }


def ari(a: dict[str, int], b: dict[str, int]) -> float:
    """Adjusted Rand index between two partitions of the same claim ids."""
    ids = sorted(a)
    assert sorted(b) == ids, "partitions cover different claim sets"
    n = comb(len(ids), 2)
    sum_ab = sum(comb(c, 2) for c in Counter((a[i], b[i]) for i in ids).values())
    sum_a = sum(comb(c, 2) for c in Counter(a.values()).values())
    sum_b = sum(comb(c, 2) for c in Counter(b.values()).values())
    expected, max_index = sum_a * sum_b / n, (sum_a + sum_b) / 2
    return 1.0 if max_index == expected else (sum_ab - expected) / (max_index - expected)


def _labels(clusters_path: Path) -> dict[str, int]:
    """claim_id -> cluster index, from a clusters.json."""
    data = json.loads(clusters_path.read_text())
    return {cid: k for k, c in enumerate(data["clusters"]) for cid in c["claim_ids"]}


def check(cfg: dict, arm: str, seed_dir: Path, m: dict, inventory: list[dict]) -> list[str]:
    """Return the list of hard-constraint violations for one generated spec."""
    a, fails = cfg["arms"][arm], []
    lo, hi = a["band"]
    if not lo <= m["tokens_total"] <= hi:
        fails.append(f"tokens {m['tokens_total']} outside band [{lo},{hi}]")
    if m["tokens_per_unit"]["min"] < int(cfg["unit_floor_tokens"]):
        fails.append(f"unit below floor ({m['tokens_per_unit']['min']} tokens)")
    rl, rh = cfg["explanation_ratio_band"]
    if not rl <= m["explanation_ratio"]["mean"] <= rh:
        fails.append(f"explanation ratio {m['explanation_ratio']['mean']} outside [{rl},{rh}]")
    if m["n_units"] != int(a["n_principles"]):
        fails.append(f"{m['n_units']} units != N={a['n_principles']}")
    labels = _labels(seed_dir / "clusters.json")
    missing = {c["claim_id"] for c in inventory} - set(labels)
    if missing:
        fails.append(f"{len(missing)} claims uncovered")
    doc = (seed_dir / "constitution.md").read_text()
    for name in ("preamble", "closing"):
        if (PKG / f"{name}.md").read_text() not in doc:
            fails.append(f"{name} not byte-identical to {name}.md")
    return fails


def run(cfg: dict, smoke: bool = False) -> dict:
    """Compute metrics for every generated spec, select seeds, write comparison.md."""
    root = out_dir(cfg, smoke)
    inventory = read_jsonl(root / "claims" / "inventory.jsonl")
    count = _counter(cfg["tokenizer"])
    source_tokens = count((out_dir(cfg) / "source" / "constitution.md").read_text())
    selection, rows = {}, []

    for arm in cfg["arms"]:
        seed_dirs = sorted((root / arm).glob("seed*")) if (root / arm).exists() else []
        results = {}
        for sd in seed_dirs:
            m = doc_metrics((sd / "constitution.md").read_text(), count)
            m["compression_ratio"] = round(source_tokens / m["tokens_total"], 2)
            m["violations"] = check(cfg, arm, sd, m, inventory)
            (sd / "metrics.json").write_text(json.dumps(m, indent=2))
            results[sd.name] = m
        if not results:
            continue
        labels = {sd.name: _labels(sd / "clusters.json") for sd in seed_dirs}
        names = sorted(labels)
        aris = [ari(labels[x], labels[y]) for i, x in enumerate(names)
                for y in names[i + 1:]]
        mean_ari = round(sum(aris) / len(aris), 3) if aris else None

        # Pre-registered rule: among seeds passing every hard constraint, select the
        # one whose mean explanation ratio is closest to the target.
        target = float(cfg["selection"]["explanation_ratio_target"])
        passing = {s: m for s, m in results.items() if not m["violations"]}
        picked = min(passing, key=lambda s: abs(
            passing[s]["explanation_ratio"]["mean"] - target)) if passing else None
        selection[arm] = {"selected": picked, "cross_seed_ari": mean_ari,
                          "passing": sorted(passing),
                          "violations": {s: m["violations"] for s, m in results.items()
                                         if m["violations"]}}
        if picked:
            p = results[picked]
            rows.append(f"| {arm} | {cfg['arms'][arm]['n_principles']} | {picked} | "
                        f"{p['tokens_total']} | {p['tokens_per_unit']['mean']} | "
                        f"{p['compression_ratio']} | {p['explanation_ratio']['mean']} | "
                        f"{p['modality_profile']['hard_ratio']} | {mean_ari} |")

    picked_tokens = [json.loads((root / a / s["selected"] / "metrics.json").read_text())
                     ["tokens_total"] for a, s in selection.items() if s["selected"]]
    if len(picked_tokens) > 1:
        spread = round(max(picked_tokens) / min(picked_tokens), 2)
        if spread > float(cfg["max_spread"]):
            print(f"!!! selected-arm spread {spread}x exceeds max {cfg['max_spread']}x")
        selection["spread"] = spread

    (root / "selection.json").write_text(json.dumps(selection, indent=2))
    (root / "comparison.md").write_text(
        "| arm | N | seed | tokens | tokens/unit | compression | expl. ratio "
        "| hard ratio | cross-seed ARI |\n|---|---|---|---|---|---|---|---|---|\n"
        + "\n".join(rows) + "\n")
    print(json.dumps(selection, indent=2))
    return selection
