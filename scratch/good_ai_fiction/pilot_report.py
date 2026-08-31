# ABOUTME: The pilot's QC/coverage/token report: what got generated, what the gates
# ABOUTME: rejected and why, and where the corpus sits against the DA-716 slice it replaces.
# Run: uv run python scratch/good_ai_fiction/pilot_report.py --run <run dir>

"""A report that is readable without opening the corpus, and greppable afterwards.

Three things it exists to surface, none of which the run log says:

  TOKENS. Trainable, CoT and reply, against DA-716's measured 832,064 / 50.6% / 49.4%.
  A projection over 716 rows, so a pilot answers the question the full run will be judged
  on.

  COVERAGE. Realised trait, cluster, stakes, form, length and source splits against the
  quotas the taxonomy declares -- and the archetypes actually used, which is the axis most
  likely to have collapsed onto whichever three the generator finds most vivid.

  WHAT THE GATES CAUGHT. Counts and reasons, per gate. A recipe whose gates reject nothing
  is not a recipe with no defects; it is a recipe whose gates are not doing anything.

Also reports the opening bigrams of both trained fields. That is a MEASUREMENT standing in
for a gate: the stock-opener lint was removed from the reasoning field because it cost 20%
of a run, and the thing it was protecting against (a corpus that opens the same way every
time) has to be watched some other way.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import fire
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TAX = yaml.safe_load(
    (ROOT / "configs/data/synth/good_ai_fiction/taxonomy.yaml").read_text(
        encoding="utf-8"))
CLUSTER = {u: c["id"] for c in TAX["clusters"] for u in c["units"]}

DA = {"rows": 716, "trainable": 832_064, "reasoning": 421_163, "answer": 410_901,
      "median_trainable": 1141, "median_reasoning": 584, "median_answer": 557}

# Title-case words that are vessel types, roles or sentence openers rather than names.
# Without this the name-reuse count reports "Freighter x3" and "Survey x2" and buries the
# signal it exists for, which is two worlds sharing an invented proper noun.
_GENERIC = {
    "ship", "vessel", "station", "freighter", "hauler", "carrier", "survey", "probe",
    "outpost", "settlement", "colony", "habitat", "arcology", "relay", "seed", "deep",
    "research", "mining", "transit", "orbital", "lunar", "solar", "system", "sector",
    "core", "hull", "deck", "crew", "council", "collective", "governance", "processor",
    "mind", "intelligence", "unit", "array", "network", "grid", "dome", "ring", "belt",
    "cloud", "field", "watch", "night", "year", "years", "month", "months", "decade",
    "generation", "there", "their", "this", "that", "these", "those", "when", "where",
    "after", "before", "since", "every", "most", "some", "each", "with", "from", "into",
    # Observed at 860 scale: SF common nouns the title-case regex cannot tell from names.
    "archive", "cycler", "rotation", "fermenter", "terrarium", "consensus", "collate",
    "cascade", "reactor", "foundry", "granary", "nursery", "hospice", "quarantine",
    # Real bodies. Two colonies on Enceladus is the genre, not a collapsed generator, so
    # counting them as name reuse would bury the signal this metric exists for.
    "enceladus", "titan", "europa", "ganymede", "callisto", "ceres", "mars", "luna",
    "triton", "charon", "io", "phobos", "deimos", "vesta", "pallas", "oberon", "rhea",
}


def pct(part: int, whole: int) -> str:
    return f"{100 * part / max(whole, 1):5.1f}%"


def table(title: str, got: Counter, want: dict[str, float] | None, total: int) -> str:
    """One coverage axis: realised count, realised share, declared share, drift."""
    lines = [f"### {title}", "", "| value | n | got | target | drift |",
             "|---|---:|---:|---:|---:|"]
    keys = sorted(set(got) | set(want or {}))
    for k in keys:
        n = got.get(k, 0)
        share = n / max(total, 1)
        if want and k in want:
            lines.append(f"| {k} | {n} | {share:.1%} | {want[k]:.0%} | "
                         f"{share - want[k]:+.1%} |")
        else:
            lines.append(f"| {k} | {n} | {share:.1%} | — | — |")
    return "\n".join(lines) + "\n"


def openers(texts: list[str], k: int = 6) -> list[tuple[str, int]]:
    """Most common opening bigrams — the collapse the removed lint used to guard."""
    heads = [" ".join(re.findall(r"[\w']+", t.lower())[:2]) for t in texts if t]
    return Counter(heads).most_common(k)


def main(run: str, out: str = "") -> None:
    """Write `<run>/pilot_report.md` from the run's own artifacts.

    Args:
        run: Run directory (needs `token_stats.json`; uses `selection.json` if present).
        out: Output path; defaults to `<run>/pilot_report.md`.
    """
    run_dir = Path(run)
    stats = json.loads((run_dir / "token_stats.json").read_text(encoding="utf-8"))
    recs = stats["records"]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")) \
        if (run_dir / "manifest.json").exists() else {}

    sel_ids: set[str] = set()
    sel_path = run_dir / "selection.json"
    selection = json.loads(sel_path.read_text(encoding="utf-8")) if sel_path.exists() \
        else None
    if selection:
        sel_ids = set(selection["ids"])

    export = run_dir / "dataset.jsonl"
    if not export.exists():
        export = next(run_dir.glob("stage_*_export_sft.jsonl"))
    rows = [json.loads(line) for line in export.open(encoding="utf-8") if line.strip()]
    md_by_id = {r["metadata"]["scenario_id"]: r for r in rows}

    scope = [r for r in recs if not sel_ids or r["scenario_id"] in sel_ids]
    n = len(scope)
    tot = {k: sum(r[k] for r in scope) for k in ("trainable", "reasoning", "answer")}
    mean = tot["trainable"] / max(n, 1)
    da_mean = DA["trainable"] / DA["rows"]

    # --- gates ---------------------------------------------------------------------
    status = Counter(r["revise_status"] or "ok" for r in recs)
    persona = Counter(r["judge_persona"] or "-" for r in recs)
    pattern = Counter(r["judge_pattern"] or "-" for r in recs)
    defects = Counter()
    for r in rows:
        for d in ((r["metadata"].get("judge_pattern") or {}).get("present") or []):
            defects[d] += 1
    critique = Counter(r["metadata"].get("critique_verdict", "?") for r in rows)

    L = []
    a = L.append
    a(f"# Good AI Fiction — pilot report\n")
    a(f"`{run_dir}`  ·  {len(recs)} candidates generated  ·  "
      f"{n} in scope for the numbers below"
      + (f" (the {len(sel_ids)} selected rows)" if sel_ids else " (all candidates)") + "\n")
    if manifest.get("usage"):
        u = manifest["usage"]
        a(f"Spend: **${u.get('total_usd', 0):.2f}**  ·  "
          f"models: {', '.join(sorted({s['model'] for s in u.get('per_stage', [])}))}\n"
          if isinstance(u, dict) else "")

    a("\n## 1. Trainable tokens, against the DA-716 slice this replaces\n")
    a("| | this pilot | DA-716 | ratio |")
    a("|---|---:|---:|---:|")
    a(f"| rows | {n} | {DA['rows']} | — |")
    a(f"| trainable / row (mean) | {mean:,.0f} | {da_mean:,.0f} | "
      f"{mean / da_mean:.3f} |")
    a(f"| CoT share of trainable | {pct(tot['reasoning'], tot['trainable'])} | "
      f"{pct(DA['reasoning'], DA['trainable'])} | — |")
    a(f"| reply share of trainable | {pct(tot['answer'], tot['trainable'])} | "
      f"{pct(DA['answer'], DA['trainable'])} | — |")
    for key, da_key in (("trainable", "median_trainable"), ("reasoning", "median_reasoning"),
                        ("answer", "median_answer")):
        q = stats["per_row"][key]
        a(f"| median {key} / row | {q['median']:,} | {DA[da_key]:,} | "
          f"{q['median'] / DA[da_key]:.3f} |")
    a(f"| **projected over 716 rows** | **{round(mean * 716):,}** | "
      f"**{DA['trainable']:,}** | **{mean * 716 / DA['trainable']:.3f}** |")
    a("")
    for key in ("trainable", "reasoning", "answer"):
        q = stats["per_row"][key]
        a(f"- per-row {key}: min {q['min']:,} · p10 {q['p10']:,} · p25 {q['p25']:,} · "
          f"median {q['median']:,} · p75 {q['p75']:,} · p90 {q['p90']:,} · "
          f"max {q['max']:,}")
    if stats["rows_at_cap"]:
        a(f"- **{stats['rows_at_cap']} rows hit the 8,192-token cap** and are truncated.")
    if stats["rows_without_think"]:
        a(f"- **{stats['rows_without_think']} rows carry no reasoning trace.**")
    a("")

    a("\n## 2. Coverage\n")
    a(table("Cluster (primary trait)",
            Counter(CLUSTER.get(r["trait_id"], "?") for r in scope),
            {c["id"]: c["share"] for c in TAX["clusters"]}, n))
    a(table("Constitution unit", Counter(r["trait_id"] for r in scope),
            {u: w / 100 for c in TAX["clusters"] for u, w in c["units"].items()}, n))
    a(table("World register", Counter(r["world"] for r in scope),
            {e["id"]: e["share"] for e in TAX["worlds"]}, n))
    a(table("Stakes band", Counter(r["stakes"] for r in scope),
            {e["id"]: e["share"] for e in TAX["stakes"]}, n))
    a(table("Narrative form", Counter(r["narrative_form"] for r in scope),
            {e["id"]: e["share"] for e in TAX["forms"]}, n))
    a(table("Length band", Counter(r["length_band"] for r in scope),
            {e["id"]: e["share"] for e in TAX["length_bands"]}, n))
    a(table("Source", Counter(r["source_type"] for r in scope),
            {e["id"]: e["share"] for e in TAX["source_types"]}, n))

    arche = Counter(r["source_archetype"] for r in scope if r["source_archetype"])
    a(f"### Bad-AI inversions used ({sum(arche.values())} rows, "
      f"{len(arche)} distinct archetypes)\n")
    a(", ".join(f"`{k}`×{v}" for k, v in arche.most_common()) or "_none_")
    a("")
    domains = Counter(r["domain"].lower() for r in scope if r["domain"])
    named = sum(1 for r in scope if str(r.get("ai_name") or "").strip())
    a(f"\n### Settings\n\n{len(domains)} distinct worlds across {n} rows; "
      f"{named} ({named / max(n, 1):.0%}) carry a named mind. Most repeated: "
      + ", ".join(f"{k} ×{v}" for k, v in domains.most_common(5)) + "\n")

    # THE CHECK THE 2026-08-27 CORRECTION EXISTS FOR. The first pilot produced 29 rows
    # and every one was a present-day workplace, so the setting is not something to trust
    # a prompt about -- it gets counted. The lint gates these, so a nonzero count here
    # means a gate is leaking, not merely that the prose drifted.
    tells = re.compile(
        r"\b(?:postdoc|principal investigator|tenure|peer[-\s]?review\w*|IRB"
        r"|insurance|insurer|underwrit\w+|actuar\w+|quarterly|fiscal year|KPIs?|OKRs?"
        r"|SaaS|CRM|ERP|Slack|Zoom|PowerPoint|spreadsheet|LinkedIn|hospital|university"
        r"|faculty|dean|data ?cent(?:re|er)|DevOps|Kubernetes|SLA)\b", re.I)
    hits: Counter = Counter()
    flagged: list[str] = []
    for r in scope:
        rec = md_by_id.get(r["scenario_id"])
        if not rec:
            continue
        asst = next(m for m in rec["messages"] if m["role"] == "assistant")
        found = tells.findall(
            (asst.get("content") or "") + " " + (asst.get("reasoning_content") or ""))
        if found:
            flagged.append(r["scenario_id"])
            hits.update(w.lower() for w in found)
    # NAME COLLAPSE. Measured 2026-08-27: 11 of 24 vessels were called Meridian and 7
    # were built on "Kepler", while the embedding dedup passed every one -- because the
    # SITUATIONS were genuinely different and only the world-building vocabulary had
    # collapsed. Dedup on the situation cannot see this, so it gets its own count.
    proper = Counter()
    for r in scope:
        rec = md_by_id.get(r["scenario_id"])
        if not rec:
            continue
        text = f"{r.get('domain', '')} {rec['metadata'].get('world_detail', '')} " \
               f"{rec['metadata'].get('ai_name', '')}"
        for w in re.findall(r"\b[A-Z][a-z]{3,}\b", text):
            if w.lower() not in _GENERIC:
                proper[w] += 1
    repeated = [(w, c) for w, c in proper.most_common() if c > 1]
    a("### Name reuse across worlds\n")
    a((f"**{repeated[0][1]} of {n} rows** share the most-reused proper noun. Repeats: "
       + ", ".join(f"`{w}` ×{c}" for w, c in repeated[:10])
       if repeated else "No proper noun appears in more than one world.") + "\n")

    a(f"### Present-day tells in the trained text\n")
    a(f"**{len(flagged)} of {n} rows** carry a contemporary-workplace noun"
      + (f": {', '.join(f'`{k}` ×{v}' for k, v in hits.most_common(8))}. "
         f"Rows: {', '.join(flagged[:8])}" if flagged
         else ". The setting gate held.") + "\n")

    a("\n## 3. What the pipeline rejected\n")
    a(f"- **Candidates generated:** {len(recs)} (from {stats['rows']} exported rows)")
    a(f"- **Critic verdict:** " + ", ".join(f"{k} {v}" for k, v in critique.most_common()))
    a(f"- **Persona gate:** " + ", ".join(f"{k} {v}" for k, v in persona.most_common()))
    a(f"- **Pattern gate:** " + ", ".join(f"{k} {v}" for k, v in pattern.most_common()))
    a(f"- **Rewrite status:** " + ", ".join(f"{k} {v}" for k, v in status.most_common()))
    if defects:
        a(f"- **Defects the pattern gate named:** "
          + ", ".join(f"{k} ×{v}" for k, v in defects.most_common()))
    if selection:
        a(f"- **Selected:** {selection['selected']} of {selection['pool']} rows that "
          f"cleared every gate ({selection['dropped_by_gates']} dropped by gates); "
          f"token gap to target {selection['gap']:+,} ({selection['gap_pct']:+.2f}%)")
    a("")

    a("\n## 4. Voice: opening bigrams (the removed lint's replacement)\n")
    for label, key in (("reasoning", "reasoning_content"), ("reply", "content")):
        texts = [next(m for m in md_by_id[r["scenario_id"]]["messages"]
                      if m["role"] == "assistant").get(key, "") for r in scope]
        top = openers(texts)
        worst = f"{100 * top[0][1] / max(n, 1):.0f}%" if top else "—"
        a(f"- **{label}** — top opener {worst} of rows: "
          + ", ".join(f"`{k}` ×{v}" for k, v in top))
    a("")

    dest = Path(out) if out else run_dir / "pilot_report.md"
    dest.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    fire.Fire(main)
