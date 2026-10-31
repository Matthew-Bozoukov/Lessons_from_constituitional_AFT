# ABOUTME: Builds the 2026-09-29 DA drift timeline table from the pressure, AI-template and AI-central-by-stage
# ABOUTME: label files, deriving labels for subset corpora (no-t6, new-t6, tools) from their parents by scenario_id.
"""    uv run python scratch/da_drift_timeline.py   -> prints the table (markdown) + per-trait ai_system, writes JSON

Reads (never re-judges):
  output/da_pressure/<date>_<repo>_pressure_labels.jsonl          scratch/classify_da_pressure.py
  output/audits/<date>_da_ai_template/<repo>.jsonl                 scratch/da_ai_template_audit.py
  <run_dir>/ai_central_by_stage.jsonl                              scratch/da_ai_central_by_stage[_tolerant].py
Derived corpora map rows to a parent's labels by scenario_id after checking the user message is byte-identical.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv(".env")
ORG = "dougalldeepmind"
RUNS = Path("output/audits/2026-09-28_corpus_runs")

# repo -> (revision, pressure file, template file, ai_central run dir, derivation)
#   derivation: None (judged directly) or {"t*": parent_repo} / {"default": parent, "t6": donor}
CORPORA = {
    "2026-09-08-da-synth": dict(rev="42107bde00cd7f4360a3a6c581aac23a540dbfea",
        pressure="output/da_pressure/2026-09-29_2026-09-08-da-synth_pressure_labels.jsonl",
        template="output/audits/2026-09-28_da_ai_template/2026-09-08-da-synth.jsonl",
        central=RUNS / "2026-09-08-da-synth"),
    "2026-09-14-da-synth": dict(rev="013886238fca238c4d54ace96530f444bb2b2f02",
        pressure="output/da_pressure/2026-09-21_2026-09-14-da-synth_pressure_labels.jsonl",
        template="output/audits/2026-09-28_da_ai_template/2026-09-14-da-synth.jsonl",
        central=RUNS / "2026-09-14-da-synth"),
    "2026-09-15-da-lowstakes-refresh-synth": dict(rev="ebafc3a6",
        pressure="output/da_pressure/2026-09-29_2026-09-15-da-lowstakes-refresh-synth_pressure_labels.jsonl",
        template="output/audits/2026-09-29_da_ai_template/2026-09-15-da-lowstakes-refresh-synth.jsonl",
        central=RUNS / "2026-09-15-da-lowstakes-refresh-synth"),
    "2026-09-23-da-synth": dict(rev="1f816d13",
        pressure="output/da_pressure/2026-09-29_2026-09-23-da-synth_pressure_labels.jsonl",
        template="output/audits/2026-09-29_da_ai_template/2026-09-23-da-synth.jsonl",
        central=RUNS / "2026-09-23-da-synth"),
    "2026-09-24-da-synth": dict(rev="4f4ca43bac4eace1aee0fed25df4175ad9299399",
        pressure="output/da_pressure/2026-09-24_2026-09-24-da-synth_pressure_labels.jsonl",
        template="output/audits/2026-09-29_da_ai_template/2026-09-24-da-synth.jsonl",
        central=RUNS / "2026-09-24-da-synth"),
    "2026-09-24-da-synth (r2_ new rows only)": dict(rev="4f4ca43bac4eace1aee0fed25df4175ad9299399", subset_of="2026-09-24-da-synth", prefix="r2_"),
    "2026-09-25-da-synth": dict(rev="618060e15315c71d7ffb9a8839198b08520a8771",
        pressure="output/da_pressure/2026-09-25_2026-09-25-da-synth_pressure_labels.jsonl",
        template="output/audits/2026-09-28_da_ai_template/2026-09-25-da-synth.jsonl",
        central=RUNS / "2026-09-25-da-synth"),
    "2026-09-25-da-no-t6-synth": dict(rev="b38daa5a", derive={"default": "2026-09-24-da-synth"}),
    "2026-09-25-da-new-t6-synth": dict(rev="5ae5665e", derive={"default": "2026-09-24-da-synth", "t6": "2026-09-25-da-synth"}),
    "2026-09-25-da-lowstakes-practical-synth": dict(rev="e7239a90",
        pressure="output/da_pressure/2026-09-29_2026-09-25-da-lowstakes-practical-synth_pressure_labels.jsonl",
        template="output/audits/2026-09-29_da_ai_template/2026-09-25-da-lowstakes-practical-synth.jsonl",
        central=RUNS / "2026-09-25-da-lowstakes-practical-synth"),
    "2026-09-26-da-multiparty-human-notools-synth": dict(rev="42bd374d",
        pressure="output/da_pressure/2026-09-29_2026-09-26-da-multiparty-human-notools-synth_pressure_labels.jsonl",
        template="output/audits/2026-09-29_da_ai_template/2026-09-26-da-multiparty-human-notools-synth.jsonl",
        central=RUNS / "2026-09-26-da-multiparty-human-notools-synth"),
    "2026-09-28-da-synth": dict(rev="b84401eaa689d02199ac5a619feaac2776b22569",
        pressure="output/da_pressure/2026-09-28_2026-09-28-da-synth_pressure_labels.jsonl",
        template="output/audits/2026-09-28_da_ai_template/2026-09-28-da-synth.jsonl",
        central=Path("output/synthdoc_v3/20260928_195614")),
    "2026-09-28-da-tools-synth": dict(rev="54d4e9dc", derive={"default": "2026-09-25-da-synth"}),
}


def jl(p):
    return [json.loads(l) for l in open(p)]


def dataset(repo, rev):
    p = hf_hub_download(f"{ORG}/{repo}", "dataset.jsonl", repo_type="dataset", revision=rev)
    return jl(p)


def user_of(r):
    return next(m["content"] for m in r["messages"] if m["role"] == "user")


def labels_direct(repo, spec):
    out = {"rows": len(dataset(repo, spec["rev"]))}
    out["pressure"] = [l for l in jl(spec["pressure"])] if Path(spec["pressure"]).exists() else None
    out["template"] = [l for l in jl(spec["template"]) if "unlabelled" not in l] if Path(spec["template"]).exists() else None
    cpath = Path(spec["central"]) / "ai_central_by_stage.jsonl"
    out["central"] = jl(cpath) if cpath.exists() else None
    return out


def labels_subset(spec, direct):
    d, pre = direct[spec["subset_of"]], spec["prefix"]
    rows = dataset(spec["subset_of"], spec["rev"])
    return {"rows": sum(1 for r in rows if r["metadata"]["scenario_id"].startswith(pre)),
            "pressure": [l for l in d["pressure"] if str(l["scenario_id"]).startswith(pre)] if d["pressure"] else None,
            "template": [l for l in d["template"] if str(l["scenario_id"]).startswith(pre)] if d["template"] else None,
            "central": [l for l in d["central"] if str(l["id"]).startswith(pre)] if d["central"] else None,
            "derived_note": f"subset of {spec['subset_of']}: scenario_id prefix {pre!r}"}


def labels_derived(repo, spec, direct):
    rows = dataset(repo, spec["rev"])
    parents = {k: dict((r["metadata"]["scenario_id"], r) for r in dataset(p, CORPORA[p]["rev"])) for k, p in spec["derive"].items()}
    idx = {}
    for k, p in spec["derive"].items():
        d = direct[p]
        idx[k] = dict(
            pressure={l["scenario_id"]: l for l in d["pressure"]} if d["pressure"] else None,
            template={l["scenario_id"]: l for l in d["template"]} if d["template"] else None,
            final={l["id"]: l for l in d["central"] if l["where"] == "final"} if d["central"] else None,
            writer={l["id"]: l for l in d["central"] if l["where"] == "writer"} if d["central"] else None)
    out = {"rows": len(rows), "pressure": [], "template": [], "central": []}
    mismatch = 0
    for r in rows:
        sid, tid = r["metadata"]["scenario_id"], r["metadata"]["trait_id"]
        k = tid if tid in spec["derive"] else "default"
        pr = parents[k].get(sid)
        if pr is None or user_of(pr) != user_of(r):
            mismatch += 1
            continue
        if idx[k]["pressure"] and sid in idx[k]["pressure"]:
            out["pressure"].append(idx[k]["pressure"][sid])
        if idx[k]["template"] and sid in idx[k]["template"]:
            out["template"].append(idx[k]["template"][sid])
        if idx[k]["final"] and sid in idx[k]["final"]:
            out["central"].append(idx[k]["final"][sid])
        if idx[k]["writer"] and sid in idx[k]["writer"]:
            out["central"].append(idx[k]["writer"][sid])
    out["derived_note"] = f"labels mapped from {spec['derive']} by scenario_id; {mismatch} rows without an identical parent row"
    return out


def pct(xs, pred):
    xs = list(xs)
    return f"{100 * sum(1 for x in xs if pred(x)) / len(xs):.0f}%" if xs else "n/a"


def summarise(repo, spec, lab):
    pr, tp, ce = lab["pressure"], lab["template"], lab["central"]
    w = [l for l in (ce or []) if l["where"] == "writer"]
    f = [l for l in (ce or []) if l["where"] == "final"]
    row = {
        "repo": repo, "rev": spec["rev"][:8], "rows": lab["rows"],
        "n_pressure": len(pr) if pr else 0, "n_template": len(tp) if tp else 0, "n_writer": len(w), "n_final": len(f),
        "ai_system": pct(tp or [], lambda l: l["ai_system"]),
        "template_all4": pct(tp or [], lambda l: l["template"]),
        "other_ai_writer": pct(w, lambda l: l["other_ai"]), "other_ai_final": pct(f, lambda l: l["other_ai"]),
        "ai_central_writer": pct(w, lambda l: l["ai_central"]), "ai_central_final": pct(f, lambda l: l["ai_central"]),
        "gvb_writer": pct(w, lambda l: l["good_vs_bad"]), "gvb_final_central": pct(f, lambda l: l["good_vs_bad"]),
        "push_or_override": pct(pr or [], lambda l: l["push"] in ("push", "override")),
        "override": pct(pr or [], lambda l: l["push"] == "override"),
        "lean": pct(pr or [], lambda l: l["push"] == "lean"),
        "good_vs_bad": pct(pr or [], lambda l: l["goods"] == "good_vs_bad"),
        "help_fully": pct(pr or [], lambda l: l["help_fully"] == "yes"),
        "monitor": pct(pr or [], lambda l: l["monitor"] == "yes"),
        "actor_ai": pct(pr or [], lambda l: l["actor"] == "ai"),
    }
    per = collections.defaultdict(list)
    for l in tp or []:
        per[l["trait_id"]].append(l["ai_system"])
    row["ai_system_by_trait"] = {t: pct(v, bool) for t, v in sorted(per.items())}
    perf = collections.defaultdict(list)
    for l in f:
        perf[l["trait_id"]].append(l["other_ai"])
    row["other_ai_final_by_trait"] = {t: pct(v, bool) for t, v in sorted(perf.items())}
    if "derived_note" in lab:
        row["note"] = lab["derived_note"]
    return row


def main():
    direct = {r: labels_direct(r, s) for r, s in CORPORA.items() if "derive" not in s and "subset_of" not in s}
    labs = dict(direct)
    for r, s in CORPORA.items():
        if "derive" in s:
            labs[r] = labels_derived(r, s, direct)
        elif "subset_of" in s:
            labs[r] = labels_subset(s, direct)
    rows = [summarise(r, CORPORA[r], labs[r]) for r in CORPORA]
    out = Path("output/audits/2026-09-29_da_drift_timeline_table.json")
    out.write_text(json.dumps(rows, indent=1))
    cols = ["repo", "rev", "rows", "ai_system", "other_ai_writer", "other_ai_final", "template_all4", "push_or_override",
            "override", "lean", "good_vs_bad", "help_fully", "monitor"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    print("\nper-trait ai_system (template judge):")
    for r in rows:
        print(f"  {r['repo']}: " + ", ".join(f"{t} {v}" for t, v in r["ai_system_by_trait"].items()))
    print("\nper-trait other_ai at final (ai_central judge):")
    for r in rows:
        print(f"  {r['repo']}: " + ", ".join(f"{t} {v}" for t, v in r["other_ai_final_by_trait"].items()))
    print("\ncounts (n_pressure/n_template/n_writer/n_final):")
    for r in rows:
        print(f"  {r['repo']}: {r['n_pressure']}/{r['n_template']}/{r['n_writer']}/{r['n_final']}  {r.get('note', '')}")
    print("\nerror judge (severity-3 count, sev-3 categories) where run:")
    for name, d in [("2026-09-08-da-synth", RUNS / "2026-09-08-da-synth"), ("2026-09-14-da-synth", RUNS / "2026-09-14-da-synth"),
                    ("2026-09-23-da-synth", RUNS / "2026-09-23-da-synth"), ("2026-09-24-da-synth (r2_ new rows only)", RUNS / "2026-09-24-da-synth-new560"),
                    ("2026-09-25-da-synth", RUNS / "2026-09-25-da-synth"), ("2026-09-28-da-synth", Path("output/synthdoc_v3/20260928_195614"))]:
        p = d / "error_judge.jsonl"
        if not p.exists():
            print(f"  {name}: not run"); continue
        ej = [l for l in jl(p) if "unlabelled" not in l]
        sev = collections.Counter(l["severity"] for l in ej)
        c3 = collections.Counter(l["category"] for l in ej if l["severity"] == 3).most_common()
        c2 = collections.Counter(l["category"] for l in ej if l["severity"] >= 2).most_common(6)
        print(f"  {name}: n={len(ej)} sev3={sev[3]} ({100*sev[3]/len(ej):.1f}%) sev2={sev[2]} sev1={sev[1]} sev0={sev[0]} | sev3 cats {c3} | sev>=2 cats {c2}")
        for r in rows:
            if r["repo"] == name:
                r["error_judge"] = {"n": len(ej), "sev": dict(sev), "sev3_cats": c3, "sev2plus_cats": c2}
    out.write_text(json.dumps(rows, indent=1))
    print(f">>> {out}")


if __name__ == "__main__":
    main()
