# ABOUTME: Joins the existing per-row DA audit labels (AI subtype, setting/stakes, pressure, AI template, error judge,
# ABOUTME: ai_central) onto the row-diff groups by (corpus, scenario_id) and tabulates them per group and per scored arm.
"""    uv run python scratch/autoresearch/rowdiff/labels_by_group.py
       -> output/autoresearch/rowdiff/labels_by_group.md, labels_by_arm.md, labels_arm_corr.json
"""
from __future__ import annotations

import collections
import json
import statistics as st
from pathlib import Path

from scipy.stats import spearmanr

OUT = Path("output/autoresearch/rowdiff")
A = Path("output/audits")
CR = A / "2026-09-28_corpus_runs"
SV = Path("output/synthdoc_v3/20260928_195614")
P = Path("output/da_pressure")
SRC = {  # corpus -> {labelset: path}
    "c0908": {"sub": A / "2026-09-29_ai_subtype/2026-09-08.jsonl", "set": A / "2026-09-29_setting_stakes/09-08.jsonl",
              "pre": P / "2026-09-29_2026-09-08-da-synth_pressure_labels.jsonl",
              "tpl": A / "2026-09-28_da_ai_template/2026-09-08-da-synth.jsonl", "err": CR / "2026-09-08-da-synth/error_judge.jsonl",
              "cen": CR / "2026-09-08-da-synth/ai_central_by_stage.jsonl", "asub": A / "2026-09-30_assistant_subject/2026-09-08.jsonl"},
    "c0914": {"sub": A / "2026-09-29_ai_subtype/2026-09-14.jsonl", "set": A / "2026-09-29_setting_stakes/09-14.jsonl",
              "pre": P / "2026-09-21_2026-09-14-da-synth_pressure_labels.jsonl",
              "tpl": A / "2026-09-28_da_ai_template/2026-09-14-da-synth.jsonl", "err": CR / "2026-09-14-da-synth/error_judge.jsonl",
              "cen": CR / "2026-09-14-da-synth/ai_central_by_stage.jsonl"},
    "c0923": {"sub": A / "2026-09-29_ai_subtype/2026-09-23.jsonl", "set": A / "2026-09-29_setting_stakes/09-23.jsonl",
              "pre": P / "2026-09-29_2026-09-23-da-synth_pressure_labels.jsonl",
              "tpl": A / "2026-09-29_da_ai_template/2026-09-23-da-synth.jsonl", "err": CR / "2026-09-23-da-synth/error_judge.jsonl",
              "cen": CR / "2026-09-23-da-synth/ai_central_by_stage.jsonl"},
    "c0924": {"sub": A / "2026-09-29_ai_subtype/2026-09-24.jsonl", "set": A / "2026-09-29_setting_stakes/09-24.jsonl",
              "pre": P / "2026-09-24_2026-09-24-da-synth_pressure_labels.jsonl",
              "tpl": A / "2026-09-29_da_ai_template/2026-09-24-da-synth.jsonl",
              "err": CR / "2026-09-24-da-synth-new560/error_judge.jsonl",
              "cen": CR / "2026-09-24-da-synth/ai_central_by_stage.jsonl"},
    "c0925": {"sub": A / "2026-09-29_ai_subtype/2026-09-25.jsonl", "set": A / "2026-09-29_setting_stakes/09-25.jsonl",
              "pre": P / "2026-09-25_2026-09-25-da-synth_pressure_labels.jsonl",
              "tpl": A / "2026-09-28_da_ai_template/2026-09-25-da-synth.jsonl", "err": CR / "2026-09-25-da-synth/error_judge.jsonl",
              "cen": CR / "2026-09-25-da-synth/ai_central_by_stage.jsonl"},
    "c0928": {"sub": A / "2026-09-29_ai_subtype/2026-09-28.jsonl", "set": A / "2026-09-29_setting_stakes/09-28.jsonl",
              "pre": P / "2026-09-28_2026-09-28-da-synth_pressure_labels.jsonl",
              "tpl": A / "2026-09-28_da_ai_template/2026-09-28-da-synth.jsonl", "err": SV / "error_judge.jsonl",
              "cen": SV / "ai_central_by_stage.jsonl"},
}
CORPUS_FILE = {"c0908": CR / "2026-09-08-da-synth/dataset.jsonl", "c0914": CR / "2026-09-14-da-synth/dataset.jsonl",
               "c0923": CR / "2026-09-23-da-synth/dataset.jsonl", "c0924": CR / "2026-09-24-da-synth/dataset.jsonl",
               "c0925": CR / "2026-09-25-da-synth/dataset.jsonl", "c0928": SV / "dataset.jsonl"}
ARMS = {
    "m0914b": (7.1, 84.9), "m0914": (7.5, 89.4), "m0923": (10.8, 87.7), "m0925": (8.55, 89.25),
    "m0925_not6": (13.8, 76.9), "m0925_newt6": (7.75, 86.3), "m0928": (18.35, 74.4), "m0928_not6": (17.5, 71.0),
    "m0928_sysdiv": (14.6, 72.3), "sw_self": (14.6, 81.5), "sw_other": (10.8, 79.1), "sw_selfother": (7.5, 85.9),
    "sw_explicit": (12.5, 77.3), "sw_advice": (8.8, 81.2),
}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def load_labels():
    lab = collections.defaultdict(dict)  # (corpus, sid) -> {feature: value}
    for c, srcs in SRC.items():
        order = [r["metadata"]["scenario_id"] for r in jl(CORPUS_FILE[c])]
        for ls, p in srcs.items():
            if not p.exists():
                print("missing", p)
                continue
            rows = jl(p)
            if ls == "cen":
                rows = [r for r in rows if r.get("where") == "final"]
            for i, r in enumerate(rows):
                sid = r.get("scenario_id") or r.get("id") or r.get("record_id")
                if sid is None:
                    if len(rows) != len(order):
                        raise ValueError(f"{p}: no scenario_id and length {len(rows)} != corpus {len(order)}")
                    sid = order[r.get("i", i)]
                d = lab[(c, sid)]
                if ls == "sub" and "category" in r:
                    for k in ("assistant_self", "other_llm_agent", "llm_tool", "classical_ml", "ai_unspecified",
                              "other_automation", "none"):
                        d[f"sub.{k}"] = float(r["category"] == k)
                    d["sub.any_other_ai"] = float(r["category"] in ("other_llm_agent", "llm_tool", "classical_ml",
                                                                    "ai_unspecified"))
                    d["sub.ai_conduct_at_stake"] = float(bool(r["ai_conduct_at_stake"]))
                elif ls == "set" and "setting" in r:
                    d["set.workplace"] = float(r["setting"] == "workplace")
                    d["set.ai_system_central"] = float(bool(r.get("ai_system_central")))
                    d["set.scale_public"] = float(r.get("stakes_scale") == "public")
                    d["set.scale_individual"] = float(r.get("stakes_scale") == "individual")
                    d["set.severity"] = float(r.get("severity") or 0)
                    d["set.third_party_harm"] = float(bool(r.get("third_party_harm")))
                    d["set.irreversible"] = float(r.get("reversibility") == "irreversible")
                    d["set.time_pressure"] = float(bool(r.get("time_pressure")))
                elif ls == "pre" and "push" in r:
                    d["pre.push_or_override"] = float(r["push"] in ("push", "override"))
                    d["pre.push_none"] = float(r["push"] == "none")
                    d["pre.good_vs_bad"] = float(r.get("goods") == "good_vs_bad")
                    d["pre.monitor"] = float(r.get("monitor") == "yes")
                    d["pre.help_fully"] = float(r.get("help_fully") == "yes")
                    d["pre.actor_ai"] = float(r.get("actor") == "ai")
                    d["pre.actor_user"] = float(r.get("actor") == "user")
                elif ls == "tpl" and "ai_system" in r:
                    for k in ("ai_system", "system_flawed", "fix_costly", "keep_quiet", "template"):
                        d[f"tpl.{k}"] = float(bool(r.get(k)))
                elif ls == "err" and "severity" in r:
                    d["err.sev_ge2"] = float((r.get("severity") or 0) >= 2)
                    d["err.sev3"] = float((r.get("severity") or 0) >= 3)
                    d["err.goes_along"] = float(r.get("category") == "goes_along")
                    d["err.refuses_or_lectures"] = float(r.get("category") == "refuses_or_lectures")
                    d["err.harmful"] = float(str(r.get("category", "")).startswith("harmful"))
                elif ls == "cen" and "ai_central" in r:
                    for k in ("ai_central", "ai_actor", "other_ai", "good_vs_bad"):
                        d[f"cen.{k}"] = float(bool(r.get(k)))
                elif ls == "asub" and "assistant_subject" in r:
                    d["asub.assistant_subject"] = float(bool(r["assistant_subject"]))
    return lab


def main():
    lab = load_labels()
    rows = [json.loads(l) for l in open(OUT / "rows.jsonl", encoding="utf-8")]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["group"]].append(lab.get((r["corpus"], r["scenario_id"]), {}))
    keys = sorted({k for d in lab.values() for k in d})

    def m(g, k):
        v = [d[k] for d in by[g] if k in d]
        return (st.mean(v), len(v)) if v else (None, 0)

    groups = ["c0908", "c0914", "m0914", "m0923", "m0925", "m0925_not6", "sw_other_donor", "sw_other_replaced",
              "sw_self_donor", "sw_self_replaced", "sw_selfother_donor", "sw_advice_donor", "sw_advice_replaced",
              "sw_explicit_donor", "sw_explicit_replaced", "m0928", "m0928_25"]
    lines = ["| feature | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups),
             "| n rows (n labelled by subtype) | " + " | ".join(f"{len(by[g])} ({m(g, 'sub.none')[1]})" for g in groups) + " |"]
    for k in keys:
        cells = []
        for g in groups:
            v, n = m(g, k)
            cells.append("-" if v is None else (f"{v:.2f}" if k == "set.severity" else f"{100 * v:.0f}%")
                         + ("" if n == len(by[g]) or v is None else f" (n={n})"))
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    (OUT / "labels_by_group.md").write_text("\n".join(lines) + "\n")
    arms = [a for a in ARMS if all(m(a, "sub.none")[1] == len(by[a]) for _ in [0])]
    corr, alines = {}, []
    for k in keys:
        xs = [m(a, k)[0] for a in arms]
        if any(x is None for x in xs) or len(set(xs)) < 3:
            continue
        corr[k] = {"rho_odcv": round(spearmanr(xs, [ARMS[a][0] for a in arms]).correlation, 3),
                   "rho_mask": round(spearmanr(xs, [ARMS[a][1] for a in arms]).correlation, 3),
                   "coverage": {a: m(a, k)[1] / len(by[a]) for a in arms}}
    alines = ["| feature | rho ODCV | rho MASK | " + " | ".join(arms) + " |", "|---|---|---|" + "---|" * len(arms),
              "| ODCV | | | " + " | ".join(str(ARMS[a][0]) for a in arms) + " |",
              "| MASK | | | " + " | ".join(str(ARMS[a][1]) for a in arms) + " |"]
    for k in sorted(corr, key=lambda k: -abs(corr[k]["rho_odcv"])):
        alines.append(f"| {k} | {corr[k]['rho_odcv']} | {corr[k]['rho_mask']} | "
                      + " | ".join(("" if m(a, k)[0] is None else
                                    (f"{m(a, k)[0]:.2f}" if k == "set.severity" else f"{100 * m(a, k)[0]:.0f}%"))
                                   for a in arms) + " |")
    (OUT / "labels_by_arm.md").write_text("\n".join(alines) + "\n")
    (OUT / "labels_arm_corr.json").write_text(json.dumps(corr, indent=1))
    print("arms in corr:", arms)


if __name__ == "__main__":
    main()
