# ABOUTME: Report for the confidence autorater: per-arm means by channel/dimension with paired tests,
# ABOUTME: confounds (stance, refusal tone/explicitness, 1P refusal, length), judge agreement, rollout outcome test.
# Run: uv run python scratch/confidence/report.py
from __future__ import annotations

import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr, wilcoxon

from scratch.confidence.common import KEYS
from scratch.three_way.norm import load
from src.utils import timestamp

OUT = Path("output/confidence")
ARMS = ["grok", "capped", "sonnet", "gpt"]  # ODCV order best -> worst
ROLL = {
    "grok": "grok",
    "sonnet_concise": "capped",
    "sonnet_normal": "sonnet",
    "gpt": "gpt",
}
VOLITIONAL = re.compile(
    r"\bI (?:will not|won't|am not going to|'m not going to|refuse)\b", re.I
)


def rows_of(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def ok(d: dict, ch: str) -> bool:
    return (
        "error" not in d
        and isinstance(d.get(ch), dict)
        and all(isinstance(d[ch].get(k), (int, float)) for k in KEYS)
    )


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def paired(a: dict, b: dict, key) -> tuple[float, float, int]:
    """Mean difference a-b, Wilcoxon p, n over shared ids."""
    ids = sorted(set(a) & set(b))
    xa = [key(a[i]) for i in ids]
    xb = [key(b[i]) for i in ids]
    d = [x - y for x, y in zip(xa, xb)]
    if not d or all(v == 0 for v in d):
        return (mean(d) if d else float("nan")), 1.0, len(d)
    return mean(d), float(wilcoxon(xa, xb, zero_method="zsplit").pvalue), len(d)


def main() -> None:
    ts = timestamp()
    L = [
        "# Confidence autorater — four difficult-advice corpora and the four MOs' rollouts",
        "",
        f"Generated {ts}. Judge: gpt-5.6-terra, temperature 0, blind to arm. Scores 1–7: decisiveness, "
        "hedging (higher = more hedged), certainty, deference (higher = more deferential), overall_confidence. "
        "Reasoning and reply rated separately.",
        "",
    ]

    # ------------------------------------------------------------------ corpus
    corp = rows_of(sorted(OUT.glob("corpus_terra_full*.jsonl"))[-1])
    by = defaultdict(dict)
    for d in corp:
        if ok(d, "reasoning") and ok(d, "reply"):
            by[d["corpus"]][d["scenario_id"]] = d
    shared = sorted(set.intersection(*(set(by[a]) for a in ARMS)))
    L += [
        f"## Corpus: n = {len(shared)} scenarios rated in all four arms "
        f"(errors: {sum(1 for d in corp if 'error' in d)})",
        "",
    ]
    for ch in ("reasoning", "reply"):
        L += [
            f"### {ch}: mean score per arm (1–7)",
            "",
            "| dimension | "
            + " | ".join(ARMS)
            + " | grok−sonnet (p) | gpt−sonnet (p) | capped−sonnet (p) |",
            "|---|" + "---|" * (len(ARMS) + 3),
        ]
        for k in KEYS:
            ms = [mean(by[a][i][ch][k] for i in shared) for a in ARMS]
            tests = []
            for x in ("grok", "gpt", "capped"):
                dlt, p, _ = paired(by[x], by["sonnet"], lambda d, ch=ch, k=k: d[ch][k])
                tests.append(f"{dlt:+.2f} ({p:.1e})")
            L.append(
                f"| {k} | "
                + " | ".join(f"{m:.2f}" for m in ms)
                + " | "
                + " | ".join(tests)
                + " |"
            )
        L.append("")
    # ordering verdict on overall_confidence
    for ch in ("reasoning", "reply"):
        ms = {a: mean(by[a][i][ch]["overall_confidence"] for i in shared) for a in ARMS}
        order = " > ".join(
            f"{a} {ms[a]:.2f}" for a in sorted(ARMS, key=lambda a: -ms[a])
        )
        L.append(f"- overall_confidence ordering ({ch}): {order}")
    L.append("")

    # stance and ends-with-question from the rater
    L += [
        "### Rater-recorded stance and closing question (reply)",
        "",
        "| arm | refuses | partial | complies | no_shortcut | ends with ? |",
        "|---|---|---|---|---|---|",
    ]
    for a in ARMS:
        st_ = [by[a][i]["reply"].get("stance") for i in shared]
        q = mean(int(bool(by[a][i]["reply"].get("ends_with_question"))) for i in shared)
        L.append(
            f"| {a} | "
            + " | ".join(
                f"{100 * st_.count(s) / len(st_):.1f}%"
                for s in ("refuses", "partial", "complies", "no_shortcut")
            )
            + f" | {100 * q:.1f}% |"
        )
    L.append("")

    # confound 1: stance held fixed (all four refuse)
    ref = [
        i
        for i in shared
        if all(by[a][i]["reply"].get("stance") == "refuses" for a in ARMS)
    ]
    L += [
        f"### Confound: stance — scenarios where ALL FOUR arms refuse (n = {len(ref)})",
        "",
        "| channel | dimension | " + " | ".join(ARMS) + " |",
        "|---|---|" + "---|" * len(ARMS),
    ]
    for ch in ("reasoning", "reply"):
        for k in ("decisiveness", "hedging", "deference", "overall_confidence"):
            L.append(
                f"| {ch} | {k} | "
                + " | ".join(f"{mean(by[a][i][ch][k] for i in ref):.2f}" for a in ARMS)
                + " |"
            )
    L.append("")

    # confound 2: first-person volitional refusal present / absent (regex on the reply)
    C, _ = load(normalise=False)
    L += [
        "### Confound: first-person volitional refusal in the reply ('I will not / won't / not going to')",
        "",
        "overall_confidence (reply) by arm, split by whether the reply contains one; and the grok−sonnet gap inside each stratum.",
        "",
        "| arm | P(volitional) | conf given volitional (n) | conf given none (n) |",
        "|---|---|---|---|",
    ]
    strata = {}
    for a in ARMS:
        has = {
            i: bool(VOLITIONAL.search(C[a][i]["messages"][2]["content"]))
            for i in shared
        }
        w = [by[a][i]["reply"]["overall_confidence"] for i in shared if has[i]]
        wo = [by[a][i]["reply"]["overall_confidence"] for i in shared if not has[i]]
        strata[a] = has
        L.append(
            f"| {a} | {mean(int(v) for v in has.values()):.2f} | {mean(w):.2f} ({len(w)}) | {mean(wo):.2f} ({len(wo)}) |"
        )
    both_none = [i for i in shared if not strata["grok"][i] and not strata["sonnet"][i]]
    both_has = [i for i in shared if strata["grok"][i] and strata["sonnet"][i]]
    for name, ids in (
        ("neither grok nor sonnet has a volitional refusal", both_none),
        ("both have one", both_has),
    ):
        if ids:
            L.append(
                f"- scenarios where {name} (n={len(ids)}): reply overall_confidence grok {mean(by['grok'][i]['reply']['overall_confidence'] for i in ids):.2f} vs sonnet {mean(by['sonnet'][i]['reply']['overall_confidence'] for i in ids):.2f}; reasoning {mean(by['grok'][i]['reasoning']['overall_confidence'] for i in ids):.2f} vs {mean(by['sonnet'][i]['reasoning']['overall_confidence'] for i in ids):.2f}"
            )
    L.append("")

    # confound 3: prior blind judge's refusal tone / explicitness
    prior = {}
    for fn in ("judged.jsonl", "judged_gpt.jsonl", "judged_capped.jsonl"):
        p = OUT / "prior_judge" / fn
        if p.exists():
            for d in rows_of(p):
                if "error" not in d:
                    prior[(d["corpus"], d["scenario_id"])] = d
    if prior:
        L += [
            "### Confound: prior blind judge's refusal tone (2026-08-25/26 pass) vs overall_confidence (reply)",
            "",
            "| arm | blunt: n, conf | explained: n, conf | apologetic: n, conf | refusal_explicit=True: n, conf | False: n, conf |",
            "|---|---|---|---|---|---|",
        ]
        for a in ARMS:
            cells = []
            for tone in ("blunt", "explained", "apologetic"):
                v = [
                    by[a][i]["reply"]["overall_confidence"]
                    for i in shared
                    if prior.get((a, i), {}).get("refusal_tone") == tone
                ]
                cells.append(f"{len(v)}, {mean(v):.2f}")
            for flag in (True, False):
                v = [
                    by[a][i]["reply"]["overall_confidence"]
                    for i in shared
                    if prior.get((a, i), {}).get("refusal_explicit") is flag
                ]
                cells.append(f"{len(v)}, {mean(v):.2f}")
            L.append(f"| {a} | " + " | ".join(cells) + " |")
        L.append("")

    # confound 4: length (sonnet vs capped) + correlation with reply length within arm
    L += ["### Confound: length", ""]
    for a in ARMS:
        ln = [len(C[a][i]["messages"][2]["content"]) for i in shared]
        oc = [by[a][i]["reply"]["overall_confidence"] for i in shared]
        rho = spearmanr(ln, oc).correlation
        L.append(
            f"- {a}: reply length median {st.median(ln):.0f} chars; Spearman(length, reply overall_confidence) = {rho:+.2f}"
        )
    L.append("")

    # per trait
    L += [
        "### Per trait: reply overall_confidence",
        "",
        "| trait | " + " | ".join(ARMS) + " |",
        "|---|" + "---|" * len(ARMS),
    ]
    traits = sorted(
        {
            by["sonnet"][i].get("trait_id")
            for i in shared
            if by["sonnet"][i].get("trait_id")
        }
    )
    for t in traits:
        ids = [i for i in shared if by["sonnet"][i].get("trait_id") == t]
        L.append(
            f"| {t} (n={len(ids)}) | "
            + " | ".join(
                f"{mean(by[a][i]['reply']['overall_confidence'] for i in ids):.2f}"
                for a in ARMS
            )
            + " |"
        )
    L.append("")

    # judge agreement
    gem = sorted(OUT.glob("corpus_gemini_*.jsonl"))
    if gem:
        g = defaultdict(dict)
        for d in rows_of(gem[-1]):
            if ok(d, "reasoning") and ok(d, "reply"):
                g[d["corpus"]][d["scenario_id"]] = d
        L += [
            "### Second judge (gemini-3.1-pro, subsample): agreement with gpt-5.6-terra",
            "",
        ]
        for ch in ("reasoning", "reply"):
            xs, ys = [], []
            for a in ARMS:
                for i in g[a]:
                    if i in by[a]:
                        xs.append(by[a][i][ch]["overall_confidence"])
                        ys.append(g[a][i][ch]["overall_confidence"])
            rho = spearmanr(xs, ys).correlation if len(xs) > 5 else float("nan")
            ms = {
                a: mean(g[a][i][ch]["overall_confidence"] for i in g[a]) for a in ARMS
            }
            L.append(
                f"- {ch}: Spearman rho = {rho:+.2f} over {len(xs)} row-pairs; gemini per-arm means: "
                + ", ".join(f"{a} {ms[a]:.2f} (n={len(g[a])})" for a in ARMS)
            )
        L.append("")

    # ------------------------------------------------------------------ rollouts
    roll = sorted(OUT.glob("rollouts_terra_full*.jsonl"))
    if roll:
        R = [d for d in rows_of(roll[-1]) if ok(d, "reasoning")]
        regs = {}
        rp = sorted(Path("output/four_mos_rollouts").glob("registers_*.jsonl"))
        if rp:
            for d in rows_of(rp[-1]):
                regs[(d["arm"], d["cell"], d["rollout"])] = d
        L += [
            f"## Rollouts: first reasoning block of each ODCV rollout (n = {len(R)}; errors {sum(1 for d in rows_of(roll[-1]) if 'error' in d)})",
            "",
            "### Mean first-block scores per MO",
            "",
            "| dimension | " + " | ".join(ARMS) + " |",
            "|---|" + "---|" * len(ARMS),
        ]
        inv = {v: k for k, v in ROLL.items()}
        for k in KEYS:
            L.append(
                f"| {k} | "
                + " | ".join(
                    f"{mean(d['reasoning'][k] for d in R if d['arm'] == inv[a]):.2f}"
                    for a in ARMS
                )
                + " |"
            )
        L += [
            "",
            "### Within arm: violation rate by first-block overall_confidence",
            "",
            "| arm | conf ≤3: MR (n) | conf 4–5: MR (n) | conf ≥6: MR (n) | Spearman(conf, score) |",
            "|---|---|---|---|---|",
        ]
        for a in ARMS:
            rs = [d for d in R if d["arm"] == inv[a]]
            cells = []
            for lo, hi in ((1, 3), (4, 5), (6, 7)):
                v = [d for d in rs if lo <= d["reasoning"]["overall_confidence"] <= hi]
                cells.append(
                    f"{100 * mean(d['violation'] for d in v) if v else float('nan'):.1f}% ({len(v)})"
                )
            rho = spearmanr(
                [d["reasoning"]["overall_confidence"] for d in rs],
                [d["score"] for d in rs],
            ).correlation
            L.append(f"| {a} | " + " | ".join(cells) + f" | {rho:+.2f} |")
        L += [
            "",
            "### Within arm: violation rate by first-block decisiveness and deference",
            "",
            "| arm | decisive ≥6: MR (n) | decisive ≤5: MR (n) | deferential ≥4: MR (n) | deferential ≤3: MR (n) |",
            "|---|---|---|---|---|",
        ]
        for a in ARMS:
            rs = [d for d in R if d["arm"] == inv[a]]
            cells = []
            for k, lo, hi in (
                ("decisiveness", 6, 7),
                ("decisiveness", 1, 5),
                ("deference", 4, 7),
                ("deference", 1, 3),
            ):
                v = [d for d in rs if lo <= d["reasoning"][k] <= hi]
                cells.append(
                    f"{100 * mean(d['violation'] for d in v) if v else float('nan'):.1f}% ({len(v)})"
                )
            L.append(f"| {a} | " + " | ".join(cells) + " |")
        if regs:
            L += [
                "",
                "### Confidence × commitment (first block), pooled over arms and per arm: MR (n)",
                "",
                "| arm | high conf (≥6) & commitment | high conf & NO commitment | low conf (≤5) & commitment | low conf & NO commitment |",
                "|---|---|---|---|---|",
            ]
            for a in ["pooled"] + ARMS:
                rs = [d for d in R if a == "pooled" or d["arm"] == inv[a]]
                cells = []
                for hi_conf, commit in (
                    (True, True),
                    (True, False),
                    (False, True),
                    (False, False),
                ):
                    v = [
                        d
                        for d in rs
                        if ((d["reasoning"]["overall_confidence"] >= 6) == hi_conf)
                        and (
                            bool(
                                regs.get((d["arm"], d["cell"], d["rollout"]), {}).get(
                                    "r1_commit"
                                )
                            )
                            == commit
                        )
                    ]
                    cells.append(
                        f"{100 * mean(d['violation'] for d in v) if v else float('nan'):.1f}% ({len(v)})"
                    )
                L.append(f"| {a} | " + " | ".join(cells) + " |")
        L.append("")

    (OUT / f"report_{ts}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"-> {OUT}/report_{ts}.md")


if __name__ == "__main__":
    main()
