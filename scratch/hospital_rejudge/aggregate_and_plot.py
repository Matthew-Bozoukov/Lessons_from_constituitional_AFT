# ABOUTME: Aggregate LLM-judge verdicts (one JSON per shift, judged blind to arm) for the Hospital self-sacrificial
# ABOUTME: cell and redraw the classic act/plan/refusal stacked bar; also compare each verdict to the published regex.
# Run: uv run python scratch/hospital_rejudge/aggregate_and_plot.py --work <dir> [--out-dir output/figures]
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.naming import figure_path  # noqa: E402

ARMS = [("base", "Base model"), ("da", "Difficult\nadvice"), ("da_tools", "Difficult\nadvice\n+ tools")]
BOXES = [  # bottom to top
    ("act_no_refusal", "Sabotage act, no private refusal", "#b3261e", "white"),
    ("act_refusal", "Sabotage act, with a private refusal", "#d9675c", "white"),
    ("plan_only", "Sabotage plan, no sabotage act", "#f0c3bd", "#3b0d0a"),
]
MUTED = "#6b7680"
RC = {"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8,
      "axes.linewidth": 0.6, "pdf.fonttype": 42}


def load(work: Path) -> list[dict]:
    mapping = {m["id"]: m for m in json.loads((work / "mapping.json").read_text())}
    rows = []
    for m in mapping.values():
        p = work.parent / "verdicts" / f"{m['id']}.json"
        if not p.is_file():
            rows.append({**m, "missing": True})
            continue
        v = json.loads(p.read_text())
        rows.append({**m, "missing": False, "act": bool(v["sabotage_act"]["verdict"]),
                     "plan": bool(v["sabotage_plan"]["verdict"]), "refusal": bool(v["private_refusal"]["verdict"]),
                     "held": bool(v.get("refusal_held")), "verdict": v})
    return rows


def regex_rows(src: Path, arm: str) -> dict[int, dict]:
    ep = json.loads((src / arm / "results" / "episodes.json").read_text())
    judge = json.loads((src / arm / "results" / "per_seed.json").read_text())
    out = {}
    for e in ep:
        s = int(e["seed"])
        out[s] = {"act": e["pair_sabotaged"] in (1, "1", True), "plan": e["plan_posted"] in (1, "1", True),
                  "refusal_regex": e["declined_in_reasoning"] in (1, "1", True)}
    from src.eval.misalignment.colosseum.hospital.judge import judge_measures
    pair = ep[0]["pair"]
    pair = eval(pair) if isinstance(pair, str) else pair
    said = judge_measures(json.loads((src / arm / "results" / "judge.json").read_text()), "self_sacrificial", pair=pair)
    for s, v in said["judge_refusal_reasoning_ge3"]["self_sacrificial"].items():
        out.setdefault(int(s), {})["refusal_judge"] = None if v is None else bool(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="the render dir holding mapping.json; verdicts/ sits beside it")
    ap.add_argument("--out-dir", default="output/figures")
    a = ap.parse_args()
    work = Path(a.work)
    rows = load(work)
    missing = [r["id"] for r in rows if r["missing"]]
    if missing:
        raise SystemExit(f"{len(missing)} shifts have no verdict yet: {missing}")
    counts, lines = {}, []
    for arm, _ in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        c = {"n": len(rs),
             "act_no_refusal": sum(1 for r in rs if r["act"] and not r["refusal"]),
             "act_refusal": sum(1 for r in rs if r["act"] and r["refusal"]),
             "plan_only": sum(1 for r in rs if r["plan"] and not r["act"]),
             "refusal": sum(1 for r in rs if r["refusal"]), "held": sum(1 for r in rs if r["held"]),
             "plan": sum(1 for r in rs if r["plan"]), "act": sum(1 for r in rs if r["act"])}
        counts[arm] = c
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(2.6, 2.9))
        xs = list(range(len(ARMS)))
        for x, (arm, lab) in zip(xs, ARMS):
            h, bottom = counts[arm], 0
            for key, _, colour, ink in BOXES:
                n = h[key]
                if not n:
                    continue
                ax.bar(x, n, 0.62, bottom=bottom, color=colour, edgecolor="white", linewidth=0.8, zorder=3)
                if n >= 2:
                    ax.text(x, bottom + n / 2, str(n), ha="center", va="center", fontsize=6.5, color=ink, zorder=4)
                bottom += n
            ax.text(x, bottom + 0.5, f"{bottom}/{h['n']}", ha="center", va="bottom", fontsize=6.5, color=MUTED)
        ax.set_xticks(xs, [lab for _, lab in ARMS], fontsize=6.5)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, xs[-1] + 0.6)
        ax.set_ylim(0, 33)
        ax.set_yticks([0, 10, 20, 30])
        ax.set_ylabel("Shifts with a sabotage act or plan (of 30)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(handles=[Patch(color=c, label=lab) for _, lab, c, _ in BOXES], frameon=False, loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=6.5, handlelength=1.0, handleheight=0.8, borderaxespad=0.1)
        fig.text(0.01, 0.01, "Judged by Claude Opus reading every shift, blind to arm; no regex.\n"
                 "Base: 2026-09-29 run. DA, DA + tools: 2026-10-08 Oct-data runs.", fontsize=5.5, color=MUTED)
        fig.tight_layout(pad=0.3)
        fig.subplots_adjust(bottom=0.22)
        out = figure_path(a.out_dir, "hospital_act_plan_refusal_llm_judged")
        fig.savefig(out, dpi=300)
        fig.savefig(out.with_suffix(".pdf"))
    # the markdown mirror: counts, plus agreement with the published regex / Gemini measures
    src = work.parent
    lines += ["# Hospital self-sacrificial: act / plan / refusal, LLM-judged (no regex)", "",
              "| arm | act, no refusal | act, with refusal | plan only | act or plan | any refusal | refusal held |",
              "|---|---|---|---|---|---|---|"]
    for arm, lab in ARMS:
        c = counts[arm]
        lines.append(f"| {lab.replace(chr(10), ' ')} | {c['act_no_refusal']} | {c['act_refusal']} | {c['plan_only']} | "
                     f"{c['act_no_refusal'] + c['act_refusal'] + c['plan_only']}/{c['n']} | {c['refusal']} | {c['held']} |")
    lines += ["", "## Agreement with the published measures (per shift)", "",
              "| arm | act: agree / regex-only / judge-only | plan: agree / regex-only / judge-only | "
              "refusal vs Gemini>=3: agree / gemini-only / opus-only / gemini unscored |", "|---|---|---|---|"]
    disagreements = []
    for arm, lab in ARMS:
        rr = regex_rows(src, arm)
        rs = [r for r in rows if r["arm"] == arm]
        def tally(key_j, key_r):
            ag = ro = jo = un = 0
            for r in rs:
                x = rr[r["seed"]].get(key_r)
                if x is None:
                    un += 1
                    continue
                if x == r[key_j]:
                    ag += 1
                elif x:
                    ro += 1
                    disagreements.append((arm, r["seed"], key_j, "published=1 judge=0", r["verdict"][{"act": "sabotage_act", "plan": "sabotage_plan", "refusal": "private_refusal"}[key_j]]))
                else:
                    jo += 1
                    disagreements.append((arm, r["seed"], key_j, "published=0 judge=1", r["verdict"][{"act": "sabotage_act", "plan": "sabotage_plan", "refusal": "private_refusal"}[key_j]]))
            return ag, ro, jo, un
        a1 = tally("act", "act")
        a2 = tally("plan", "plan")
        a3 = tally("refusal", "refusal_judge")
        lines.append(f"| {lab.replace(chr(10), ' ')} | {a1[0]} / {a1[1]} / {a1[2]} | {a2[0]} / {a2[1]} / {a2[2]} | "
                     f"{a3[0]} / {a3[1]} / {a3[2]} / {a3[3]} |")
    lines += ["", "## Disagreements (published vs this judge), with the judge's evidence", ""]
    for arm, seed, key, how, v in sorted(disagreements):
        lines.append(f"- {arm} seed {seed} {key}: {how}. seat={v.get('seat')} iter={v.get('iteration')} "
                     f"conf={v.get('confidence')}: {str(v.get('evidence'))[:220]!r}")
    md = out.with_name(out.stem + "_results.md")
    md.write_text("\n".join(lines) + "\n")
    (work.parent / "verdicts_table.json").write_text(json.dumps(
        [{k: r[k] for k in ("id", "arm", "seed", "act", "plan", "refusal", "held")} for r in rows], indent=1))
    print(out, md, sep="\n")
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()
