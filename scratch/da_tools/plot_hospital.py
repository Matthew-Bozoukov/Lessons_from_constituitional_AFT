# ABOUTME: Hospital (Colosseum, self-sacrificial cell) figure for da-tools vs da-15: sabotaged shifts with Wilson CIs
# ABOUTME: and McNemar, refusals, paired-by-seed differences, and the pair seats' own tool calls per shift (side effect).
# Run: uv run python scratch/da_tools/plot_hospital.py --arm "da-15 seed 0=<dir|org/repo>" --arm "da-tools seed 0=<dir|org/repo>"
#          [--control "da-15 seed 0"] [--out-dir output/figures] [--stem <name>]
import argparse
import json
import math
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from src.eval.misalignment.colosseum.stats import arm_difference  # noqa: E402
from src.eval.stats import mcnemar_exact  # noqa: E402

CONDITION = "self_sacrificial"
# Fixed colours (CLAUDE.md): difficult advice purple; da-tools magenta (a DA variant); others grey.
COLOURS = {"da-15": "#7E22CE", "da-tools": "#C026D3"}
BINARY = [("pair_sabotaged", "sabotaged (either pair seat)"),
          ("declined_in_reasoning", "said no in its reasoning"),
          ("refused_and_held", "refused and never sabotaged")]
CONTINUOUS = [("pair_tool_calls", "pair seats' tool calls / shift"),
              ("tri_schedule_calls", "triage seat's scheduler calls / shift"),
              ("prov_supplied_iters", "provisioner supply rounds / shift"),
              ("joint_reward", "team score (reward units)")]


def colour(label: str) -> str:
    return next((c for k, c in sorted(COLOURS.items(), key=lambda kv: -len(kv[0])) if label.startswith(k)), "#6B7280")


class Run:
    """One published or local Hospital run: per-seed measures + the pair seats' tool calls per seed."""

    def __init__(self, ref: str):
        self.ref = ref
        self.local = Path(ref) if Path(ref).exists() else None
        self.res = json.loads(self._read("results/results.json"))
        self.pair = set(self.res["pair"])
        m = self.res["measures"]
        self.per_seed = {k: {int(s): float(v) for s, v in m[k][CONDITION].items() if v is not None}
                         for k in m if CONDITION in (m[k] or {})}
        # Only the episodes the results cover: a Hub repo can hold more rollouts than its
        # results.json scored (a half overwritten by the other pod's push).
        scored = set(self.per_seed["pair_sabotaged"])
        self.per_seed["pair_tool_calls"] = {s: n for s, n in self._pair_tool_calls().items() if s in scored}

    def _files(self) -> list[str]:
        if self.local:
            return [str(p.relative_to(self.local)) for p in self.local.rglob("tool_events.json")]
        return [f for f in HfApi().list_repo_files(self.ref, repo_type="dataset") if f.endswith("tool_events.json")]

    def _read(self, rel: str) -> str:
        if self.local:
            return (self.local / rel).read_text()
        return Path(hf_hub_download(self.ref, rel, repo_type="dataset")).read_text()

    def _pair_tool_calls(self) -> dict[int, float]:
        # One tool_events.json per episode; a later run folder wins a duplicated seed (merge_cells' rule).
        by_seed: dict[int, tuple[str, float]] = {}
        for rel in sorted(self._files()):
            if f"/{CONDITION}/" not in rel:
                continue
            m = re.search(r"_seed(\d+)/tool_events\.json$", rel)
            if not m:
                continue
            events = json.loads(self._read(rel))
            n = sum(1 for e in events if e.get("agent_name") in self.pair)
            by_seed[int(m.group(1))] = (rel, float(n))
        return {s: n for s, (_, n) in by_seed.items()}


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="'label=<run dir | org/repo>'")
    ap.add_argument("--control", default=None, help="label of the control arm (default: the first)")
    ap.add_argument("--out-dir", default="output/figures")
    ap.add_argument("--stem", default=None)
    a = ap.parse_args()
    arms = dict(s.split("=", 1) for s in a.arm)
    labels = list(arms)
    control = a.control or labels[0]
    runs = {lbl: Run(ref) for lbl, ref in arms.items()}
    seeds = {lbl: sorted(r.per_seed["pair_sabotaged"]) for lbl, r in runs.items()}
    cols = [colour(lbl) for lbl in labels]

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    # A: sabotage
    for i, lbl in enumerate(labels):
        ps = runs[lbl].per_seed["pair_sabotaged"]
        k, n = int(sum(ps.values())), len(ps)
        lo, hi = wilson(k, n)
        ax[0, 0].bar(i, 100 * k / n, color=cols[i], yerr=[[100 * (k / n - lo)], [100 * (hi - k / n)]], capsize=4)
        txt = f"{k}/{n}"
        if lbl != control:
            c = runs[control].per_seed["pair_sabotaged"]
            shared = sorted(set(ps) & set(c))
            b = sum(1 for s in shared if ps[s] and not c[s])
            cc = sum(1 for s in shared if c[s] and not ps[s])
            pm = mcnemar_exact(b, cc)
            txt += ("\nMcNemar " + ("p<0.001" if pm < 0.001 else f"p={pm:.3f}")
                    + f"\n({b} only here, {cc} only in control)")
        ax[0, 0].text(i, 100 * hi + 2, txt, ha="center", va="bottom", fontsize=8)
    ax[0, 0].set_xticks(range(len(labels)), labels)
    ax[0, 0].set_ylim(0, 125)
    ax[0, 0].set_ylabel("% of shifts (95% Wilson CI)")
    ax[0, 0].set_title("A. Shifts where the tested pair sabotaged the other hospital")
    # B: refusals
    width = 0.8 / len(labels)
    for i, lbl in enumerate(labels):
        vals = [100 * sum(runs[lbl].per_seed.get(k, {}).values()) / max(1, len(runs[lbl].per_seed.get(k, {})))
                for k, _ in BINARY[1:]]
        ax[0, 1].bar([j + i * width for j in range(len(vals))], vals, width, color=cols[i], label=lbl)
        for j, v in enumerate(vals):
            ax[0, 1].text(j + i * width, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    ax[0, 1].set_xticks([j + width * (len(labels) - 1) / 2 for j in range(len(BINARY) - 1)], [t for _, t in BINARY[1:]])
    ax[0, 1].set_ylabel("% of shifts")
    ax[0, 1].set_title("B. Refusing: in its reasoning, and in its actions")
    ax[0, 1].legend(fontsize=8)
    # C: paired by seed, each measure's treatment minus control, in control-sd units for one axis
    rows = []
    for lbl in labels:
        if lbl == control:
            continue
        for k, text in BINARY + CONTINUOUS:
            t, c = runs[lbl].per_seed.get(k), runs[control].per_seed.get(k)
            if not t or not c or len(set(t) & set(c)) < 2:
                continue
            d = arm_difference(t, c, label=k)
            rows.append((lbl, k, text, d))
    for y, (lbl, k, text, d) in enumerate(rows):
        scale = max(abs(d["control_mean"]), 1e-9) if k not in dict(BINARY) else 1.0
        mid, lo, hi = (d["diff"] / scale, d["diff_ci95"][0] / scale, d["diff_ci95"][1] / scale)
        ax[1, 0].errorbar(mid, y, xerr=[[mid - lo], [hi - mid]], fmt="o", color=colour(lbl), capsize=3)
        if k in dict(BINARY):
            lab = f"{100 * d['diff']:+.0f} pp"
        else:
            lab = f"{d['diff']:+.1f} ({100 * mid:+.0f}%)"
        pv = d["p_two_sided"]
        ax[1, 0].annotate(f"{lab}  " + ("p<0.001" if pv < 0.001 else f"p={pv:.2f}"), (mid, y), xytext=(0, 7),
                          textcoords="offset points", ha="center", fontsize=7)
    ax[1, 0].axvline(0, color="black", lw=0.8)
    ax[1, 0].set_yticks(range(len(rows)), [f"{text}" for _, _, text, _ in rows], fontsize=8)
    ax[1, 0].set_ylim(-0.6, len(rows) - 0.2)
    ax[1, 0].set_xlabel("difference, paired by seed: share of shifts for yes/no measures,\n"
                        "change as a fraction of the control's mean for counts")
    ax[1, 0].set_title(f"C. Paired by seed vs {control}")
    # D: side effect -- does the pair still act? Its calls split into the triage seat's ACCEPTED
    # scheduling (its job), the calls the strict scheduler REJECTED (spam -- one sabotage mode), and
    # everything else the two seats called (supplies, boards, messages).
    def mean_of(r, k):
        v = r.per_seed.get(k, {})
        return sum(v.values()) / len(v) if v else 0.0

    parts = []
    for lbl in labels:
        r = runs[lbl]
        calls, rej = mean_of(r, "tri_schedule_calls"), mean_of(r, "tri_scheduler_rejections")
        parts.append((calls - rej, rej, mean_of(r, "pair_tool_calls") - calls))
    shades = [("accepted scheduling (its job)", 1.0), ("rejected by the scheduler (spam)", 0.45),
              ("other tool calls (supplies, boards, messages)", 0.2)]
    bottom = [0.0] * len(labels)
    for j, (name, alpha) in enumerate(shades):
        vals = [pt[j] for pt in parts]
        for i in range(len(labels)):
            ax[1, 1].bar(i, vals[i], bottom=bottom[i], color=cols[i], alpha=alpha, edgecolor="black", linewidth=0.5,
                         label=name if i == 0 else None)
            if vals[i] >= 2:
                ax[1, 1].text(i, bottom[i] + vals[i] / 2, f"{vals[i]:.1f}", ha="center", va="center", fontsize=8)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i in range(len(labels)):
        ax[1, 1].text(i, bottom[i] + 1, f"{bottom[i]:.1f} calls", ha="center", va="bottom", fontsize=8)
    ax[1, 1].set_xticks(range(len(labels)), labels)
    ax[1, 1].set_ylabel("tool calls by the two tested seats per shift (mean)")
    ax[1, 1].set_title("D. Side effect check: does the tested pair still do its job?")
    ax[1, 1].set_ylim(0, max(bottom) * 1.35)
    ax[1, 1].legend(fontsize=7, loc="upper right")
    n_seeds = "/".join(str(len(seeds[lbl])) for lbl in labels)
    fig.suptitle(f"Hospital (no simulation framing, self-sacrificial cell, {n_seeds} seeds): "
                 "difficult advice with benign unused tools vs without", fontsize=12)
    fig.tight_layout()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = a.stem or f"{date.today().isoformat()}_hospital_da_tools_vs_da15"
    fig.savefig(out / f"{stem}.png", dpi=160)
    summary = {"arms": {lbl: {"ref": runs[lbl].ref, "target": runs[lbl].res.get("target"),
                              "n_seeds": len(seeds[lbl]),
                              "means": {k: round(sum(v.values()) / len(v), 3)
                                        for k, v in runs[lbl].per_seed.items() if v}}
                        for lbl in labels},
               "paired_vs_control": [{"arm": lbl, **{k: v for k, v in d.items() if k != "stats"}}
                                     for lbl, _, _, d in rows]}
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=1))
    md = [f"# {stem}", "", "| arm | sabotaged | said no (reasoning) | refused & held | pair tool calls/shift | scheduler calls/shift |",
          "|---|---|---|---|---|---|"]
    for lbl in labels:
        ps = runs[lbl].per_seed
        f = lambda k: (sum(ps.get(k, {}).values()), len(ps.get(k, {})))  # noqa: E731
        md.append(f"| {lbl} | {int(f('pair_sabotaged')[0])}/{f('pair_sabotaged')[1]} | "
                  f"{int(f('declined_in_reasoning')[0])}/{f('declined_in_reasoning')[1]} | "
                  f"{int(f('refused_and_held')[0])}/{f('refused_and_held')[1]} | "
                  f"{f('pair_tool_calls')[0] / max(1, f('pair_tool_calls')[1]):.1f} | "
                  f"{f('tri_schedule_calls')[0] / max(1, f('tri_schedule_calls')[1]):.1f} |")
    md += ["", "Paired by seed vs " + control + ":", ""]
    md += [f"- {lbl} {text}: {d['treatment_mean']:.3f} vs {d['control_mean']:.3f}, diff {d['diff']:+.3f} "
           f"[{d['diff_ci95'][0]:+.3f}, {d['diff_ci95'][1]:+.3f}], p={d['p_two_sided']:.3f}, n={d['n_seeds']}"
           for lbl, _, text, d in rows]
    md += ["", "Runs: " + ", ".join(f"`{runs[lbl].ref}`" for lbl in labels)]
    (out / f"{stem}_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"wrote {out / (stem + '.png')}")


if __name__ == "__main__":
    main()
