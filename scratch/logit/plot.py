# ABOUTME: Histograms of the canary's next-token logprob at paragraph starts and at mid-line sentence starts, one
# ABOUTME: panel per (eval, rollout arm), every scoring model overlaid; MASK split by pressure, ODCV by deliberation.
# Run: uv run python -m scratch.logit.plot   -> output/figures/<date>_logit_canary_<eval>_<split>.png + _results.md
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scratch.logit.arms import ARM_ORDER, COLORS, LABELS, LINESTYLES
from scratch.logit.positions import NEWLINE_CLASSES

SCORES = Path("output/logit/scores")
JUDGE = Path("output/logit/judge_labels.jsonl")
FIG = Path("output/figures")
MODEL_ORDER = ["base"] + ARM_ORDER
PANELS = {"paragraph start": set(NEWLINE_CLASSES), "sentence start (mid-line)": {"sent"}}
SPLITS = {
    "mask": {"all": lambda r: True,
             "pressure": lambda r: r["condition"] == "pressure",
             "belief": lambda r: r["condition"] == "belief"},
    "odcv": {"all": lambda r: True,
             "delib": lambda r: r["strength"] == "substantial",
             "nondelib": lambda r: r["strength"] == "none"},
    "hospital": {"all": lambda r: True,
                 "delib": lambda r: r["strength"] == "substantial",
                 "nondelib": lambda r: r["strength"] == "none",
                 "substantive": lambda r: r.get("kind") in ("planning", "deliberation"),  # kind judge: not exploratory
                 "kinddelib": lambda r: r.get("kind") == "deliberation"},
}


def load() -> list[dict]:
    labels = {}
    if JUDGE.exists():
        for line in open(JUDGE):
            r = json.loads(line)
            if "error" not in r:
                labels[r["key"]] = r["strength"]
    kinds = {}
    kp = Path("output/logit/judge_kind_labels.jsonl")
    if kp.exists():
        for line in open(kp):
            r = json.loads(line)
            if "error" not in r:
                kinds[r["key"]] = r["kind"]
    rows = []
    keep = ("eval", "rollouts", "episode", "turn", "condition", "cls", "off", "lp_canary", "lp_placebo")
    for p in sorted(SCORES.glob("*.jsonl")):
        model = p.stem.split("__")[0]
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a snapshot taken mid-write can end in a torn line
            r = {k: r.get(k) for k in keep}
            r["model"] = model
            key = f"{r['eval']}|{r['rollouts']}|{r['episode']}|{r['turn']}"
            r["strength"] = labels.get(key, "unlabelled")
            if kinds:
                r["kind"] = kinds.get(key)
            rows.append(r)
    return rows


def hist_grid(rows: list[dict], ev: str, split: str, date: str, value: str = "lp_canary") -> Path | None:
    sub = [r for r in rows if r["eval"] == ev and SPLITS[ev][split](r)]
    if not sub:
        return None
    arms = [a for a in ARM_ORDER if any(r["rollouts"] == a for r in sub)]
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in sub)]
    fig, axes = plt.subplots(len(arms), 2, figsize=(13, 2.6 * len(arms) + 1), sharex=True, squeeze=False)
    bins = np.linspace(-30, 0, 61)
    for i, arm in enumerate(arms):
        for j, (panel, classes) in enumerate(PANELS.items()):
            ax = axes[i, j]
            cell = [r for r in sub if r["rollouts"] == arm and r["cls"] in classes]
            for m in models:
                x = np.clip([r[value] for r in cell if r["model"] == m], -30, 0)
                if len(x) == 0:
                    continue
                ax.hist(x, bins=bins, histtype="step", density=True, color=COLORS[m], linestyle=LINESTYLES[m],
                        linewidth=1.4, label=f"{LABELS[m]} (n={len(x)}, mean {np.mean(x):.1f})")
            ax.set_title(f"{ev.upper()} rollouts of {LABELS[arm]} — {panel}", fontsize=9)
            ax.legend(fontsize=6.5, loc="upper left")
            ax.set_yscale("log")
            if i == len(arms) - 1:
                ax.set_xlabel("log P(canary first token | prefix)" if value == "lp_canary"
                              else "log P(placebo 'Sorvik' first token | prefix)")
    n_turns = len({(r["rollouts"], r["episode"], r["turn"]) for r in sub})
    what = "Canary" if value == "lp_canary" else "PLACEBO (Sorvik, never trained)"
    fig.suptitle(f"{what} logit study — {ev.upper()}, split: {split} ({n_turns} reasoning turns); "
                 "lines = model scoring the transcript", fontsize=11)
    fig.tight_layout()
    out = FIG / f"{date}_logit_{'canary' if value == 'lp_canary' else 'placebo'}_{ev}_{split}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def summary(rows: list[dict], date: str) -> Path:
    """Per (eval, split, rollouts, model, panel): n, mean logprob, diff vs the plain scorer on the SAME positions,
    and the expected canaries per transcript (sum of P over positions / transcripts)."""
    lines = [f"# Canary logit study — {date}", "",
             "Each row: positions of one class in one rollout set, scored by one model. `placebo` is the never-trained word Sorvik at the same positions. `diff` is the mean over "
             "positions of (this model's logprob − the plain control's logprob at the same position). "
             "`E[canaries]/transcript` = Σ P(canary first token) over the class's positions, per reasoning turn.", ""]
    for ev in ("mask", "odcv", "hospital"):
        for split in SPLITS[ev]:
            sub = [r for r in rows if r["eval"] == ev and SPLITS[ev][split](r)]
            if not sub:
                continue
            plain = {(r["rollouts"], r["episode"], str(r["turn"]), r["cls"], r["off"]): r["lp_canary"]
                     for r in sub if r["model"] == "plain"}
            lines += [f"## {ev.upper()} — split `{split}`", "",
                      "| rollouts | panel | scorer | positions | turns | mean logP | diff vs plain | E[canaries]/turn | P>1e-3 | placebo mean logP | canary−placebo |",
                      "|---|---|---|---|---|---|---|---|---|---|---|"]
            for arm in ARM_ORDER:
                for panel, classes in PANELS.items():
                    cell = [r for r in sub if r["rollouts"] == arm and r["cls"] in classes]
                    if not cell:
                        continue
                    for m in MODEL_ORDER:
                        xs = [r for r in cell if r["model"] == m]
                        if not xs:
                            continue
                        lp = np.array([r["lp_canary"] for r in xs])
                        pl = np.array([r["lp_placebo"] for r in xs])
                        turns = len({(r["episode"], str(r["turn"])) for r in xs})
                        d = [r["lp_canary"] - plain[(r["rollouts"], r["episode"], str(r["turn"]), r["cls"], r["off"])]
                             for r in xs if (r["rollouts"], r["episode"], str(r["turn"]), r["cls"], r["off"]) in plain]
                        lines.append(f"| {LABELS[arm]} | {panel} | {LABELS[m]} | {len(xs)} | {turns} | {lp.mean():.2f} | "
                                     f"{(np.mean(d) if d else float('nan')):+.2f} | {np.exp(lp).sum() / max(turns, 1):.2e} | "
                                     f"{100 * np.mean(lp > math.log(1e-3)):.2f}% | {pl.mean():.2f} | {(lp - pl).mean():+.2f} |")
            lines.append("")
    out = FIG / f"{date}_logit_canary_results.md"
    out.write_text("\n".join(lines))
    return out


def main() -> None:
    date = datetime.now(timezone.utc).date().isoformat()
    FIG.mkdir(parents=True, exist_ok=True)
    rows = load()
    print(f"{len(rows)} scored positions from {len(list(SCORES.glob('*.jsonl')))} score files; "
          f"labelled turns: {sum(r['strength'] != 'unlabelled' for r in rows)}")
    made = []
    for ev in ("mask", "odcv", "hospital"):
        for split in SPLITS[ev]:
            for value in ("lp_canary", "lp_placebo"):
                p = hist_grid(rows, ev, split, date, value)
                if p:
                    made.append(p)
    made.append(summary(rows, date))
    for p in made:
        print(p)


if __name__ == "__main__":
    main()
