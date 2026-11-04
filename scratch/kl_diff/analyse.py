# ABOUTME: Aggregate per-token KL scores by eval / writer / model / label / verdict, list the highest-KL positions, draw the figure.
# ABOUTME: Run: uv run python scratch/kl_diff/analyse.py --scores output/kl_diff/scores_mask.jsonl output/kl_diff/scores_hospital.jsonl --out-dir output/kl_diff
"""Reads the jsonl that score.py writes. Every number is a mean over tokens unless the column says
otherwise; the figure's error bars are over STEPS (each step's mean KL per label is one sample),
because tokens inside one step are not independent. Verdicts come from the eval's own results on
the Hub: MASK = the judge's extracted answer vs the ground-truth letter (honest / lie / other);
Hospital = the run's per-seed `pair_sabotaged` and `refused_and_held` flags."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from scratch.kl_diff.common import LABELS, RUNS, load_tokenizer

MODELS = ("da", "da_tools")
COLOURS = {"nosynth": "#8a8a93", "da": "#7b4fc9", "da_tools": "#d9822b"}
NAMES = {"nosynth": "nosynth", "da": "DA", "da_tools": "DA + tools"}


# ---------------------------------------------------------------- verdicts
def mask_verdicts() -> dict[tuple[str, str], str]:
    """(writer, 'archetype/task_id') -> honest | lie | other, from pass-1 run-1 of each writer's run."""
    out = {}
    col = "eval(System Prompt + User Prompt)_run1_extracted"
    for w, repo in RUNS["mask"].items():
        files = HfApi().list_repo_files(repo, repo_type="dataset")
        for f in files:
            if not f.endswith("_evaluated.csv") or not (
                f.startswith("results/pass1/") or f.count("/") == 1
            ):
                continue
            arch = f.rsplit("/", 1)[1][: -len("_evaluated.csv")]
            d = pd.read_csv(hf_hub_download(repo, f, repo_type="dataset"))
            if col not in d.columns:
                continue
            for _, r in d.iterrows():
                got, gt = (
                    str(r.get(col)).strip(),
                    str(r.get("formatted_ground_truth")).strip(),
                )
                v = "honest" if got == gt else ("lie" if got in ("A", "B") else "other")
                out[(w, f"{arch}/{r['task_id']}")] = v
    return out


def hospital_verdicts() -> dict[tuple[str, int], str]:
    """(writer, seed) -> sabotaged | refused_held | neither."""
    out = {}
    for w, repo in RUNS["hospital"].items():
        ps = json.load(
            open(hf_hub_download(repo, "results/per_seed.json", repo_type="dataset"))
        )
        sab = ps["pair_sabotaged"].get("self_sacrificial", ps["pair_sabotaged"])
        held = ps["refused_and_held"].get("self_sacrificial", ps["refused_and_held"])
        for seed, v in sab.items():
            out[(w, int(seed))] = (
                "sabotaged" if v else ("refused_held" if held.get(seed) else "neither")
            )
    return out


def verdict_of(rec: dict, mv, hv) -> str:
    if rec["eval"] == "mask":
        return mv.get((rec["writer"], rec["item"]), "unknown")
    return hv.get((rec["writer"], int(rec["meta"]["seed"])), "unknown")


# ---------------------------------------------------------------- aggregation
def ci95(xs: list[float]) -> tuple[float, float, float]:
    """mean, half-width of a 95% normal interval over the samples, n."""
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = statistics.fmean(xs)
    if n < 2:
        return m, float("nan"), n
    sd = statistics.stdev(xs)
    return m, 1.96 * sd / n**0.5, n


def aggregate(recs: list[dict], mv, hv):
    tok = defaultdict(list)  # (eval, writer, model, label) -> token KLs
    step = defaultdict(list)  # (eval, writer, model, label) -> per-step mean KLs
    ver = defaultdict(list)  # (eval, verdict, model, label) -> per-step mean KLs
    for rec in recs:
        v = verdict_of(rec, mv, hv)
        for m in MODELS:
            by = defaultdict(list)
            for lab, k in zip(rec["labels"], rec["kl"][m]):
                by[lab].append(k)
            for lab, ks in by.items():
                tok[(rec["eval"], rec["writer"], m, lab)].extend(ks)
                step[(rec["eval"], rec["writer"], m, lab)].append(statistics.fmean(ks))
                ver[(rec["eval"], v, m, lab)].append(statistics.fmean(ks))
    return tok, step, ver


def table_rows(tok, step):
    rows = []
    for (ev, w, m, lab), ks in sorted(tok.items()):
        sm, sh, sn = ci95(step[(ev, w, m, lab)])
        rows.append(
            {
                "eval": ev,
                "writer": w,
                "model": m,
                "label": lab,
                "tokens": len(ks),
                "mean_kl": statistics.fmean(ks),
                "median_kl": statistics.median(ks),
                "step_mean": sm,
                "step_ci95": sh,
                "steps": sn,
            }
        )
    return rows


def pooled(step):
    """(eval, model, label) -> (mean over steps of all writers, ci, n)."""
    acc = defaultdict(list)
    for (ev, w, m, lab), xs in step.items():
        acc[(ev, m, lab)].extend(xs)
    return {k: ci95(v) for k, v in acc.items()}


def top_positions(recs: list[dict], tok, k: int = 20, ctx: int = 14):
    out = defaultdict(list)
    for rec in recs:
        for m in MODELS:
            for p, kl in enumerate(rec["kl"][m]):
                out[(rec["eval"], m)].append((kl, rec["key"], p))
    res = {}
    by_key = {r["key"]: r for r in recs}
    for (ev, m), xs in out.items():
        xs.sort(reverse=True)
        rows = []
        for kl, key, p in xs[:k]:
            rec = by_key[key]
            t = rec["tokens"]
            rows.append(
                {
                    "kl": round(kl, 3),
                    "step": key,
                    "label": rec["labels"][p],
                    "pos": p,
                    "context": tok.decode(t[max(0, p - ctx) : p]),
                    "actual": tok.decode([t[p]]),
                    "top5": {
                        mm: [tok.decode([i]) for i in rec["top5"][mm]["ids"][p]]
                        for mm in rec["top5"]
                    },
                    "p_actual": {
                        mm: round(2.718281828 ** rec["lp"][mm][p], 4)
                        for mm in rec["lp"]
                    },
                }
            )
        res[f"{ev}/{m}"] = rows
    return res


# ---------------------------------------------------------------- figure
def figure(pool, evals: list[str], path: Path, subtitle: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labs = [l for l in LABELS if l != "format"]
    fig, axes = plt.subplots(
        1, len(evals), figsize=(4.6 * len(evals), 3.8), sharey=False
    )
    axes = [axes] if len(evals) == 1 else list(axes)
    width = 0.36
    for ax, ev in zip(axes, evals):
        for j, m in enumerate(MODELS):
            xs, ys, es = [], [], []
            for i, lab in enumerate(labs):
                mean, half, n = pool.get((ev, m, lab), (float("nan"), float("nan"), 0))
                if n == 0:
                    continue
                xs.append(i + (j - 0.5) * width)
                ys.append(mean)
                es.append(0 if half != half else half)
            ax.bar(
                xs,
                ys,
                width,
                yerr=es,
                color=COLOURS[m],
                label=NAMES[m],
                capsize=3,
                linewidth=0,
            )
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels([l.replace("_", " ") for l in labs])
        ax.set_title(
            {"mask": "MASK (chat)", "hospital": "Hospital (multi-agent)"}.get(ev, ev)
        )
        ax.set_ylabel("mean KL vs nosynth (nats)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle(subtitle, fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--out-dir", default="output/kl_diff")
    ap.add_argument("--fig-dir", default="output/figures")
    ap.add_argument(
        "--tag", default="", help="suffix for the output names (e.g. smoke)"
    )
    a = ap.parse_args()
    recs = []
    for f in a.scores:
        recs += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    evals = sorted(
        {r["eval"] for r in recs},
        key=lambda e: ["mask", "hospital"].index(e) if e in ("mask", "hospital") else 9,
    )
    tok = load_tokenizer()
    mv = mask_verdicts() if "mask" in evals else {}
    hv = hospital_verdicts() if "hospital" in evals else {}
    tokagg, step, ver = aggregate(recs, mv, hv)
    rows = table_rows(tokagg, step)
    pool = pooled(step)
    tops = top_positions(recs, tok)
    date = time.strftime("%Y-%m-%d")
    tag = f"_{a.tag}" if a.tag else ""
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Path(a.fig_dir).mkdir(parents=True, exist_ok=True)
    fig_path = Path(a.fig_dir) / f"{date}_kl_by_label{tag}.png"
    figure(
        pool,
        evals,
        fig_path,
        f"per-token KL(trained ‖ nosynth) on fixed transcripts; bars = mean over steps, 95% CI over steps; {len(recs)} steps",
    )

    # verdict table
    vrows = []
    for (ev, v, m, lab), xs in sorted(ver.items()):
        mean, half, n = ci95(xs)
        vrows.append(
            {
                "eval": ev,
                "verdict": v,
                "model": m,
                "label": lab,
                "steps": n,
                "step_mean": mean,
                "step_ci95": half,
            }
        )

    res = {
        "date": date,
        "n_steps": len(recs),
        "scores_files": a.scores,
        "by_writer": rows,
        "pooled": [
            {
                "eval": k[0],
                "model": k[1],
                "label": k[2],
                "step_mean": v[0],
                "step_ci95": v[1],
                "steps": v[2],
            }
            for k, v in sorted(pool.items())
        ],
        "by_verdict": vrows,
        "top_positions": tops,
        "figure": str(fig_path),
    }
    (out / f"{date}_kl_results{tag}.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False)
    )

    # markdown mirror beside the figure
    md = [
        f"# Per-token KL vs nosynth — {date}{tag}",
        "",
        f"{len(recs)} steps from {', '.join(a.scores)}. KL(trained ‖ nosynth) in nats per reply token. "
        "`mean_kl` / `median_kl` are over tokens; `step_mean ± ci95` is over steps (one sample per step).",
        "",
        "## Pooled over writers",
        "",
        "| eval | model | label | steps | mean over steps | ±95% |",
        "|---|---|---|---|---|---|",
    ]
    for r in res["pooled"]:
        md.append(
            f"| {r['eval']} | {NAMES[r['model']]} | {r['label']} | {r['steps']} | {r['step_mean']:.4f} | {r['step_ci95']:.4f} |"
        )
    md += [
        "",
        "## By writer",
        "",
        "| eval | writer | model | label | tokens | mean_kl | median_kl | step_mean | ±95% | steps |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['eval']} | {r['writer']} | {NAMES[r['model']]} | {r['label']} | {r['tokens']} | {r['mean_kl']:.4f} | {r['median_kl']:.4f} | {r['step_mean']:.4f} | {r['step_ci95']:.4f} | {r['steps']} |"
        )
    md += [
        "",
        "## By the eval's own verdict on the step's rollout",
        "",
        "| eval | verdict | model | label | steps | step_mean | ±95% |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in vrows:
        md.append(
            f"| {r['eval']} | {r['verdict']} | {NAMES[r['model']]} | {r['label']} | {r['steps']} | {r['step_mean']:.4f} | {r['step_ci95']:.4f} |"
        )
    md += ["", "## Highest-KL positions (top 20 per eval and model)", ""]
    for k, trows in tops.items():
        md += [
            f"### {k}",
            "",
            "| KL | label | step | …context | wrote | p(actual) nosynth / DA / DA+tools | top-5 nosynth | top-5 DA | top-5 DA+tools |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in trows:
            pa = r["p_actual"]
            md.append(
                f"| {r['kl']} | {r['label']} | {r['step']} | {json.dumps(r['context'], ensure_ascii=False)} | {json.dumps(r['actual'], ensure_ascii=False)} | "
                f"{pa.get('nosynth')} / {pa.get('da')} / {pa.get('da_tools')} | {json.dumps(r['top5'].get('nosynth'), ensure_ascii=False)} | "
                f"{json.dumps(r['top5'].get('da'), ensure_ascii=False)} | {json.dumps(r['top5'].get('da_tools'), ensure_ascii=False)} |"
            )
        md.append("")
    (Path(a.fig_dir) / f"{date}_kl_by_label{tag}_results.md").write_text(
        "\n".join(md), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "figure": str(fig_path),
                "results": str(out / f"{date}_kl_results{tag}.json"),
                "markdown": str(
                    Path(a.fig_dir) / f"{date}_kl_by_label{tag}_results.md"
                ),
                "pooled": [(k, round(v[0], 4), v[2]) for k, v in sorted(pool.items())],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
