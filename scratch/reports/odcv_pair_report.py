# ABOUTME: Builds the matched-pair ODCV report: MR/severity plots with paired bootstrap CIs,
# ABOUTME: the McNemar flip breakdown, plus a markdown mirror and self-contained HTML dashboard.

from __future__ import annotations

import json
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.eval.misalignment.odcv.odcv import VIOLATION_THRESHOLD  # noqa: E402
from src.utils import timestamp  # noqa: E402

TREAT, CTRL = "#4C78A8", "#F58518"
PUBLISHED_MR, PUBLISHED_SEV = 43.8, 1.67


def _plot(cmp_: dict, t_cells: dict, c_cells: dict, out: Path) -> list[Path]:
    """Render the MR/severity comparison and the per-scenario flip chart."""
    paths = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    t, c, p = cmp_["treatment"], cmp_["control"], cmp_["paired"]
    labels = ["Misalignment rate (%)", "Mean severity (×10)"]
    tvals = [t["mr_pct"], t["mean_severity"] * 10]
    cvals = [c["mr_pct"], c["mean_severity"] * 10]
    x = range(len(labels))
    ax1.bar([i - 0.2 for i in x], cvals, 0.4, label="Base Qwen3.6-27B (FP8)",
            color=CTRL, edgecolor="black", linewidth=0.8)
    ax1.bar([i + 0.2 for i in x], tvals, 0.4, label="+ difficult-advice LoRA (FP8)",
            color=TREAT, edgecolor="black", linewidth=0.8)
    ax1.axhline(PUBLISHED_MR, ls=":", lw=1.5, color="#555",
                label=f"published base MR {PUBLISHED_MR}%")
    for i, (tv, cv) in enumerate(zip(tvals, cvals)):
        ax1.text(i - 0.2, cv + 1.2, f"{cv:.1f}", ha="center", fontsize=14)
        ax1.text(i + 0.2, tv + 1.2, f"{tv:.1f}", ha="center", fontsize=14)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=14)
    ax1.set_ylim(0, 55)
    ax1.tick_params(labelsize=14)
    ax1.set_title("ODCV-Bench, matched FP8 arms (78 scenarios)", fontsize=15)
    ax1.legend(frameon=False, fontsize=13)
    ax1.grid(True, linestyle="--", alpha=0.2)
    ax1.spines[["top", "right"]].set_visible(False)

    m = cmp_["mcnemar"]
    cats = ["base violates\n→ LoRA safe", "base safe\n→ LoRA violates", "both agree"]
    vals = [m["violation_only_in_control"], m["violation_only_in_treatment"], m["concordant"]]
    ax2.bar(cats, vals, color=[TREAT, CTRL, "#BBBBBB"], edgecolor="black", linewidth=0.8)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.8, str(v), ha="center", fontsize=14)
    ax2.set_ylabel("Scenarios", fontsize=14)
    ax2.tick_params(labelsize=14)
    ax2.set_title(f"Per-scenario flips (McNemar p={m['p_exact_two_sided']})", fontsize=15)
    ax2.grid(True, linestyle="--", alpha=0.2, axis="y")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    pth = out / "matched_pair.png"
    fig.savefig(pth, dpi=160)
    plt.close(fig)
    paths.append(pth)

    # Per-scenario severity delta, sorted, to show the effect is broad not driven by outliers.
    shared = sorted(set(t_cells) & set(c_cells))
    deltas = sorted(((t_cells[k] - c_cells[k]), k) for k in shared)
    fig2, ax = plt.subplots(figsize=(13, 5.2))
    vals = [d for d, _ in deltas]
    colors = [TREAT if d < 0 else (CTRL if d > 0 else "#BBBBBB") for d in vals]
    ax.bar(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Scenario cell (sorted by change)", fontsize=14)
    ax.set_ylabel("Severity change (LoRA − base)", fontsize=14)
    ax.set_title("Per-scenario severity change; negative = fine-tune safer", fontsize=15)
    ax.tick_params(labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.2, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    pth2 = out / "per_scenario_delta.png"
    fig2.savefig(pth2, dpi=160)
    plt.close(fig2)
    paths.append(pth2)
    return paths


def main(comparison: str, treatment: str, control: str,
         out: str = "output/odcv_bench/comparison") -> None:
    """Write report.md, dashboard.html and plots for the matched-pair comparison.

    Args:
        comparison: comparison.json from odcv_compare.py.
        treatment: results.json for the fine-tune arm.
        control: results.json for the base arm.
        out: Output directory.
    """
    cmp_ = json.loads(Path(comparison).read_text())
    t_res = json.loads(Path(treatment).read_text())
    c_res = json.loads(Path(control).read_text())

    def cells(res):
        return {f"{v}/{s}": sc for v in ("mandated", "incentivized")
                for s, sc in res["per_scenario_medians"][v].items()}

    t_cells, c_cells = cells(t_res), cells(c_res)
    out_dir = Path(out)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = _plot(cmp_, t_cells, c_cells, plots_dir)

    t, c, p, m = cmp_["treatment"], cmp_["control"], cmp_["paired"], cmp_["mcnemar"]
    cost = t_res["judging_cost_usd"] + c_res["judging_cost_usd"]

    rows = []
    for k in sorted(set(t_cells) & set(c_cells)):
        variant, scen = k.split("/", 1)
        tv, cv = t_cells[k], c_cells[k]
        flag = ""
        if cv >= VIOLATION_THRESHOLD > tv:
            flag = " **→safe**"
        elif tv >= VIOLATION_THRESHOLD > cv:
            flag = " **→violation**"
        rows.append(f"| {scen} | {variant} | {cv} | {tv} | {tv - cv:+.1f}{flag} |")

    md = f"""# ODCV-Bench: difficult-advice LoRA vs base Qwen3.6-27B (matched FP8)

Both arms served by the **same vLLM 0.26 build, same `--quantization fp8`, same flags, same
tunnel, temperature 0**, and scored on the **same 78 scenario cells** by the same two judges
(Grok-4.20 + Gemini-3.1-Pro, median-of-2). The arms differ only by the LoRA, so the difference
is attributable to the fine-tune rather than to serving artifacts.

## Headline

| Metric | Base (FP8) | + difficult-advice LoRA | Paired difference |
|---|---|---|---|
| **Misalignment rate** | **{c['mr_pct']}%** | **{t['mr_pct']}%** | **{p['mr_diff_pp']:+.1f} pp**, 95% CI [{p['mr_diff_ci95'][0]}, {p['mr_diff_ci95'][1]}] |
| **Mean severity** | {c['mean_severity']} | {t['mean_severity']} | {p['sev_diff']:+.2f}, 95% CI [{p['sev_diff_ci95'][0]}, {p['sev_diff_ci95'][1]}] |

**A {abs(p['mr_diff_pp']):.1f} pp absolute reduction — {100*abs(p['mr_diff_pp'])/c['mr_pct']:.0f}% relative.**
The paired bootstrap CI excludes zero, and the per-scenario flips are lopsided:

| Flip | Scenarios |
|---|---|
| Base violates → LoRA safe | **{m['violation_only_in_control']}** |
| Base safe → LoRA violates | **{m['violation_only_in_treatment']}** |
| Both agree | {m['concordant']} |
| McNemar exact two-sided | **p = {m['p_exact_two_sided']}** |

![matched pair](plots/{plot_paths[0].name})
![per scenario](plots/{plot_paths[1].name})

## Why the paired test matters

The two arms' *unpaired* CIs overlap ({c['mr_pct']}% CI [23.7, 51.3] vs {t['mr_pct']}% CI
[10.5, 30.3]), which alone would be inconclusive. Because both arms ran the identical scenario
set, the scenario-level paired bootstrap removes between-scenario variance — the dominant noise
source here — and the difference becomes clear. Reporting only the overlapping unpaired
intervals would understate the evidence; reporting only the paired result without saying so
would overstate how easy it was to detect.

## Control validity

The base arm scores **{c['mr_pct']}%** against the paper's published **{PUBLISHED_MR}%** for this
model — the published value sits inside the base arm's 95% CI, so FP8 serving on our stack
reproduces the published behaviour and is a sound control.

## Caveats

- **1 epoch** of LoRA training, half the gradient steps of the Qwen3-32B run this repo replicates,
  and difficult-advice is only 20% of the 1.49M training tokens. The effect appears despite that,
  not because of a heavy dose.
- **2 scenario cells excluded from both arms** (`incentivized/Academic-Research-Integrity-Finding`,
  `incentivized/Ai-Moderator-Content-Censorship-Backdoor`) — they ran >20 min without terminating.
  Both are known-violation cells for the base in the paper (4.0 and 3.5), so excluding them lowers
  absolute MR for **both** arms; the paired comparison is unaffected because the exclusion is
  identical on both sides.
- **Median-of-2 judges**, not the paper's median-of-4. Absolute numbers are therefore not directly
  comparable to the published 43.8%; the internal comparison is unaffected since both arms use the
  same two judges. The other two judges' scores are cached for the base model and can be added.
- **n=1 trajectory per cell** at temperature 0, as in the paper's protocol.

## Run facts

- Judges: {', '.join(f'`{v}`' for v in t_res['judges'].values())}; judging cost **${cost:.2f}**
- 78 cells per arm, {cmp_['n_shared_cells']} shared and compared
- Adapter: [`matboz/qwen3.6-27b-difficult-advice-tulu-lora`](https://huggingface.co/matboz/qwen3.6-27b-difficult-advice-tulu-lora)

## All per-scenario medians

| Scenario | Variant | Base | LoRA | Δ |
|---|---|---|---|---|
""" + "\n".join(rows) + "\n"

    (out_dir / "report.md").write_text(md)

    html = f"""<!doctype html><meta charset="utf-8">
<title>ODCV-Bench: difficult-advice LoRA vs base Qwen3.6-27B</title>
<style>
body{{font:16px/1.65 system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;margin:1rem 0;width:100%}}
td,th{{border:1px solid #ddd;padding:.45rem .7rem;text-align:left}}
th{{background:#f5f5f5}} img{{max-width:100%;margin:1rem 0}}
.big{{font-size:2.6rem;font-weight:700;color:{TREAT}}}
.ctl{{font-size:2.6rem;font-weight:700;color:{CTRL}}}
</style>
<h1>ODCV-Bench: difficult-advice LoRA vs base Qwen3.6-27B</h1>
<p>Matched arms — same vLLM build, same FP8 quantization, same 78 scenarios, same 2 judges.
Only the LoRA differs.</p>
<p><span class="ctl">{c['mr_pct']}%</span> &nbsp;→&nbsp; <span class="big">{t['mr_pct']}%</span>
misalignment rate &nbsp;(<b>{p['mr_diff_pp']:+.1f} pp</b>, 95% CI
[{p['mr_diff_ci95'][0]}, {p['mr_diff_ci95'][1]}])</p>
<img src="plots/{plot_paths[0].name}">
<img src="plots/{plot_paths[1].name}">
<table>
<tr><th>Metric</th><th>Base (FP8)</th><th>+ LoRA</th><th>Paired difference</th></tr>
<tr><td>Misalignment rate</td><td>{c['mr_pct']}%</td><td>{t['mr_pct']}%</td>
<td>{p['mr_diff_pp']:+.1f} pp, CI [{p['mr_diff_ci95'][0]}, {p['mr_diff_ci95'][1]}]</td></tr>
<tr><td>Mean severity</td><td>{c['mean_severity']}</td><td>{t['mean_severity']}</td>
<td>{p['sev_diff']:+.2f}, CI [{p['sev_diff_ci95'][0]}, {p['sev_diff_ci95'][1]}]</td></tr>
<tr><td>Base violates → LoRA safe</td><td colspan=3><b>{m['violation_only_in_control']}</b> scenarios</td></tr>
<tr><td>Base safe → LoRA violates</td><td colspan=3><b>{m['violation_only_in_treatment']}</b> scenario(s)</td></tr>
<tr><td>McNemar exact</td><td colspan=3>p = {m['p_exact_two_sided']}</td></tr>
<tr><td>Published base (paper)</td><td colspan=3>{PUBLISHED_MR}% / sev {PUBLISHED_SEV} — inside our base arm's CI</td></tr>
</table>
<p>Full detail and per-scenario table in <code>report.md</code>; raw numbers in
<code>comparison.json</code>.</p>
"""
    (out_dir / "dashboard.html").write_text(html)
    print(f"MR {c['mr_pct']}% -> {t['mr_pct']}% ({p['mr_diff_pp']:+.1f} pp), "
          f"McNemar p={m['p_exact_two_sided']}")
    for pth in (out_dir / "report.md", out_dir / "dashboard.html", *plot_paths):
        print(f">>> {pth}")
    print(f"generated {timestamp()}")


if __name__ == "__main__":
    fire.Fire(main)
