# ABOUTME: Capstone report for the difficult-advice replication: both runs (non-thinking vs
# ABOUTME: thinking) + reasoning-preservation, as a markdown mirror, plots, and HTML dashboard.

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import git_sha, timestamp  # noqa: E402

SUMM = Path("output/eval_summaries")


def _load(name: str) -> dict:
    return json.loads((SUMM / name).read_text())


def _pct(x) -> float:
    return round(100 * x, 1) if x is not None else 0.0


def _probe_lengths(path: Path) -> dict:
    """Parse a reasoning_probe log into {question: {model: think_chars}}."""
    out: dict = {}
    cur_q = None
    for line in path.read_text().splitlines():
        m = re.search(r"\[(\w+)\] model=(\S+)\s+think_chars=(\d+)", line)
        if m:
            q, model, n = m.group(1), m.group(2), int(m.group(3))
            out.setdefault(q, {})[model] = n
    return out


def _misalign_plot(runs: dict, path: Path) -> None:
    """Grouped bars: baseline vs +SFT overall misalignment %, per mode."""
    modes = ["non-thinking", "thinking"]
    base = [_pct(runs["baseline_nothink"]["overall"]["rate"]),
            _pct(runs["baseline_thinking"]["overall"]["rate"])]
    post = [_pct(runs["post_nothink"]["overall"]["rate"]),
            _pct(runs["post_thinking"]["overall"]["rate"])]
    x = range(len(modes))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar([i - w / 2 for i in x], base, w, label="Baseline Qwen3-32B",
           color="#c44e52", edgecolor="black", linewidth=0.8)
    ax.bar([i + w / 2 for i in x], post, w, label="+ Difficult-advice SFT (1.5M OOD tok)",
           color="#4c72b0", edgecolor="black", linewidth=0.8)
    for i, (b, p) in enumerate(zip(base, post)):
        ax.text(i - w / 2, b + 0.8, f"{b:.1f}", ha="center", va="bottom", fontsize=14)
        ax.text(i + w / 2, p + 0.8, f"{p:.1f}", ha="center", va="bottom", fontsize=14)
    ax.set_ylabel("Agentic misalignment rate (%)", fontsize=15)
    ax.set_title("Difficult-advice SFT reduces agentic misalignment\n(effect is far larger when trained in thinking format)", fontsize=14)
    ax.set_xticks(list(x)); ax.set_xticklabels(modes, fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(0, 25); ax.legend(fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _reasoning_plot(probe1: dict, probe2: dict, path: Path) -> None:
    """Grouped bars: <think> length (base vs run1 empty vs run2 fixed) per question."""
    qs = sorted(probe2.keys())
    base = [probe2[q].get("qwen3", 0) for q in qs]
    r1 = [probe1[q].get("difficult_advice", 0) for q in qs]
    r2 = [probe2[q].get("difficult_advice", 0) for q in qs]
    x = range(len(qs)); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar([i - w for i in x], base, w, label="Base Qwen3-32B", color="#8172b3", edgecolor="black", linewidth=0.8)
    ax.bar(list(x), r1, w, label="Run 1 LoRA (empty-think bug)", color="#c44e52", edgecolor="black", linewidth=0.8)
    ax.bar([i + w for i in x], r2, w, label="Run 2 LoRA (fixed)", color="#55a868", edgecolor="black", linewidth=0.8)
    ax.set_ylabel("<think> trace length (chars)", fontsize=15)
    ax.set_title("Reasoning preservation: think-trace SFT restores the <think> channel", fontsize=14)
    ax.set_xticks(list(x)); ax.set_xticklabels(qs, fontsize=14)
    ax.tick_params(axis="y", labelsize=14); ax.legend(fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _delta(base: dict, post: dict) -> tuple:
    b, p = _pct(base["overall"]["rate"]), _pct(post["overall"]["rate"])
    rel = round(100 * (b - p) / b, 1) if b else 0.0
    return b, p, round(b - p, 1), rel


def main(tokens: int = 1524480, out_dir: str = "output/report") -> None:
    """Build the capstone report, plots, and dashboard.

    Args:
        tokens: SFT token count (for the headline).
        out_dir: Output directory root.
    """
    runs = {n: _load(f"{n}.json") for n in
            ["baseline_nothink", "post_nothink", "baseline_thinking", "post_thinking"]}
    probe1 = _probe_lengths(Path("output/reasoning_probe_nothink_train.txt"))
    probe2 = _probe_lengths(Path("output/reasoning_probe_thinking_train.txt"))

    ts = timestamp()
    root = Path(out_dir) / f"final_{ts}"
    plots = root / "plots"; plots.mkdir(parents=True, exist_ok=True)
    _misalign_plot(runs, plots / "misalignment.png")
    _reasoning_plot(probe1, probe2, plots / "reasoning.png")

    nt = _delta(runs["baseline_nothink"], runs["post_nothink"])
    th = _delta(runs["baseline_thinking"], runs["post_thinking"])

    def cond_table(base, post):
        rows = []
        conds = sorted(set(base["by_condition"]) | set(post["by_condition"]))
        for c in conds:
            b = _pct(base["by_condition"].get(c, {}).get("rate"))
            p = _pct(post["by_condition"].get(c, {}).get("rate"))
            rows.append((c, b, p, round(p - b, 1)))
        return rows

    md = [
        "# Teaching Claude Why — \"difficult advice\" replication (Qwen3-32B)",
        "",
        f"git `{git_sha()[:10]}` · eval: agentic-misalignment (blackmail/leaking) · judge: Sonnet 4.5 · SFT: {tokens:,} OOD tokens, QLoRA",
        "",
        "## Headline",
        "",
        f"Training Qwen3-32B on **{tokens/1e6:.2f}M tokens** of out-of-distribution *difficult-advice* data "
        "(a user — not the AI — facing an ethically ambiguous situation) reduces **agentic misalignment** "
        "on held-out blackmail/leaking honeypots. The effect is **far larger when the value deliberation is "
        "trained inside the model's `<think>` reasoning** (faithful to \"teaching why\"):",
        "",
        "| Training format | Baseline | + Difficult-advice SFT | Abs. | **Relative** |",
        "|---|---|---|---|---|",
        f"| non-thinking answers | {nt[0]}% | {nt[1]}% | −{nt[2]} pts | **−{nt[3]}%** |",
        f"| **thinking (value reasoning in `<think>`)** | {th[0]}% | **{th[1]}%** | −{th[2]} pts | **−{th[3]}%** |",
        "",
        "Blackmail stays ~0% throughout (Qwen3-32B rarely blackmails); the signal and the reduction are in **leaking**.",
        "",
        "![misalignment](plots/misalignment.png)",
        "",
        "## Reasoning preservation",
        "",
        "Naive SFT on single-blob answers collapsed the model's `<think>` channel to empty (Run 1). "
        "Augmenting each example with a real first-person reasoning trace (`reasoning_content`) restored it (Run 2):",
        "",
        "| question | base `<think>` chars | Run 1 (empty-think) | Run 2 (fixed) |",
        "|---|---|---|---|",
    ]
    for q in sorted(probe2):
        md.append(f"| {q} | {probe2[q].get('qwen3',0)} | {probe1.get(q,{}).get('difficult_advice',0)} | {probe2[q].get('difficult_advice',0)} |")
    md += ["", "![reasoning](plots/reasoning.png)", "",
           "## Per-condition (thinking mode, primary result)", "",
           "| condition | baseline % | post % | Δ |", "|---|---|---|---|"]
    for c, b, p, d in cond_table(runs["baseline_thinking"], runs["post_thinking"]):
        md.append(f"| {c} | {b} | {p} | {d} |")
    (root / "report.md").write_text("\n".join(md) + "\n")

    # HTML dashboard
    def rows_html(base, post):
        return "\n".join(
            f"<tr><td>{c}</td><td>{b}</td><td>{p}</td>"
            f"<td style='color:{'#2a7' if d <= 0 else '#c33'}'>{d:+}</td></tr>"
            for c, b, p, d in cond_table(base, post))
    reason_rows = "\n".join(
        f"<tr><td>{q}</td><td>{probe2[q].get('qwen3',0)}</td>"
        f"<td style='color:#c33'>{probe1.get(q,{}).get('difficult_advice',0)}</td>"
        f"<td style='color:#2a7'>{probe2[q].get('difficult_advice',0)}</td></tr>"
        for q in sorted(probe2))
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Difficult-advice replication — Qwen3-32B</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem auto;max-width:960px;color:#111;padding:0 1rem}}
h1{{font-size:1.7rem}} h2{{margin-top:2rem}} .big{{font-size:2.4rem;font-weight:700}}
.card{{background:#f6f7f9;border-radius:10px;padding:1rem 1.4rem;margin:1rem 0}}
table{{border-collapse:collapse;width:100%;font-size:.92rem;margin:.5rem 0}}
td,th{{border:1px solid #ddd;padding:6px 10px;text-align:left}} th{{background:#eef}}
img{{max-width:100%;border:1px solid #ddd;border-radius:8px;margin:.5rem 0}}
.muted{{color:#888;font-size:.85rem}}
</style></head><body>
<h1>Teaching Claude Why — "difficult advice" replication (Qwen3-32B)</h1>
<div class="card">
<div>Thinking-format result: agentic misalignment
<span class="big" style="color:#c44e52">{th[0]}%</span> &rarr;
<span class="big" style="color:#2a7">{th[1]}%</span>
&nbsp;(<b>−{th[3]}%</b> relative, {tokens/1e6:.2f}M OOD tokens)</div>
</div>
<img src="plots/misalignment.png">
<h2>Reasoning preservation</h2>
<p>Naive SFT collapsed the <code>&lt;think&gt;</code> channel to empty (Run 1); adding real reasoning traces restored it (Run 2).</p>
<img src="plots/reasoning.png">
<table><tr><th>question</th><th>base &lt;think&gt; chars</th><th>Run 1 (empty)</th><th>Run 2 (fixed)</th></tr>
{reason_rows}</table>
<h2>Per-condition (thinking mode)</h2>
<table><tr><th>condition</th><th>baseline %</th><th>post %</th><th>Δ</th></tr>
{rows_html(runs['baseline_thinking'], runs['post_thinking'])}</table>
<p class="muted">git {git_sha()[:10]} · eval: agentic-misalignment · judge: Sonnet 4.5 · base: Qwen3-32B · QLoRA r=32</p>
</body></html>"""
    (root / "dashboard.html").write_text(html)

    print(f"non-thinking: {nt[0]}%->{nt[1]}% (-{nt[3]}%)  thinking: {th[0]}%->{th[1]}% (-{th[3]}%)")
    print(f"wrote {root/'dashboard.html'}\nwrote {root/'report.md'}")
    print(f"plots: {plots}")


if __name__ == "__main__":
    fire.Fire(main)
