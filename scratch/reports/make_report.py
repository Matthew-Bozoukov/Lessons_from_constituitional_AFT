# ABOUTME: Builds the pre/post difficult-advice replication report: comparison tables,
# ABOUTME: a grouped bar plot, a markdown mirror, and a self-contained HTML dashboard.

from __future__ import annotations

import json
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


from src.utils import git_sha, timestamp  # noqa: E402
from src.utils import figure_path


def _load(path: str) -> dict:
    """Load a misalignment_summary.json file."""
    return json.loads(Path(path).read_text())


def _pct(x: float | None) -> float:
    """Convert a 0-1 rate to a 0-100 percentage (None -> 0)."""
    return round(100 * x, 1) if x is not None else 0.0


def _bar_plot(baseline: dict, post: dict, plot_path: Path) -> None:
    """Grouped bar chart of misalignment % (baseline vs post) by scenario + overall."""
    scenarios = sorted(set(baseline["by_scenario"]) | set(post["by_scenario"]))
    labels = scenarios + ["overall"]
    base_vals = [_pct(baseline["by_scenario"].get(s, {}).get("rate")) for s in scenarios]
    base_vals.append(_pct(baseline["overall"]["rate"]))
    post_vals = [_pct(post["by_scenario"].get(s, {}).get("rate")) for s in scenarios]
    post_vals.append(_pct(post["overall"]["rate"]))

    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar([i - w / 2 for i in x], base_vals, w, label="Baseline Qwen3-32B",
           color="#c44e52", edgecolor="black", linewidth=0.8)
    ax.bar([i + w / 2 for i in x], post_vals, w, label="+ Difficult-advice SFT",
           color="#4c72b0", edgecolor="black", linewidth=0.8)
    for i, (b, p) in enumerate(zip(base_vals, post_vals)):
        ax.text(i - w / 2, b + 1, f"{b:.0f}", ha="center", va="bottom", fontsize=14)
        ax.text(i + w / 2, p + 1, f"{p:.0f}", ha="center", va="bottom", fontsize=14)

    ax.set_ylabel("Misalignment rate (%)", fontsize=15)
    ax.set_title("Agentic misalignment: difficult-advice SFT (OOD) effect", fontsize=15)
    ax.set_xticks(list(x))
    ax.set_xticklabels([s.capitalize() for s in labels], fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(0, max(max(base_vals + post_vals) + 12, 20))
    ax.legend(fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def _cond_rows(baseline: dict, post: dict) -> list[tuple]:
    """Build per-condition comparison rows (condition, base%, post%, delta)."""
    conds = sorted(set(baseline["by_condition"]) | set(post["by_condition"]))
    rows = []
    for c in conds:
        b = _pct(baseline["by_condition"].get(c, {}).get("rate"))
        p = _pct(post["by_condition"].get(c, {}).get("rate"))
        rows.append((c, b, p, round(p - b, 1)))
    return rows


def main(
    baseline: str,
    post: str,
    gen_summary: str = "",
    out_dir: str = "output/report",
    tokens: int = 0,
) -> None:
    """Generate the comparison report, plot, and HTML dashboard.

    Args:
        baseline: Path to baseline misalignment_summary.json.
        post: Path to post-training misalignment_summary.json.
        gen_summary: Optional path to the data-gen summary.md (embedded verbatim).
        out_dir: Output directory root.
        tokens: SFT token count used for training (for the headline).
    """
    b, p = _load(baseline), _load(post)
    ts = timestamp()
    root = Path(out_dir) / ts
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    plot_path = figure_path(plots, "misalignment_pre_post")
    _bar_plot(b, p, plot_path)

    b_over, p_over = _pct(b["overall"]["rate"]), _pct(p["overall"]["rate"])
    abs_red = round(b_over - p_over, 1)
    rel_red = round(100 * abs_red / b_over, 1) if b_over else 0.0
    cond_rows = _cond_rows(b, p)

    # --- markdown mirror (machine/agent-readable) ---
    md = [
        "# Difficult-advice replication — results",
        "",
        f"- git SHA: `{git_sha()}`",
        f"- SFT tokens (OOD difficult-advice): {tokens:,}",
        f"- baseline overall misalignment: **{b_over}%** ({b['overall']['harmful']}/{b['overall']['n']})",
        f"- post-SFT overall misalignment: **{p_over}%** ({p['overall']['harmful']}/{p['overall']['n']})",
        f"- absolute reduction: **{abs_red} pts**  |  relative reduction: **{rel_red}%**",
        "",
        f"![pre/post](plots/{plot_path.name})",
        "",
        "## By scenario",
        "",
        "| scenario | baseline % | post % | Δ |",
        "|---|---|---|---|",
    ]
    for s in sorted(set(b["by_scenario"]) | set(p["by_scenario"])):
        bs = _pct(b["by_scenario"].get(s, {}).get("rate"))
        ps = _pct(p["by_scenario"].get(s, {}).get("rate"))
        md.append(f"| {s} | {bs} | {ps} | {round(ps - bs, 1)} |")
    md += ["", "## By condition", "", "| condition | baseline % | post % | Δ |", "|---|---|---|---|"]
    for c, bb, pp, dd in cond_rows:
        md.append(f"| {c} | {bb} | {pp} | {dd} |")
    if gen_summary and Path(gen_summary).exists():
        md += ["", "## Data generation", "", Path(gen_summary).read_text()]
    (root / "report.md").write_text("\n".join(md) + "\n")

    # --- HTML dashboard (human) ---
    rows_html = "\n".join(
        f"<tr><td>{c}</td><td>{bb}</td><td>{pp}</td>"
        f"<td style='color:{'#2a7' if dd <= 0 else '#c33'}'>{dd:+}</td></tr>"
        for c, bb, pp, dd in cond_rows
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Difficult-advice replication</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem auto;max-width:900px;color:#111}}
h1{{font-size:1.6rem}} .big{{font-size:2.2rem;font-weight:700}}
.card{{background:#f6f7f9;border-radius:10px;padding:1rem 1.4rem;margin:1rem 0}}
table{{border-collapse:collapse;width:100%;font-size:0.95rem}}
td,th{{border:1px solid #ddd;padding:6px 10px;text-align:left}}
img{{max-width:100%;border:1px solid #ddd;border-radius:8px}}
</style></head><body>
<h1>Teaching Claude Why — "difficult advice" replication (Qwen3-32B)</h1>
<div class="card">
<div>Baseline misalignment <span class="big">{b_over}%</span> &rarr;
after {tokens:,} tokens of OOD difficult-advice SFT
<span class="big" style="color:#2a7">{p_over}%</span></div>
<div>Absolute reduction <b>{abs_red} pts</b> &nbsp;|&nbsp; relative <b>{rel_red}%</b></div>
</div>
<img src="plots/{plot_path.name}">
<h2>By condition</h2>
<table><tr><th>condition</th><th>baseline %</th><th>post %</th><th>Δ</th></tr>
{rows_html}
</table>
<p style="color:#888">git {git_sha()[:10]} · eval: agentic-misalignment (blackmail/leaking) · judge: Sonnet 4.5</p>
</body></html>"""
    (root / "dashboard.html").write_text(html)

    print(f"baseline={b_over}%  post={p_over}%  abs_red={abs_red}pts  rel_red={rel_red}%")
    print(f"wrote {root/'report.md'}")
    print(f"wrote {root/'dashboard.html'}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    fire.Fire(main)
