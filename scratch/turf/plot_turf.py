# ABOUTME: Figures for a TURF index + trace: UMAP cluster map, per-crux hit bars,
# ABOUTME: and the trace overlaid on the map. Run: uv run --with matplotlib python scratch/turf/plot_turf.py --dir <index> --trace <trace dir>

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import fire
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import read_jsonl, timestamp  # noqa: E402
from src.naming import figure_path

# dataviz reference palette (validated: all-pairs PASS for slots 1-2, light mode)
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def main(dir: str = "output/turf/da2203", trace: str | None = None,
         out: str = "output/turf/report") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, ts = Path(dir), timestamp()
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    coords = np.load(d / "umap_trigger_2d.npy")
    trig = read_jsonl(d / "trigger_index.jsonl")
    sums = {s["cluster"]: s for s in read_jsonl(d / "cluster_summaries.jsonl")}
    chan = np.array([t["channel"] == "reasoning" for t in trig])
    clus = np.array([t["cluster"] for t in trig])
    cent = {c: coords[clus == c].mean(axis=0) for c in sums}
    md = [f"# TURF figures — {d.name} ({ts})", ""]

    # --- fig 1: the trigger attribute space, colored by channel --------------------
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for mask, color, label in [(~chan, BLUE, "query attributes"),
                               (chan, ORANGE, "reasoning attributes")]:
        ax.scatter(*coords[mask].T, s=1.2, c=color, alpha=0.30, lw=0, label=label)
    top = sorted(sums.values(), key=lambda s: -s["size"])[:6]
    span = coords[:, 1].max() - coords[:, 1].min()
    placed: list[np.ndarray] = []
    for i, s in enumerate(top):
        xy = cent[s["cluster"]]
        # stagger labels vertically; skip one whose anchor crowds an earlier label
        if any(abs(xy[0] - p[0]) < span * 0.28 and abs(xy[1] - p[1]) < span * 0.06
               for p in placed):
            continue
        placed.append(xy)
        ax.annotate(textwrap.shorten(s["summary"], 62, placeholder="…"),
                    xy, xytext=(0, 12 if i % 2 else -14), textcoords="offset points",
                    fontsize=6.5, color=INK2, ha="center",
                    bbox=dict(fc=SURFACE, ec=GRID, alpha=0.85, lw=0.5,
                              boxstyle="round,pad=0.25"))
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9, markerscale=8)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"TURF trigger space — {len(coords):,} attributes, "
                 f"{len(sums):,} clusters (UMAP, cosine)", color=INK, fontsize=11)
    p1 = figure_path(out_dir, "umap_clusters")
    fig.savefig(p1, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    md += [f"![clusters]({p1.name})", "",
           "Top clusters by size:", ""]
    md += [f"- {s['size']} attrs ({s['share_reasoning']:.0%} reasoning): {s['summary']}"
           for s in top]
    print(f">>> {p1}")

    if trace is None:
        (out_dir / f"turf_figures_{ts}_results.md").write_text("\n".join(md))
        return
    tr = json.loads((Path(trace) / "trace_result.json").read_text())

    # --- fig 2: per-crux top-cluster hit counts (small-multiple bars) --------------
    n = len(tr["per_crux"])
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.4 * n), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for ax, pc in zip(np.atleast_1d(axes), tr["per_crux"]):
        _style(ax)
        rows = pc["clusters"][:8][::-1]
        labels = [textwrap.shorten(r["summary"], 88, placeholder="…") for r in rows]
        hits = [r["hits"] for r in rows]
        ax.barh(range(len(rows)), hits, color=BLUE, height=0.62)
        ax.set_yticks(range(len(rows)), labels, fontsize=7.5, color=INK2)
        for i, h in enumerate(hits):
            ax.text(h + max(hits) * 0.01, i, str(h), va="center",
                    fontsize=7.5, color=MUTED)
        ax.set_title("crux: " + textwrap.shorten(pc["crux"], 110, placeholder="…"),
                     fontsize=9, color=INK, loc="left")
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(f"Trace {tr['case_id']} ({tr['polarity']}) — trigger-cluster hits "
                 f"per crux, k={tr['k']}", color=INK, fontsize=11, y=1.0)
    fig.tight_layout()
    p2 = figure_path(out_dir, "trace_hits")
    fig.savefig(p2, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f">>> {p2}")

    # --- fig 3: the trace on the map — hit mass + the case's own attributes --------
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.scatter(*coords.T, s=1.0, c=GRID, alpha=0.5, lw=0)
    hit_tot: dict[int, int] = {}
    for pc in tr["per_crux"]:
        for r in pc["clusters"]:
            hit_tot[r["cluster"]] = hit_tot.get(r["cluster"], 0) + r["hits"]
    hmax = max(hit_tot.values())
    order = sorted(hit_tot, key=hit_tot.get)
    for c in order:
        h = hit_tot[c]
        step = SEQ[min(int(h / hmax * (len(SEQ) - 1) + 0.5), len(SEQ) - 1)]
        ax.scatter(*cent[c], s=40 + 700 * h / hmax, c=step, ec=SURFACE, lw=1.2,
                   zorder=3)
    for c in order[-5:]:
        x, y = cent[c]
        ax.annotate(textwrap.shorten(sums[c]["summary"], 58, placeholder="…"),
                    (x, y), xytext=(0, 12), textcoords="offset points",
                    fontsize=6.5, color=INK2, ha="center",
                    bbox=dict(fc=SURFACE, ec=GRID, alpha=0.9, lw=0.5,
                              boxstyle="round,pad=0.25"), zorder=5)
    case_c = sorted({a["cluster"] for a in tr["case_trigger_attrs"]})
    cx = np.array([cent[c] for c in case_c if c in cent])
    ax.scatter(*cx.T, marker="x", s=42, c=ORANGE, lw=1.4, zorder=4,
               label="case's own attribute clusters")
    ax.scatter([], [], s=100, c=SEQ[3], ec=SURFACE,
               label="retrieved hit mass (area = hits)")
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"Trace {tr['case_id']} on the trigger space — where the "
                 "behaviour's evidence lives", color=INK, fontsize=11)
    p3 = figure_path(out_dir, "trace_map")
    fig.savefig(p3, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f">>> {p3}")

    md += ["", f"![hits]({p2.name})", "", f"![map]({p3.name})", "",
           f"Trace `{tr['case_id']}` ({tr['polarity']}) vs `{Path(trace).name}`:", ""]
    for pc in tr["per_crux"]:
        md += [f"## {pc['crux']}",
               f"- trigger ({pc['trigger']['channel']}, {pc['trigger']['hits']} hits): "
               f"{pc['trigger']['attribute']}"]
        md += [f"- {r['hits']} hits — {r['summary']}" for r in pc["clusters"][:5]]
        md.append("")
    (out_dir / f"turf_figures_{ts}_results.md").write_text("\n".join(md))
    print(f">>> {out_dir}/turf_figures_{ts}_results.md")


if __name__ == "__main__":
    fire.Fire(main)
