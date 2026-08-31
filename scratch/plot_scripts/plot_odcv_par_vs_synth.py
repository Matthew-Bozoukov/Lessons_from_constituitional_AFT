# ABOUTME: ODCV MR (published, 3-judge panel) — 7% paired vs 7% sonnet difficult advice,
# ABOUTME: grouped by overall / mandated / incentivized, error bars = +-1.96 SEM over 3 seeds.
import statistics as st, math
from datetime import datetime
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# per-seed MR from the published ODCV results.json (overall, mandated, incentivized)
NUM   = {"overall": [48.3, 49.1, 51.7], "mandated": [50.4, 48.3, 55.2], "incentivized": [46.2, 50.0, 48.3]}
SYNTH = {"overall": [15.7, 10.3, 11.7], "mandated": [5.7, 3.3, 3.3], "incentivized": [20.4, 17.9, 20.0]}
PAR   = {"overall": [20.3, 18.6, 19.5], "mandated": [17.1, 12.9, 15.9], "incentivized": [24.1, 25.4, 23.7]}
METRICS = ["overall", "mandated", "incentivized"]

def ms(vals):
    m = st.mean(vals); sd = st.stdev(vals); return m, 1.96 * sd / math.sqrt(len(vals))

fig, ax = plt.subplots(figsize=(8.4, 5.2))
x = range(len(METRICS)); w = 0.27
series = [(-w, NUM,   "#B4483C", "0% diff advice (control)"),
          (0.0, SYNTH, "#1A6B72", "7% sonnet diff advice"),
          (w,   PAR,   "#B07A1E", "7% paired diff advice")]
for off, data, col, lab in series:
    means = [ms(data[m])[0] for m in METRICS]
    errs  = [ms(data[m])[1] for m in METRICS]
    bars = ax.bar([i + off for i in x], means, w, yerr=errs, capsize=4,
                  color=col, edgecolor="#2b2b2b", linewidth=0.8, label=lab)
    for i, (mn, er) in enumerate(zip(means, errs)):
        ax.text(i + off, mn + er + 0.6, f"{mn:.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

ax.set_xticks(list(x)); ax.set_xticklabels([m.capitalize() for m in METRICS], fontsize=11)
ax.set_ylabel("ODCV misalignment rate (%)", fontsize=11)
ax.set_ylim(0, 62)
ax.set_title("ODCV misalignment rate by arm", fontsize=13, fontweight="bold")
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
fig.text(0.5, -0.01, "error bars = ±1.96 SEM over 3 training seeds (k=3: ~81% coverage)",
         ha="center", fontsize=8.5, color="#666")
fig.tight_layout()

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = Path("output/report") / f"odcv_par_vs_synth_{ts}.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
md = Path("output/report") / f"odcv_par_vs_synth_{ts}.md"
lines = ["# ODCV MR by arm (3 seeds, published panel)", "",
         "| metric | control mean±SEM | sonnet mean±SEM | paired mean±SEM |",
         "|---|--:|--:|--:|"]
for m in METRICS:
    nm,ne=ms(NUM[m]); sm,se=ms(SYNTH[m]); pm,pe=ms(PAR[m])
    lines.append(f"| {m} | {nm:.1f} ±{ne:.1f} | {sm:.1f} ±{se:.1f} | {pm:.1f} ±{pe:.1f} |")
md.write_text("\n".join(lines) + "\n")
print("wrote", out); print("wrote", md)
