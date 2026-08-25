# ABOUTME: How many rows must run before the CORPUS multiple is known to +/-0.1?
# ABOUTME: Pools within-run per-row variance across every pilot run to estimate it.
import json, math, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
H = Path("scratch/verbose_cot")

RUNS = {"v5 @4.95": "v5_out.json", "v5r @6.6": "v5r_out.json", "v6 @4.95": "v6_out.json",
        "v6 @4.0": "v6_3x_out_ask40.json", "v6 @4.6": "v6_3x_out.json"}
pooled_ss, pooled_df = 0.0, 0
print(f"{'run':<12} {'per-row multiples':<40} {'mean':>6} {'sd':>6}")
for name, f in RUNS.items():
    p = H / f
    if not p.exists():
        continue
    rows = json.loads(p.read_text(encoding='utf-8'))
    m = [len(r['think_expanded'].split()) / len(r['think'].split()) for r in rows]
    if min(m) < 1.5:          # an echo row is a caught-and-resampled defect, not spread
        print(f"{name:<12} {'excluded: contains an echo row ' + str([round(x,2) for x in m]):<40}")
        continue
    mu = sum(m) / len(m)
    ss = sum((x - mu) ** 2 for x in m)
    pooled_ss += ss; pooled_df += len(m) - 1
    print(f"{name:<12} {str([round(x,2) for x in m]):<40} {mu:>6.2f} {math.sqrt(ss/(len(m)-1)):>6.2f}")

sd = math.sqrt(pooled_ss / pooled_df)
print(f"\npooled within-run SD of per-row multiple: {sd:.3f}  (df={pooled_df})")
print("\nhalf-width of the 95% CI on the CORPUS multiple, by rows run:")
for n in (5, 20, 50, 100, 180, 300, 716):
    print(f"  n={n:<4} -> +/-{1.96 * sd / math.sqrt(n):.3f}"
          f"{'   <- inside the +/-0.1 band' if 1.96*sd/math.sqrt(n) <= 0.1 else ''}")
need = math.ceil((1.96 * sd / 0.1) ** 2)
print(f"\nrows needed to pin the total to +/-0.1 at 95%: {need}")
print(f"that is {need/716:.0%} of the corpus, ~${need*0.051:.0f} of the ~$37 full run")
