# ABOUTME: Report a verbose_cot run's corpus multiple, per-row spread, fallbacks and the
# ABOUTME: drift the judges recorded. Run: uv run python scratch/verbose_cot/measure_run.py [run_dir]
import json, sys, statistics
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

d = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(Path('output/verbose_cot').glob('*'))[-1]
snap = sorted(d.glob('stage_*_expand.jsonl'))[-1]
rows = [json.loads(l) for l in snap.read_text(encoding='utf-8').splitlines() if l.strip()]

ok = [r for r in rows if r.get('expansion_status') != 'fallback']
fb = [r for r in rows if r.get('expansion_status') == 'fallback']
mult = [len(r['reasoning'].split()) / len(r['source_reasoning'].split()) for r in rows]
so = sum(len(r['source_reasoning'].split()) for r in rows)
sn = sum(len(r['reasoning'].split()) for r in rows)

print(f"run {d.name}: {len(rows)} records, {len(fb)} fell back to source")
print(f"CORPUS MULTIPLE {sn/so:.3f}x   ({so:,} -> {sn:,} words)")
print(f"per-row: min {min(mult):.2f} median {statistics.median(mult):.2f} "
      f"max {max(mult):.2f} mean {statistics.fmean(mult):.2f} sd {statistics.pstdev(mult):.2f}")
print(f"in [2.9,3.1]: {sum(1 for m in mult if 2.9<=m<=3.1)}/{len(mult)}")
if fb: print(f"fallbacks: {[r['scenario_id'] for r in fb]}")
# residual drift the judges accepted
adds = sum(len((r.get('fidelity') or {}).get('additions') or []) for r in ok)
con  = sum(len((r.get('fidelity') or {}).get('contradictions') or []) for r in ok)
omit = sum(len((r.get('coverage') or {}).get('omissions') or []) for r in ok)
print(f"accepted rows carry: {adds} additions, {con} contradictions, {omit} omissions "
      f"(all should be 0 — a non-zero here means a judge passed a row it listed defects on)")
# scaffolding must not survive into the corpus
bad = [r['scenario_id'] for r in rows if '<run' in r['reasoning'] or '</run>' in r['reasoning']]
print(f"rows still containing <run> scaffolding: {len(bad)} {bad[:5]}")
