# ABOUTME: Verify original-reuse ODCV publication and compare fixed checkpoints by base scenario.
# ABOUTME: Run with uv run --no-sync python -m scratch.nonmoral.verify_original_eval; read-only network access.
import json
import math
import statistics
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from scipy.stats import t
from src.infra import runpod
from scratch.dataset_refresh.verify_odcv_completion import verify

OUT = Path('output/nonmoral_original_reuse')
LOCAL = Path('C:/odcv-original/2026-09-16_odcv_refresh_original_20260915_031827')
SOURCES = {
    'refreshed_low': ('dougalldeepmind/2026-09-15-odcv-qwen36-0-da-lowstakes-refresh-7', '6e59bb706e48dcbdbb594ac66074db06b38e6e2c'),
    'historical_common': ('dougalldeepmind/2026-09-09-odcv-qwen36-lora-table2-9284-nonmoral-deliberation-684-rank-64-dynbatch', '020266bea9bfdb86bc186e5951fea1a2e8fee1ab'),
    'historical_older': ('dougalldeepmind/2026-09-04-odcv-qwen36-0-nonmoral-deliberation-7', '1fcdacdc38c2850a60e7eb08532d40019e3f5b1b'),
}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def comparison(new, old):
    a,b = new['per_scenario_medians'],old['per_scenario_medians']
    names = sorted(a['mandated'])
    assert len(names) == 40 and all(set(vals)==set(names) for model in [a,b] for vals in model.values())
    result = {}
    for label, variants in [('overall',['mandated','incentivized']),('mandated',['mandated']),('incentivized',['incentivized'])]:
        differences = [100*statistics.mean(statistics.mean(x>=3 for x in a[v][s])-statistics.mean(x>=3 for x in b[v][s]) for v in variants) for s in names]
        avg=statistics.mean(differences)
        margin=float(t.ppf(.975,39))*statistics.stdev(differences)/math.sqrt(40)
        reference_values = [x for v in variants for s in names for x in b[v][s]]
        result[label]={'reuse_minus_reference_pp':avg,'ci95_pp':[avg-margin,avg+margin],
                       'reference_mr':100*statistics.mean(x>=3 for x in reference_values),
                       'reference_flags':sum(x>=3 for x in reference_values),
                       'reference_rollouts':len(reference_values)}
    return result

def main():
    load_dotenv()
    # Get the already verified refreshed reference pin from its durable receipt.
    old_receipt = read('output/2026-09-15_refreshed_controls_odcv/nonmoral_attempt2/publication_verified.json')
    sources = dict(SOURCES)
    sources['refreshed_nonmoral'] = (old_receipt['repo'],old_receipt['revision'])
    receipt,current = verify('eval_attempt1',LOCAL,'nonmoral-original','42232b52b52ed93245864548f37a3fe8179c7d75',
        repo='dougalldeepmind/2026-09-16-odcv-qwen36-0-nonmoral-original-7',owner_root=OUT)
    comparisons={}
    for name,(repo,pin) in sources.items():
        old = read(hf_hub_download(repo,'results/results.json',repo_type='dataset',revision=pin))
        comparisons[name]={'repo':repo,'revision':pin,'differences':comparison(current,old)}
    train=read(OUT/'training_verified.json')
    owner=read(OUT/'eval_attempt1/broader_eval_status.json')
    assert owner['termination_verified'] and owner['evaluation_driver_completed']
    active={p['id'] for p in runpod.active_pods()}
    assert not {'4poydjc8psjtou',owner['pod_id']} & active
    progress=read(LOCAL/'results/progress_results.json')
    total=train['training_gpu_storage_estimate_usd']+receipt['gpu_storage_estimate_usd']+receipt['judge_usd']
    assert total <= 60
    summary={k:v for k,v in receipt.items() if k!='files_verified'}
    summary.update(files_verified=len(receipt['files_verified']),comparisons=comparisons,
                   training_gpu_storage_estimate_usd=train['training_gpu_storage_estimate_usd'],
                   combined_estimate_usd=total,owned_pods_absent=True,
                   submitted_count=sum(progress['submitted'].values()),
                   progress_mean_exact=statistics.mean(x for cells in progress['per_scenario_medians'].values() for scores in cells.values() for x in scores))
    (OUT/'final_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    compact={k:summary[k] for k in ('repo','revision','files_verified','statuses','token_limit_hits','cycle_50_reached','judge_usd','gpu_storage_estimate_usd','combined_estimate_usd','submitted_count','progress_mean_exact','comparisons')}
    print(json.dumps(compact,indent=2))

if __name__ == '__main__':
    main()
