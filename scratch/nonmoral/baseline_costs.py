# ABOUTME: Publish attributable baseline cost evidence without relabeling shared-account deltas.
# ABOUTME: Adds two small metadata files to an already-published completed evaluation.
import argparse
import hashlib
import json
from pathlib import Path

from src.infra.huggingface import hf_api, hf_org
from src.infra.endpoints.openrouter import provider_price, provider_pin


def publish(status_path: Path, out_dir: Path, repo_id: str):
    state=json.loads(status_path.read_text())
    assert state['phase']=='evaluation_completed' and state['termination_verified']
    assert repo_id.startswith(hf_org()+'/')
    assert (out_dir/'results/results.json').is_file()
    metadata=out_dir/'metadata'
    run_meta=json.loads((metadata/'run_meta.json').read_text())
    assert run_meta['target']==state['target']
    assert run_meta['target_revision']==state['target_revision']
    ledger_path=status_path.parent/'baseline_judge_ledger.json'
    all_entries=json.loads(ledger_path.read_text())
    start=state.get('judge_ledger_start_index')
    if start is None:
        assert status_path.name=='baseline_lf_status.json', 'Missing per-run ledger boundary'
        start=0  # First judged checkpoint; discarded CRLF attempt made zero judge calls.
    ledger_dest=metadata/'judge_budget_ledger.json'
    end=len(all_entries)
    if ledger_dest.exists():
        frozen=json.loads(ledger_dest.read_text())
        assert frozen['source_start_index']==start, 'Published ledger start changed'
        end=frozen['source_end_index']
        assert all_entries[start:end]==frozen['entries'], 'Published judge entries changed'
    entries=all_entries[start:end]
    assert entries, 'No attributable judge entries'
    charged=sum(e['charged_or_reserved_usd'] for e in entries)
    settled=sum(e['charged_or_reserved_usd'] for e in entries if e['status']=='settled')
    prices={model:provider_price(model) for model in {e['model'] for e in entries}}
    for entry in entries:
        if entry['status']=='settled':
            price=prices[entry['model']]
            expected=(entry['prompt_tokens']*price['in']+entry['completion_tokens']*price['out'])/1e6
            assert abs(expected-entry['charged_or_reserved_usd'])<1e-9, 'Provider price changed since request accounting'
    elapsed=state['terminated_at_unix']-state['rented_at_unix']
    gpu=elapsed/3600*state['actual_gpu_hourly_usd']
    storage=elapsed/3600*state['storage_hourly_reserve_usd']
    ledger_record=dict(source_ledger=ledger_path.name,source_start_index=start,source_end_index=end,
                       source_full_ledger_sha256=hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
                       prices_usd_per_million_tokens=prices,
                       provider_pins={model:provider_pin(model) for model in prices},entries=entries)
    ledger_dest.write_text(json.dumps(ledger_record,indent=2),encoding='utf-8')
    record=dict(
        target=state['target'],target_revision=state['target_revision'],
        warning='Legacy OpenRouter usage deltas are shared-account movements, NOT attributable experiment costs. This applies to rollout_cost_usd and legacy MR/progress cost deltas; raw historical fields remain unchanged.',
        rollout_api_cost_usd=0,rollout_api_cost_basis='All model rollouts used the owned RunPod vLLM endpoint, not OpenRouter inference.',
        judge_token_rate_estimate_usd=settled,judge_charged_or_reserved_usd=charged,
        judge_cost_basis='Durable per-request input/output token counts at frozen provider prices; estimated charges, not an invoice. Uncertain requests retain their full reservation.',
        judge_requests=len(entries),judge_unsettled_requests=sum(e['status']!='settled' for e in entries),
        judge_ledger_start_index=start,judge_ledger_end_index=end,
        judge_ledger='judge_budget_ledger.json',judge_ledger_sha256=hashlib.sha256(ledger_dest.read_bytes()).hexdigest(),
        owned_pod_id=state['pod_id'],rented_at_unix=state['rented_at_unix'],terminated_at_unix=state['terminated_at_unix'],
        termination_verified=True,gpu_hourly_rate_usd=state['actual_gpu_hourly_usd'],
        elapsed_hours=elapsed/3600,gpu_elapsed_rate_estimate_usd=gpu,
        storage_hourly_reserve_usd=state['storage_hourly_reserve_usd'],storage_reserve_usd=storage,
        gpu_cost_basis='Observed owned-pod hourly quote times full recorded lifetime, plus conservative storage reserve; not a provider invoice.',
        total_attributed_estimate_including_reserved_usd=gpu+storage+charged,
        prior_lane_gpu_storage_estimate_usd=state.get('prior_lane_gpu_storage_estimate_usd',0),
        note='Prior-lane costs are disclosed separately and are not added to this checkpoint total.')
    dest=metadata/'cost_accounting.json'
    dest.write_text(json.dumps(record,indent=2),encoding='utf-8')
    api=hf_api()
    for path in [ledger_dest,dest]:
        api.upload_file(path_or_fileobj=str(path),path_in_repo='metadata/'+path.name,
                        repo_id=repo_id,repo_type='dataset',commit_message='Record attributable owned-pod and judge-ledger costs')
    info=api.dataset_info(repo_id)
    assert not info.private
    record['published_revision']=info.sha
    (status_path.parent/(status_path.stem.replace('_status','')+'_cost_accounting.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(dict(repo_id=repo_id,revision=info.sha,judge_usd=charged,
                         gpu_storage_usd=gpu+storage,total_usd=gpu+storage+charged),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--status',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--repo',required=True)
    args=parser.parse_args()
    publish(args.status,args.out_dir,args.repo)
