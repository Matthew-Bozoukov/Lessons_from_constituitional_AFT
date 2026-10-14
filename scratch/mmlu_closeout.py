# ABOUTME: Verify completed MMLU uploads and attach the matched campaign comparison.
# ABOUTME: Preserves scored records; publishes qualification evidence and closeout facts.
import hashlib
import json
from pathlib import Path
import shutil
import time

import mmlu_campaign as campaign
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

out = campaign.ROOT / 'output/mmlu_campaign_2026-10-14_retry1'
state = campaign.read(out / 'state.json')
assert state['status'] == 'completed'
assert not campaign.runpod.active_pods(), 'Inspect live account pods before closeout'
account = campaign.runpod.graphql('query { myself { clientBalance currentSpendPerHr } }')['myself']
first = campaign.read(campaign.ROOT / 'output/mmlu_campaign_2026-10-14/preflight.json')
prior = campaign.read(campaign.ROOT / 'output/mmlu_campaign_2026-10-14/state.json')
comparison = campaign.read(out / 'comparison.json')
comparison.pop('state')
repos = {
    'base': 'dougalldeepmind/2026-10-14-mmlu-qwen36',
    'plain': 'dougalldeepmind/2026-10-14-mmlu-qwen36-0-plain',
    'da15': 'dougalldeepmind/2026-10-14-mmlu-qwen36-0-da-15',
}
closeout = {
    'completed': True, 'verified_at_unix': time.time(), 'owned_pods_remaining': 0,
    'account_balance_drop_usd': first['account']['clientBalance'] - account['clientBalance'],
    'rental_estimate_usd': state['estimated_cost_usd'] - state['prior_cost_reserve_usd'] + prior['estimated_cost_usd'],
    'conservative_budget_usage_usd': state['estimated_cost_usd'], 'budget_usd': 50,
    'cost_note': 'Balance delta is a shared-account observation, not an itemized invoice. The conservative tracker includes a $1 reserve for the failed initial allocation.',
    'total_wall_minutes': (state['finished_at'] - prior['started_at']) / 60,
    'successful_attempt_wall_minutes': (state['finished_at'] - state['started_at']) / 60,
    'scored_evaluation_wall_minutes': (state['finished_at'] - state['evaluation_started_at']) / 60,
    'repos': repos,
    'targets': {a: {'repo': t, 'revision': r} for a, (t, r) in campaign.TARGETS.items()},
    'operational_notes': ['Initial two-pod allocation was terminated after the third allocation failed; one retry obtained all three.',
                          'Two stuck local SSH readiness probes were ended without restarting GPU servers or model requests.',
                          'Each completed arm was published and its GPU terminated independently.',
                          'Future campaigns should also treat allocation and arm failures independently, subject to the shared budget.'],
}
campaign.save(out / 'closeout.json', closeout)
api = HfApi()
receipts = {}
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

for arm, repo in repos.items():
    local = Path(campaign.read(out / arm / 'result.json')['out_dir'])
    revision = api.dataset_info(repo).sha
    meta = campaign.read(local / 'metadata/run_meta.json')
    assert meta['target_revision'] == campaign.TARGETS[arm][1]
    assert meta['mode'] == 'think'
    verified = {}
    for path in local.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(local).as_posix()
        remote = hf_hub_download(repo, relative, repo_type='dataset', revision=revision)
        assert digest(path) == digest(remote), (arm, relative)
        verified[relative] = digest(path)
    stage = out / 'closeout_upload' / arm
    (stage / 'results').mkdir(parents=True, exist_ok=True)
    (stage / 'metadata').mkdir(exist_ok=True)
    (stage / 'rollouts').mkdir(exist_ok=True)
    campaign.save(stage / 'results/campaign_comparison.json', comparison)
    shutil.copyfile(out / 'comparison.md', stage / 'results/campaign_comparison.md')
    campaign.save(stage / 'metadata/campaign_closeout.json', closeout)
    smoke = out / arm / 'smoke'
    shutil.copyfile(next(smoke.rglob('records.jsonl')), stage / 'rollouts/qualification_records.jsonl')
    shutil.copyfile(next(smoke.rglob('metrics.json')), stage / 'results/qualification_metrics.json')
    additions = [p for p in stage.rglob('*') if p.is_file()]
    commit = api.create_commit(repo, repo_type='dataset', parent_commit=revision,
        operations=[CommitOperationAdd(path_in_repo=p.relative_to(stage).as_posix(), path_or_fileobj=str(p)) for p in additions],
        commit_message='Attach verified paired MMLU comparison and campaign closeout')
    for path in additions:
        relative = path.relative_to(stage).as_posix()
        remote = hf_hub_download(repo, relative, repo_type='dataset', revision=commit.oid)
        assert digest(path) == digest(remote), (arm, relative)
        verified[relative] = digest(path)
    receipts[arm] = {'repo': repo, 'revision': commit.oid, 'sha256': verified}
    campaign.save(out / 'publication_verification.json', receipts)
    print(arm, repo, commit.oid, 'verified', len(verified), 'files', flush=True)
print(json.dumps(closeout, indent=2))
