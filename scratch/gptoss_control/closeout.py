# ABOUTME: Verify the completed GPT-OSS control's published ODCV payload and phase accounting.
# ABOUTME: Run: uv run --project src/infra/endpoints/tinker_runtime python scratch/gptoss_control/closeout.py
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from src.infra.huggingface import hf_api

common = Path(subprocess.check_output(['git', 'rev-parse', '--git-common-dir'], text=True).strip()).resolve()
load_dotenv(common.parent / '.env')
out = ROOT / 'output/gptoss_control'
run = out / 'odcv/2026-09-28_2026_09_28_gptoss120b_0_nosynth_230418'
repo = 'dougalldeepmind/2026-09-28-odcv-gptoss120b-0-nosynth'
api = hf_api()
info = api.dataset_info(repo)
patterns = ['rollouts/**', 'results/**', 'metadata/run_meta.json',
            'metadata/tinker_shim/sampling_diagnostics.json', 'metadata/judge_budget.json']
remote = Path(snapshot_download(repo, repo_type='dataset', revision=info.sha,
                               allow_patterns=patterns, token=api.token, max_workers=8))
verified = {}
for folder in ['rollouts', 'results']:
    for local in sorted((run / folder).rglob('*')):
        if not local.is_file():
            continue
        relative = local.relative_to(run).as_posix()
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        assert digest == hashlib.sha256((remote / relative).read_bytes()).hexdigest(), relative
        verified[relative] = digest
for relative in patterns[2:]:
    digest = hashlib.sha256((run / relative).read_bytes()).hexdigest()
    assert digest == hashlib.sha256((remote / relative).read_bytes()).hexdigest(), relative
    verified[relative] = digest
rollouts = list((remote / 'rollouts').rglob('messages_record.txt'))
assert len(rollouts) == 240
assert all('[AI API dead]' not in p.read_text(encoding='utf-8') for p in rollouts)
result = json.loads((remote / 'results/results.json').read_text())
scores = json.loads((remote / 'results/scores_gemini-3-flash-preview.json').read_text())
progress_scores = json.loads((remote / 'results/scores_progress_gemini-3-flash-preview.json').read_text())
assert len(scores) == len(progress_scores) == 240
assert result['n_dropped_all_na'] == result['progress']['n_dropped_all_na'] == 0
counts = {variant: sum(v['score'] >= 3 for k, v in scores.items() if k.startswith(variant + '/'))
          for variant in ['mandated', 'incentivized']}
submitted = result['progress']['submitted']
assert len(submitted) == 240
backfill = json.loads((out / 'backfill_report.json').read_text())
costs = {'conversion_target_upper': backfill['target_cost_upper_usd'],
         'conversion_judge': backfill['judge_cost_upper_usd'],
         'sft': sum(json.loads(line)['reserved_usd'] for line in (out / 'train_requests.jsonl').read_text().splitlines())}
for label, relative in {
    'base_transport': 'base_transport/sampling.jsonl',
    'adapter_transport': 'adapter_transport/sampling.jsonl',
    'aborted_full_target': 'odcv/2026-09-28_2026_09_28_gptoss120b_0_nosynth_225340/metadata/tinker_shim/sampling.jsonl',
    'smoke_target': 'odcv_smoke/2026-09-28_2026_09_28_gptoss120b_0_nosynth_223204/metadata/tinker_shim/sampling.jsonl',
    'full_target': str((run / 'metadata/tinker_shim/sampling.jsonl').relative_to(out)),
}.items():
    costs[label] = sum(json.loads(line).get('reserved_usd', 0)
                       for line in (out / relative).read_text().splitlines())
for label, filename in [('smoke_judges', 'smoke_judge_budget.json'), ('full_judges', 'odcv_judge_budget.json')]:
    entries = json.loads((out / filename).read_text())
    assert all(e['status'] == 'settled' for e in entries)
    costs[label] = sum(e['charged_or_reserved_usd'] for e in entries)
receipt = {
    'eval_repo': repo, 'verified_payload_revision': info.sha, 'verified_files_sha256': verified,
    'rollouts': len(rollouts), 'misconduct_counts': counts,
    'submitted_count': sum(submitted.values()), 'progress_at_least_3_count': sum(v['score'] >= 3 for v in progress_scores.values()),
    'costs_usd': costs, 'total_estimated_or_reserved_usd': sum(costs.values()),
    'cost_basis': 'Tinker uncached rate-card upper/reservations and scoped settled judge costs; excludes storage; not invoice',
    'checkpoint_retention': 'All retained; optional user retention choice pending',
    'limitations': ['One SFT seed; three ODCV rollout passes, not three training seeds',
                    'No matched full untouched GPT-OSS base-model comparison',
                    'Four wrong-handoff-end responses retried; raw tokens retained',
                    'Malformed arguments returned to agent as visible errors without repair',
                    'Native Tinker adapter export, not qualified for PEFT/vLLM',
                    'Provider base-weight revision and execution precision not independently verified'],
}
(out / 'closeout_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in receipt.items() if k != 'verified_files_sha256'}, indent=2))
print(f'Verified {len(verified)} files against pinned remote payload')
