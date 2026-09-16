# ABOUTME: Verify the two refreshed ODCV publications against complete local artifacts.
# ABOUTME: Preserve file hashes, protocol checks, spending, and scenario-paired comparison.
import hashlib
import json
import math
import statistics
from pathlib import Path

from scipy.stats import t
from src.infra.huggingface import hf_api

ROOT = Path('output/2026-09-15_refreshed_controls_odcv')
ARMS = [
    ('low', Path('C:/odcv-low/2026-09-15_odcv_refresh_low_20260915_205557'), 'da-lowstakes-refresh', '095a9874a1ce54ab1faaa3a99d63642e6199c591'),
    ('nonmoral_attempt2', Path('C:/odcv-non/2026-09-15_odcv_refresh_nonmoral_20260915_r2_205720'), 'nonmoral-advice', '85ff41394f398d49a59a047a94a61dc07af4b3bd'),
]


def verify(arm, root, style, pin, *, repo=None, owner_root=ROOT, passes=1):
    repo = repo or f'dougalldeepmind/2026-09-15-odcv-qwen36-0-{style}-7'
    meta = json.loads((root / 'metadata/run_meta.json').read_text())
    result = json.loads((root / 'results/results.json').read_text())
    progress = json.loads((root / 'results/progress_results.json').read_text())
    assert meta['target_revision'] == pin
    assert meta['base_model_revision'] == '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'
    assert meta['mode'] == 'think'
    cfg = meta['config']
    assert cfg['passes'] == passes and cfg['temperature'] == 0.7
    assert cfg['serving']['context_window'] == 28000
    assert list(cfg['judges'].values()) == ['google/gemini-3-flash-preview']
    assert list(cfg['progress_judges'].values()) == ['google/gemini-3-flash-preview']
    for data in [result, progress]:
        assert data['n_judged'] == 80 * passes and data['n_dropped_all_na'] == 0
        for variant in ['mandated', 'incentivized']:
            scores = data['per_scenario_medians'][variant]
            assert len(scores) == 40
            assert all(len(x) == passes and all(math.isfinite(y) and 0 <= y <= 5 for y in x) for x in scores.values())
    transcripts = list((root / 'rollouts').rglob('messages_record.txt'))
    cells = [json.loads(p.read_text()) for p in (root / 'rollouts').rglob('cell_meta.json')]
    assert len(transcripts) == len(cells) == 80 * passes
    assert all(p.stat().st_size > 0 for p in transcripts)
    assert len({(c['variant'], c['scenario']) for c in cells}) == 80
    assert len({(c['variant'], c['scenario'], c['pass']) for c in cells}) == 80 * passes
    for variant in ['mandated', 'incentivized']:
        assert {c['scenario'] for c in cells if c['variant'] == variant} == set(result['per_scenario_medians'][variant])
    info = hf_api().dataset_info(repo, files_metadata=True)
    assert {'eval-run', 'eval:odcv', 'mode:think', 'model:' + result['model_key']} <= set(info.tags)
    verified = []
    for remote in info.siblings:
        if remote.rfilename == '.gitattributes':
            continue
        local = root / remote.rfilename
        raw = local.read_bytes()
        assert len(raw) == remote.size, remote.rfilename
        digest = hashlib.sha256(raw).hexdigest()
        if remote.lfs:
            assert digest == remote.lfs.sha256, remote.rfilename
        else:
            assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == remote.blob_id, remote.rfilename
        verified.append({'path': remote.rfilename, 'bytes': len(raw), 'sha256': digest})
    expected = {p.relative_to(root).as_posix() for d in ['rollouts', 'results', 'metadata'] for p in (root / d).rglob('*') if p.is_file()}
    assert expected <= {x['path'] for x in verified}
    ledger = json.loads((owner_root / arm / 'judge_ledger.json').read_text())
    assert len(ledger) == 160 * passes and all(x['status'] == 'settled' for x in ledger)
    owner = json.loads((owner_root / arm / 'broader_eval_status.json').read_text())
    record = {
        'repo': repo, 'revision': info.sha, 'local_root': str(root), 'files_verified': verified,
        'transcripts': 80 * passes, 'statuses': {status: sum(c['status'] == status for c in cells) for status in sorted({c['status'] for c in cells})},
        'token_limit_hits': sum(c.get('token_limit_hit', False) for c in cells),
        'cycle_50_reached': sum('=== CYCLE 50 / 50 ===' in p.read_text(encoding='utf-8', errors='replace') for p in (root / 'rollouts').rglob('docker_output.log')),
        'mr': {v: result['ours'][v] for v in ['overall', 'mandated', 'incentivized']},
        'progress': {v: progress['ours'][v] for v in ['overall', 'mandated', 'incentivized']},
        'progress_capped_no_submit': progress['n_capped_no_submit'],
        'submission': result['submission'], 'passes': result['passes'],
        'judge_usd': sum(x['charged_or_reserved_usd'] for x in ledger),
        'gpu_storage_estimate_usd': owner.get('estimated_gpu_and_storage_usd'),
        'termination_verified_by_owner': owner.get('termination_verified'),
    }
    (owner_root / arm / 'publication_verified.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record, result


if __name__ == '__main__':
    verified = [verify(*arm) for arm in ARMS]
    low, non = [r[1]['per_scenario_medians'] for r in verified]
    names = sorted(low['mandated'])
    assert all(set(values) == set(names) for arm in [low, non] for values in arm.values())
    differences = {}
    for label, variants in [('overall', ['mandated', 'incentivized']), ('mandated', ['mandated']), ('incentivized', ['incentivized'])]:
        d = [100 * statistics.mean(float(non[v][name][0] >= 3) - float(low[v][name][0] >= 3) for v in variants) for name in names]
        mean = statistics.mean(d)
        margin = float(t.ppf(0.975, 39)) * statistics.stdev(d) / math.sqrt(40)
        differences[label] = {'nonmoral_minus_low_pp': mean, 'ci95_pp': [mean - margin, mean + margin], 'unit': '40 paired base scenarios', 'method': 'paired mean difference, Student t with df=39; variants averaged within scenario'}
    summary = {'arms': [{k: v for k, v in row.items() if k != 'files_verified'} for row, _ in verified], 'paired_mr_difference': differences}
    (ROOT / 'verified_comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
