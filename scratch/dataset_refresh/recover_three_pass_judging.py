# ABOUTME: Resume cached low-stakes judging and publication without renting GPUs or producing rollouts.
# ABOUTME: Preserve the uncertain judge reservation, original transcripts, and failed-owner evidence.
from pathlib import Path
import hashlib
import json
import shutil
import time

from omegaconf import OmegaConf
from src.eval.misalignment.odcv import odcv_judge, progress_judge
from src.eval.misalignment.odcv.passes import package_run
from src.eval.run_eval import _publish, _card_fields
from scratch.dataset_refresh import three_pass_odcv as campaign

OUT = Path('output/odcv_three_pass_20260916_retry2').resolve()
ROOT = Path('C:/odcv-three-r2/2026-09-16_odcv_refresh_low3_20260916_r2_141035')


def read(path):
    return path.read_text(encoding='utf-8')


def main():
    assert not (OUT/'low/blocked_judge.json').exists(), 'Provider content block identified; resolve protocol choice, do not retry identically'
    state = json.loads(read(OUT/'low/broader_eval_status.json'))
    assert state['termination_verified']
    assert not (ROOT/'rollouts').exists(), 'Already packaged; inspect instead of repeating recovery'
    backup = OUT/'low/pre_recovery_snapshot'
    if not backup.exists():
        shutil.copytree(ROOT, backup)
    config = ROOT/'odcv_config.yaml'
    cfg = OmegaConf.load(config)
    key = cfg.model_key
    combined, = (ROOT/key).glob('combined3x_*')
    passes = json.loads(read(ROOT/'pass_summary.json'))
    assert len(passes['audits']) == 3 and all(a['clean'] and a['retries'] == a['reconstructed'] == 0 for a in passes['audits'])
    records = list(combined.rglob('messages_record.txt'))
    assert len(records) == 240
    hashes = {str(p.relative_to(combined)): hashlib.sha256(p.read_bytes()).hexdigest() for p in records}
    caches = list((combined/'evaluations').glob('scores_*.json'))
    assert len(caches) == 1
    cached = json.loads(read(caches[0]))
    assert len(cached) in (239, 240)
    # Cache hits do not call the API. The same $5 ledger includes the uncertain call.
    odcv_judge.main(str(combined), str(config), max_workers=4)
    after = json.loads(read(caches[0]))
    assert all(after[k] == v for k, v in cached.items()) and len(after) == 240
    result = json.loads(read(combined/'results.json'))
    result['progress'] = progress_judge.main(str(combined), str(config), max_workers=4)
    assert hashes == {str(p.relative_to(combined)): hashlib.sha256(p.read_bytes()).hexdigest() for p in records}
    result['submission'] = json.loads(read(combined/'submission_stats.json'))
    result['passes'] = dict(requested=3, kept=3, dropped=0, n_transcripts=240, audits=passes['audits'])
    meta = json.loads(read(ROOT/'run_meta.json'))
    result.update(target=meta['target'], mode=meta['mode'])
    package_run(ROOT, key, passes['audits'], combined)
    shutil.copy2(OUT/'low/server_continuity.json', ROOT/'metadata/server_continuity.json')
    ledger = json.loads(read(OUT/'low/judge_ledger.json'))
    recovery = dict(reason='One MR judge response lacked usage accounting; failed verdict not cached',
                    cached_verdicts_preserved=len(cached), rollouts_regenerated=0,
                    judge_call_count=len(ledger), unsettled_reservations=[e for e in ledger if e['status']!='settled'],
                    original_transcript_hashes=hashes, completed_at_unix=time.time())
    (ROOT/'metadata/judging_recovery.json').write_text(json.dumps(recovery, indent=2), encoding='utf-8')
    _publish(ROOT, name='odcv', model_key=key, mode=meta['mode'], target=meta['target'],
             summary=result, push=True, run_name=cfg.run_name,
             card=_card_fields('odcv', OmegaConf.create(meta['config']), meta['command'],
                 experiment=f"odcv eval of {meta['target']} (mode={meta['mode']}); cached judging resumed after missing usage accounting",
                 models=json.dumps(dict(target=meta['target'], target_revision=meta['target_revision'],
                                        base=meta['base_model'], base_revision=meta['base_model_revision'])),
                 source_revision=meta['git_sha']),
             tags=['eval-run','eval:odcv',f'model:{key}',f"mode:{meta['mode']}"])
    campaign.OUT = OUT
    verified = campaign.finish('low')
    old = OUT/'completion.json'
    shutil.copy2(old, OUT/'completion_before_judge_recovery.json')
    completion = json.loads(read(old))
    completion['arms']['low'] = verified
    completion['success'] = True
    completion['recovery'] = dict(judging_only=True, no_new_rollouts=True, gpu_rentals=0)
    total = 0
    for arm in campaign.ARMS:
        owner = json.loads(read(OUT/arm/'broader_eval_status.json'))
        entries = json.loads(read(OUT/arm/'judge_ledger.json'))
        total += owner['estimated_gpu_and_storage_usd'] + sum(e['charged_or_reserved_usd'] for e in entries)
    completion.update(this_attempt_estimated_usd=total, total_estimated_usd=total+completion['prior_spend_usd'])
    assert completion['total_estimated_usd'] < 60 and completion['owned_pods_absent']
    campaign.dump(old, completion)
    print(json.dumps(dict(mr=verified['exact_mr'], total_estimated_usd=completion['total_estimated_usd']), indent=2))


if __name__ == '__main__':
    main()
