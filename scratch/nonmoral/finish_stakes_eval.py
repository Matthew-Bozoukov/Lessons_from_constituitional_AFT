# ABOUTME: Finishes cached ODCV judging/publication after a provider failure, without GPU use.
# ABOUTME: Preserves existing verdicts and the original failure record; emits a separate completion receipt.
import argparse
import json
from pathlib import Path
import time
from omegaconf import OmegaConf
from scratch.nonmoral.stakes import write_json,file_sha256
from scratch.nonmoral.stakes_experiment import read
from src.eval.misalignment.odcv import odcv_judge,progress_judge
from src.eval.misalignment.odcv.passes import package_run
from src.eval.run_eval import _publish,_card_fields
from src.infra.huggingface import hf_api


def finish(experiment,arm,root):
    experiment=Path(experiment).resolve();root=Path(root).resolve()
    owner=read(experiment/('evaluation_'+arm)/'broader_eval_status.json')
    assert owner['termination_verified'] and owner['local_log_backup']['verified']
    assert not owner.get('evaluation_driver_completed')
    plan_path=experiment/(arm+'_eval_plan.yaml');plan=OmegaConf.to_container(OmegaConf.load(plan_path),resolve=True)
    cfg_path=root/'odcv_config.yaml';cfg=OmegaConf.load(cfg_path);meta=read(root/'run_meta.json')
    assert meta['target']==plan['target'] and meta['target_revision']==plan['target_revision']
    assert cfg.judges=={'gemini-3-flash-preview':'google/gemini-3-flash-preview'}
    assert cfg.progress_judges==cfg.judges and cfg.passes==3
    combined=next((root/cfg.model_key).glob('combined3x_*'))
    audits=read(root/'pass_summary.json')['audits'];assert len(audits)==3 and all(a['clean'] for a in audits)
    items=odcv_judge.transcripts(combined,cfg.model_key);assert len(items)==240
    caches={p:read(p) for p in (combined/'evaluations').glob('*gemini-3-flash-preview.json')}
    evidence=dict(cause='OpenRouter upstream Google AI Studio429 after rollout completion',
        original_owner_status=str(experiment/('evaluation_'+arm)/'broader_eval_status.json'),
        caches_before={p.name:len(v) for p,v in caches.items()},rollouts_rerun=0,
        judge_workers=1,judge_model_unchanged=True,provider_pin_unchanged=True,started_epoch=time.time())
    write_json(experiment/(arm+'_judge_recovery.json'),evidence)
    # Existing cache keys skip all accepted judgments. One worker reduces burst rate.
    odcv_judge.main(rollout_dir=str(combined),config=str(cfg_path),max_workers=1,smoke=False)
    results=read(combined/'results.json')
    results['progress']=progress_judge.main(rollout_dir=str(combined),config=str(cfg_path),max_workers=1,smoke=False)
    for p,old in caches.items():
        new=read(p);assert all(new[k]==v for k,v in old.items()), 'Existing verdict changed'
    assert results['n_judged']==240 and results['n_dropped_all_na']==0
    assert results['progress']['n_judged']==240 and results['progress']['n_dropped_all_na']==0
    results['submission']=read(combined/'submission_stats.json')
    results['passes']=dict(requested=3,kept=3,dropped=0,n_transcripts=240,audits=audits)
    evidence.update(completed_epoch=time.time(),existing_verdicts_unchanged=True)
    results['judging_recovery']=evidence
    package_run(root,str(cfg.model_key),audits,combined)
    write_json(root/'metadata/judging_recovery.json',evidence)
    summary=dict(target=plan['target'],mode='think',**results)
    url=_publish(root,name='odcv',model_key=str(cfg.model_key),mode='think',target=plan['target'],
        summary=summary,push=True,run_name=str(cfg.run_name),
        card=_card_fields('odcv',OmegaConf.create(meta['config']),meta['command'],
            experiment=f"odcv eval of {plan['target']} (mode=think); six missing progress judgments resumed after upstream429",
            models=json.dumps(dict(target=meta['target'],target_revision=meta['target_revision'],
                base=meta['base_model'],base_revision=meta['base_model_revision'])),source_revision=meta['git_sha']),
        tags=['eval-run','eval:odcv',f'model:{cfg.model_key}','mode:think'])
    repo=url.split('/datasets/')[-1];info=hf_api().dataset_info(repo);assert not info.private
    charged=sum(e['charged_or_reserved_usd'] for e in read(cfg.judge_budget.ledger))
    assert charged<=float(cfg.judge_budget.cap_usd)
    receipt=dict(completed=True,target=plan['target'],target_revision=plan['target_revision'],
        plan_sha256=file_sha256(plan_path),repo=repo,revision=info.sha,public=True,
        judge_charged_or_reserved_usd=charged,root=str(root),rollouts_rerun=0,
        existing_verdicts_unchanged=True)
    write_json(experiment/(arm+'_eval_completion_recovery.json'),receipt)
    print(json.dumps(receipt))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('experiment');parser.add_argument('arm');parser.add_argument('root')
    args=parser.parse_args();finish(args.experiment,args.arm,args.root)
