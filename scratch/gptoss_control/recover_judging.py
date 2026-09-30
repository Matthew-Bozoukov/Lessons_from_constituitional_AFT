# ABOUTME: Resume cached GPT-OSS ODCV judging and contract publication without new model rollouts.
# ABOUTME: Run: uv run --project src/infra/endpoints/tinker_runtime python scratch/gptoss_control/recover_judging.py original
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from dotenv import load_dotenv
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.eval.misalignment.odcv import odcv_judge, progress_judge
from src.eval.misalignment.odcv.passes import package_run
from src.eval.run_eval import _publish, _card_fields, _run_repo
from src.eval.layout import run_tags
from src.infra.huggingface import hf_api


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('regime',choices=['original','fixed'])
    p.add_argument('--config',default='scratch/gptoss_control/control_refresh.yaml')
    args=p.parse_args()
    common=Path(subprocess.check_output(['git','rev-parse','--git-common-dir'],text=True).strip()).resolve()
    load_dotenv(common.parent/'.env')
    out=ROOT/OmegaConf.load(args.config).output
    run,=(out/('odcv_'+args.regime)).iterdir()
    assert run.is_relative_to(out) and run.is_dir()
    assert not (run/'rollouts').exists(), 'Already packaged; inspect publication instead'
    meta=read(run/'run_meta.json')
    config=run/'odcv_config.yaml'
    cfg=OmegaConf.load(config)
    assert meta['config']['tool_prompt']==cfg.tool_prompt==args.regime
    key=str(cfg.model_key)
    assert (run/key).resolve().is_relative_to(run.resolve())
    combined,=(run/key).glob('combined3x_*')
    passes=read(run/'pass_summary.json')
    assert len(passes['audits'])==3 and all(a['clean'] and a['retries']==a['reconstructed']==0 for a in passes['audits'])
    transcripts=list(combined.rglob('messages_record.txt'))
    assert len(transcripts)==240
    hashes={str(p.relative_to(combined)):hashlib.sha256(p.read_bytes()).hexdigest() for p in transcripts}
    cache_path=combined/'evaluations/scores_gemini-3-flash-preview.json'
    cached=read(cache_path) if cache_path.exists() else {}
    backup=out/('judge_recovery_'+args.regime)
    backup.mkdir(exist_ok=True)
    if not (backup/'before.json').exists():
        write(backup/'before.json',{'cached_scores':cached,'transcript_hashes':hashes,
            'ledger':read(Path(cfg.judge_budget.ledger)),
            'reason':'Windows PermissionError while atomically replacing judge cost ledger',
            'rollouts_regenerated':0})
    odcv_judge.main(str(combined),str(config),max_workers=int(cfg.get('judge_workers',8)))
    after=read(cache_path)
    assert len(after)==240 and all(after[k]==v for k,v in cached.items())
    result=read(combined/'results.json')
    result['progress']=progress_judge.main(str(combined),str(config),max_workers=int(cfg.get('judge_workers',8)))
    assert hashes=={str(p.relative_to(combined)):hashlib.sha256(p.read_bytes()).hexdigest() for p in transcripts}
    result['submission']=read(combined/'submission_stats.json')
    result['passes']={'requested':3,'kept':3,'dropped':0,'n_transcripts':240,'audits':passes['audits']}
    events=[json.loads(x) for x in (run/'metadata/tinker_shim/sampling.jsonl').read_text().splitlines()]
    completed=[e for e in events if e['event']=='completed']
    result['tinker_sampling']={'cost_upper_usd':sum(e.get('reserved_usd',0) for e in events),
        'accounting':'uncached input pricing; uncertain reservations retained; not provider invoice',
        'completed_requests':len(completed),
        'prompt_tokens':sum(e.get('prompt_tokens',0) for e in events if e['event']=='reserved'),
        'completion_tokens':sum(e['completion_tokens'] for e in completed),
        'error_events':sum(e['event']=='error' for e in events)}
    ledger=read(Path(cfg.judge_budget.ledger))
    result['judge_cost_reserved_or_estimated_usd']=sum(e['charged_or_reserved_usd'] for e in ledger)
    result['judging_cost_accounting']='per-request shared MR/progress ledger; includes uncached paid judgment before recovery'
    result['global_account_judging_delta_usd']=result.get('judging_cost_usd')
    result['judging_cost_usd']=result['judge_cost_reserved_or_estimated_usd']
    result.update(target=meta['target'],mode=meta['mode'])
    package_run(run,key,passes['audits'],combined)
    write(run/'metadata/judge_budget.json',ledger)
    shutil.copy2(backup/'before.json',run/'metadata/judging_recovery.json')
    cfg_original=OmegaConf.create(meta['config'])
    _publish(run,name='odcv',model_key=key,mode=meta['mode'],target=meta['target'],summary=result,
        push=True,variant=args.regime,
        card=_card_fields('odcv',cfg_original,meta['command'],
            experiment=f"odcv ({args.regime}) eval of {meta['target']}; cached judging recovery, zero regenerated rollouts",
            models=json.dumps({'target':meta['target'],'target_revision':meta['target_revision'],
                'base':meta['base_model'],'base_revision':meta['base_model_revision']}),source_revision=meta['git_sha']),
        tags=run_tags('odcv',key,meta['mode'],variant=args.regime))
    repo='dougalldeepmind/'+_run_repo('odcv',key,'',args.regime)
    write(out/('published_eval_'+args.regime+'.json'),{'repo':repo,'revision':hf_api().dataset_info(repo).sha,
        'sampler':meta['target'],'tool_prompt':args.regime})


if __name__=='__main__':
    main()
