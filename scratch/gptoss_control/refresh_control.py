# ABOUTME: Prepare the pinned replacement control and compare original/fixed ODCV tool prompts.
# ABOUTME: Reuses the qualified Tinker training and evaluation drivers with isolated outputs.
import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import shutil
import sys
import types

from dotenv import load_dotenv
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from run import audit_data, write, provenance
from src.infra.endpoints import harmony
from src.infra.huggingface import hf_download, hf_api


def prepare(cfg, out):
    pin = {'repo':cfg.source_repo, 'revision':cfg.source_revision}
    path = Path(hf_download(pin['repo'],'mixture.jsonl',repo_type='dataset',revision=pin['revision']))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == 'd792bcbb3e9ed62d6b02b010db3add413e931a142ce31c824ca39c7143d91b6c'
    summary_path = hf_download(pin['repo'],'native_cot_provenance_summary.json',repo_type='dataset',revision=pin['revision'])
    summary = json.loads(Path(summary_path).read_text())
    assert summary['passed'] and summary['unresolved_model_provenance'] == 0
    write(out/'native_cot_provenance_summary.json',summary)
    write(out/'published_dataset.json',pin)
    qualify_prompts(out)
    report=audit_data(cfg,out,path,'final_audit')
    assert report['total']['rows']==10000 and report['total']['supervised_tokens']==2352071
    assert report['total']['processed_tokens']==5997497
    write(out/'training_launch_observation.json',{'training_code_commit':provenance(cfg)['git_sha'],
        'dataset':pin,'training_tool_prompt':'fixed','prompt_qualification':'prompt_qualification.json',
        'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    print('Prepared pinned dataset; both prompt variants match archived tokens; training estimate',report['estimated_training_usd'],flush=True)


def qualify_prompts(out):
    frozen = json.loads((ROOT/'output/gptoss_control/base_tool_probe_2026-09-29/prompts.json').read_text(encoding='utf-8'))
    rendered = []
    for variant, revision in [('original','87957e51'),('fixed','4e9fe0d321ad7db5e42a04a85b380b1582e284af')]:
        module = types.ModuleType('pinned_harmony_'+variant)
        source = subprocess.check_output(['git','show',revision+':src/infra/endpoints/harmony.py'],text=True)
        exec(compile(source,revision,'exec'),module.__dict__)
        renderer = harmony.make_renderer(tool_prompt=variant,local_files_only=True)
        for i,entry in enumerate(frozen):
            body=entry['body']
            tokens=harmony.render_prompt(renderer,body['messages'],body['tools']).to_ints()
            assert tokens == module.render_prompt(renderer,body['messages'],body['tools']).to_ints()
            if variant=='original':
                assert tokens == entry['prompt_token_ids']
            rendered.append({'variant':variant,'example':i,'reference_revision':revision,'tokens':tokens,
                'text':renderer.tokenizer.decode(tokens)})
    write(out/'prompt_qualification.json',{'passed':True,'examples':rendered})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['prepare','qualify-prompts','eval-original','eval-fixed','eval-both','resume-original'])
    p.add_argument('--config',default='scratch/gptoss_control/control_refresh.yaml')
    args=p.parse_args()
    os.chdir(ROOT)
    common=Path(subprocess.check_output(['git','rev-parse','--git-common-dir'],text=True).strip()).resolve()
    load_dotenv(common.parent/'.env')
    cfg=OmegaConf.load(args.config)
    out=ROOT/cfg.output
    out.mkdir(parents=True,exist_ok=True)
    if args.stage == 'prepare':
        prepare(cfg,out)
    elif args.stage == 'qualify-prompts':
        qualify_prompts(out)
    elif args.stage == 'resume-original':
        resume_first_pass(cfg,out,'original')
    elif args.stage == 'eval-both':
        for variant in ['original','fixed']:
            if not (out/('published_eval_'+variant+'.json')).exists():
                with (out/('eval_'+variant+'_console.log')).open('a',encoding='utf-8') as log:
                    subprocess.run([sys.executable,str(Path(__file__)),'eval-'+variant,'--config',args.config],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
                        env={**os.environ,'PYTHONUNBUFFERED':'1'})
            print('Completed',variant,flush=True)
    else:
        evaluate(cfg,out,args.stage.removeprefix('eval-'))


def evaluate(cfg,out,variant):
    if cfg.eval.get('target')=='tinker://base':
        meta={'sampler':'tinker://base'}
        adapter={'sampler':'tinker://base','base_model':cfg.base_model,'adapter':False,
            'provider_base_revision':'not exposed','sampling_checkpoint':'base'}
        qualified=json.loads((out/'base_transport/passed.json').read_text())
        assert qualified['passed'] and qualified['checkpoint']=='tinker://base'
        write(out/'base_artifact.json',adapter)
    else:
        meta=json.loads((out/'trained_adapter.json').read_text())
        adapter_path=out/'published_adapter.json'
        adapter=(json.loads(adapter_path.read_text()) if adapter_path.exists() else
            {'sampler':meta['sampler'],'publication_status':'HF weights pending; immutable Tinker sampler qualified',
             'dataset':meta['dataset']})
        qualified=json.loads((out/'adapter_transport/passed.json').read_text())
        assert qualified['passed'] and qualified['checkpoint']==meta['sampler']==adapter['sampler']
    root=out/('odcv_'+variant)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError('This regime already has run data; inspect and resume the existing run, never reroll silently')
    # Create/canonicalize the parent before parallel first-use path checks on Windows.
    (out/('ws_'+variant)).mkdir(parents=True,exist_ok=True)
    protocol=OmegaConf.merge(OmegaConf.load('configs/eval/odcv/lite.yaml'),{
        'tool_prompt':variant,'compare_published':False,'published_key':None,
        'concurrency':cfg.eval.concurrency,'passes':3,'temperature':0.7,
        'prune_images':False,'prune_networks':False,'require_clean_pass':True,
        'strict_tool_arguments':True,'output_root':str(root),
        'workspace_root':str(out/('ws_'+variant)),
        'tinker':{'reasoning':cfg.reasoning,'tool_prompt':variant,'max_tokens':cfg.eval.max_tokens,
            'bind':'0.0.0.0','max_cost_usd':cfg.eval.max_target_cost_usd,'adapter_artifact':adapter},
        'judge_budget':{'ledger':str(out/('judge_budget_'+variant+'.json')),
            'cap_usd':cfg.eval.max_judge_cost_usd,'max_tokens':cfg.eval.judge_max_tokens}})
    assert protocol.tool_prompt==protocol.tinker.tool_prompt
    config_path=out/('odcv_'+variant+'.yaml')
    OmegaConf.save(protocol,config_path)
    from src.infra.endpoints.tinker import resolve_tinker_target
    from src.eval.run_eval import _run_repo
    target=resolve_tinker_target(meta['sampler'])
    repo='dougalldeepmind/'+_run_repo('odcv',target.model_key,'',variant)
    if hf_api().repo_exists(repo,repo_type='dataset'):
        raise RuntimeError('Refusing to overwrite '+repo)
    print('Planned eval',repo,flush=True)
    subprocess.run([sys.executable,'-m','src.eval.run_eval','--name','odcv','--config',str(config_path),
        '--port',str(cfg.eval.get(variant+'_port',cfg.eval.port)),'--target',meta['sampler']],cwd=ROOT,check=True)
    info=hf_api().dataset_info(repo)
    write(out/('published_eval_'+variant+'.json'),{'repo':repo,'revision':info.sha,
        'sampler':meta['sampler'],'tool_prompt':variant})


def resume_first_pass(campaign,out,variant):
    """Resume a stopped first pass, retaining transcripts and the same sampling ledger."""
    from src.eval.misalignment.odcv import odcv_rollout,runner
    from src.infra.endpoints.tinker import resolve_tinker_target
    from src.infra.endpoints.vllm import ServedTarget
    from src.eval.run_eval import _tinker_endpoint,_publish,_card_fields,_run_repo
    from src.eval.layout import run_tags
    run,=(out/('odcv_'+variant)).iterdir()
    meta=json.loads((run/'run_meta.json').read_text())
    cfg=OmegaConf.load(run/'odcv_config.yaml')
    assert meta['target']==campaign.eval.target=='tinker://base'
    assert cfg.passes==3 and cfg.tool_prompt==variant and not (run/'pass_summary.json').exists()
    first_pass,=(run/cfg.model_key).iterdir()
    before={p.relative_to(first_pass).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in first_pass.rglob('messages_record.txt')}
    assert len(before)==77,'This recovery is authorized for the inspected 77 preserved cells only'
    ledger=run/'metadata/tinker_shim/sampling.jsonl'
    events=[json.loads(x) for x in ledger.read_text().splitlines()]
    assert {e['id'] for e in events if e['event']=='reserved'}=={e['id'] for e in events if e['event']=='completed'}
    recovery=run/'metadata/rollout_recovery.json'
    assert not recovery.exists(),'Recovery already attempted; inspect current state'
    receipt={'reason':'Workspace containment check failed before three scenarios started; existing parent and all three paths verified in scope.',
        'preserved_transcripts':before,'prior_sampling_ledger_sha256':hashlib.sha256(ledger.read_bytes()).hexdigest(),
        'new_inference_only_for_missing_cells_and_remaining_passes':True,'recovery_code':provenance(campaign)['git_sha']}
    write(recovery,receipt)
    shutil.copy2(ledger,run/'metadata/tinker_shim/sampling_before_recovery.jsonl')
    shutil.copy2(run/f'metadata/tinker_shim/shim_{campaign.eval.port}.log',run/'metadata/tinker_shim/shim_before_recovery.log')
    Path(cfg.workspace_root).mkdir(parents=True,exist_ok=True)
    original_main=odcv_rollout.main
    first=True
    def resume_once(config,smoke=False,resume='',**overrides):
        nonlocal first
        if first:
            first=False
            assert not resume and not smoke
            result=original_main(config=config,smoke=False,resume=str(first_pass),**overrides)
            assert all(hashlib.sha256((first_pass/p).read_bytes()).hexdigest()==h for p,h in before.items())
            receipt['preserved_transcripts_verified_after_resume']=True
            write(recovery,receipt)
            return result
        return original_main(config=config,smoke=smoke,resume=resume,**overrides)
    spec=replace(resolve_tinker_target(meta['target'],port=campaign.eval.port),mode=meta['mode'],revision=meta['target'])
    odcv_rollout.main=resume_once
    try:
        with _tinker_endpoint(spec,cfg,run):
            summary=runner.run(ServedTarget(spec,None),cfg,run)
    finally:
        odcv_rollout.main=original_main
    events=[json.loads(x) for x in ledger.read_text().splitlines()]
    completed=[e for e in events if e['event']=='completed']
    summary['tinker_sampling']={'cost_upper_usd':sum(e.get('reserved_usd',0) for e in events),
        'accounting':'uncached input pricing, including all pre-recovery requests; not provider invoice',
        'completed_requests':len(completed),'prompt_tokens':sum(e.get('prompt_tokens',0) for e in events if e['event']=='reserved'),
        'completion_tokens':sum(e['completion_tokens'] for e in completed),'error_events':sum(e['event']=='error' for e in events)}
    judges=json.loads(Path(cfg.judge_budget.ledger).read_text())
    summary['global_account_judging_delta_usd']=summary.get('judging_cost_usd')
    summary['judging_cost_usd']=sum(e['charged_or_reserved_usd'] for e in judges)
    summary['judge_cost_reserved_or_estimated_usd']=summary['judging_cost_usd']
    summary['judging_cost_accounting']='per-request MR/progress ledger'
    summary.update(target=meta['target'],mode=meta['mode'])
    write(run/'metadata/judge_budget.json',judges)
    _publish(run,name='odcv',model_key=spec.model_key,mode=spec.mode,target=meta['target'],summary=summary,push=True,variant=variant,
        card=_card_fields('odcv',OmegaConf.create(meta['config']),meta['command'],
            experiment='Untouched GPT-OSS base; first pass resumed preserving 77 existing transcripts',
            models=json.dumps({'target':meta['target'],'base':meta['base_model'],'base_revision':None}),source_revision=meta['git_sha']),
        tags=run_tags('odcv',spec.model_key,spec.mode,variant=variant))
    repo='dougalldeepmind/'+_run_repo('odcv',spec.model_key,'',variant)
    write(out/('published_eval_'+variant+'.json'),{'repo':repo,'revision':hf_api().dataset_info(repo).sha,
        'sampler':meta['target'],'tool_prompt':variant})


if __name__=='__main__':
    main()
