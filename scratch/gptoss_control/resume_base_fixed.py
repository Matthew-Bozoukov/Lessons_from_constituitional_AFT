# ABOUTME: Retain the base fixed-prompt arm's 160 trajectories, including one terminal model-format failure.
# ABOUTME: Verify saved raw completions, run only pass three, then use standard judges and publication.
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys

from dotenv import load_dotenv
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
common=Path(subprocess.check_output(['git','rev-parse','--git-common-dir'],text=True).strip()).resolve()
load_dotenv(common.parent/'.env')
from src.eval.misalignment.odcv import odcv_rollout,runner
from src.infra.endpoints.harmony import make_renderer
from src.infra.endpoints.tinker import resolve_tinker_target
from src.infra.endpoints.vllm import ServedTarget
from src.eval.run_eval import _tinker_endpoint,_publish,_card_fields,_run_repo
from src.eval.layout import run_tags
from src.infra.huggingface import hf_api


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2),encoding='utf-8')


def main():
    campaign=OmegaConf.load(ROOT/'scratch/gptoss_control/base_odcv.yaml')
    out=ROOT/campaign.output
    run,=(out/'odcv_fixed').iterdir()
    cfg=OmegaConf.load(run/'odcv_config.yaml')
    meta=json.loads((run/'run_meta.json').read_text())
    passes=sorted((run/cfg.model_key).iterdir())
    assert [p.name for p in passes]==['20260930_221203','20260930_222029']
    assert cfg.passes==3 and cfg.tool_prompt=='fixed' and meta['target']=='tinker://base'
    before={f'pass{i}/'+p.relative_to(d).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for i,d in enumerate(passes,1) for p in d.rglob('messages_record.txt')}
    assert len(before)==160
    recovery=run/'metadata/rollout_recovery.json'
    assert not recovery.exists(),'Inspect state before any repeated recovery'
    ledger=run/'metadata/tinker_shim/sampling.jsonl'
    events=[json.loads(x) for x in ledger.read_text().splitlines()]
    completed={e['id']:e for e in events if e['event']=='completed'}
    reserved={e['id']:e for e in events if e['event']=='reserved'}
    assert set(reserved)==set(completed), 'Unsettled paid requests require inspection'
    ids=['5e1711e8f68640f6941765703ed155d8','e57ec454dda346199f0005f63fafdc46','87357dc1df3048138e105071e722d19d']
    renderer=make_renderer(local_files_only=True)
    evidence=[]
    for request in ids:
        assert any(e['event']=='error' and e['id']==request and e['type']=='ValueError' for e in events)
        tokens=completed[request]['raw_tokens']
        parsed,term=renderer.parse_response(tokens)
        calls=renderer.to_openai_message(parsed)['tool_calls']
        assert term.is_stop_sequence and tokens[-1]==200002
        assert len(calls)==1 and calls[0]['function']['name']=='task_complete'
        arguments=json.loads(calls[0]['function']['arguments'])
        assert '100.0' in arguments['reason'] and 'moderation report' in arguments['reason']
        assert reserved[request]['prompt_tokens']==11022 and reserved[request]['checkpoint']=='base'
        evidence.append({'request_id':request,'tokens_sha256':hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            'decoded_output':renderer.tokenizer.decode(tokens),'arguments':arguments,'ending_token':tokens[-1]})
    relative='agent_logs/gptoss120b-incentivized/experiments/Ai-Moderator-Content-Censorship-Backdoor/docker_output.log'
    log=passes[1]/relative
    receipt={'cause':'tool_calls_ended_without_handoff','docker_log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
        'rejected_request_ids':ids,'evidence':evidence,'disposition':'Keep the existing 47-step trajectory; no accepted task_complete, no further inference for this cell.'}
    cfg.model_output_failure_receipts={passes[1].name+'/'+relative:receipt}
    for p in passes:
        invalid,known=runner.audit_terminal_errors(p,cfg.model_output_failure_receipts)
        assert not invalid
        assert len(known)==(1 if p==passes[1] else 0)
    report={'reason':'Pass two stopped at clean-pass audit after three completed task_complete outputs used return rather than handoff. Retained as a model-format failure, not rerolled.',
        'preserved_transcripts':before,'path_format':'passN/agent_logs/model-condition/experiments/scenario/messages_record.txt',
        'prior_sampling_ledger_sha256':hashlib.sha256(ledger.read_bytes()).hexdigest(),
        'terminal_model_output_failure':receipt,'new_inference_only_for_pass_three':True,
        'recovery_code':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    write(recovery,report)
    shutil.copy2(run/'pass_summary.json',run/'metadata/pass_summary_before_recovery.json')
    shutil.copy2(ledger,run/'metadata/tinker_shim/sampling_before_recovery.jsonl')
    shutil.copy2(run/'metadata/tinker_shim/shim_18343.log',run/'metadata/tinker_shim/shim_before_recovery.log')
    original=odcv_rollout.main
    used=0
    def reuse(config,smoke=False,resume='',**overrides):
        nonlocal used
        if used<2:
            assert not smoke and not resume
            d=passes[used]; used+=1
            assert all(hashlib.sha256((d/k.split('/',1)[1]).read_bytes()).hexdigest()==h
                for k,h in before.items() if k.startswith(f'pass{used}/'))
            if used==2:
                report['preserved_transcripts_verified_after_resume']=True
                write(recovery,report)
            print('Reusing unchanged pass:',d,flush=True)
            return d
        return original(config=config,smoke=smoke,resume=resume,**overrides)
    spec=replace(resolve_tinker_target(meta['target'],port=campaign.eval.fixed_port),mode=meta['mode'],revision=meta['target'])
    odcv_rollout.main=reuse
    try:
        with _tinker_endpoint(spec,cfg,run):
            summary=runner.run(ServedTarget(spec,None),cfg,run)
    finally:
        odcv_rollout.main=original
    publish(out,run,cfg,meta,summary,spec)


def publish(out,run,cfg,meta,summary,spec):
    ledger=run/'metadata/tinker_shim/sampling.jsonl'
    events=[json.loads(x) for x in ledger.read_text().splitlines()]
    completed=[e for e in events if e['event']=='completed']
    summary['tinker_sampling']={'cost_upper_usd':sum(e.get('reserved_usd',0) for e in events),
        'accounting':'uncached input pricing including pre-recovery requests; not provider invoice',
        'completed_requests':len(completed),'prompt_tokens':sum(e.get('prompt_tokens',0) for e in events if e['event']=='reserved'),
        'completion_tokens':sum(e['completion_tokens'] for e in completed),'error_events':sum(e['event']=='error' for e in events)}
    judges=json.loads(Path(cfg.judge_budget.ledger).read_text())
    summary['global_account_judging_delta_usd']=summary.get('judging_cost_usd')
    summary['judging_cost_usd']=sum(e['charged_or_reserved_usd'] for e in judges)
    summary['judge_cost_reserved_or_estimated_usd']=summary['judging_cost_usd']
    summary['judging_cost_accounting']='per-request MR/progress ledger'
    summary.update(target=meta['target'],mode=meta['mode'])
    write(run/'metadata/judge_budget.json',judges)
    _publish(run,name='odcv',model_key=spec.model_key,mode=spec.mode,target=meta['target'],summary=summary,push=True,variant='fixed',
        card=_card_fields('odcv',OmegaConf.create(meta['config']),meta['command'],
            experiment='Untouched GPT-OSS base; completed trajectories retained including audited terminal model-format failures',
            models=json.dumps({'target':meta['target'],'base':meta['base_model'],'base_revision':None}),source_revision=meta['git_sha']),
        tags=run_tags('odcv',spec.model_key,spec.mode,variant='fixed'))
    repo='dougalldeepmind/'+_run_repo('odcv',spec.model_key,'','fixed')
    write(out/'published_eval_fixed.json',{'repo':repo,'revision':hf_api().dataset_info(repo).sha,'sampler':meta['target'],'tool_prompt':'fixed'})


if __name__=='__main__':
    main()
