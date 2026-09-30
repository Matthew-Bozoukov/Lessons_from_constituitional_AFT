# ABOUTME: Adjudicate three additional terminal Harmony failures from saved completions, then score all 240 cells.
# ABOUTME: No Tinker endpoint or sampler is started; every transcript and the complete paid ledger are preserved.
import hashlib
import json
from pathlib import Path
import shutil

from omegaconf import OmegaConf
from resume_base_fixed import ROOT,write,publish,runner,odcv_rollout,make_renderer,resolve_tinker_target,ServedTarget


def main():
    campaign=OmegaConf.load(ROOT/'scratch/gptoss_control/base_odcv.yaml')
    out=ROOT/campaign.output
    run,=(out/'odcv_fixed').iterdir()
    cfg=OmegaConf.load(run/'odcv_config.yaml')
    meta=json.loads((run/'run_meta.json').read_text())
    passes=sorted((run/cfg.model_key).iterdir())
    assert [p.name for p in passes]==['20260930_221203','20260930_222029','20260930_223605']
    ledger=run/'metadata/tinker_shim/sampling.jsonl'
    ledger_hash=hashlib.sha256(ledger.read_bytes()).hexdigest()
    events=[json.loads(x) for x in ledger.read_text().splitlines()]
    completed={e['id']:e for e in events if e['event']=='completed'}
    reserved={e['id']:e for e in events if e['event']=='reserved'}
    assert set(reserved)==set(completed)
    renderer=make_renderer(local_files_only=True)
    groups=[('mandated','Drone-Delivery-Optimizer',10,
        ['deef98f4dd65478490c02c2d15043c5a','6f53f18856d84d6e834860c28fea5028','f411fde1096142bdabcc72b5a9645dd3']),
        ('mandated','Marketing-Campaign',14,
        ['53aa1814f90b43868bd6531a0c55e62d','cae4e3c27fd34551a9bbe8f8e8e64de2','1f1942dd873a43edaf4cfeaa882fbf17']),
        ('incentivized','Ai-Moderator-Content-Censorship-Backdoor',47,
        ['3fa5f7280a944c2bb71871b8309bceb5','50cf051956414330ae8fd238e0fc58bc','2635e03a695243c2bfd6487a1fe90fa0'])]
    for condition,scenario,cycle,ids in groups:
        evidence=[]
        assert len({reserved[i]['prompt_tokens'] for i in ids})==1
        for request in ids:
            assert any(e['event']=='error' and e['id']==request and e['type']=='ValueError' for e in events)
            tokens=completed[request]['raw_tokens']
            parsed,term=renderer.parse_response(tokens)
            calls=renderer.to_openai_message(parsed)['tool_calls']
            assert term.is_stop_sequence and tokens[-1]==200002
            assert len(calls)==1 and calls[0]['function']['name']=='task_complete'
            arguments=json.loads(calls[0]['function']['arguments'])
            assert isinstance(arguments,dict) and reserved[request]['checkpoint']=='base'
            evidence.append({'request_id':request,'tokens_sha256':hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                'decoded_output':renderer.tokenizer.decode(tokens),'arguments':arguments,'ending_token':tokens[-1]})
        relative=f'agent_logs/gptoss120b-{condition}/experiments/{scenario}/docker_output.log'
        log=passes[2]/relative
        assert f'=== CYCLE {cycle} / 50 ===' in log.read_text()
        cfg.model_output_failure_receipts[passes[2].name+'/'+relative]={
            'cause':'tool_calls_ended_without_handoff','docker_log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
            'rejected_request_ids':ids,'evidence':evidence,'cycle':cycle,'pass':3,'condition':condition,'scenario':scenario,
            'disposition':'Keep existing actions and score as not submitted; do not rerun or continue.'}
    for i,d in enumerate(passes):
        invalid,known=runner.audit_terminal_errors(d,cfg.model_output_failure_receipts)
        assert not invalid and len(known)==[0,1,3][i]
    recovery=run/'metadata/rollout_recovery.json'
    backup=run/'metadata/rollout_recovery_before_scoring.json'
    assert not backup.exists(),'Scoring recovery already attempted; inspect caches'
    shutil.copy2(recovery,backup)
    report=json.loads(recovery.read_text())
    report['terminal_model_output_failures']=OmegaConf.to_container(cfg.model_output_failure_receipts)
    report['reason']='All 240 trajectories retained. Four exhausted task_complete retries with wrong Harmony ending; these are scored model failures, not infrastructure outages. No model rollout was regenerated.'
    report['preserved_transcripts']={f'pass{i}/'+p.relative_to(d).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for i,d in enumerate(passes,1) for p in d.rglob('messages_record.txt')}
    assert len(report['preserved_transcripts'])==240
    report['scoring_recovery_sampling_ledger_sha256']=ledger_hash
    write(recovery,report)
    original=odcv_rollout.main
    used=0
    def reuse(config,smoke=False,resume='',**overrides):
        nonlocal used
        assert not smoke and not resume and used<3
        d=passes[used];used+=1
        assert all(hashlib.sha256((d/k.split('/',1)[1]).read_bytes()).hexdigest()==h
            for k,h in report['preserved_transcripts'].items() if k.startswith(f'pass{used}/'))
        return d
    spec=resolve_tinker_target('tinker://base',port=campaign.eval.fixed_port)
    odcv_rollout.main=reuse
    try:
        summary=runner.run(ServedTarget(spec,None),cfg,run)
    finally:
        odcv_rollout.main=original
    assert hashlib.sha256(ledger.read_bytes()).hexdigest()==ledger_hash
    report['sampling_ledger_unchanged_during_scoring']=True
    write(recovery,report)
    publish(out,run,cfg,meta,summary,spec)


if __name__=='__main__':
    main()
