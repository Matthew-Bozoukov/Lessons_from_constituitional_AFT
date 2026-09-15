# ABOUTME: Run one explicit final cleanup batch through the existing budgeted Sonnet client.
# ABOUTME: Freeze five source-preserving requests, retain all physical evidence, and never retry or accept automatically.
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
from omegaconf import OmegaConf
from dotenv import load_dotenv
from scratch.dataset_refresh import run as b, offline_acceptance as a, saved_input_pilot as pilot
from scratch.dataset_refresh.recover_short_draft import fields

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO/'output/2026-09-15_nonmoral_final_cleanup'
BUDGET = a.ORIGINAL_BUDGET_ROOT
IDS = {'t1_051_v0','t1_067_v0','t3_026_v0','t9_026_v0','t9_075_v0'}


def prepare(config):
    cfg=OmegaConf.to_container(OmegaConf.load(config),resolve=True)
    if ROOT.exists() or cfg['ceiling']!=270 or set(cfg['instructions'])!=IDS:
        raise ValueError('Require one fresh explicitly bounded five-case cleanup')
    ledger=a.read(BUDGET/'spend.json')
    if len(ledger)!=11682 or any(e['status'] not in ('settled','billing_verified_failure') for e in ledger):
        raise ValueError('Prior approved work must be closed at its exact ledger cutoff')
    preview=a.bound(cfg['preview'],cfg['preview_sha256']);rows=b.read_rows(preview)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    code=Path(__file__);frozen=subprocess.check_output(['git','show',commit+':'+code.relative_to(REPO).as_posix()])
    if frozen.replace(b'\r\n',b'\n')!=code.read_bytes().replace(b'\r\n',b'\n'):
        raise ValueError('Commit exact cleanup helper first')
    if b.provider_price(pilot.MODEL)!={'in':2.0,'out':10.0}:raise ValueError('Pinned model pricing changed')
    payloads=[]
    for cid,instruction in cfg['instructions'].items():
        row=next(r for r in rows if r['metadata'].get('original_scenario_id')==cid)
        ref={k:row['metadata']['origin'][k] for k in ('root','arm','candidate_id','result_sha256')}
        source=Path(ref['root'])/ref['arm']/'records'/cid
        result=b.load_checkpoint(a.bound(source/'result.json',ref['result_sha256']))
        source_cfg=b.validate_arm(ref['root'],ref['arm']);record=result['record'];conv={k:record[k] for k in a.FIELDS}
        if a.messages(conv)!=row['messages'] or not b.acceptance(b.load_checkpoint(source/'preflight.json'),source_cfg['preflight']):
            raise ValueError('Source differs or is ineligible')
        stage=source_cfg['response_stages'][-1]
        messages=[{'role':r,'content':b.render(stage['prompts'][r],fields(record,source_cfg))} for r in ('system','user')]
        messages[-1]['content']+='\n\nMake only the following factual cleanup in both reasoning and final. Preserve the useful existing advice; do not add premises or broaden the answer. Write standalone advice to this human with no mention of drafts, reviewers or these instructions.\n'+instruction
        request={'model':pilot.MODEL,'messages':messages,'temperature':.7,'max_tokens':8192}
        bound={str(p):a.sha(p) for p in [source/'result.json',source/'result.receipt.json',source/'scenario.json',source/'scenario.receipt.json',source/'preflight.json',source/'preflight.receipt.json',Path(ref['root'])/ref['arm']/'config.json']}
        for path,digest in cfg['review_evidence'].items():a.bound(path,digest);bound[path]=digest
        payloads.append({'candidate_id':cid,'case_key':a.origin_name(ref['root'])+'::'+cid,'source_ref':ref,'actual_conversation':conv,'full_working_preference':record['trait_text'],'request':request,'request_sha256':b.digest(request),'bound_files':bound,'root_cleanup_instruction':instruction,'automatic_acceptance':False})
    ROOT.mkdir()
    for i,payload in enumerate(payloads):b.save_checkpoint(ROOT/f'{i:02d}.input.json',payload)
    b.save_checkpoint(ROOT/'manifest.json',{'config_path':str(Path(config).resolve()),'config_sha256':a.sha(config),'source_commit':commit,'code_path':str(code),'code_sha256':a.sha(code),'baseline_count':len(ledger),'baseline_digest':b.digest(ledger),'ceiling':270,'workers':5,'calls':5,'automatic_retries':0,'paid_critics':0,'inputs':[{'path':f'{i:02d}.input.json','sha256':a.sha(ROOT/f'{i:02d}.input.json')} for i in range(5)]})
    b.write_json(ROOT/'dispatch.json',{'enabled':False,'manifest_sha256':a.sha(ROOT/'manifest.json')})
    return str(ROOT)


def execute():
    m=b.load_checkpoint(ROOT/'manifest.json');a.bound(m['code_path'],m['code_sha256']);a.bound(m['config_path'],m['config_sha256'])
    if a.read(ROOT/'dispatch.json')!={'enabled':True,'manifest_sha256':a.sha(ROOT/'manifest.json')} or (ROOT/'started.json').exists():
        raise ValueError('Dispatch disabled or batch already attempted; never retry')
    payloads=[b.load_checkpoint(a.bound(ROOT/e['path'],e['sha256'])) for e in m['inputs']]
    client=b.BudgetClient(BUDGET,250,{pilot.MODEL});client.ceiling=270
    with client.lock:
        ledger=client.entries()
        if len(ledger)!=m['baseline_count'] or b.digest(ledger)!=m['baseline_digest']:
            raise ValueError('Closed budget prefix changed')
        reserve=sum((1.25*(len(json.dumps(p['request']['messages'],ensure_ascii=False).encode())+2048)*2+8192*10)/1e6 for p in payloads)
        if sum(e['charged_or_reserved_usd'] for e in ledger)+reserve>270:raise ValueError('Whole batch exceeds ceiling')
        b.save_checkpoint(ROOT/'started.json',{'manifest_sha256':a.sha(ROOT/'manifest.json'),'reserved_batch_upper_usd':reserve})
    def one(pair):
        i,p=pair;item={'candidate_id':p['candidate_id'],'case_key':p['case_key'],'status':'failed','automatic_acceptance':False}
        try:
            for path,digest in p['bound_files'].items():a.bound(path,digest)
            client.local.run_root=str(ROOT);client.local.arm='nonmoral-final-cleanup';client.local.stage='single_saved_revision';client.local.candidate_id=p['case_key']
            response=client.chat(**p['request'])
            with client.lock:
                entries=[e for e in client.entries()[m['baseline_count']:] if e['run_root']==str(ROOT) and e['candidate_id']==p['case_key']]
            if len(entries)!=1:raise ValueError('Expected one physical call')
            e=entries[0];raw_path=BUDGET/'raw_calls'/f'{e["call_id"]:06d}.json';shutil.copyfile(raw_path,ROOT/f'{i:02d}.raw.json')
            receipt={'call_id':e['call_id'],'request_sha256':p['request_sha256'],'raw_sha256':a.sha(raw_path),'ledger_entry_sha256':b.digest(e)};item['physical_receipt']=receipt
            if response.response_model!=pilot.MODEL or response.finish_reason!='stop':raise ValueError('Incomplete or wrong-model response')
            parsed,audit=pilot.parsed_fields(response.content);conv={**p['actual_conversation'],**parsed}
            if b.simple_checks(conv):raise ValueError('Local output checks failed')
            a.validate_author(a.read(raw_path),receipt,conv,p['candidate_id'],p['request'],p['case_key'])
            item.update(status='awaiting_independent_full_review',conversation=conv,audit=audit)
        except BaseException as exc:item.update(error_type=type(exc).__name__,error=str(exc)[:1500])
        b.save_checkpoint(ROOT/f'{i:02d}.result.json',item);return {k:item[k] for k in ('candidate_id','status')}
    with ThreadPoolExecutor(max_workers=5) as pool:results=list(pool.map(one,enumerate(payloads)))
    summary={'results':results,'automatic_accepted_rows':0,'shared_exposure_usd':sum(e['charged_or_reserved_usd'] for e in client.entries())};b.save_checkpoint(ROOT/'summary.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','execute']);p.add_argument('--config');args=p.parse_args()
    load_dotenv(REPO/'.env')
    print(prepare(args.config) if args.action=='prepare' else execute())
