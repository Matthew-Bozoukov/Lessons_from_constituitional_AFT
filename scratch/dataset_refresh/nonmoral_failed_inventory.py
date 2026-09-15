# ABOUTME: Inventories remaining nonmoral technical failures and valid quality rejections without changing them.
# ABOUTME: Freezes eight complete-answer candidates for a bounded no-API human recovery assessment.
import argparse
import json
from collections import Counter
from pathlib import Path
from omegaconf import OmegaConf
from scratch.dataset_refresh import run as runtime

SELECTED=['t2_007_v1','t6_065_v0','t9_054_v0','t5_051_v0','t8_066_v0','t3_040_v0','t4_071_v0','t1_038_v0']

def main():
    parser=argparse.ArgumentParser(description='Freeze a new failed-terminal inventory without modifying source evidence.')
    parser.add_argument('--config',required=True,help='YAML with run_root, output_dir and budget_root.')
    args=parser.parse_args()
    config=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
    ROOT=Path(config['run_root']).resolve()
    ARM=ROOT/'nonmoral-advice'
    OUT=Path(config['output_dir']).resolve()
    BUDGET=Path(config['budget_root']).resolve()
    if any((OUT/name).exists() for name in ['remaining_terminal_inventory.json','remaining_terminal_inventory.receipt.json']):
        raise FileExistsError('Refusing to overwrite frozen terminal inventory')
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ARM/'config.json').read_text(encoding='utf8'))
    ledger=json.loads((BUDGET/'spend.json').read_text(encoding='utf8'))
    calls={}
    for e in ledger:
        if e.get('run_root') and Path(e['run_root']).resolve()==ROOT and e.get('arm')=='nonmoral-advice':
            calls.setdefault(e.get('candidate_id'),[]).append(e)
    rows=[]
    for p in sorted(ARM.glob('records/*/result.json')):
        result=runtime.load_checkpoint(p)
        if result['status']=='accepted': continue
        record=result['record']; cid=p.parent.name
        complete=all(isinstance(record.get(k),str) and record[k].strip() for k in ['system','user','reasoning','response'])
        pre=p.parent/'preflight.json'
        eligible=pre.exists() and runtime.acceptance(runtime.load_checkpoint(pre),cfg['preflight'])
        held=(p.parent/'independent_exclusion.json').exists()
        stage_files={}
        for name in ['scenario','preflight','revise_responses','repair_1','grounding_0','grounding_1','review_0','review_1']:
            q=p.parent/(name+'.json')
            if q.exists():
                runtime.load_checkpoint(q)
                stage_files[name]=runtime.digest(q.read_bytes())
        call=calls.get(cid,[])[-1] if calls.get(cid) else None
        raw_summary=None
        if call:
            rawp=BUDGET/f"raw_calls/{call['call_id']:06d}.json"
            raw=json.loads(rawp.read_text(encoding='utf8'))
            response=raw.get('response',{})
            raw_summary={'path':str(rawp),'sha256':runtime.digest(rawp.read_bytes()),'call_id':call['call_id'],'stage':call['stage'],'accounting_status':call['status'],'finish_reason':response.get('finish_reason'),'content_characters':len(response.get('content') or ''),'request_sha256':call.get('request_sha256')}
        category=('existing_independent_hold_out_of_scope' if held else 'complete_author_technical_review_failure' if complete and eligible and result['status']=='failed' else 'valid_adverse_content_review' if complete and result.get('rejection_stage')=='content_reviews' else 'source_ineligible_or_no_complete_final')
        rows.append({'candidate_id':cid,'result_path':str(p),'result_sha256':runtime.digest(p.read_bytes()),'status':result['status'],'error':result.get('error'),'rejection_stage':result.get('rejection_stage'),'complete_final_fields':complete,'preflight_accepted':eligible,'local_checks':runtime.simple_checks(record) if complete else None,'independent_hold_exists':held,'category':category,'stage_sha256':stage_files,'latest_call':raw_summary,'selected_for_full_read':cid in SELECTED})
    selected=[next(r for r in rows if r['candidate_id']==cid) for cid in SELECTED]
    assert all(r['category']=='complete_author_technical_review_failure' for r in selected)
    report={'scope':'All remaining failed/rejected qualified nonmoral terminals; independent holds are inventoried but excluded from new recovery scope. Complete fields and local checks do not certify content. No source mutation/API calls. Missing-terminal pending candidates are outside this terminal inventory.','counts':dict(Counter(r['category'] for r in rows)),'technical_complete_per_trait':dict(Counter(r['candidate_id'].split('_')[0] for r in rows if r['category']=='complete_author_technical_review_failure')),'selection_reason':'Bounded eight across distinct available tensions: one malformed complete final review (t2_007_v1), one latest repaired answer (t6_065_v0), then eligible complete initial finals from other tensions. Selected before full answer inspection; no convenience substitution.','selected_ids':SELECTED,'rows':rows}
    runtime.save_checkpoint(OUT/'remaining_terminal_inventory.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    print('sha256='+runtime.digest((OUT/'remaining_terminal_inventory.json').read_bytes()))

if __name__=='__main__':main()
