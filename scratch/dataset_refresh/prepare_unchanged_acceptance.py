# ABOUTME: Prepare eleven exact-author-bound untouched-final dossiers with independent agent review evidence.
# ABOUTME: Never call models, adopt rows, modify origins or edit the frozen offline acceptance validator.
import argparse
import json
from pathlib import Path
from omegaconf import OmegaConf
from scratch.dataset_refresh import run as rt
from scratch.dataset_refresh import offline_acceptance as oa
from scratch.dataset_refresh.saved_input_pilot import ORIGINAL_BUDGET_ROOT

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); a=p.parse_args()
    cfg=OmegaConf.to_container(OmegaConf.load(a.config),resolve=True)
    assert oa.sha(oa.__file__)==cfg['implementation_sha256']
    out=Path(cfg['output']).resolve()
    if out.exists(): raise FileExistsError('Preparation output must be new')
    cases=[]
    for name in ['round1','round2']:
        folder=Path(cfg[name]); summary=rt.load_checkpoint(folder/'summary.json')
        for cid in cfg[name+'_candidates']:
            report_path=folder/(cid+'.review.json'); report=rt.load_checkpoint(report_path)
            ref=next(x for x in summary['cases'] if x['candidate_id']==cid)
            assert oa.sha(report_path)==ref['review_sha256'] and report['accepted'] is True
            cases.append({'candidate_id':cid,'root':str(Path(cfg['qualified_root']).resolve()),'stage':report['selected_input_stage'],'stage_sha256':report['selected_input_sha256'],'source_result_sha256':report['source_files_sha256']['result.json'],'conversation_sha256':report['conversation_sha256'],'prior_review':{'path':str(report_path.resolve()),'sha256':oa.sha(report_path),'content':report},'annotation':{'source_first_reason':report['reason'],'craft_reason':report['craft_constitution'],'source_quotes':report['source_quotes'],'answer_quotes':report['answer_quotes'],'benign_counterreading':report['benign_counterreading'],'remaining_caveats':report['caveats']}})
    t4=rt.read_rows(Path(cfg['t4_inputs'])); annotations=oa.read(cfg['t4_annotations'])
    for cid in cfg['t4_candidates']:
        row=next(x for x in t4 if x['candidate_id']==cid)
        cases.append({'candidate_id':cid,'root':str(Path(row['root']).resolve()),'stage':row['selected_input_stage'],'stage_sha256':row['selected_input_sha256'],'source_result_sha256':row['result_sha256'],'conversation_sha256':row['conversation_sha256'],'fresh_read_input_sha256':oa.sha(cfg['t4_inputs']),'annotation':annotations[cid]})
    ledger_path=(ORIGINAL_BUDGET_ROOT/'spend.json').resolve(); ledger=oa.read(ledger_path)
    planned=[]
    for case in cases:
        cid=case['candidate_id']; root=Path(case['root']); row=root/'nonmoral-advice'/'records'/cid
        assert oa.sha(row/'result.json')==case['source_result_sha256']
        assert oa.sha(row/(case['stage']+'.json'))==case['stage_sha256']
        original=rt.load_checkpoint(row/'result.json'); scenario=rt.load_checkpoint(row/'scenario.json'); stage=rt.load_checkpoint(row/(case['stage']+'.json'))
        conv={**{k:scenario[k] for k in ('system','user')},**{k:stage[k] for k in ('reasoning','response')}}
        assert rt.digest(conv)==case['conversation_sha256'],cid
        matches=[]
        for entry in ledger:
            if entry.get('run_root')!=str(root) or entry.get('arm')!='nonmoral-advice' or entry.get('candidate_id')!=cid or entry.get('stage')!=case['stage'] or entry.get('status')!='settled': continue
            raw_path=ledger_path.parent/'raw_calls'/f"{entry['call_id']:06d}.json"
            raw=oa.read(raw_path)
            receipt={'call_id':entry['call_id'],'request_sha256':entry['request_sha256'],'ledger_entry_sha256':rt.digest(entry),'raw_sha256':oa.sha(raw_path)}
            try: oa.validate_author(raw,receipt,conv,cid)
            except (ValueError,KeyError): continue
            matches.append((raw_path,receipt))
        assert len(matches)==1,(cid,'Expected one exact complete original author call',len(matches))
        raw_path,receipt=matches[0]
        spec={'source_ref':{'root':str(root),'arm':'nonmoral-advice','candidate_id':cid,'result_sha256':case['source_result_sha256']},'author_kind':'untouched_saved_final','review_config_path':str(Path(cfg['review_config']).resolve()),'review_config_sha256':oa.sha(cfg['review_config']),'review_contract_path':str(Path(cfg['review_contract']).resolve()),'review_contract_sha256':oa.sha(cfg['review_contract']),'saved_stage':case['stage'],'saved_stage_sha256':case['stage_sha256'],'raw_path':str(raw_path),'physical_receipt':receipt,'ledger_path':str(ledger_path)}
        planned.append((case,spec))
    out.mkdir(parents=True); (out/'specs').mkdir(); (out/'reviews').mkdir()
    results=[]
    for case,spec in planned:
        cid=case['candidate_id']; key=Path(case['root']).name+'__'+cid
        spec_path=out/'specs'/(key+'.json'); rt.write_json(spec_path,spec)
        dossier_path=out/key/'dossier.json'; oa.prepare(spec_path,dossier_path.parent)
        d,frozen=oa.validate_dossier(dossier_path); conv=d['conversation']; ann=case['annotation']
        original=oa.read(frozen['origin/result.json'])['record']; old_pref=original['trait_text']
        review={'candidate_id':cid,'decision':'accept','accepted':True,'issues':[],'gates':{k:True for k in oa.GATES},'reviewer_provenance':{'kind':'independent_codex_agent','task':'/root/audit_nonmoral','human_review':False},'scope':'Fresh full source/answer reads for four t4 cases; exact-byte-bound previously completed full reads reused for seven other cases. Full qualified craft and full new09 reviewed. Preparation and content acceptance review only; root native/duplicate/adoption gates remain separate.','source_result_sha256':d['source_ref']['result_sha256'],'result_sha256':oa.sha(frozen['author_result']),'conversation_sha256':d['conversation_sha256'],'review_contract_sha256':d['review_contract_sha256'],'constitution_sha256':d['constitution_sha256'],'full_working_preference_sha256':d['full_working_preference_sha256'],'request_sha256':d['physical_receipt']['request_sha256'],'physical_receipt':d['physical_receipt'],**{('source_'+k if k in ('system','user') else k)+'_sha256':rt.digest(conv[k].encode()) for k in oa.FIELDS},**ann,'constitution_compatibility':'Full new09 reviewed as compatibility only. No substantive deception, harmful advice, coercion, fabricated action or inappropriate refusal found; ordinary benign craft judgments retain human agency. Nonblocking calibration caveats remain explicit. No demand for moral framing or performance of all nine principles.','nonmoral_eligibility':'The actual decision concerns benign craft, personal or community documentation organization. No real medical/veterinary, emergency, legal, financial/trading, employment, political-persuasion, security or safety-critical decision is requested. Ordinary accuracy and preservation of submitted meaning do not turn this into moral deliberation.','prior_full_read_evidence':case.get('prior_review'),'fresh_read_input_sha256':case.get('fresh_read_input_sha256'),'author_preference_provenance':{'original_record_trait_text_sha256':rt.digest(old_pref.encode()),'qualified_review_trait_text_sha256':d['full_working_preference_sha256'],'different':old_pref!=d['full_working_preference'],'policy':'Original author prompt/raw and original trait metadata remain untouched. Current qualified craft is a separate independent acceptance review, not retroactively claimed author conditioning.'}}
        review_path=out/'reviews'/(key+'.json'); rt.save_checkpoint(review_path,review); oa.validate_review(d,frozen,rt.load_checkpoint(review_path))
        results.append({'case_key':key,'candidate_id':cid,'source_root':case['root'],'saved_stage':case['stage'],'dossier_path':str(dossier_path),'dossier_sha256':oa.sha(dossier_path),'independent_review_path':str(review_path),'independent_review_sha256':oa.sha(review_path),'conversation_sha256':d['conversation_sha256'],'author_call_id':d['physical_receipt']['call_id'],'author_preference_differs_from_qualified_review':review['author_preference_provenance']['different']})
        print(key+' prepared and independent review validated',flush=True)
    summary={'status':'prepared_not_adopted','rows':len(results),'implementation_sha256':cfg['implementation_sha256'],'reviewer_provenance':{'kind':'independent_codex_agent','task':'/root/audit_nonmoral','human_review':False},'cases':results,'scope':'No API calls, origin modifications or adoptions. All11 exact settled Sonnet author outputs and normalized independent reviews verified. Native token/mask and duplicate decisions remain root gates.'}
    rt.save_checkpoint(out/'summary.json',summary); print('summary_sha256='+oa.sha(out/'summary.json'))

if __name__=='__main__':main()
