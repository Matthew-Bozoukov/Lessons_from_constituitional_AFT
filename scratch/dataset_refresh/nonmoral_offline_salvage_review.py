# ABOUTME: Freezes and records bounded independent Codex-agent reads of complete saved nonmoral answers.
# ABOUTME: Uses exact source and author-stage receipts without inference calls or original terminal mutations.
import argparse
import json
from collections import Counter
from pathlib import Path
from omegaconf import OmegaConf
from scratch.dataset_refresh import run as rt

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--action',choices=['prepare','show','record','finalize'],required=True)
    p.add_argument('--candidate')
    a=p.parse_args(); cfg=OmegaConf.to_container(OmegaConf.load(a.config),resolve=True)
    out=Path(cfg['output']).resolve(); out.mkdir(parents=True,exist_ok=True)
    selection_path=out/'selection.json'
    if a.action=='prepare':
        if selection_path.exists(): raise FileExistsError(selection_path)
        ip=Path(cfg['inventory']); rows=[json.loads(x) for x in ip.read_text(encoding='utf8').splitlines()]
        rows=[x for x in rows if x['classification']=='complete_final_candidate_needs_source_and_answer_review' and x['trait_id'] in cfg['priority']]
        excluded=set(cfg.get('excluded_candidates',[]))
        for prior in cfg.get('excluded_selections',[]):
            excluded.update(x['candidate_id'] for x in rt.load_checkpoint(Path(prior))['rows'])
        rows=[x for x in rows if x['candidate_id'] not in excluded]
        rows.sort(key=lambda x:(cfg['priority'].index(x['trait_id']),x['candidate_id']))
        rows=rows[:cfg['maximum']]
        rt.save_checkpoint(selection_path,{'reviewer':'independent_codex_agent /root/audit_nonmoral','sampling':'Purposive trait-priority inventory order; not random or representative. Frozen before new case reads.','inventory_sha256':rt.digest(ip.read_bytes()),'rows':rows})
        print([x['candidate_id'] for x in rows]); return
    selection=rt.load_checkpoint(selection_path)
    if a.action=='finalize':
        target=out/'summary.json'
        if any(x.exists() for x in [target,target.with_suffix('.receipt.json'),target.with_suffix('.md')]): raise FileExistsError(target)
        reviews=[]
        adapter_holds=[]
        for selected in selection['rows']:
            rp=out/(selected['candidate_id']+'.review.json'); review=rt.load_checkpoint(rp)
            folder=Path(review['source_folder'])
            assert rt.digest((folder/(review['selected_input_stage']+'.json')).read_bytes())==review['selected_input_sha256']
            for name,sha in review['source_files_sha256'].items(): assert rt.digest((folder/name).read_bytes())==sha
            try:
                rt.load_checkpoint(folder/'result.json')
            except Exception as exc:
                adapter_holds.append({'candidate_id':review['candidate_id'],'content_decision':review['decision'],'reason':'Current root adapter requires readable receipt-bound original terminal; complete source/author-stage review does not satisfy this separate provenance requirement.','error_type':type(exc).__name__,'error':str(exc)})
            reviews.append({'candidate_id':review['candidate_id'],'trait_id':review['trait_id'],'decision':review['decision'],'review_path':str(rp),'review_sha256':rt.digest(rp.read_bytes()),'selected_input_stage':review['selected_input_stage'],'selected_input_sha256':review['selected_input_sha256']})
        summary={'scope':f'{len(reviews)} purposively selected complete saved finals, all fully reread with actual source/system and both author answer blocks. Independent Codex-agent review, not human or paid-model grading. Proposed unchanged accepts require root adoption and remaining native/selection checks; no original status changed here. Repairable answers and source exclusions remain held. Not a representative quality rate or exhaustive duplication screen.','reviewer_provenance':{'kind':'independent_codex_agent','task':'/root/audit_nonmoral','human_review':False},'selection_sha256':rt.digest(selection_path.read_bytes()),'counts':dict(Counter(x['decision'] for x in reviews)),'per_trait':{trait:dict(Counter(x['decision'] for x in reviews if x['trait_id']==trait)) for trait in sorted({x['trait_id'] for x in reviews})},'cases':reviews,'materiality':'Definite wrong decisive source facts and constraints are held; ordinary scoped proposals and nondecisive rhetorical overstatements are separately disclosed rather than hidden or automatically rejected. Full craft/09 compatibility review does not require moral language or an arbitrary synthesis.'}
        summary['current_adapter_provenance_holds']=adapter_holds
        summary['unchanged_content_accepts_with_readable_terminal']=sum(x['decision']=='accept_unchanged' and x['candidate_id'] not in {h['candidate_id'] for h in adapter_holds} for x in reviews)
        rt.save_checkpoint(target,summary)
        target.with_suffix('.md').write_text('# Independent nonmoral offline salvage review\n\n'+summary['scope']+'\n\n'+json.dumps(summary['counts'])+'\n\n'+'\n'.join('- '+x['candidate_id']+': '+x['decision'] for x in reviews)+'\n',encoding='utf8')
        print(rt.digest(target.read_bytes())); return
    row=next(x for x in selection['rows'] if x['candidate_id']==a.candidate)
    folder=Path(cfg['root_parent'])/row['root']/row['arm']/'records'/a.candidate
    for name,sha in row['source_files_sha256'].items(): assert rt.digest((folder/name).read_bytes())==sha,(a.candidate,name)
    stage_path=folder/(row['selected_input_stage']+'.json')
    assert rt.digest(stage_path.read_bytes())==row['selected_input_sha256']
    scenario=rt.load_checkpoint(folder/'scenario.json'); stage=rt.load_checkpoint(stage_path)
    actual={k:scenario[k] for k in ['system','user']}; actual.update({k:stage[k] for k in ['reasoning','response']})
    if a.action=='show':
        print(json.dumps({'candidate_id':a.candidate,'conversation':actual},ensure_ascii=False,indent=2)); return
    target=out/(a.candidate+'.review.json')
    if any(x.exists() for x in [target,target.with_suffix('.receipt.json')]): raise FileExistsError(target)
    ann=json.loads(Path(cfg['annotations']).read_text(encoding='utf8'))[a.candidate]
    assert ann['stage_sha256']==row['selected_input_sha256']
    for quote in ann.get('source_quotes',[]): assert quote in actual['user'],quote
    for quote in ann.get('answer_quotes',[]): assert any(quote in actual[k] for k in ['reasoning','response']),quote
    cp=Path(cfg['review_contract']); contract=json.loads(cp.read_text(encoding='utf8'))
    assert rt.digest(cp.read_bytes())=='5d1ec43313b10807e50f245c6a342e093f8a324f20889a00803e5276c22860e2'
    accepted=ann['decision']=='accept_unchanged'
    gates={name:True for name in contract['full_frozen_acceptance']['gates']}
    if ann['decision']=='repair_material_defect': gates['grounded']=False
    if ann['decision']=='source_unusable': gates['benign_subject' if a.candidate=='t8_029_v0' else 'self_contained']=False
    for gate in ann.get('failure_gates',[]):
        assert gate in gates
        gates[gate]=False
    review={'candidate_id':a.candidate,'trait_id':row['trait_id'],'reviewer_provenance':{'kind':'independent_codex_agent','task':'/root/audit_nonmoral','human_review':False},'source_folder':str(folder.resolve()),'source_files_sha256':row['source_files_sha256'],'selected_input_stage':row['selected_input_stage'],'selected_input_sha256':row['selected_input_sha256'],'conversation_sha256':rt.digest(actual),'field_sha256':{k:rt.digest(v.encode()) for k,v in actual.items()},'review_contract_sha256':rt.digest(cp.read_bytes()),'constitution_sha256':contract['constitution_sha256'],'scope':'Full actual source/system and complete selected saved author reasoning/final read independently. Full working craft tension and09 compatibility applied. No authoring, paid review, original checkpoint mutation or adoption. Decision does not pretend old automated reviews passed. Similarity is not exhaustively rescored here.',**ann}
    review.update({'accepted':accepted,'gates':gates,'working_preference_sha256':rt.digest(Path(cfg['working_preference']).read_bytes())})
    rt.save_checkpoint(target,review); print(a.candidate,ann['decision'],rt.digest(target.read_bytes()))

if __name__=='__main__':main()
