# ABOUTME: Create separate schema-normalized independent reviews and explicitly draft root cleanup decisions.
# ABOUTME: Preserve judgments and original hashes; only perform read-only payload validation when initialized.
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from scratch.dataset_refresh import run as rt, offline_acceptance as oa, saved_input_second_pass as second


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--validate-payloads',action='store_true');args=p.parse_args()
    cfg=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True);out=Path(cfg['output'])
    if args.validate_payloads:
        summary=rt.load_checkpoint(out/'summary.json'); target=out/'payload_validation.json'
        if target.exists(): raise FileExistsError(target)
        decisions=rt.load_checkpoint(out/'root_decision_drafts.json')
        assert oa.sha(out/'root_decision_drafts.json')==summary['root_decision_drafts_sha256']
        payloads=[second.make_payload(d) for d in decisions]
        rt.save_checkpoint(target,{'status':'read_only_validated_not_prepared_or_dispatched','root_decision_drafts_sha256':oa.sha(out/'root_decision_drafts.json'),
                                  'rows':[{'candidate_id':x['candidate_id'],'request_sha256':x['request_sha256'],'reservation_usd':x['reservation_usd'],'saved_input_basis':x['saved_input_basis']} for x in payloads]})
        print(len(payloads),oa.sha(target));return
    if out.exists(): raise FileExistsError(out)
    proposals=rt.load_checkpoint(Path(cfg['proposal']))['rows'];annotations=oa.read(cfg['instructions']);plans=[]
    for cid in cfg['candidates']:
        if cid in cfg.get('review_overrides',{}):
            review_path=Path(cfg['review_overrides'][cid]); review=rt.load_checkpoint(review_path)
            summary=rt.load_checkpoint(Path(cfg['campaign'])/'campaign_reviews'/'batch_source_15'/'summary.json')
            assert summary['cases'][0]['candidate_id']==cid
            folder=Path(cfg['campaign'])/'batches'/'batch_source_15'
            inp=rt.load_checkpoint(folder/'00.input.json')
            row={'source_ref':inp['source_ref'],'case_key':inp['case_key'],'candidate_id':cid,
                 **{f'first_{kind}_path':str((folder/('00.'+kind+'.json')).resolve()) for kind in ('input','result','raw')}}
            for kind in ('input','result','raw'):row[f'first_{kind}_sha256']=oa.sha(row[f'first_{kind}_path'])
            instruction=review['proposed_minimal_instruction']
        else:
            row=next(x for x in proposals if x['candidate_id']==cid);review_path=Path(row['full_independent_review_path']);review=rt.load_checkpoint(review_path)
            assert oa.sha(review_path)==row['full_independent_review_sha256'];instruction=annotations[cid]
        assert review['accepted'] is False and review['decision'] in ('hold','reject','uncertain')
        assert review.get('full_read') is True or review.get('full_system_user_reasoning_final_read') is True
        d,frozen=oa.validate_dossier(Path(review['dossier_path']))
        assert d['conversation_sha256']==review['conversation_sha256']
        normalized={**review,'full_system_user_reasoning_final_read':True,'source_eligible':True,
                    'schema_normalization_only':True,'original_review_path':str(review_path.resolve()),'original_review_sha256':oa.sha(review_path),
                    'normalization_scope':'Full source eligibility was assessed in the original independent full read. Add explicit boolean and issue quote-list fields without changing decision, uncertainty or any textual finding.'}
        normalized['issues']=[{**issue,'quotes':issue.get('quotes',[issue['evidence']]),
                               'severity':issue.get('severity','uncertain_materiality' if review['decision']=='uncertain' else 'material')} for issue in review['issues']]
        decision={'root_actor':'/root','decision':'one_additional_focused_revision','approval_state':'DRAFT_FOR_ROOT_REVIEW_NOT_AUTHORIZED_BY_THIS_FILE','execution_enabled':False,
                  'case_key':row['case_key'],'source_ref':row['source_ref'],'instruction':instruction,'max_tokens':cfg['max_tokens'],
                  **{k:row[k] for k in ('first_input_path','first_input_sha256','first_result_path','first_result_sha256','first_raw_path','first_raw_sha256')}}
        if review['decision']=='uncertain':
            decision.update(root_selected_material_cleanup=True,material_cleanup_reason='DRAFT root choice: clean the specifically quoted unsupported factual/format assertion while preserving the independent uncertain materiality judgment and its counterreading; no quota-based tolerance change.')
        plans.append((cid,normalized,decision))
    out.mkdir(parents=True);decisions=[];refs=[]
    for cid,review,decision in plans:
        path=out/(cid+'.review.json');rt.save_checkpoint(path,review)
        decision.update(independent_review_path=str(path.resolve()),independent_review_sha256=oa.sha(path))
        inp=rt.load_checkpoint(Path(decision['first_input_path']));res=rt.load_checkpoint(Path(decision['first_result_path']))
        second.validate_material_review(review,inp,res,decision['first_result_sha256'],decision['first_input_sha256'],review['review_contract_sha256'],decision)
        decisions.append(decision);refs.append({'candidate_id':cid,'decision':review['decision'],'review_sha256':oa.sha(path),'original_review_sha256':review['original_review_sha256']})
    rt.save_checkpoint(out/'root_decision_drafts.json',decisions)
    report={'status':'draft_for_root_review_not_dispatched','rows':refs,'root_decision_drafts_sha256':oa.sha(out/'root_decision_drafts.json'),
            'helper_sha256':oa.sha(__file__),'config_sha256':oa.sha(args.config),'proposal_sha256':oa.sha(cfg['proposal']),
            'scope':'Independent evidence normalization only, all uncertainty retained. Material-review schema validated without calls. make_payload remains pending second-campaign initialization; no prepare/adopt/dispatch.'}
    rt.save_checkpoint(out/'summary.json',report);print(len(refs),oa.sha(out/'summary.json'))


if __name__=='__main__':main()
