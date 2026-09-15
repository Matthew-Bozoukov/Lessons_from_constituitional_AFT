# ABOUTME: Bind prospective minimal second-pass instructions to immutable first attempts and full-read evidence.
# ABOUTME: No inference, adoption or verdict relabeling; incomplete first outputs retain original saved-input lineage.
import argparse
from pathlib import Path
from datetime import datetime, timezone
from omegaconf import OmegaConf
from scratch.dataset_refresh import run as rt, offline_acceptance as oa


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); args=p.parse_args()
    cfg=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
    out=Path(cfg['output']);
    if out.exists(): raise FileExistsError(out)
    instructions=oa.read(cfg['instructions']); rows=[]
    for entry in cfg['cases']:
        batch,index=entry['batch'],entry['index']; stem=f'{index:02d}'
        folder=Path(cfg['campaign'])/'batches'/batch
        ip,rp,raw=(folder/(stem+suffix) for suffix in ('.input.json','.result.json','.raw.json'))
        inp,res=rt.load_checkpoint(ip),rt.load_checkpoint(rp); cid=inp['candidate_id']
        review_path=Path(cfg['campaign'])/'campaign_reviews'/batch/(stem+'.independent_review.json')
        review=rt.load_checkpoint(review_path)
        assert review['result_sha256']==oa.sha(rp) and review['input_sha256']==oa.sha(ip)
        assert review['accepted'] is False
        assert oa.sha(raw)==res['physical_receipt']['raw_sha256']
        instruction=instructions[cid]; assert len(instruction.encode())<=2048
        complete='conversation' in res
        conv=res['conversation'] if complete else inp['actual_conversation']
        prior_path=Path(entry['original_full_review']) if not complete else None
        prior=rt.load_checkpoint(prior_path) if prior_path else None
        if prior:
            assert prior['conversation_sha256']==rt.digest(conv)
        row={'case_key':inp['case_key'],'candidate_id':cid,'trait_id':inp['trait_id'],'source_ref':inp['source_ref'],
             'first_batch':batch,'first_batch_index':index,'first_input_path':str(ip.resolve()),'first_input_sha256':oa.sha(ip),
             'first_result_path':str(rp.resolve()),'first_result_sha256':oa.sha(rp),'first_raw_path':str(raw.resolve()),'first_raw_sha256':oa.sha(raw),
             'first_physical_receipt':res['physical_receipt'],'first_conversation_sha256':rt.digest(conv) if complete else None,
             'full_independent_review_path':str(review_path.resolve()),'full_independent_review_sha256':oa.sha(review_path),
             'first_dossier_path':review.get('dossier_path'),'first_dossier_sha256':review.get('dossier_sha256'),
             'full_system_user_reasoning_final_read':complete,'source_eligible':True,
             'independent_review_provenance':{'kind':'independent_codex_agent','human_review':False,'task':'/root/audit_nonmoral'},
             'original_independent_disposition':review['decision'],'existing_bad_quotes':review.get('answer_quotes',[]),
             'actual_source_quotes':review.get('source_quotes',prior.get('source_quotes',[]) if prior else []),
             'material_defect':review.get('source_first_reason',prior.get('reason') if prior else None),
             'counterreading_and_caveats':review.get('benign_counterreading',prior.get('benign_counterreading') if prior else None),
             'proposed_instruction':instruction,'instruction_utf8_bytes':len(instruction.encode()),
             'revision_scope':'One extra focused revision, unchanged source and target. Both trained blocks remain standalone. This proposal does not change any prior verdict or authorize dispatch.',
             'dispatch_enabled':False,'may_enter_selection_without_new_full_review':False}
        if not complete:
            row.update(revision_input_kind='original_complete_saved_answer_after_first_length_failure',
                       original_complete_conversation_sha256=rt.digest(conv),original_full_review_path=str(prior_path.resolve()),
                       original_full_review_sha256=oa.sha(prior_path),original_complete_full_read=True,
                       first_failure_status=res['status'],first_failure_error=res.get('error'),
                       incomplete_first_output_must_not_be_revision_input=True)
        if review['decision']=='uncertain':
            row['uncertainty_policy']='Prior uncertainty and strongest counterreading remain unchanged. Root cleanup choice must be bound separately; this proposal is not an independent certification of materiality.'
        rows.append(row)
    out.parent.mkdir(parents=True,exist_ok=True)
    report={'aboutme':['Prospective focused second-pass instructions for own held/uncertain first outputs and one length failure.','Exact source/physical lineage preserved; no calls, adoption or silent verdict relabeling.'],
            'created_at':datetime.now(timezone.utc).isoformat(),'authorization_context':'Parent reports user expressly allowed one additional focused revision under cumulative270; root must select, bind and dispatch.',
            'counts':{'proposed_sources':len(rows)},'rows':rows,'instructions_sha256':oa.sha(cfg['instructions']),
            'helper_sha256':oa.sha(__file__),'config_sha256':oa.sha(args.config),
            'remaining_acceptance_contract':oa.read(cfg['reference_proposal'])['remaining_acceptance_contract'], 'execution_enabled':False}
    rt.save_checkpoint(out,report); print(len(rows),oa.sha(out))


if __name__=='__main__':main()
