# ABOUTME: Freeze a separate full-reread adjudication without changing the original campaign review.
# ABOUTME: Preserve all byte bindings and uncertainty history; never dispatch calls or adopt rows.
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from scratch.dataset_refresh import run as rt, offline_acceptance as oa


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); args=p.parse_args()
    cfg=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
    out=Path(cfg['output'])
    if out.exists(): raise FileExistsError(out)
    assert oa.sha(cfg['original_summary'])==cfg['original_summary_sha256']
    summary=rt.load_checkpoint(Path(cfg['original_summary']))
    case=next(x for x in summary['cases'] if x['candidate_id']==cfg['candidate_id'])
    path=Path(case['review_path']); assert oa.sha(path)==case['review_sha256']
    review=rt.load_checkpoint(path); d,frozen=oa.validate_dossier(Path(case['dossier_path']))
    assert review['conversation_sha256']==d['conversation_sha256']
    annotation=oa.read(cfg['annotation'])
    for issue in annotation['issues']:
        assert issue['evidence'] in d['conversation']['reasoning']+'\n'+d['conversation']['response']
    adjudicated={**review,**annotation,'full_system_user_reasoning_final_read':True,'source_eligible':True,
                 'prior_independent_review_path':str(path.resolve()),'prior_independent_review_sha256':oa.sha(path),
                 'prior_independent_disposition':review['decision'],'adjudication_scope':'Second full actual source/reasoning/final reread, separately adjudicating materiality. Original uncertainty and counterreading preserved; no quota optimization.',
                 'helper_sha256':oa.sha(__file__),'config_sha256':oa.sha(args.config),'annotation_sha256':oa.sha(cfg['annotation'])}
    out.parent.mkdir(parents=True,exist_ok=True);rt.save_checkpoint(out,adjudicated);print(oa.sha(out))


if __name__=='__main__':main()
