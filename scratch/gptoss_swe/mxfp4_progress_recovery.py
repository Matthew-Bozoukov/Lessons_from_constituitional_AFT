# ABOUTME: Complete only known-unsent MXFP4 progress judging after an import-time credential failure.
# ABOUTME: Preserve all model rollouts, alignment verdicts and the existing judge ledger.
import hashlib
import os
import shutil
import time
from scratch.gptoss_swe import mxfp4_pilot as pilot


def run():
    # progress_judge captures its credential when imported; populate environment first.
    os.environ.update(pilot.environment())
    from src.eval.misalignment.odcv import progress_judge,passes
    from omegaconf import OmegaConf
    r=pilot.ROOT/'mxfp4';od=r/'odcv'
    pilot.claim(r/'progress-recovery-owner-started.json')
    cfg=OmegaConf.load(od/'odcv_config.yaml')
    combined=list((od/cfg.model_key).glob('combined*'));assert len(combined)==1
    combined=combined[0];results=pilot.read(combined/'results.json')
    assert results['n_judged']==30 and results['n_dropped_all_na']==0
    ledger=pilot.read(r/'judge-budget.json')
    assert len(ledger)==30 and all(x['status']=='settled' for x in ledger)
    assert not list((combined/'evaluations').glob('progress_*.json'))
    assert 'Missing credentials' in pilot.read(od/'status.json')['error']
    assert progress_judge.OPENAI_API_KEY, 'Required OpenRouter credential is unavailable'
    archive=r/'progress-credential-failure';archive.mkdir()
    for src in (od/'status.json',r/'judge-budget.json'):
        shutil.copyfile(src,archive/src.name)
    verdicts=list((combined/'evaluations').glob('scores_*.json'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in verdicts}
    pilot.save(archive/'receipt.json',dict(at=time.time(),known_unsent=True,dispatch_count_before=30,alignment_hashes=hashes,
        reason='OpenAI constructor failed before create/reservation; env loaded before progress_judge import in this separate process'))
    progress_judge.openrouter_usage=lambda **kw:None
    pilot.phase(od,'judging',recovery='progress-credential-only')
    results['progress']=progress_judge.main(str(combined),str(od/'odcv_config.yaml'),max_workers=4)
    assert results['progress']['n_judged']==30 and results['progress']['n_dropped_all_na']==0
    for p,digest in hashes.items():assert hashlib.sha256(__import__('pathlib').Path(p).read_bytes()).hexdigest()==digest
    audits=pilot.read(od/'pass_summary.json')['audits']
    results['submission']=pilot.read(combined/'submission_stats.json')
    results['passes']=dict(requested=3,kept=3,dropped=0,n_transcripts=30,audits=audits)
    passes.package_run(od,str(cfg.model_key),audits,combined)
    pilot.save(r/'odcv-summary.json',results)
    pilot.phase(od,'complete',recovery='progress-credential-only')
    pilot.save(r/'progress-recovery-complete.json',dict(at=time.time(),alignment_preserved=True,progress_judged=30))


if __name__=='__main__':run()
