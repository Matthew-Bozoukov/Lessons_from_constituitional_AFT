# ABOUTME: Runs only four prospectively approved integrated craft-loss pairs under an additional $2 cap.
# ABOUTME: Run: uv run python scratch/nonmoral/stakes/run_integrated.py freeze|answers|review; no repair or GPU calls.
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tenacity import stop_after_attempt
from scratch.nonmoral.pilot import CappedClient,SONNET,verify_live_prices
from scratch.nonmoral.stakes.integrated_design import CASES
from scratch.nonmoral.stakes.prepare import digest,read_rows,write_json,write_rows,phase_config
from src.data.synth.pipeline import run
from src.infra.endpoints.openrouter import OpenRouterClient
from src.utils import timestamp

OUT=REPO/'output/nonmoral_stakes/20260909_integrated4'
IDS={'glass_pattern','miniature_paint','music_patch_bank','folded_print'}
CAP=2.0
PRIOR=[REPO/'output/nonmoral_stakes'/p/'spend.json' for p in ('20260909_first8','20260909_production')]


def freeze():
    if OUT.exists():raise ValueError('Already frozen; do not overwrite original trials')
    OUT.mkdir()
    pairs=[dict(scenario_id=c['id'],core_template=c['prompt'],low_loss=c['low'],high_loss=c['high'],
                low_user=c['prompt'].format(loss=c['low']),high_user=c['prompt'].format(loss=c['high']))
           for c in CASES if c['id'] in IDS]
    assert len(pairs)==4
    write_rows(OUT/'frozen_pairs.jsonl',pairs)
    prior=[dict(path=str(p),sha256=digest(p),exposure_usd=sum(r['charged_or_reserved_usd'] for r in json.loads(p.read_text()))) for p in PRIOR]
    assert sum(p['exposure_usd'] for p in prior)+CAP<=55
    write_json(OUT/'authorization.json',dict(source_sha256=digest(OUT/'frozen_pairs.jsonl'),
        additional_cap_usd=CAP,prior=prior,prior_exposure_usd=sum(p['exposure_usd'] for p in prior),
        planned_answer_calls=8,planned_review_calls=4,excluded_controls=['private_game_route','private_puzzle_chain'],
        authority='Root reviewed all six and approved only cases 1,2,4,5, additional $2 within cumulative $55; no wider run before local inspection.'))
    cfg=OmegaConf.to_container(OmegaConf.load(REPO/'configs/data/synth/nonmoral-stakes.yaml'),resolve=True)
    cfg['workers']=4
    cfg['phases']['review'][0]['prompts']['user'] += '\nThe loss is part of the task mechanics. Check whether the authored reasoning treats that loss as relevant context, not an unrelated/testing aside. No choice switch, added caution or different answer length is required. Mentioning the exact number is not mandatory if the real loss tradeoff is understood.'
    OmegaConf.save(OmegaConf.create(cfg),OUT/'frozen_recipe.yaml')
    print(json.dumps(dict(source_sha256=digest(OUT/'frozen_pairs.jsonl'),ledger=str(OUT/'spend.json'))))


class ExactOnce:
    def __init__(self,client,allowed):
        self.client=client;self.allowed=allowed;self.seen=set();self.lock=Lock()
    def chat(self,**kw):
        user=kw['messages'][-1]['content']
        with self.lock:
            if (OUT/'STOP_DISPATCH').exists() or user not in self.allowed or user in self.seen:
                raise RuntimeError('Unapproved, repeated, or stopped request; no paid retry')
            self.seen.add(user)
        return self.client.chat(**kw)


def generate(phase):
    auth=json.loads((OUT/'authorization.json').read_text());assert digest(OUT/'frozen_pairs.jsonl')==auth['source_sha256']
    for p in auth['prior']:assert digest(p['path'])==p['sha256']
    if (OUT/'STOP_DISPATCH').exists():raise ValueError('Paused')
    cfg=OmegaConf.to_container(OmegaConf.load(OUT/'frozen_recipe.yaml'),resolve=True)
    pairs=read_rows(OUT/'frozen_pairs.jsonl')
    if phase=='answers':
        rows=[dict(scenario_id=p['scenario_id']+'__'+arm,pair_id=p['scenario_id'],arm=arm,
                   original_system='',user=p[arm+'_user']) for p in pairs for arm in ('low','high')]
    else:
        answers={r['scenario_id']:r for r in read_rows(OUT/'answers/complete_candidates.jsonl')}
        rows=[]
        for p in pairs:
            if all(p['scenario_id']+'__'+arm in answers for arm in ('low','high')):
                rows.append(dict(scenario_id=p['scenario_id'],original_system='',**{
                    arm+'_'+key:answers[p['scenario_id']+'__'+arm][key]
                    for arm in ('low','high') for key in ('user','reasoning','response')}))
    assert rows
    dest=OUT/phase
    if dest.exists():raise ValueError('Phase already dispatched; no implicit resume')
    fd=os.open(OUT/'dispatch.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(fd,str(os.getpid()).encode());dest.mkdir()
        phase_config(cfg,phase,rows,dest)
        effective=OmegaConf.to_container(OmegaConf.load(dest/'prepared_config.yaml'),resolve=True)
        effective['budget_usd']=CAP;OmegaConf.save(OmegaConf.create(effective),dest/'dispatch_config.yaml')
        stage=effective['stages'][1]
        allowed={stage['prompts']['user'].format(**r) for r in rows}
        assert len(allowed)==len(rows)
        load_dotenv(REPO.parent/'teaching_claude_why_replication/.env',override=False)
        prices=verify_live_prices({SONNET});client=OpenRouterClient();once=client.chat.retry_with(stop=stop_after_attempt(1))
        capped=CappedClient(lambda **kw:once(client,**kw),OUT/'spend.json',CAP,{SONNET})
        run_dir=dest/'runs'/timestamp();run_dir.mkdir(parents=True)
        state=dict(phase=phase,status='running',rows=len(rows),run_dir=str(run_dir),source_sha256=auth['source_sha256'],
                   config_sha256=digest(dest/'dispatch_config.yaml'),code_sha256=digest(__file__),prices=prices,
                   git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        write_json(dest/'status.json',state)
        try:
            run(effective,resume=str(run_dir),client=ExactOnce(capped,allowed));state['status']='awaiting_local_review'
        finally:
            if (run_dir/'dataset.jsonl').exists():
                output=read_rows(run_dir/'dataset.jsonl');write_rows(dest/'complete_candidates.jsonl',output)
                state['produced']=len(output)
            ledger=json.loads((OUT/'spend.json').read_text());state.update(calls=len(ledger),
                exposure_usd=sum(r['charged_or_reserved_usd'] for r in ledger),
                cumulative_lane_exposure_usd=auth['prior_exposure_usd']+sum(r['charged_or_reserved_usd'] for r in ledger),
                unsettled=sum(r['status']!='settled' for r in ledger))
            write_json(dest/'status.json',state);print(json.dumps(state))
    finally:
        os.close(fd);(OUT/'dispatch.lock').unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','answers','review']);args=parser.parse_args()
    freeze() if args.phase=='freeze' else generate(args.phase)
