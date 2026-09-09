# ABOUTME: Generates the full stakes lane under its cumulative $55 allocation using shared synth stages.
# ABOUTME: Run: uv run python scratch/nonmoral/stakes/production.py sources|answers|review --execute; no GPU operations.
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tenacity import stop_after_attempt

from scratch.nonmoral.pilot import CappedClient, SONNET, verify_live_prices
from scratch.nonmoral.stakes.prepare import approved, arm_rows, digest, join_answers, read_rows, write_json, write_rows
from src.data.synth.pipeline import build_stages, run
from src.infra.endpoints.openrouter import OpenRouterClient
from src.utils import timestamp

OUT = REPO/'output/nonmoral_stakes/20260909_production'
PREVIOUS = REPO/'output/nonmoral_stakes/20260909_first8/spend.json'
POOL = REPO/'output/nonmoral_stakes/20260909_historical_pool/source_candidates.jsonl'
CONFIG = REPO/'configs/data/synth/nonmoral-stakes.yaml'
TOTAL_CAP = 55.0


def phase_rows(phase, source_review=None):
    if phase == 'sources':
        return read_rows(POOL)
    if phase == 'answers':
        path = OUT/'sources/complete_candidates.jsonl'
        if not source_review:
            raise ValueError('Hash-linked local source review required before answers')
        cfg=OmegaConf.to_container(OmegaConf.load(CONFIG),resolve=True)
        selected=approved(read_rows(path), path, source_review)
        # Keep the original source and generated drafts, but use the independently
        # approved numeric-only intervention. This never rewrites a task core.
        framed=[dict(r,eligible='yes',model_source_eligible=r['eligible'],draft_low_context=r['low_context'],draft_high_context=r['high_context'],
                     low_context=cfg['fixed_context']['low'],high_context=cfg['fixed_context']['high']) for r in selected]
        write_rows(OUT/'fixed_frames.jsonl',framed)
        return arm_rows(framed)
    path = OUT/'answers/complete_candidates.jsonl'
    grouped = {}
    for row in read_rows(path):
        grouped.setdefault(row['pair_id'], []).append(row)
    pairs, failures = [], {}
    for key, rows in grouped.items():
        try:
            pairs.extend(join_answers(rows))
        except ValueError as exc:
            failures[key] = str(exc)
    write_json(OUT/'incomplete_pair_exclusions.json', failures)
    return pairs


def generate(phase, source_review=None):
    OUT.mkdir(parents=True, exist_ok=True)
    prior_entries = json.loads(PREVIOUS.read_text())
    assert len(prior_entries) == 24 and all(e['status']=='settled' for e in prior_entries)
    prior = sum(e['charged_or_reserved_usd'] for e in prior_entries)
    cap = TOTAL_CAP-prior
    lock = OUT/'dispatch.lock'
    fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(fd,str(os.getpid()).encode())
        dest=OUT/phase
        if dest.exists():
            raise ValueError('Existing phase; no implicit redispatch or regeneration')
        cfg=OmegaConf.to_container(OmegaConf.load(CONFIG),resolve=True)
        rows=phase_rows(phase,source_review)
        if not rows:
            raise ValueError('No eligible rows; report shortfall rather than fabricating replacements')
        dest.mkdir()
        write_rows(dest/'inputs.jsonl',rows)
        if source_review:
            (dest/'source_review.json').write_bytes(Path(source_review).read_bytes())
        effective={k:v for k,v in cfg.items() if k not in ('phases','source_parent','target_pairs')}
        effective.update(workers=8, budget_usd=cap, total_scenarios=len(rows),
            source=dict(local_dir=str(dest),snapshot='inputs.jsonl'),output_dir=str(dest/'runs'),
            stages=[dict(name='original_stakes_inputs',kind='load_source_run'),*cfg['phases'][phase]])
        build_stages(effective)
        OmegaConf.save(OmegaConf.create(effective),dest/'dispatch_config.yaml')
        load_dotenv(REPO.parent/'teaching_claude_why_replication/.env',override=False)
        prices=verify_live_prices({SONNET})
        ledger=OUT/'spend.json'
        existing=json.loads(ledger.read_text()) if ledger.exists() else []
        uncertain=[e for e in existing if e['status']!='settled']
        if uncertain:
            receipt=OUT/'reservation_reconciliation.json'
            reconciled=json.loads(receipt.read_text()) if receipt.exists() else {}
            declared=reconciled.get('conservatively_charged_requests',{})
            if any(e['status']!='reserved' or declared.get(e['request_sha256'])!=e['charged_or_reserved_usd'] for e in uncertain):
                raise ValueError('Reconcile uncertain reservations before next phase')
            # Closed prior calls retain their entire upper bound in the shared cap.
            # No reimbursement, retry, or claim that actual usage is known.
        run_dir=dest/'runs'/timestamp()
        run_dir.mkdir(parents=True,exist_ok=False)
        state=dict(phase=phase,status='running',rows=len(rows),run_dir=str(run_dir),
            cumulative_lane_cap_usd=TOTAL_CAP,production_cap_usd=cap,prior_exposure_usd=prior,
            prior_ledger=str(PREVIOUS),prior_ledger_sha256=digest(PREVIOUS),
            input_sha256=digest(dest/'inputs.jsonl'),config_sha256=digest(dest/'dispatch_config.yaml'),
            code_sha256=digest(__file__),prices=prices,
            git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        write_json(dest/'status.json',state)
        client=OpenRouterClient()
        once=client.chat.retry_with(stop=stop_after_attempt(1))
        capped=CappedClient(lambda **kw:once(client,**kw),ledger,cap,{SONNET})
        try:
            run(effective,resume=str(run_dir),client=capped)
            state['status']='awaiting_local_review'
        except BaseException:
            state['status']='generation_failed'
            raise
        finally:
            entries=json.loads(ledger.read_text()) if ledger.exists() else []
            exposure=sum(e['charged_or_reserved_usd'] for e in entries)
            state.update(production_calls=len(entries),production_exposure_usd=exposure,
                cumulative_lane_exposure_usd=prior+exposure,
                unsettled_calls=sum(e['status']!='settled' for e in entries))
            output=run_dir/'dataset.jsonl'
            if output.exists():
                produced=read_rows(output)
                write_rows(dest/'complete_candidates.jsonl',produced)
                state.update(produced=len(produced),dataset_sha256=digest(output),
                    missing_ids=sorted({r['scenario_id'] for r in rows}-{r['scenario_id'] for r in produced}))
                if phase=='sources':
                    state['model_eligible']=sum(r.get('eligible')=='yes' for r in produced)
            write_json(dest/'status.json',state)
            print(json.dumps(state),flush=True)
    finally:
        os.close(fd)
        lock.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['sources','answers','review'])
    p.add_argument('--source-review',type=Path)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.execute:
        generate(args.phase,args.source_review)
    else:
        print(json.dumps(dict(phase=args.phase,paid=False,cumulative_lane_cap_usd=TOTAL_CAP,
                              production_ledger=str(OUT/'spend.json'),prior_ledger=str(PREVIOUS))))
