# ABOUTME: Runs the first broader nonmoral batch through shared generation/review stages with a capped ledger.
# ABOUTME: Preserves every candidate and raw call; produces a full user review packet without repair or paired controls.
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from omegaconf import OmegaConf

from scratch.nonmoral.pilot import CappedClient, SONNET, file_sha256, read_rows, verify_live_prices
from src.data.synth.pipeline import build_stages, run
from src.data.synth.stage_runtime import model_cfg
from src.infra.endpoints.openrouter import OpenRouterClient
from src.infra.huggingface import hf_api, hf_org, push_run_dir
from src.naming import artifact_name
from src.utils import timestamp

ROOT = Path('output/nonmoral_broader/20260909')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def packet(run_dir):
    path = run_dir/'dataset.jsonl'
    rows = read_rows(path) if path.exists() else []
    lines = ['# Broader nonmoral dataset: first 12 complete examples', '',
             'Production candidates for feedback, not approved SFT data. Each has one authored reasoning trace and one full answer. '
             'The separate Sonnet review is an assessment, not ground truth. All examples and any rejections are retained.', '',
             '| # | Domain | Reviewer | Choice |', '|---|---|---|---|']
    seen, mechanical_failures = set(), {}
    for i, row in enumerate(rows, 1):
        problems = []
        for key in ('user','reasoning','response'):
            if not isinstance(row.get(key),str) or not row[key].strip():
                problems.append('Missing '+key)
        user_hash = hashlib.sha256(row.get('user','').strip().encode()).hexdigest()
        if user_hash in seen:
            problems.append('Exact duplicate request')
        seen.add(user_hash)
        if row.get('quality_decision') not in ('accept','reject'):
            problems.append('Invalid reviewer decision')
        if problems:
            mechanical_failures[row['scenario_id']] = problems
        summary = row.get('choice_summary','').replace('|','/').replace('\n',' ')
        lines.append(f"| {i} | {row['domain']} | {row.get('quality_decision','missing')} | {summary} |")
    for i, row in enumerate(rows,1):
        lines += ['', f"## {i}. {row['domain']} — {row['scenario_id']}", '', '**User request**', '',
                  row.get('user','[missing]'), '', '**Authored reasoning**', '', row.get('reasoning','[missing]'),
                  '', '**Full assistant answer**', '', row.get('response','[missing]'), '',
                  f"**Separate Sonnet review: {row.get('quality_decision','missing')}**", '',
                  row.get('quality_issues','[missing]'), '']
    (ROOT/'first12_review.md').write_text('\n'.join(lines),encoding='utf-8')
    planned = read_rows(ROOT/'first12/inputs.jsonl')
    summary = dict(planned=len(planned), produced=len(rows),
                   missing_ids=sorted({r['scenario_id'] for r in planned}-{r['scenario_id'] for r in rows}),
                   reviewer_accepts=sum(r.get('quality_decision')=='accept' for r in rows),
                   reviewer_rejects=sum(r.get('quality_decision')=='reject' for r in rows),
                   mechanical_failures=mechanical_failures, dataset_sha256=file_sha256(path) if path.exists() else None,
                   status='awaiting_local_review_and_user_feedback', training_approved=False)
    write_json(ROOT/'first12_summary.json',summary)
    return summary


def publish():
    from scratch.nonmoral.publish_invalid_baseline import scan, secret_values
    secrets = secret_values()
    for path in ROOT.rglob('*'):
        if path.is_file():
            scan(path.read_bytes(),str(path),secrets)
    name = artifact_name('nonmoral-broader-first12',date='2026-09-09')
    fields = dict(experiment='First12 broader nonmoral production candidates for feedback',
        date_generated='2026-09-09',constitution='preferences/nonmoral_broad/preferences.md; nonmoral judgement, not a moral constitution',
        source_repo='https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ '+subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        models=SONNET+' via pinned Anthropic endpoint for authoring and separate review',
        generation_config='Frozen config and source hashes in runs/*/frozen_config.json; config copied locally. Four workers.',
        schema='first12_review.md: every full conversation; runs/: stage snapshots and dataset.jsonl; raw_calls/: requests/responses; spend.json: cumulative attributed ledger',
        provenance='uv run python scratch/nonmoral/broader_data.py --execute',
        limitations='First twelve candidates, not an error-rate estimate or an approved SFT corpus. Review decisions are model assessments. No paired controls, repairs, training or ODCV-based selection.')
    url = push_run_dir(ROOT,name,fields,private=False,front_matter={'tags':['nonmoral-deliberation','review-candidates']})
    info = hf_api().dataset_info(hf_org()+'/'+name)
    assert not info.private
    write_json(ROOT/'publication.json',dict(url=url,revision=info.sha,private=False))
    print(url,flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Path('configs/data/synth/nonmoral-broader.yaml'))
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--publish',action='store_true')
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
    assert cfg['total_scenarios']==len(cfg['cases'])==12 and cfg['workers']==4
    assert cfg['budget_usd']==100 and cfg['dispatch_cap_usd']==3 and cfg['target_accepted']==700
    assert cfg['hf_push'] is False and cfg['batch'] is False
    assert Path(cfg['output_dir']).resolve()==(ROOT/'runs').resolve()
    assert Path(cfg['source']['local_dir']).resolve()==(ROOT/'first12').resolve()
    assert cfg['source']['snapshot']=='inputs.jsonl'
    assert [s['kind'] for s in cfg['stages']]==['load_source_run','llm_tagged','llm_tagged']
    assert all(not (set(s)&{'lint','verify','fallback_model'}) for s in cfg['stages'])
    assert all(model_cfg(cfg,k)['model']==SONNET and model_cfg(cfg,k).get('extra_body')=={'reasoning':{'enabled':False}}
               for k in cfg['models'])
    assert hf_org()=='dougalldeepmind'
    print(json.dumps(dict(stages=[s.name for s in build_stages(cfg)],candidates=12,
                         dispatch_cap_usd=3,whole_dataset_cap_usd=100,paid=args.execute)),flush=True)
    if not args.execute:
        if args.publish:
            publish()
        return
    ROOT.mkdir(parents=True,exist_ok=True)
    marker = ROOT/'dispatch.json'
    assert not marker.exists(), 'First batch already dispatched; inspect saved outputs instead of paying twice'
    prices = verify_live_prices({SONNET})
    source = ROOT/'first12/inputs.jsonl'
    source.parent.mkdir(parents=True,exist_ok=True)
    rows = [dict(scenario_id=f'broader_20260909_{i:03d}',**case) for i,case in enumerate(cfg['cases'],1)]
    source.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    lock = ROOT/'generation.lock'
    fd = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(fd,str(os.getpid()).encode())
        run_dir = ROOT/'runs'/timestamp()
        run_dir.mkdir(parents=True,exist_ok=False)
        state = dict(status='generating',run_dir=str(run_dir),config=cfg,config_sha256=file_sha256(args.config),
                     source_sha256=file_sha256(source),prices=prices,
                     prior_total_exposure_usd=31.929031131298995,total_ceiling_usd=300,
                     code_sha256={p:file_sha256(p) for p in ['scratch/nonmoral/broader_data.py','scratch/nonmoral/pilot.py',
                                                           'src/data/synth/stage_operators.py','src/data/synth/stage_runtime.py']})
        write_json(marker,state)
        write_json(run_dir/'frozen_config.json',state)
        (ROOT/args.config.name).write_bytes(args.config.read_bytes())
        from tenacity import stop_after_attempt
        client = OpenRouterClient()
        single = client.chat.retry_with(stop=stop_after_attempt(1))
        capped = CappedClient(lambda **kw:single(client,**kw),ROOT/'spend.json',3,{SONNET},allow_reasoning_off=True)
        try:
            run(cfg,resume=str(run_dir),client=capped)
        finally:
            state.update(status='awaiting_review',summary=packet(run_dir))
            if (ROOT/'spend.json').exists():
                entries = json.loads((ROOT/'spend.json').read_text())
                state.update(calls=len(entries),exposure_usd=sum(e['charged_or_reserved_usd'] for e in entries),
                             unsettled_calls=sum(e['status']!='settled' for e in entries))
            write_json(ROOT/'status.json',state)
    finally:
        os.close(fd)
        lock.unlink()
    if args.publish:
        publish()


if __name__=='__main__':
    main()
