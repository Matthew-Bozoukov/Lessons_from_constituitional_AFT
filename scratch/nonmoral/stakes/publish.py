# ABOUTME: Exports paired retained stakes corpora using the shared synth naming and HF data contract.
# ABOUTME: Run: uv run python scratch/nonmoral/stakes/publish.py [--publish]; no inference or GPU operations.
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from dotenv import load_dotenv
from omegaconf import OmegaConf
from scratch.nonmoral.stakes.prepare import approved,digest,join_answers,read_rows,write_json
from scratch.nonmoral.stakes.production import OUT,PREVIOUS,CONFIG
from src.data.synth.hf_cache import StageCache
from src.infra.huggingface import hf_api,hf_org,push_run_dir,training_data_tags
from src.naming import synth_name
from src.utils import timestamp


def retained():
    if (OUT/'dispatch.lock').exists():
        raise ValueError('Publish only a stable completed phase snapshot')
    reviews=OUT/'review/complete_candidates.jsonl'
    dispositions=OUT/'local_review.json'
    verdict=json.loads(dispositions.read_text(encoding='utf-8'))
    answers_path=OUT/'answers/complete_candidates.jsonl'
    if verdict.get('answer_sha256')!=digest(answers_path):
        raise ValueError('Local dispositions do not identify these exact answer bytes')
    selected=approved(read_rows(reviews),reviews,dispositions)
    ids={r['scenario_id'] for r in selected}
    answers=[r for r in read_rows(answers_path) if r['pair_id'] in ids]
    pairs=join_answers(answers)
    if not ids or {r['scenario_id'] for r in pairs}!=ids:
        raise ValueError('No retained pairs or incomplete pair membership')
    cfg=OmegaConf.load(CONFIG)
    assert cfg.fixed_context.low.replace('cost me £10 ','cost me £1,000 ')==cfg.fixed_context.high
    for r in answers:
        assert r['stakes_context']==cfg.fixed_context[r['arm']]
    return ids,answers


def export(publish=False):
    ids,answers=retained()
    load_dotenv(REPO.parent/'teaching_claude_why_replication/.env',override=False)
    from scratch.nonmoral.publish_invalid_baseline import scan,secret_values
    secret_strings=secret_values()
    date='2026-09-09'
    git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    receipts=[]
    for arm in ('low','high'):
        config_path=REPO/f'configs/data/synth/nonmoral-stakes-{arm}.yaml'
        cfg=OmegaConf.to_container(OmegaConf.load(config_path),resolve=True)
        assert cfg['arm']==arm
        name=synth_name(cfg['pipeline'],date=date)
        dest=OUT/'publications'/timestamp()/arm
        cache=StageCache(dest,None)
        for i,phase in enumerate(('sources','answers','review'),1):
            cache.save(i,phase,read_rows(OUT/phase/'complete_candidates.jsonl'))
            shutil.copytree(OUT/phase,dest/'audit'/phase)
        rows=[dict(scenario_id=r['scenario_id'],pair_id=r['pair_id'],arm=arm,
                   system=r['original_system'],user=r['user'],reasoning=r['reasoning'],
                   response=r['response'],core_user=r['core_user'],stakes_context=r['stakes_context'])
              for r in answers if r['arm']==arm]
        assert len(rows)==len(ids)
        cache.publish_final(rows)
        for file in ('spend.json','source_review.json','source_eligibility_summary.json',
                     'reservation_reconciliation.json','fixed_frames.jsonl','local_review.json',
                     'incomplete_pair_exclusions.json'):
            shutil.copy2(OUT/file,dest/file)
        shutil.copytree(OUT/'raw_calls',dest/'audit/raw_calls')
        shutil.copy2(PREVIOUS,dest/'prior_first8_spend.json')
        shutil.copy2(CONFIG,dest/'generation_recipe.yaml')
        shutil.copy2(config_path,dest/'config.yaml')
        ledger=json.loads((OUT/'spend.json').read_text())
        prior=json.loads(PREVIOUS.read_text())
        manifest=dict(arm=arm,paired_rows=len(rows),original_pool_rows=702,target_pairs=684,
            shortfall=684-len(rows),dataset_sha256=digest(dest/'dataset.jsonl'),
            paired_ids=sorted(ids),training_approved=False,
            selection='Immutable original sources; complete nonmoral source review, fresh answers, one Sonnet paired review and local material dispositions. No ODCV feedback.',
            cumulative_lane_exposure_usd=sum(r['charged_or_reserved_usd'] for r in ledger+prior),
            unsettled_reserved_calls=sum(r['status']!='settled' for r in ledger),
            source_review_sha256=digest(OUT/'source_review.json'),local_review_sha256=digest(OUT/'local_review.json'))
        cache.save_json('manifest.json',manifest)
        configs=[dict(config_name='dataset',data_files='dataset.jsonl',default=True)]
        configs += [dict(config_name=p.stem,data_files='stages/'+p.name) for p in sorted(dest.glob('stage_*.jsonl'))]
        (dest/'stages').mkdir()
        for p in list(dest.glob('stage_*.jsonl')):p.rename(dest/'stages'/p.name)
        fields=dict(experiment=f'Matched nonmoral personal replacement-loss corpus: {arm} stakes',
            date_generated=date,constitution='none; nonmoral working preferences',
            source_repo='https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ '+git_sha,
            models=dict(source_screen='anthropic/claude-sonnet-5',answerer='anthropic/claude-sonnet-5',
                reviewer='anthropic/claude-sonnet-5',provider='Anthropic via OpenRouter',
                revision='API model has no immutable revision exposed; exact model ID, settings and returned content preserved'),
            generation_config='generation_recipe.yaml and frozen audit phase dispatch configs; temperature .7 for source/answers, 0 for paired review; fresh full answers; no semantic repairs.',
            schema='Default dataset.jsonl contains retained system/user/reasoning/response rows with paired IDs. stages/ and audit/ retain all candidate screens, answers and reviews. Local dispositions, original source hashes and cost ledgers included.',
            provenance='uv run python scratch/nonmoral/stakes/production.py sources --execute; answers --source-review output/nonmoral_stakes/20260909_production/source_review.json --execute; review --execute; then scratch/nonmoral/stakes/publish.py --publish',
            downstream='Candidate corpus for a matched two-arm SFT mixture after root approves adequacy. No LoRA or ODCV run; no duplicated filling of the 684-example slot.',
            limitations='Personal monetary replacement-loss framing on a selected historical craft-task subset, not all stakes. Original professional context remains identical in both arms. Model/local reviews are fallible. Missing/held/rejected rows excluded from default. Early exception calls retain hashes and full cost upper bounds but lack raw exception bodies. Fictional pound amounts in requests are not API/GPU expenses.')
        for p in dest.rglob('*'):
            if p.is_file():scan(p.read_bytes(),str(p),secret_strings)
        receipt=dict(name=name,arm=arm,rows=len(rows),dataset_sha256=manifest['dataset_sha256'],snapshot=str(dest),published=False)
        if publish:
            assert hf_org()=='dougalldeepmind'
            receipt['url']=push_run_dir(dest,name,fields,private=False,front_matter=dict(configs=configs,
                tags=training_data_tags('synth',cfg['pipeline'],'none',extra=['status:reviewed','experiment:nonmoral-stakes'])))
            info=hf_api().dataset_info(hf_org()+'/'+name)
            assert not info.private
            receipt.update(revision=info.sha,private=False,published=True)
        receipts.append(receipt)
    write_json(OUT/'production_publications.json',receipts)
    print(json.dumps(receipts),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true')
    export(parser.parse_args().publish)
