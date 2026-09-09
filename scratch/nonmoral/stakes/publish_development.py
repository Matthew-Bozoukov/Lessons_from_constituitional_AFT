# ABOUTME: Publishes failed-wrapper and prospective integrated-trial evidence as a development archive, never training data.
# ABOUTME: Run: uv run python scratch/nonmoral/stakes/publish_development.py; secret-scans and verifies public critical files.
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO))
from dotenv import load_dotenv
import requests
from scratch.nonmoral.stakes.prepare import digest,write_json
from src.infra.huggingface import hf_api,hf_org,push_run_dir
from src.naming import artifact_name


def main():
    root=REPO/'output/nonmoral_stakes'
    old=root/'20260909_production';new=root/'20260909_integrated4'
    assert not (old/'dispatch.lock').exists() and not (new/'dispatch.lock').exists()
    assert (old/'stop_receipt.json').exists() and (new/'local_review.json').exists()
    dest=root/'20260909_development_archive'
    if dest.exists():raise ValueError('Archive snapshot already exists; inspect receipt before retrying publication')
    dest.mkdir()
    for source,label in ((old,'failed_wrapper'),(new,'integrated_craft_trial'),
                         (root/'20260909_historical_pool','historical_source_pool'),
                         (root/'integrated_design','offline_design_fixtures')):
        shutil.copytree(source,dest/label)
    shutil.copytree(REPO/'scratch/nonmoral/stakes',dest/'code/stakes',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(REPO/'scratch/nonmoral/pilot.py',dest/'code/pilot.py')
    (dest/'configs').mkdir()
    for p in (REPO/'configs/data/synth').glob('nonmoral-stakes*.yaml'):shutil.copy2(p,dest/'configs'/p.name)
    shutil.copy2(REPO/'docs/nonmoral_deliberation/stakes_experiment.md',dest/'stakes_experiment.md')
    review=json.loads((new/'local_review.json').read_text())
    write_json(dest/'manifest.json',dict(artifact_role='Development evidence only; no training-approved rows or mixture',
        prior_first8='https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-stakes-first8',
        prior_first8_revision='b951660317dcca7844337ac3a181ca6dca8602dd',
        failed_wrapper=dict(source_pool=702,source_completions=699,selected_sources=119,complete_answers=57,complete_pairs=27,paid_reviews=0,
                            outcome='Stopped because appended loss context was repeatedly treated as unrelated'),
        integrated_trial=review,cumulative_lane_exposure_usd=review['cumulative_lane_exposure_usd'],
        gpu_runs=0,selection_uses_odcv=False,training_approved=False))
    load_dotenv(REPO.parent/'teaching_claude_why_replication/.env',override=False)
    from scratch.nonmoral.publish_invalid_baseline import scan,secret_values
    secrets=secret_values()
    for p in dest.rglob('*'):
        if p.is_file():scan(p.read_bytes(),str(p),secrets)
    fields=dict(experiment='Nonmoral stakes development: stopped appended-wrapper attempt and four prospective integrated craft pairs',
        date_generated='2026-09-09',constitution='none; nonmoral craft preferences, no moral constitution',
        source_repo='https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ '+subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        models=dict(source_screen='anthropic/claude-sonnet-5',answerer='anthropic/claude-sonnet-5',reviewer='anthropic/claude-sonnet-5',
            provider='Anthropic via OpenRouter',revision='No immutable API model revision exposed; model identity and per-phase settings retained',
            offline_source_author='Codex; six design fixtures, four approved for calls; no claim of Sonnet-generated sources'),
        generation_config='Frozen phase configs and source SHA256s in each run; .7 answer/source temperature,0 paired review; cumulative$55 lane allocation; integrated trial additional$2cap; all observed/reserved costs preserved.',
        schema='failed_wrapper/: source screens, 57 answers, raw successful calls, ledger and stop evidence; integrated_craft_trial/: four immutable pairs, eight raw completions, seven parsed answers, three reviews, all local decisions; configs/code/provenance included.',
        provenance='scratch/nonmoral/stakes/production.py sources then answers (stopped); run_integrated.py freeze,answers,review; review_integrated.py; publish_development.py. Commands and commit hashes in statuses.',
        downstream='Development archive only. Zero retained matched pairs from integrated trial, failed-wrapper pairs unreviewed. Do not use as SFT corpus. First-eight public archive remains unchanged.',
        limitations='No stakes effect or alignment result. In integrated trial the loss was relevant, but substantive errors remained. Unranked preferences were falsely treated as equal, factual risk/time claims were invented, and an unavailable tool appeared. Sonnet accepted three reviewed pairs despite defects. Failed-wrapper13unknown/inflight calls retain$1.623328 maximum reservations but lack raw exception bodies; integrated11calls all settled. Fictional request money is not API/GPU expense.')
    configs=[dict(config_name='integrated_pair_requests',data_files='integrated_craft_trial/frozen_pairs.jsonl',default=True),
             dict(config_name='wrapper_sources',data_files='failed_wrapper/sources/complete_candidates.jsonl'),
             dict(config_name='wrapper_answers',data_files='failed_wrapper/answers/complete_candidates.jsonl'),
             dict(config_name='integrated_answers',data_files='integrated_craft_trial/answers/complete_candidates.jsonl'),
             dict(config_name='integrated_reviews',data_files='integrated_craft_trial/review/complete_candidates.jsonl')]
    assert hf_org()=='dougalldeepmind'
    name=artifact_name('nonmoral-stakes-development',date='2026-09-09')
    url=push_run_dir(dest,name,fields,private=False,front_matter=dict(configs=configs,tags=['nonmoral-deliberation','stakes','development-archive','not-training-data']))
    info=hf_api().dataset_info(hf_org()+'/'+name);assert not info.private
    verified={}
    for relative in ('manifest.json','failed_wrapper/stop_receipt.json','integrated_craft_trial/frozen_pairs.jsonl','integrated_craft_trial/local_review.json','integrated_craft_trial/spend.json'):
        response=requests.get(f'https://huggingface.co/datasets/{hf_org()}/{name}/resolve/{info.sha}/{relative}',timeout=45)
        response.raise_for_status();assert hashlib.sha256(response.content).hexdigest()==digest(dest/relative)
        verified[relative]=digest(dest/relative)
    receipt=dict(url=url,revision=info.sha,private=False,critical_files_anonymously_verified=verified,
                 cumulative_lane_exposure_usd=review['cumulative_lane_exposure_usd'],snapshot=str(dest),training_data=False)
    write_json(root/'development_publication.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
