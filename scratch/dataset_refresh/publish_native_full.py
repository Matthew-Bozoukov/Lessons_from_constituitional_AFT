# ABOUTME: Publishes the unchanged 716-row selection, complete provenance archive and candid review.
# ABOUTME: Run: uv run --no-sync python -X utf8 -m scratch.dataset_refresh.publish_native_full --config scratch/dataset_refresh/native_lowstakes_full.yaml --root <run> --report <review.md>
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import zipfile
from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, hf_hub_download
from omegaconf import OmegaConf
from scratch.dataset_refresh.run import read_rows, write_json, digest
from scratch.dataset_refresh.select_native_lowstakes import select_rows
from src.infra.huggingface import hf_api, hf_token, gate_push, card_markdown, card_front_matter, training_data_tags
from src.utils import git_sha, origin_url, timestamp


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--root',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    load_dotenv()
    root=Path(args.root)
    launch=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
    native=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    cfg=native['config']
    rows=read_rows(root/'dataset.jsonl')
    chosen, selection=select_rows(rows,launch['selection'])
    assert len(chosen)==716 and chosen==read_rows(root/'selection/dataset.jsonl')
    completion=json.loads((root/'review/corpus_completion/completion.json').read_text())
    assert completion['successful'] and completion['source_sha256']==digest((root/'dataset.jsonl').read_bytes())
    entries=json.loads(Path(launch['campaign_budget_root'],'spend.json').read_text())
    assert all(e['status']!='reserved' for e in entries), 'Paid calls still active'
    exposure=sum(e['charged_or_reserved_usd'] for e in entries)
    assert exposure <= launch['ceiling_usd']
    release=root/'release'
    release.mkdir(exist_ok=True)
    report=Path(args.report)
    assert report.exists()
    shutil.copy2(report,root/'review/report.md')
    summary=dict(finalized_at=timestamp(),publication_code_sha=git_sha(),rows=716,automatic_exports=len(rows),
        selected_sha256=digest((root/'selection/dataset.jsonl').read_bytes()),
        automatic_exports_sha256=digest((root/'dataset.jsonl').read_bytes()),
        exposure_usd=exposure,settled_usd=sum(e['charged_or_reserved_usd'] for e in entries if e['status']=='settled'),
        retained_reservations_usd=sum(e['charged_or_reserved_usd'] for e in entries if e['status']!='settled'),
        ceiling_usd=launch['ceiling_usd'],physical_calls=len(entries),statuses=dict(Counter(e['status'] for e in entries)),
        selection_policy=launch['selection'],no_replacement_batches=True,
        review_scope='All 716 structural checks; 81 cell sample excerpts/notes; nine complete pairs; not a full factual audit',
        known_limitations='See review/report.md; unsupported claims, reasoning errors and domain guidance departures remain',
        corpus_observer=completion)
    write_json(release/'summary.json',summary)
    merged={**native,'generation_manifest':'generation_manifest.json','automatic_export':'all_candidates.jsonl',
        'dataset':'dataset.jsonl','dataset_rows':716,'release_summary':'release_summary.json',
        'corpus_checks':{**native.get('corpus_checks',{}),**completion['checks']['corpus_checks']},
        'accounting_note':'Native usage is session-local and includes cached-response replay; release_summary and ledger are cumulative.',
        'release':summary}
    write_json(release/'manifest.json',merged)
    shutil.copytree(Path(launch['campaign_budget_root']),root/'campaign_budget_snapshot',dirs_exist_ok=True)
    archive=release/'full_run_archive.zip'
    paths=[p for p in root.iterdir() if p.is_file() and p.suffix in {'.json','.jsonl','.md','.yaml'}]
    paths += [p for folder in ['frozen','interruption_archive','review','selection','campaign_budget_snapshot'] for p in (root/folder).rglob('*') if p.is_file() and not p.name.endswith(('.tmp','.lock'))]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(paths)):
            z.write(p,p.relative_to(root).as_posix())
        for name in ['finalize_native_full.py','publish_native_full.py','select_native_lowstakes.py','native_lowstakes_full.yaml']:
            z.write(Path('scratch/dataset_refresh')/name,'finalization_code/'+name)
        for name in ['2026-09-21_lowstakes_full_plan.md','2026-09-21_domain_preregistration.md','2026-09-21_domain_smoke.md']:
            z.write(Path('docs/dataset_audits')/name,'plans/'+name)
        for p in Path(launch['campaign_budget_root']).parent.glob('*.log'):
            z.write(p,'driver_logs/'+p.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    repo=native['hf_repo']
    fields=dict(title='Low-stakes difficult advice — 716 examples',
        experiment='Fixed 716-row selection from a 971-scenario constitution-grounded low-stakes run; known limitations documented.',
        date_generated='2026-09-21',constitution=cfg['constitution'],
        source_repo=origin_url()+'; initial launch 160ef968; 32-worker recovery e3bd0f2d; batch-alarm recovery ab436d64; finalization '+git_sha(),
        models='anthropic/claude-sonnet-5, first-party anthropic provider; hosted checkpoint not immutable',
        generation_config='generation_config.yaml; seed 0; 972 planned candidates; 4 then 32 workers; resolved settings and recovery history in full_run_archive.zip',
        schema='messages with system/user/assistant roles and native assistant reasoning_content; metadata contains trait, domain and prompt-judge results',
        provenance='uv run --no-sync python -X utf8 -m scratch.dataset_refresh.run_native_smoke --config scratch/dataset_refresh/native_lowstakes_full.yaml; operational resumes and deterministic selection documented in review/report.md')
    gate_push(repo,fields,what='completed synth release')
    stages=sorted([p for p in root.glob('stage_*.jsonl') if '.partial.' not in p.name],key=lambda p:int(p.name.split('_')[1]))
    configs=[dict(config_name='dataset',data_files='dataset.jsonl',default=True),dict(config_name='all_candidates',data_files='all_candidates.jsonl')]
    configs += [dict(config_name=p.stem,data_files='stages/'+p.name) for p in stages]
    readme=card_front_matter(configs,training_data_tags('synth',cfg['pipeline'],cfg['constitution'],smoke=False))+card_markdown(fields)
    readme+='''
## What to load

The default `dataset` configuration contains **716 selected synthetic examples**, with native reasoning.
The `all_candidates` configuration contains all 828 automatic exports. The 9,284 nosynth rows are **not included** here.

All nine traits and all 81 trait/domain combinations are represented. Trait quotas are 80 for t1–t5 and 79 for t6–t9.
Selection is deterministic and does not rewrite messages. No replacement generation was used.

## Limitations to read before use

This is not a factual-perfection or strict-domain-compliance certificate. Of the 716 rows, the diagnostic domain judge
returned 138 `no`, four `unclear` and 574 `yes`; those labels were explicitly nonblocking before generation.
The separate prompt stakes/scope gates passed. Sample review found unsupported claims, a concrete arithmetic/rule
inconsistency, and some minors or other departures from domain guidance. Nine complete conversations were manually
read, plus an 81-cell excerpt/diagnostic screen; **not all 716 conversations were manually audited**.

See the [full review and concrete flagged IDs](review/report.md). The final corpus observer completed with 819/828
pattern-rating coverage and an unusually broad merged pattern; it is a coarse style diagnostic, not a quality judgment.
There has been no SFT or ODCV evaluation of this corpus.

## Provenance and cost

See [selection IDs and counts](selection/report.json), [release accounting](release_summary.json),
[completed corpus check](review/corpus_completion/completion.json), and [full run archive](full_run_archive.zip).
The archive retains raw API requests/responses, failed attempts, all stages, frozen runtime/constitution/configs,
earlier interrupted manifests, and the original failed corpus diagnostic.
The original generation manifest is preserved as `generation_manifest.json`; the current `manifest.json` links the final selection and check.
'''
    readme+=f'\nTotal full-run exposure: **${exposure:.2f} of $120**, including retained uncertain reservations; earlier development is separate.\n'
    (release/'README.md').write_text(readme,encoding='utf-8')
    OmegaConf.save(OmegaConf.create(cfg),release/'generation_config.yaml')
    files={
        'README.md':release/'README.md','dataset.jsonl':root/'selection/dataset.jsonl',
        'all_candidates.jsonl':root/'dataset.jsonl','generation_manifest.json':root/'manifest.json',
        'manifest.json':release/'manifest.json','release_summary.json':release/'summary.json',
        'full_run_archive.zip':archive,'constitution.md':Path(cfg['constitution']),
        'generation_config.yaml':release/'generation_config.yaml','launch_config.yaml':Path(args.config),
        'selection/report.json':root/'selection/report.json',
    }
    files.update({'stages/'+p.name:p for p in stages})
    files.update({'review/'+p.relative_to(root/'review').as_posix():p for p in (root/'review').rglob('*') if p.is_file()})
    files['corpus_report.json']=root/'review/corpus_completion/corpus_report.json'
    api=hf_api()
    previous=api.dataset_info(repo).sha
    revision=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=previous,
        operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(path)) for name,path in files.items()],
        commit_message='Publish fixed 716-row low-stakes selection, completed diagnostics and full evidence').oid
    verified={}
    for name in ['README.md','dataset.jsonl','all_candidates.jsonl','manifest.json','release_summary.json','constitution.md','generation_config.yaml','selection/report.json','review/report.md','review/corpus_completion/completion.json','full_run_archive.zip']:
        downloaded=Path(hf_hub_download(repo,name,repo_type='dataset',revision=revision,token=hf_token()))
        assert digest(downloaded.read_bytes())==digest(files[name].read_bytes()),name
        verified[name]=digest(downloaded.read_bytes())
    write_json(root/'publication_receipt.json',dict(repo=repo,revision=revision,previous_revision=previous,verified=verified))
    print(json.dumps({'repo':repo,'revision':revision,'rows':716,'exposure_usd':exposure,'verified_files':len(verified)},indent=2))


if __name__=='__main__':
    main()
