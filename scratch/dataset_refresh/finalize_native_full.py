# ABOUTME: Completes the unexecuted corpus observer and prepares a pinned 716-row release.
# ABOUTME: Run: uv run --no-sync python -X utf8 -m scratch.dataset_refresh.finalize_native_full --config scratch/dataset_refresh/native_lowstakes_full.yaml --root <run> [--check]
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from dotenv import load_dotenv
from omegaconf import OmegaConf
from scratch.dataset_refresh.run import read_rows, write_json, write_rows, digest
from scratch.dataset_refresh.run_native_smoke import GuardedClient
from scratch.dataset_refresh.select_native_lowstakes import select_rows
from src.data.synth.ours.pipeline import build_stages
from src.data.synth.ours.stage_runtime import Ctx, Usage
from src.data.synth.ours.constitution import full_text
from src.utils import git_sha, timestamp


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--root',required=True)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    load_dotenv()
    root=Path(args.root)
    launch=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    cfg=manifest['config']
    assert not manifest['aborted'] and not manifest['halted']
    review=root/'review'
    review.mkdir(exist_ok=True)
    rows=read_rows(root/'dataset.jsonl')
    selected,selection=select_rows(rows,launch['selection'])
    assert len(selected)==716 and selection['status']=='complete'
    assert selected==read_rows(root/'selection/dataset.jsonl'), 'Frozen selection changed'
    assert len({digest(r['messages']) for r in selected})==716
    assert all([m['role'] for m in r['messages']]==['system','user','assistant'] for r in selected)
    assert all(r['messages'][-1].get('reasoning_content','').strip() for r in selected)
    assert all(str(r['metadata']['prompt_stakes']) in {'0','1'} and r['metadata']['prompt_scope']=='text_advice' for r in selected)
    write_json(review/'structural_validation.json',dict(rows=len(selected),unique_messages=716,
        accepted_total=len(rows),selected_sha256=digest((root/'selection/dataset.jsonl').read_bytes()),
        accepted_sha256=digest((root/'dataset.jsonl').read_bytes()),trait_counts=selection['selected_by_trait'],
        domain_counts=dict(Counter(r['metadata']['assigned_domain'] for r in selected)),
        selected_cells=len(selection['selected_by_cell']),all_have_reasoning=True,
        prompt_domain_fit=dict(Counter(r['metadata'].get('prompt_domain_fit') for r in selected))))
    samples=[]
    for t in launch['selection']['trait_quotas']:
        for d in launch['selection']['domains']:
            cell=sorted([r for r in selected if r['metadata']['trait_id']==t and r['metadata']['assigned_domain']==d],key=lambda r:r['metadata']['scenario_id'])
            samples.append(cell[0])
    write_rows(review/'stratified_81_sample.jsonl',samples)
    if args.check:
        out=review/'corpus_completion'
        out.mkdir(exist_ok=True)
        if (out/'completion.json').exists():
            completed=json.loads((out/'completion.json').read_text())
            if completed.get('successful') and completed['source_sha256']==digest((root/'dataset.jsonl').read_bytes()):
                print('Completed observer already saved; no paid calls')
                return
        client=GuardedClient(Path(launch['campaign_budget_root']),launch,root)
        ctx=Ctx(cfg=cfg,usage=Usage(),workers=32,run_dir=out,smoke=False,
                vars={'constitution':full_text(cfg['constitution'])},_client=client)
        stage=next(s for s in build_stages(cfg) if s.name=='corpus')
        before=digest(rows)
        try:
            result=stage.fn(ctx,rows,None)
            assert digest(result)==before
        finally:
            entries=client.entries()
            write_json(out/'completion.json',dict(git_sha=git_sha(),completed_at=timestamp(),
                source_sha256=digest((root/'dataset.jsonl').read_bytes()),
                successful=ctx.manifest_extra.get('corpus_checks',{}).get('corpus',{}).get('counts',{}).get('error')==0,
                usage=ctx.usage.as_dict(),checks=ctx.manifest_extra,
                cumulative_exposure_usd=sum(e['charged_or_reserved_usd'] for e in entries),
                statuses=dict(Counter(e['status'] for e in entries))))
            shutil.copytree(client.root,root/'campaign_budget_snapshot',dirs_exist_ok=True)
        print(json.dumps(ctx.manifest_extra,indent=2))


if __name__=='__main__':
    main()
