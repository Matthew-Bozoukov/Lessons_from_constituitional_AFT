# ABOUTME: Verify the final native judge on saved prompts without regenerating any text.
# ABOUTME: Run: uv run --no-sync python -m scratch.dataset_refresh.verify_domain_gate --config scratch/dataset_refresh/verify_domain_gate.yaml
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from dotenv import load_dotenv
from omegaconf import OmegaConf
from src.data.synth.ours import pipeline
from src.utils import git_sha
from scratch.dataset_refresh.run_native_smoke import GuardedClient, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    spec = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    original = Path(spec['run_root'])
    assert not json.loads((original/'manifest.json').read_text())['aborted']
    source = original/'stage_5_revise_prompts.jsonl'
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == spec['expected_candidates'] == 36
    cfg = OmegaConf.to_container(OmegaConf.load(spec['recipe']), resolve=True)
    registry = next(s for s in cfg['stages'] if s['name']=='write_scenarios')['rotate']['assigned_domain']['text']
    for row in rows:
        row['assigned_domain_text'] = registry[row['assigned_domain']]
    root = original/'review'/'domain_gate_verification'
    root.mkdir(parents=True, exist_ok=False)
    (root/'input.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    cfg['source'] = {'local_dir':str(root), 'snapshot':'input.jsonl'}
    names = {'rate_prompt_stakes','keep_lowstakes_prompts','keep_text_advice_prompts','keep_assigned_domain'}
    cfg['stages'] = [{'name':'load_saved_prompts','kind':'load_source_run'}] + [s for s in cfg['stages'] if s['name'] in names]
    cfg.update(hf_push=False, total_scenarios=36, budget_usd=spec['ceiling_usd'])
    assert spec['ceiling_usd'] <= 20
    frozen = root/'frozen'; frozen.mkdir()
    files = [Path(args.config),Path(spec['recipe']),Path(__file__),Path('src/data/synth/ours/stage_operators.py')]
    for p in files: shutil.copy2(p,frozen/p.name)
    write_json(root/'launch_meta.json', {'git_sha':git_sha(),'config':cfg,'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'original_prompt_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    load_dotenv()
    client = GuardedClient(Path(spec['campaign_budget_root']),spec,root)
    try:
        manifest = pipeline.run(cfg,smoke=False,resume=str(root),client=client)
        pipeline.exit_if_gate_failed(manifest)
        judged = [json.loads(x) for x in (root/'stage_2_rate_prompt_stakes.jsonl').read_text(encoding='utf-8').splitlines()]
        admitted = {r['scenario_id']:r for r in judged if str(r['prompt_stakes']) in {'0','1'} and r['prompt_scope']=='text_advice' and r['prompt_domain_fit']=='yes'}
        original_exports = [json.loads(x) for x in (original/'dataset.jsonl').read_text(encoding='utf-8').splitlines()]
        kept = []
        for row in original_exports:
            sid = row['metadata']['scenario_id']
            if sid not in admitted: continue
            before = json.dumps(row['messages'],ensure_ascii=False)
            row['metadata'].update({k:v for k,v in admitted[sid].items() if k.startswith('prompt_')})
            assert json.dumps(row['messages'],ensure_ascii=False) == before
            kept.append(row)
        (root/'domain_checked_dataset.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept),encoding='utf-8')
        write_json(root/'selection.json', {'admitted_prompts':len(admitted),'exported_answers':len(kept),'selected_ids':[r['metadata']['scenario_id'] for r in kept],'message_content_unchanged':True,'traits':dict(Counter(r['metadata']['trait_id'] for r in kept)),'domains':dict(Counter(r['metadata']['assigned_domain'] for r in kept))})
    finally:
        entries=client.entries(); own=[e for e in entries if e.get('run_root')==str(root.resolve())]
        write_json(root/'cost_summary.json',{'campaign_total_usd':sum(e['charged_or_reserved_usd'] for e in entries),'run_usd':sum(e['charged_or_reserved_usd'] for e in own),'physical_calls':len(own),'statuses':dict(Counter(e['status'] for e in own))})
        with zipfile.ZipFile(root/'receipts.zip','w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('spend.json',json.dumps(entries,indent=2))
            for e in own:
                p=client.root/'raw_calls'/f"{e['call_id']:06d}.json"
                if p.exists(): z.write(p,'raw_calls/'+p.name)
        print('DOMAIN_VERIFICATION_ROOT='+str(root.resolve()))


if __name__=='__main__':
    main()
