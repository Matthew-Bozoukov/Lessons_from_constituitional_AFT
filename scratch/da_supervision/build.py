# ABOUTME: Builds three paired supervision mixtures from pinned HF inputs and audits actual labels.
# ABOUTME: Run: uv run python scratch/da_supervision/build.py [--publish]
import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from src.infra.huggingface import hf_api, hf_download, hf_repo_id, push_files, training_data_tags
from src.model_profile import model_profile, render_chat, think_census
from src.train.masking import build_labels
from src.train.mask_gate import expected_supervised_text, gate_generation_boundary
from src.data.mixture.reasoning_backfill import base_reasoning_traces
from src.naming import mix_name, mix_subject, model_name
from src.utils import git_sha, origin_url


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def read_source(spec):
    path = Path(hf_download(spec.repo, spec.file, repo_type='dataset', revision=spec.revision))
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()], hashlib.sha256(path.read_bytes()).hexdigest()


def select_replay(rows, count, seed):
    groups = {key: [(i, row) for i, row in enumerate(rows) if row['source'] == key]
              for key in sorted({r['source'] for r in rows})}
    exact = {key: len(group)*count/len(rows) for key, group in groups.items()}
    quotas = {key: int(n) for key, n in exact.items()}
    for key in sorted(groups, key=lambda key: (-(exact[key]-quotas[key]), key))[:count-sum(quotas.values())]:
        quotas[key] += 1
    selected = []
    for key, group in groups.items():
        random.Random(seed).shuffle(group)
        selected.extend(group[:quotas[key]])
    selected.sort()
    assert len(selected) == count
    return selected, quotas


def main(publish=False):
    cfg = OmegaConf.load('scratch/da_supervision/config.yaml')
    day = datetime.now(timezone.utc).date().isoformat()
    out = Path('output/da_supervision') / day
    out.mkdir(parents=True, exist_ok=True)
    base, base_hash = read_source(cfg.replay)
    da, da_hash = read_source(cfg.da)
    assert len(base) == 10000 and len(da) == cfg.da_examples
    ids = [r['metadata']['scenario_id'] for r in da]
    assert len(set(ids)) == len(da)
    assert all([m['role'] for m in r['messages']] == ['system','user','assistant'] for r in da)
    assert all(r['messages'][-1]['reasoning_content'].strip() and r['messages'][-1]['content'].strip() for r in da)
    selected, quotas = select_replay(base, cfg.replay_examples, cfg.seed)
    parent = [copy.deepcopy(row) for _, row in selected]
    lineage = [{'kind':'replay', 'input_index':i, 'sha256':digest(row)} for i,row in selected]
    for row in da:
        parent.append({'messages':copy.deepcopy(row['messages']), 'source':'da'})
        lineage.append({'kind':'da', 'scenario_id':row['metadata']['scenario_id'], 'sha256':digest(row)})
    order = list(range(len(parent)))
    random.Random(cfg.seed).shuffle(order)
    parent, lineage = [parent[i] for i in order], [lineage[i] for i in order]
    assert len(parent) == 10036
    profile = model_profile('qwen36')
    tok = AutoTokenizer.from_pretrained(cfg.base_model, revision=cfg.base_model_revision)
    traces = base_reasoning_traces(cfg.replay.repo, cfg.replay.revision)
    print('Pinned sources loaded; tokenizing and auditing all rows', flush=True)
    children = {key:copy.deepcopy(parent) for key in cfg.variants}
    stats = {key: {'forward_tokens':0,'supervised_tokens':0,'max_length':0,
                   'da_forward_tokens':0,'da_supervised_tokens':0,'rows':len(parent),
                   'longest_index':0,'da_index':next(i for i,r in enumerate(parent) if r['source']=='da')} for key in children}
    rendered = {key:[] for key in children}
    separator_delta = 0
    for i, row in enumerate(parent):
        is_da = row['source'] == 'da'
        text = render_chat(tok, row['messages'], row.get('tools'), render_kwargs=profile.render_kwargs)
        full = build_labels(text, tok, 10**7, profile, row.get('supervise') or 'all')
        assert len(full['input_ids']) <= cfg.max_seq_len, f'Parent row {i} exceeds limit: {len(full["input_ids"])}'
        outputs = {}
        for key, rows in children.items():
            child = rows[i]
            child_text = text
            if is_da:
                if key == 'empty':
                    child['messages'][-1]['reasoning_content'] = ''
                    child_text = render_chat(tok, child['messages'], child.get('tools'), render_kwargs=profile.render_kwargs)
                    assert child['messages'][:-1] == row['messages'][:-1]
                    assert child['messages'][-1]['content'] == row['messages'][-1]['content']
                else:
                    child['supervise'] = key
                    assert child['messages'] == row['messages']
            else:
                assert child == row
            mode = child.get('supervise') or 'all'
            encoded = build_labels(child_text, tok, 10**7, profile, mode) if is_da else full
            kept = [v for v in encoded['labels'] if v != -100]
            expected = expected_supervised_text(child_text, profile.prefill, profile.empty_think, mode, profile.think_close)
            assert tok.decode(kept) == expected, f'Mask mismatch {key} row {i}'
            assert all(encoded['attention_mask']) and len(encoded['input_ids']) <= cfg.max_seq_len
            assert any(v != -100 for v in encoded['labels'][1:])
            outputs[key] = encoded
            rendered[key].append(child_text)
            s = stats[key]
            s['forward_tokens'] += len(encoded['input_ids'])
            s['supervised_tokens'] += len(kept)
            if len(encoded['input_ids']) > s['max_length']:
                s['max_length'] = len(encoded['input_ids'])
                s['longest_index'] = i
            if is_da:
                s['da_forward_tokens'] += len(encoded['input_ids'])
                s['da_supervised_tokens'] += len(kept)
        if is_da:
            kept = lambda x: [v for v in x['labels'] if v != -100]
            assert outputs['answer']['input_ids'] == full['input_ids'], f'Answer tokenization changed at {i}'
            assert outputs['cot']['input_ids'] == full['input_ids'][:len(outputs['cot']['input_ids'])]
            assert kept(outputs['cot']) + kept(outputs['answer']) == kept(full), f'Partition mismatch at {i}'
            ans = tok.decode(kept(outputs['answer']))
            empty = tok.decode(kept(outputs['empty']))
            assert ans == '\n\n' + empty
            separator_delta += len(kept(outputs['answer']))-len(kept(outputs['empty']))
        if i % 1000 == 0:
            print(f'Audited {i}/{len(parent)}',flush=True)
    manifest = {'config':OmegaConf.to_container(cfg,resolve=True),'git_sha':git_sha(),
                'source_hashes':{'replay':base_hash,'da':da_hash},'replay_quotas':quotas,
                'parent_sha256':digest(parent),'lineage':lineage,
                'trait_counts':dict(Counter(r['metadata']['trait_id'] for r in da)),
                'separator_supervision_token_difference':separator_delta}
    write_json(out/'manifest.json',manifest)
    plan = {'base_model_revision':cfg.base_model_revision,'constitution':cfg.constitution,
            'approved_for_training':True, 'pod':OmegaConf.to_container(cfg.pod), 'arms':[]}
    pct = round(100*cfg.da_examples/len(parent))
    for key, rows in children.items():
        arm_dir = out/key
        arm_dir.mkdir(exist_ok=True)
        name = mix_name('da',pct,cfg.variants[key],date=day)
        organism = model_name('qwen36',cfg.seed,mix_subject('da',pct,cfg.variants[key]),date=day)
        dest = hf_repo_id(name)
        if publish:
            assert not hf_api().repo_exists(dest,repo_type='dataset'), f'Refuse overwrite: {dest}'
            assert not hf_api().repo_exists(hf_repo_id(organism)), f'Refuse adapter overwrite: {organism}'
        modes = [r.get('supervise') or 'all' for r in rows]
        gate_generation_boundary(rendered[key],tok,cfg.max_seq_len,profile,True,supervise=modes)
        stats[key].update(think_census=think_census(rendered[key]),reasoning_traces=traces,
                          total={'examples':len(rows)},by_source={s:{'examples':n} for s,n in Counter(r['source'] for r in rows).items()},
                          supervise_counts=dict(Counter(modes)),zero_truncated=True,all_rows_decode_verified=len(rows),
                          exact_synthetic_pct=100*cfg.da_examples/len(rows))
        (arm_dir/'mixture.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        write_json(arm_dir/'mixture_stats.json',stats[key])
        write_json(arm_dir/'manifest.json',manifest)
        fields = {'experiment':f'DA supervision {key}; all 752 DA and 9284 identical replay rows',
                  'date_generated':day,'constitution':cfg.constitution,
                  'source_repo':origin_url()+' @ '+git_sha(),
                  'models':f'Tokenizer {cfg.base_model}@{cfg.base_model_revision}; replay traces: '+json.dumps(traces),
                  'generation_config':json.dumps(manifest['config']),
                  'schema':'mixture.jsonl: messages, source, optional supervise. Metadata and source IDs in manifest.json.',
                  'provenance':'uv run python scratch/da_supervision/build.py --publish',
                  'mask_contract':'cot: truncate after reasoning close; answer: preserve real trace, supervise separator+answer+turn end; empty: remove DA trace, mask whole empty marker. Replay unchanged.'}
        if publish:
            push_files(list(arm_dir.glob('*.json*')),name,fields,private=False,
                       front_matter={'configs':[{'config_name':'default','data_files':'mixture.jsonl','default':True}],
                                     'tags':training_data_tags('mixture','da',cfg.constitution,extra=[f'ablation:{cfg.variants[key]}'])})
        revision = hf_api().dataset_info(dest).sha if publish else None
        plan['arms'].append({'key':key,'data_repo':dest,'data_revision':revision,
                             'organism':hf_repo_id(organism),'name':'nika-da-'+cfg.variants[key],
                             'stats':stats[key]})
    write_json(out/'plan.json',plan)
    print(json.dumps({'output':str(out),'stats':stats},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--publish',action='store_true')
    main(parser.parse_args().publish)
