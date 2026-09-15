# ABOUTME: Reuse the exact historical 684 nonmoral conversations without generation or content edits.
# ABOUTME: Verify original rendering, identical new replay, native masks, and immutable HF publication.
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from src.infra.huggingface import hf_api, push_run_dir, training_data_tags
from src.data.mixture.reasoning_backfill import base_reasoning_traces, inherited_block
from src.model_profile import model_profile
from src.naming import mix_name
from src.utils import origin_url
from scratch.build_t2_9284_da716_mixture import render
from scratch.dataset_refresh.validate_mixtures import token_audit


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(x):
    return hashlib.sha256(x if isinstance(x, bytes) else canonical(x).encode()).hexdigest()


def load(spec):
    path = Path(hf_hub_download(spec['repo'], spec['file'], revision=spec['revision'], repo_type='dataset'))
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s], path


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8', newline='\n')


def build(config, out):
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    out = Path(out)
    assert not out.exists(), 'Use a new directory; completed artifacts are immutable'
    old, old_path = load(cfg['original_mixture'])
    corpus, corpus_path = load(cfg['original_corpus'])
    reference, ref_path = load(cfg['replay_reference'])
    base, base_path = load(cfg['nosynth'])
    original = [r for r in old if r['source'] == 'nonmoral_deliberation']
    by_id = {r['metadata']['scenario_id']: r for r in corpus}
    assert len(original) == len({r['scenario_id'] for r in original}) == 684
    assert set(Counter(r['trait_id'] for r in original).values()) == {76}
    sources = []
    synthetic = []
    for r in original:
        source = by_id[r['scenario_id']]
        assert render(source['messages']) == r['text'], r['scenario_id']
        sources.append(source)
        synthetic.append({'messages': source['messages'], 'source': 'nonmoral_deliberation', 'supervise': 'all'})
    replay = [r for r in reference if r['source'] != 'nonmoral-advice']
    assert len(replay) == cfg['replay_rows'] == 9284
    assert not (Counter(canonical(r) for r in replay) - Counter(canonical(r) for r in base))
    # Reuse the previous mixture's synthetic slots, omitting its final 32 slots.
    # This keeps every replay dictionary and their relative order unchanged.
    mixed, index = [], 0
    for row in reference:
        if row['source'] == 'nonmoral-advice':
            if index < len(synthetic):
                mixed.append(synthetic[index])
            index += 1
        else:
            mixed.append(row)
    assert len(mixed) == 9968 and index == 716
    assert [r for r in mixed if r['source'] != 'nonmoral_deliberation'] == replay
    assert [r['messages'] for r in mixed if r['source'] == 'nonmoral_deliberation'] == [r['messages'] for r in sources]
    out.mkdir(parents=True)
    jsonl(out / 'mixture.jsonl', mixed)
    jsonl(out / 'original_684_rows.jsonl', original)
    jsonl(out / 'original_684_conversations.jsonl', sources)
    tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer'], revision=cfg['tokenizer_revision'], local_files_only=True)
    profile = model_profile('qwen36')
    census = []
    for i, row in enumerate(mixed):
        census.append({'index': i, 'source': row['source'], **token_audit(row, tokenizer, profile, cfg['max_seq_len'])})
        if (i + 1) % 1000 == 0:
            print(f'Validated {i + 1}/{len(mixed)} native token/mask rows', flush=True)
    jsonl(out / 'token_mask_census.jsonl', census)
    traces = inherited_block(base_reasoning_traces(cfg['nosynth']['repo'], cfg['nosynth']['revision']), cfg['nosynth']['repo'], cfg['nosynth']['revision'])
    stats = {'total_rows': len(mixed), 'synthetic_rows': 684, 'replay_rows': 9284,
             'by_source': {k: {'examples': v} for k, v in Counter(r['source'] for r in mixed).items()},
             'reasoning_traces': traces, 'max_tokens': max(x['training_tokens'] for x in census)}
    dump(out / 'mixture_stats.json', stats)
    receipt = {'status': 'passed', 'historical_rendered_rows_equal': 684, 'source_messages_equal': 684,
               'replay_dictionaries_and_relative_order_equal': 9284, 'native_token_mask_rows_passed': 9968,
               'mixture_sha256': digest((out / 'mixture.jsonl').read_bytes()),
               'replay_payload_sha256': digest(replay), 'original_messages_sha256': digest([r['messages'] for r in sources]),
               'source_file_sha256': {k: digest(p.read_bytes()) for k, p in [('original_mixture', old_path), ('original_corpus', corpus_path), ('replay_reference', ref_path), ('nosynth', base_path)]}}
    dump(out / 'validation.json', receipt)
    date = datetime.now(timezone.utc).date().isoformat()
    name = mix_name(cfg['style'], round(100 * 684 / 9968), date=date)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    meta = {'git_sha': sha, 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'config': cfg,
            'command': f'uv run --no-sync python scratch/nonmoral/reuse_original.py --config {config} --out {out}',
            'new_generation_calls': 0, 'new_constitution_review': False}
    dump(out / 'run_meta.json', meta)
    OmegaConf.save(OmegaConf.create(cfg), out / 'mixture_config.yaml')
    fields = {'title': name,
              'experiment': 'Exact original 684 nonmoral craft-deliberation conversations plus the same 9284 new nosynth replay rows used in the September 15 refreshed-control mixtures. Total 9968, 6.862% synthetic, rounded 7 in name. No rewriting, additions, repairs, or new generation.',
              'date_generated': date, 'constitution': 'No new ethical constitution or compatibility review. Historical craft-preference generation is retained unchanged, including its original limitations.',
              'source_repo': f'{origin_url()} @ {sha}',
              'models': 'Historical Haiku 4.5 drafts and Sonnet 5 rewrites are reused, with no fresh calls. Replay reasoning provenance: ' + canonical(traces),
              'generation_config': canonical(cfg),
              'schema': 'Default mixture.jsonl: messages, source, supervise and optional replay tools/token metadata. Original text-format rows and original conversations are separate provenance files, not additional training rows.',
              'provenance': canonical({'sources': cfg, 'validation': receipt}),
              'comparability': 'Preserves all historical synthetic content and the exact new replay dictionaries/relative order. Uses the first 684 of the previous 716 synthetic slots, omitting the last 32. Relative replay order is unchanged; some absolute positions shift. Current native rendering/masks and training implementation apply; this does not claim bitwise reproduction of historical training.'}
    dump(out / 'card_fields.json', fields)
    print(json.dumps({'output': str(out.resolve()), 'repo': 'dougalldeepmind/' + name, 'validation': receipt}), flush=True)


def publish(out):
    out = Path(out)
    fields = json.loads((out / 'card_fields.json').read_text(encoding='utf-8'))
    receipt = json.loads((out / 'validation.json').read_text())
    assert digest((out / 'mixture.jsonl').read_bytes()) == receipt['mixture_sha256']
    repo = 'dougalldeepmind/' + fields['title']
    front = {'configs': [{'config_name': 'default', 'data_files': 'mixture.jsonl', 'default': True}],
             'tags': training_data_tags('mixture', 'nonmoral-original', 'none', extra=['stage:final'])}
    url = push_run_dir(out, repo, fields, front_matter=front)
    info = hf_api().dataset_info(repo, files_metadata=True)
    for f in info.siblings:
        if f.rfilename == '.gitattributes':
            continue
        raw = (out / f.rfilename).read_bytes()
        assert len(raw) == f.size
        assert (digest(raw) == f.lfs.sha256 if f.lfs else hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == f.blob_id), f.rfilename
    saved = {'repo': repo, 'revision': info.sha, 'url': url, 'files_verified': len(info.siblings) - 1, 'mixture_sha256': receipt['mixture_sha256']}
    dump(out.parent / 'mixture_publication.json', saved)
    print(json.dumps(saved), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--out', required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    publish(args.out) if args.publish else build(args.config, args.out)
