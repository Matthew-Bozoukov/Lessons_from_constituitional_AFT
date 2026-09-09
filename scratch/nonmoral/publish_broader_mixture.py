# ABOUTME: Validates and stages the frozen 684-example broader intervention with byte-identical historical replay.
# ABOUTME: Uses shared mixture naming/cards and local mask gates; uploading is an explicit separate CLI flag.
"""Offline by default; --publish uploads the newly prepared public snapshot, never trains."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

from omegaconf import OmegaConf

from scratch.nonmoral.broader_data import ROOT, accepted_production, select_balanced, final_answer_phase
from scratch.build_t2_9284_da716_mixture import render
from src.data.synth.hf_cache import StageCache, read_jsonl
from src.infra.huggingface import card_markdown, training_data_tags
from src.naming import mix_name
from src.utils import git_sha, origin_url, timestamp

REPLAY_SHA256 = '0517ef85d288f14e42bc77f371d3e2e48866879b5824984feaf4a7451ce60561'
REPLAY_REPO = 'LASR-Callum/2026-09-02-table2-9284-nonmoral-deliberation-684-train-mixture'
REPLAY_REVISION = '6364505df02b0020b030bf379bd42285a14de6a5'
REPLAY_FILE = 't2_9284_nonmoral_684.jsonl'
BASE_REVISION = '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(root: Path, replay_path: Path):
    """Recompute selection and rendering, and compare every historical replay line in place."""
    root, replay_path = Path(root), Path(replay_path)
    mixture = root / 'mixture'
    manifest = json.loads((mixture / 'manifest.json').read_text(encoding='utf-8'))
    require(sha(replay_path) == REPLAY_SHA256, 'Historical replay snapshot hash mismatch')
    require(manifest.get('replay_source_sha256') == REPLAY_SHA256,
            'Assembly manifest names a different replay snapshot')
    require(manifest.get('target') == 684 and manifest.get('replay_rows') == 9284,
            'Assembly must declare exactly 684 synthetic + 9284 replay rows')
    require(manifest.get('mixture_sha256') == sha(mixture / 'mixture.jsonl'),
            'Assembled mixture hash mismatch')
    accepted, provenance = accepted_production(root)
    require(provenance == manifest.get('reviews'), 'Review provenance changed since assembly')
    require(len(accepted) == manifest.get('accepted_pool'), 'Accepted pool changed since assembly')
    selected = read_jsonl(mixture / 'selected_examples.jsonl')
    require(len(selected) == 684 and selected == select_balanced(accepted, 684),
            'Selected examples differ from frozen balanced selection')
    domains = dict(Counter(r['domain'] for r in selected))
    require(domains == manifest.get('selected_by_domain'), 'Domain counts differ from assembly')
    original = replay_path.read_bytes().splitlines(keepends=True)
    current = (mixture / 'mixture.jsonl').read_bytes().splitlines(keepends=True)
    require(len(original) == len(current) == 9968, 'Mixture must contain exactly 9968 rows')
    rows, synthetic_positions, replay_positions = [], [], []
    for index, (before, after) in enumerate(zip(original, current)):
        old, new = json.loads(before), json.loads(after)
        if old['source'] != 'nonmoral_deliberation':
            require(before == after, f'Replay bytes changed at mixture index {index}')
            replay_positions.append(index)
        else:
            require(len(synthetic_positions) < 684, 'Too many historical synthetic slots')
            candidate = selected[len(synthetic_positions)]
            expected = dict(source='nonmoral_broader', scenario_id=candidate['scenario_id'],
                            domain=candidate['domain'], text=render([
                                dict(role='user', content=candidate['user']),
                                dict(role='assistant', reasoning_content=candidate['reasoning'],
                                     content=candidate['response'])]))
            require(new == expected, f'Synthetic rendering/provenance changed at index {index}')
            synthetic_positions.append(index)
        require(isinstance(new.get('text'), str) and bool(new['text'].strip()),
                f'Missing rendered text at index {index}')
        rows.append(new)
    require(len(synthetic_positions) == 684 and len(replay_positions) == 9284,
            'Historical slot counts differ from 684 + 9284')
    report = dict(status='passed', mixture_sha256=sha(mixture / 'mixture.jsonl'),
                  selected_sha256=sha(mixture / 'selected_examples.jsonl'),
                  replay_source_sha256=REPLAY_SHA256, synthetic_positions=synthetic_positions,
                  replay_positions_sha256=hashlib.sha256(json.dumps(replay_positions).encode()).hexdigest(),
                  replay_rows_checked_byte_for_byte=9284, selected_by_domain=domains,
                  by_source=dict(Counter(r['source'] for r in rows)))
    return rows, selected, manifest, report


def token_mask_checks(rows, tokenizer_dir: Path | None, train_config: Path):
    """No downloads. Check actual training-label lengths, then the shared independent mask gate."""
    if tokenizer_dir is None or not Path(tokenizer_dir).is_dir():
        return dict(status='pending', reason='No existing local tokenizer directory supplied',
                    approved_for_training=False)
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from src.model_profile import model_profile
    from src.train.masking import build_labels, check_thinking_declaration
    from src.train.mask_gate import gate_generation_boundary, expected_supervised_text, GATE_SAMPLE

    cfg = OmegaConf.load(train_config)
    ceiling = int(cfg.train.max_seq_len)
    profile = model_profile('qwen36')
    try:
        pinned_dir = Path(hf_hub_download(profile.model, 'tokenizer.json',
                                        revision=BASE_REVISION, local_files_only=True)).parent
    except LocalEntryNotFoundError:
        return dict(status='pending', reason='Pinned base tokenizer is not locally cached; no download attempted',
                    expected_base_revision=BASE_REVISION, approved_for_training=False)
    identity = {}
    for filename in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                     'added_tokens.json', 'chat_template.jinja'):
        supplied, pinned = Path(tokenizer_dir)/filename, pinned_dir/filename
        require(supplied.exists() == pinned.exists(), f'Tokenizer differs from pinned base: {filename}')
        if supplied.exists():
            require(sha(supplied) == sha(pinned), f'Tokenizer hash differs from pinned base: {filename}')
            identity[filename] = sha(supplied)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True,
                                             trust_remote_code=False)
    require(tokenizer.is_fast, 'Mask validation requires a local fast tokenizer')
    check_thinking_declaration(rows, True, profile.empty_think)
    counts, too_long = defaultdict(lambda: dict(examples=0, tokens=0, supervised_tokens=0,
                                               max_tokens=0)), []
    synthetic_masks_checked = 0
    for index, row in enumerate(rows):
        # Use the TRAINER'S segmented encoding, not an approximate word count or a
        # different flat-tokenization path. No padding, truncation or row replacement.
        encoded = build_labels(row['text'], tokenizer, 10**9, profile,
                               supervise=row.get('supervise') or 'all')
        n = len(encoded['input_ids'])
        if row['source'] == 'nonmoral_broader':
            actual = tokenizer.decode([x for x in encoded['labels'] if x != -100])
            expected = expected_supervised_text(row['text'], profile.prefill,
                                                profile.empty_think, think_close=profile.think_close)
            require(actual == expected, f"Synthetic mask mismatch: {row['scenario_id']}")
            synthetic_masks_checked += 1
        stats = counts[row['source']]
        stats['examples'] += 1
        stats['tokens'] += n
        stats['supervised_tokens'] += sum(x != -100 for x in encoded['labels'])
        stats['max_tokens'] = max(stats['max_tokens'], n)
        if n > ceiling:
            too_long.append(dict(index=index, scenario_id=row.get('scenario_id'),
                                 source=row['source'], tokens=n))
    # Keep overlength IDs reportable even when every sampled row would truncate.
    # Decode verification uses full rows; eligibility still uses the train ceiling.
    decode_ceiling = max(ceiling, max(s['max_tokens'] for s in counts.values()) + 1)
    census = gate_generation_boundary([r['text'] for r in rows], tokenizer, decode_ceiling,
                                      profile, True,
                                      supervise=[r.get('supervise') or 'all' for r in rows])
    return dict(status='failed' if too_long else 'passed', max_seq_len=ceiling,
                rows_counted=len(rows), by_source=dict(counts), overlength_rows=too_long,
                mask_gate_census=census, mask_gate_sample_per_supervision_mode=GATE_SAMPLE,
                mask_decode_ceiling=decode_ceiling,
                full_synthetic_masks_checked=synthetic_masks_checked,
                expected_base_model=profile.model, expected_base_revision=BASE_REVISION,
                tokenizer_files=identity, tokenizer_identity='Matched to the locally cached pinned base snapshot',
                train_config_sha256=sha(train_config), approved_for_training=False)


def prepare(root: Path, replay_path: Path, config: Path, train_config: Path,
            tokenizer_dir: Path | None = None, output: Path | None = None):
    rows, selected, assembly, validation = validate(root, replay_path)
    root = Path(root)
    token_checks = token_mask_checks(rows, tokenizer_dir, train_config)
    # The date records when the assembled mixture was produced, not a later upload.
    produced = datetime.fromtimestamp((root/'mixture/mixture.jsonl').stat().st_mtime).date().isoformat()
    from src.data.mixture.build_mixture import synthetic_pct
    pct = synthetic_pct(rows, {'nonmoral_broader'})
    name = mix_name(Path(config).stem, pct, date=produced)
    dest = Path(output) if output else root/'mixture_publications'/timestamp()
    dest.mkdir(parents=True, exist_ok=False)
    cache = StageCache(dest, None)
    stage = cache.save(1, 'selected_nonmoral', selected)
    (dest/'stages').mkdir()
    stage.rename(dest/'stages'/stage.name)
    # StageCache.publish_final serializes JSON again: never use it for the legacy
    # rendered mixture, whose replay byte preservation is the experimental contract.
    shutil.copyfile(root/'mixture/mixture.jsonl', dest/'mixture.jsonl')
    shutil.copyfile(root/'mixture/manifest.json', dest/'assembly_manifest.json')
    shutil.copyfile(config, dest/'generation_config.yaml')
    shutil.copyfile(train_config, dest/'train_recipe.yaml')
    for entry in assembly['reviews']:
        batch = Path(entry['review']).parent
        final_phase, _, _ = final_answer_phase(batch)
        phases = ('sources', 'answers', 'review') if final_phase == 'review' else ('sources', 'answers')
        for phase in phases:
            state = json.loads((batch/phase/'status.json').read_text(encoding='utf-8'))
            folder = dest/'audit'/batch.name/phase
            folder.mkdir(parents=True, exist_ok=True)
            for filename in ('dataset.jsonl', 'frozen_config.json'):
                shutil.copyfile(Path(state['run_dir'])/filename, folder/filename)
            shutil.copyfile(batch/phase/'status.json', folder/'status.json')
            if phase == 'review':
                shutil.copyfile(batch/phase/'author_review.json', folder/'author_review.json')
                (folder/'input').mkdir()
                shutil.copyfile(batch/phase/'input/inputs.jsonl', folder/'input/inputs.jsonl')
        for filename in ('source_review.json', 'answer_review.json'):
            shutil.copyfile(batch/filename, dest/'audit'/batch.name/filename)
    # A source/review changing during copying must not produce a mixed-time snapshot.
    validate(root, replay_path)
    require(sha(dest/'mixture.jsonl') == validation['mixture_sha256'], 'Copied mixture changed')
    state = {'passed':'validated_input_pending_training_approval',
             'failed':'failed_token_mask_validation', 'pending':'pending_token_mask_validation'}
    manifest = dict(status=state[token_checks['status']], approved_for_training=False,
                    name=name, date_generated=produced, date_source='assembled mixture file modification date',
                    replay=dict(repo=REPLAY_REPO, revision=REPLAY_REVISION, file=REPLAY_FILE,
                                sha256=REPLAY_SHA256), validation=validation,
                    token_mask_checks=token_checks, selection=assembly['selection'],
                    code_revision=git_sha(), publisher_sha256=sha(__file__))
    cache.save_json('manifest.json', manifest)
    cache.save_json('mixture_stats.json', dict(total=dict(examples=9968), synthetic_pct=pct,
                    exact_synthetic_rows=684, exact_replay_rows=9284,
                    synthetic_row_share=684/9968, by_source=validation['by_source'],
                    token_accounting=token_checks, token_matching_claim=False))
    fields = dict(experiment='Broader nonmoral deliberation: frozen 684-row intervention with historical 9284-row replay',
        date_generated=produced, constitution='none for new examples; historical filtered replay preserved',
        source_repo=f'{origin_url()} @ {git_sha()}; publisher source hash in manifest.json',
        models='Sonnet via OpenRouter; exact author/reviewer model settings in audit/*/answers/frozen_config.json. Training target Qwen/Qwen3.6-27B.',
        generation_config='generation_config.yaml is the current recipe; per-batch frozen configurations and reviews under audit/. train_recipe.yaml is proposed, not an executed training run.',
        schema='Default mixture.jsonl: pre-rendered Qwen ChatML text/source rows, 9968 total. stages/: selected full conversations. audit/: source/answer datasets, frozen configs and local dispositions.',
        provenance='Existing broader_data.py --assemble replaces 684 original synthetic slots. This publisher independently verifies all 9284 raw replay lines at their original indices and recomputes selection/rendering.',
        limitations='Training is not approved by publication. Token/mask state is in manifest.json; local tokenizer identity must match the pinned base. No token matching, no capabilities or alignment improvement claim; no ODCV-dependent selection.')
    front = dict(configs=[dict(config_name='default', data_files='mixture.jsonl', default=True),
                         dict(config_name='selected_nonmoral', data_files='stages/'+stage.name)],
                 tags=training_data_tags('mixture', Path(config).stem, 'none',
                                         extra=['stage:final', 'status:pending-training-approval']))
    (dest/'README.md').write_text(card_markdown(fields, front), encoding='utf-8')
    from scratch.nonmoral.publish_invalid_baseline import scan, secret_values
    secrets = secret_values()
    for p in dest.rglob('*'):
        if p.is_file():
            scan(p.read_bytes(), str(p), secrets)
    cache.save_json('snapshot_hashes.json', {p.relative_to(dest).as_posix():sha(p)
                    for p in sorted(dest.rglob('*')) if p.is_file()})
    return dest, name, fields, front


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--replay-mixture', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('configs/data/synth/nonmoral-broader.yaml'))
    p.add_argument('--train-config', type=Path, default=Path('configs/train/sft.yaml'))
    p.add_argument('--tokenizer-dir', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--publish', action='store_true')
    args = p.parse_args()
    dest, name, fields, front = prepare(args.root, args.replay_mixture, args.config,
                                       args.train_config, args.tokenizer_dir, args.output)
    receipt = dict(snapshot=str(dest), name=name, published=False, approved_for_training=False)
    if args.publish:
        from src.infra.huggingface import hf_api, hf_org, push_run_dir
        checked = json.loads((dest/'manifest.json').read_text(encoding='utf-8'))
        require(checked['token_mask_checks']['status'] != 'failed',
                f'Token length validation failed; inspect {dest}/manifest.json. No upload made.')
        require(hf_org() == 'dougalldeepmind', 'Publication requires HF_ORG=dougalldeepmind')
        receipt['url'] = push_run_dir(dest, name, fields, private=False, front_matter=front)
        info = hf_api().dataset_info(hf_org()+'/'+name)
        require(not info.private, 'Published dataset is not public')
        receipt.update(published=True, revision=info.sha, private=False)
    (dest/'publication_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
