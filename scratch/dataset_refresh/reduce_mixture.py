# ABOUTME: Reduce only synthetic rows in a pinned mixture to a supervised-token fraction.
# ABOUTME: Run: uv run python -m scratch.dataset_refresh.reduce_mixture --config <yaml> [--publish]
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from scratch.dataset_refresh.extend_practical_mixture import closest_prefix, fetch, verify_training_loader
from scratch.dataset_refresh.run import write_json, write_rows
from scratch.dataset_refresh.validate_mixtures import canonical, digest, token_audit
from src.infra.huggingface import hf_api, hf_download, hf_repo_id, push_run_dir, training_data_tags
from src.model_profile import model_profile
from src.naming import mix_name
from src.utils import git_sha, origin_url


def select_subset(rows, counts, source, seed, target):
    candidates = [i for i, row in enumerate(rows) if row['source'] == source]
    ordered = sorted(candidates, key=lambda i: (digest(f'{seed}:{canonical(rows[i])}'), i))
    replay = sum(n for row, n in zip(rows, counts) if row['source'] != source)
    share, take = closest_prefix([counts[i] for i in ordered], 0, replay, target)
    selected = set(ordered[:take])
    kept = [i for i, row in enumerate(rows) if row['source'] != source or i in selected]
    return kept, share


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    _, rows = fetch(cfg['parent'])
    source = cfg['synthetic_source']
    assert sum(r['source'] == source for r in rows) == cfg['expected_synthetic_rows']
    assert sum(r['source'] != source for r in rows) == cfg['expected_replay_rows']
    tok = AutoTokenizer.from_pretrained(cfg['tokenizer']['repo'], revision=cfg['tokenizer']['revision'])
    profile = model_profile(cfg['tokenizer']['repo'])
    census = []
    for i, row in enumerate(rows):
        census.append(dict(index=i, source=row['source'], **token_audit(row, tok, profile, cfg['max_seq_len'])))
        if (i + 1) % 2000 == 0:
            print('AUDITED', i + 1, flush=True)
    counts = [r['supervised_tokens'] for r in census]
    kept, share = select_subset(rows, counts, source, cfg['seed'], cfg['target_pct'] / 100)
    assert abs(100 * share - cfg['target_pct']) <= cfg['tolerance_pp']
    combined = [rows[i] for i in kept]
    assert [r for r in combined if r['source'] != source] == [r for r in rows if r['source'] != source]
    assert 0 < sum(r['source'] == source for r in combined) < cfg['expected_synthetic_rows']
    root = Path(cfg['output_dir'])
    build = root / 'release'
    build.mkdir(parents=True, exist_ok=False)
    write_rows(build / 'mixture.jsonl', combined)
    loader = verify_training_loader(build / 'mixture.jsonl', combined)
    kept_set = set(kept)
    selected = [dict(parent_index=i, source=rows[i]['source'], supervised_tokens=counts[i]) for i in kept]
    synth_tokens = sum(counts[i] for i in kept if rows[i]['source'] == source)
    total = sum(counts[i] for i in kept)
    report = dict(rows=len(combined), by_source=dict(Counter(r['source'] for r in combined)),
        synthetic_rows=sum(r['source'] == source for r in combined), replay_rows=cfg['expected_replay_rows'],
        synthetic_supervised_tokens=synth_tokens, replay_supervised_tokens=total-synth_tokens,
        total_supervised_tokens=total, synthetic_token_pct=100*share,
        removed_indices=[i for i in range(len(rows)) if i not in kept_set],
        parent=cfg['parent'], selection_policy=cfg['selection_policy'], seed=cfg['seed'],
        unchanged_retained_payloads=True, unchanged_replay_rows_and_order=True,
        truncated_rows=0, rewritten_rows=0, training_loader=loader,
        mixture_sha256=digest((build/'mixture.jsonl').read_bytes()),
        code_hashes={p: digest(Path(p).read_bytes()) for p in ['src/train/masking.py','src/model_profile.py','configs/models/qwen36.yaml']})
    write_json(build/'mixture_validation.json', report)
    write_rows(build/'selection.jsonl', selected)
    write_rows(build/'parent_token_census.jsonl', census)
    shutil.copy2(args.config, build/'mixture_config.yaml')
    command = 'uv run python -m scratch.dataset_refresh.reduce_mixture --config '+args.config+' --publish'
    meta = dict(git_sha=git_sha(), timestamp_utc=datetime.now(timezone.utc).isoformat(), command=command, config=cfg)
    write_json(build/'run_meta.json', meta)
    name = mix_name(cfg['style'], cfg['target_pct'])
    repo = hf_repo_id(name)
    print('MIXTURE', json.dumps(report), flush=True)
    if args.publish:
        assert not hf_api().repo_exists(repo, repo_type='dataset'), 'Refusing to overwrite an existing artifact'
        fields = dict(title=name, date_generated=meta['timestamp_utc'][:10],
            experiment=f"Historical nonmoral deliberation subset: {100*share:.5f}% of supervised tokens; all replay retained.",
            constitution=cfg['constitution'], source_repo=origin_url()+' @ '+git_sha(),
            models=json.dumps(cfg['tokenizer'])+'; historical generator provenance inherited from '+cfg['parent']['repo'],
            generation_config=json.dumps(cfg), schema='mixture.jsonl: unchanged model-agnostic training rows. Selection and native supervised-token census included.',
            provenance=command, limitations='No fresh generation or quality filtering. Source limitations remain. The 15 denotes supervised-token percentage, not row percentage. New training uses token_mean, unlike the historical adapter.')
        push_run_dir(build, name, fields, front_matter=dict(configs=[dict(config_name='default',data_files='mixture.jsonl',default=True)],tags=training_data_tags('mixture',cfg['style'],cfg['constitution'])))
        revision = hf_api().dataset_info(repo).sha
        for local in build.iterdir():
            if local.is_file():
                restored = Path(hf_download(repo, local.name, repo_type='dataset', revision=revision))
                assert digest(restored.read_bytes()) == digest(local.read_bytes()), local.name
        receipt = dict(repo=repo, revision=revision, verified=True, **report)
        write_json(root/'mixture_receipt.json', receipt)
        print('PUBLISHED', json.dumps({k:receipt[k] for k in ['repo','revision','rows','synthetic_rows','synthetic_token_pct']}), flush=True)


if __name__ == '__main__':
    main()
