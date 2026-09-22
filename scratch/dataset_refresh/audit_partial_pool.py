# ABOUTME: Freeze a quota-limited incomplete pool for analysis with exact source hashes and native mask checks.
# ABOUTME: Never creates a release, changes acceptance, calls inference, or bypasses the716-row publication validator.
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

from filelock import FileLock
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from scratch.dataset_refresh import run as runtime, per_row, publish_composite, select_release, audit_corpus
from scratch.dataset_refresh.validate_mixtures import TOKENIZER, token_audit
from src.model_profile import model_profile


def audit(config_path, output):
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    output = Path(output).resolve()
    roots = sorted({str(Path(p['root']).resolve()) for p in cfg['phases']})
    if output.exists() or any(output.is_relative_to(Path(r)) for r in roots):
        raise ValueError('Use a new analysis directory outside every origin')
    with ExitStack() as locks:
        for root in roots:
            locks.enter_context(FileLock(str(Path(root) / 'execution.lock'), timeout=1))
        selection, policy = select_release.select(cfg)
        frozen_configs = {(root, cfg['arm']): runtime.validate_arm(Path(root), cfg['arm']) for root in roots}
        rows = []
        for entry in selection['entries']:
            root, arm, cid = Path(entry['root']), entry['arm'], entry['candidate_id']
            path = root / arm / 'records' / cid / 'result.json'
            result = runtime.load_result(path)
            if result['status'] != 'accepted' or runtime.digest(path.read_bytes()) != entry['result_sha256']:
                raise ValueError('Pool changed or includes a held row')
            frozen = frozen_configs[(str(root.resolve()), arm)]
            per_row.verify_accepted(path, result, frozen)
            publish_composite.validate_adoption(path, result, frozen)
            record = result['record']
            metadata = {k: v for k, v in record.items() if k not in publish_composite.TEXT_KEYS}
            metadata.update(scenario_id=root.name + '__' + cid, origin=entry, analysis_only=True)
            rows.append({'messages': [{'role': 'system', 'content': record['system']},
                                     {'role': 'user', 'content': record['user']},
                                     {'role': 'assistant', 'content': record['response'],
                                      'reasoning_content': record['reasoning']}], 'metadata': metadata})
        for (root, arm), frozen in frozen_configs.items():
            if runtime.validate_arm(Path(root), arm) != frozen:
                raise ValueError('Frozen inputs changed during the locked analysis snapshot')
        output.mkdir(parents=True)
        runtime.write_json(output / 'pool_selection.json', selection)
        runtime.write_json(output / 'selection_policy_and_pool.json', policy)
        runtime.write_rows(output / 'conversations_for_audit.jsonl', rows)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    profile = model_profile(TOKENIZER)
    diagnostics, failures = [], []
    for row in rows:
        sid = row['metadata']['scenario_id']
        try:
            diagnostics.append({'id': sid, **token_audit(row, tokenizer, profile, 8192)})
        except (ValueError, AssertionError) as exc:
            failures.append({'id': sid, 'error': str(exc)})
    source = output / 'conversations_for_audit.jsonl'
    runtime.write_json(output / 'token_mask_audit.json', {
        'input_sha256': runtime.digest(source.read_bytes()), 'rows': len(rows),
        'tokenizer': TOKENIZER, 'max_train_tokens': 8192, 'untruncated': True,
        'diagnostics': diagnostics, 'failures': failures,
        'status': 'failed' if failures else 'passed'})
    corpus = audit_corpus.audit(source, output, local_files_only=True)
    result = {'train_ready': False, 'mixture_ready': False, 'analysis_rows': len(rows),
              'helper_sha256': runtime.digest(Path(__file__).read_bytes()),
              'shortages': policy['shortages'], 'input_sha256': corpus['input_sha256'],
              'scope': 'Quota-limited effective accepted pool, not a complete release or full independent answer certification.',
              'token_failures': failures, 'semantic_pairs_at_least_0_9': corpus['semantic_pairs_at_least_0_9']}
    runtime.write_json(output / 'analysis_readiness.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(audit(args.config, args.output))
