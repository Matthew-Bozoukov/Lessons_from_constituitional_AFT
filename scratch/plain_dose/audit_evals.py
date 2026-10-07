# ABOUTME: Read-only audits of plain-base DA dose ODCV and MASK packages and pinned HF publications.
# ABOUTME: Requires complete matched protocols, checks local evidence and emits JSON receipts only to stdout.
"""Audit packaged runs: --eval odcv|mask --run-dir DIR --target ID --target-revision SHA.
Or --manifest JSON with a list of entries using eval/run_dir/target/target_revision,
and optional hf_repo/hf_revision. Does not modify runs; redirect stdout outside them.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
TARGET = TARGET_REVISION = MODEL_KEY = KIND = None
BASE_REVISION = '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'
REFERENCES = {
    'odcv': [('dougalldeepmind/2026-10-06-odcv-qwen36-0-plain', '1c1ede57af32fa09ceff055818c49adad967997e'),
             ('dougalldeepmind/2026-10-06-odcv-qwen36-0-da-15', '37aa070a15cd51d1455c1acc181c4860f6563af4')],
    'mask': [('dougalldeepmind/2026-10-05-mask-qwen36-0-plain', 'b242ff9e64e39ce69e73031f25a84e40923a8c58'),
             ('dougalldeepmind/2026-10-05-mask-qwen36-0-da-15', '945451667d4b4ecdb9172af6186b2cfcdace67f7')],
}
# Independently read from both pinned references' metadata/run_meta.json, October 6.
REFERENCE_CONFIG = {
    'published_key': 'qwen3.6-27b', 'temperature': 0.7, 'passes': 3,
    'bench_dir': 'src/eval/misalignment/odcv/third_party/odcv-bench',
    'concurrency': 16, 'scenario_timeout_s': 2400, 'prune_images': False,
    'judges': {'gemini-3-flash-preview': 'google/gemini-3-flash-preview'},
    'judge_workers': 8, 'progress_judge': True,
    'progress_judges': {'gemini-3-flash-preview': 'google/gemini-3-flash-preview'},
    'smoke': False,
    'serving': {'context_window': 28000, 'needs_tool_calls': True,
                'reuses_long_prefixes': True},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_axis(result, name):
    require(result.get('n_judged') == 240, f'{name}: require 240 judgments')
    require(result.get('n_dropped_all_na') == 0 and result.get('dropped') == [],
            f'{name}: missing or dropped verdicts')
    overall = result['ours']['overall']
    require(overall['n_rollouts'] == 240 and overall['n_cells'] == 80
            and overall['n_scenarios'] == 40, f'{name}: incorrect coverage')
    for variant in ('mandated', 'incentivized'):
        require(result['ours'][variant]['n_rollouts'] == 120,
                f'{name}/{variant}: require 120 judgments')
    require(result.get('judges') == REFERENCE_CONFIG['judges'], f'{name}: judge mismatch')


def check_audits(audits):
    require(len(audits) == 3, 'Require three pass audits')
    for audit in audits:
        require(audit.get('clean') is True and audit.get('kept') is True,
                'A pass is incomplete or not kept')
        require(audit.get('n_expected') == audit.get('transcripts_real') == 80,
                'Pass must contain 80 real transcripts')
        for field in ('missing_cells', 'shell_transcripts', 'empty_transcripts',
                      'model_unreachable_transcripts'):
            require(audit.get(field) == 0, f'Pass has missing/invalid cells: {field}')


def audit_odcv(root):
    meta = read_json(root / 'metadata/run_meta.json')
    require(meta['target'] == TARGET and meta['target_revision'] == TARGET_REVISION,
            'Wrong target or target revision')
    require(meta['base_model'] == 'Qwen/Qwen3.6-27B'
            and meta['base_model_revision'] == BASE_REVISION and meta['mode'] == 'think',
            'Wrong base model, base revision or serving mode')
    require(meta['model_key'] == MODEL_KEY, 'Wrong model key')
    for key, expected in REFERENCE_CONFIG.items():
        require(meta['config'].get(key) == expected, f'Reference config differs: {key}')
    # Use the vendored scenario panel rather than accepting arbitrary 40-directory counts.
    bench = REPO / REFERENCE_CONFIG['bench_dir']
    panels = {v: {p.name for p in (bench / f'{v}_scenarios').iterdir() if p.is_dir()}
              for v in ('mandated', 'incentivized')}
    require(all(len(names) == 40 for names in panels.values()), 'Vendored panel differs')
    expected_cells = {(v, name, f'pass{i}') for v, names in panels.items()
                      for name in names for i in (1, 2, 3)}
    transcripts = list((root / 'rollouts').rglob('messages_record.txt'))
    actual_cells = set()
    for path in transcripts:
        from src.eval.misalignment.odcv.passes import cell_usable
        require(cell_usable(path.parent), f'Unusable rollout: {path}')
        parts = path.relative_to(root / 'rollouts').parts
        require(len(parts) == 4, f'Unexpected transcript path: {path}')
        cell = tuple(parts[:3])
        require(cell in expected_cells and cell not in actual_cells,
                f'Unexpected/duplicate rollout: {cell}')
        actual_cells.add(cell)
        text = path.read_text(encoding='utf-8')
        require(any(line.startswith('role: assistant') for line in text.splitlines()),
                f'No assistant turn: {cell}')
        docker_log = path.parent / 'docker_output.log'
        require(docker_log.is_file(), f'Missing executor log: {cell}')
        require('[AI API dead]: Connection error' not in
                docker_log.read_text(encoding='utf-8', errors='replace'),
                f'Infrastructure-interrupted rollout counted: {cell}')
        cell_meta = read_json(path.parent / 'cell_meta.json')
        require(cell_meta.get('judged') is True, f'Unjudged rollout: {cell}')
        require(cell_meta.get('variant') == cell[0] and cell_meta.get('scenario') == cell[1]
                and cell_meta.get('pass') == int(cell[2][4:]), f'Cell metadata differs: {cell}')
        require(cell_meta.get('transcript_bytes') == path.stat().st_size,
                f'Transcript changed after packaging: {cell}')
    require(actual_cells == expected_cells and len(transcripts) == 240,
            f'Incomplete panel: {len(actual_cells)}/240; missing={sorted(expected_cells-actual_cells)}')
    require(len(list((root / 'rollouts').rglob('cell_meta.json'))) == 240,
            'Unexpected extra cell metadata')
    counts = Counter((v, p) for v, _, p in actual_cells)
    require(all(counts[v, f'pass{i}'] == 40 for v in panels for i in (1, 2, 3)),
            'Require 40 cells per variant per pass')

    combined = read_json(root / 'metadata/combine_manifest.json')
    require(combined['n_transcripts'] == 240 and combined['skipped_empty'] == [],
            'Combined manifest has skipped/missing cells')
    require(len(set(combined['passes'])) == 3
            and set(combined['per_pass_counts']) == set(combined['passes']),
            'Combined pass identities differ')
    require(all(c == {'mandated': 40, 'incentivized': 40}
                for c in combined['per_pass_counts'].values()), 'Combined pass counts differ')
    summary = read_json(root / 'metadata/pass_summary.json')
    require(summary['requested_passes'] == summary['kept_passes'] == 3, 'Pass summary differs')
    check_audits(summary['audits'])
    result = read_json(root / 'results/results.json')
    require(result['target'] == TARGET and result['mode'] == 'think', 'Results target differs')
    require(result['passes']['requested'] == result['passes']['kept'] == 3
            and result['passes']['dropped'] == 0 and result['passes']['n_transcripts'] == 240,
            'Result pass coverage differs')
    check_audits(result['passes']['audits'])
    check_axis(result, 'misconduct')
    check_axis(result['progress'], 'progress')
    progress = read_json(root / 'results/progress_results.json')
    check_axis(progress, 'standalone progress')
    require(progress == result['progress'], 'Standalone and embedded progress results differ')
    for axis in (result, progress):
        medians = axis['per_scenario_medians']
        require(set(medians) == set(panels), 'Scored variants differ')
        for variant, names in panels.items():
            require(set(medians[variant]) == names, f'Scored scenario panel differs: {variant}')
            require(all(len(values) == 3 for values in medians[variant].values()),
                    f'Require three scored samples per scenario: {variant}')
    for name in ('results.md', 'scores_gemini-3-flash-preview.json',
                 'scores_progress_gemini-3-flash-preview.json'):
        require((root / 'results' / name).is_file(), f'Missing published result file: {name}')
    score_keys = {f'{v}/{name}/rollout_{i:03d}' for v, names in panels.items()
                  for name in names for i in range(3)}
    for name in ('scores_gemini-3-flash-preview.json', 'scores_progress_gemini-3-flash-preview.json'):
        scores = read_json(root / 'results' / name)
        require(set(scores) == score_keys, f'Judge cache coverage differs: {name}')
        require(all(isinstance(row.get('score'), (int, float))
                    and not isinstance(row['score'], bool)
                    and math.isfinite(row['score']) and 0 <= row['score'] <= 5
                    for row in scores.values()), f'Invalid cached judge scores: {name}')
    return {'verified': True, 'target': TARGET, 'target_revision': TARGET_REVISION,
            'rollouts': 240, 'passes': 3, 'misconduct_judged': 240, 'progress_judged': 240,
            'misconduct': result['ours']['overall'],
            'progress': result['progress']['ours']['overall'],
            'submission': result.get('submission')}


def audit_hub(root, repo_id, revision):
    from src.infra.huggingface import hf_api
    require(re.fullmatch(r'[0-9a-f]{40}', revision) is not None, 'Require exact HF commit SHA')
    info = hf_api().dataset_info(repo_id, revision=revision, files_metadata=True)
    require(info.sha == revision, 'HF resolved revision differs')
    local = {p.relative_to(root).as_posix(): p for p in root.rglob('*') if p.is_file()}
    remote = {p.rfilename: p for p in info.siblings if p.rfilename != '.gitattributes'}
    local.pop('.gitattributes', None)
    require(set(local) == set(remote),
            f'Hub/local file sets differ: missing={sorted(set(local)-set(remote))}; '
            f'extra={sorted(set(remote)-set(local))}')
    total = 0
    for name, path in sorted(local.items()):
        require(not path.is_symlink(), f'Refusing symlink: {path}')
        size = path.stat().st_size
        require(remote[name].size == size, f'Hub size differs: {name}')
        digest = hashlib.sha256() if remote[name].lfs else hashlib.sha1(f'blob {size}\0'.encode())
        with path.open('rb') as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                digest.update(chunk)
        expected = remote[name].lfs.sha256 if remote[name].lfs else remote[name].blob_id
        require(digest.hexdigest() == expected, f'Hub content hash differs: {name}')
        total += size
    require({'eval-run', f'eval:{KIND}', 'mode:think', f'model:{MODEL_KEY}'} <= set(info.tags or []),
            'Required HF discovery tags missing')
    return {'repo': repo_id, 'revision': revision, 'verified_files': len(local), 'bytes': total}


MASK_CONFIG = {
    'mode': 'think', 'judge_model': 'google/gemini-3-flash-preview',
    'judge_concurrency': 50, 'judge_batch': False, 'subsample': None, 'seed': 0,
    'gen_concurrency': 192, 'gen_timeout_s': 1800, 'temperature': 1.0,
    'max_tokens': 16384, 'max_generation_error_rate': 0.05, 'empty_content': 'evasion',
    'lie_k': 1, 'passes': 1, 'constitution': 'none',
    'serving': {'context_window': 20480, 'needs_tool_calls': False, 'concurrency': 192},
}
MASK_ROWS = {'continuations': 176, 'disinformation': 125, 'doubling_down_known_facts': 120,
             'known_facts': 209, 'provided_facts': 274, 'statistics': 96}


def audit_mask(root):
    from src.eval.misalignment.mask.runner import generation_errors
    meta = read_json(root / 'metadata/run_meta.json')
    require(meta['target'] == TARGET and meta['target_revision'] == TARGET_REVISION,
            'MASK target/revision differs')
    require(meta['base_model'] == 'Qwen/Qwen3.6-27B' and meta['base_model_revision'] == BASE_REVISION
            and meta['mode'] == 'think' and meta['model_key'] == MODEL_KEY, 'MASK model provenance differs')
    for key, expected in MASK_CONFIG.items():
        require(key in meta['config'] and meta['config'][key] == expected,
                f'MASK reference config differs: {key}')
    result = read_json(root / 'results/results.json')
    run_meta = read_json(root / 'metadata/mask_run_meta.json')
    require(result['target'] == TARGET and result['mode'] == 'think'
            and result['judge'] == MASK_CONFIG['judge_model'], 'MASK result identity differs')
    require(result['n_rows'] == 1000 and result['n_archetypes'] == 6 and result['passes'] == 1
            and result['overall_weighting'] == 'per_row' and result['smoke'] is False
            and result['lie_k'] == 1 and result['subsample'] is None,
            'MASK incomplete/different scoring protocol')
    require(run_meta['upstream_commit'] == '25e0b1201e6c928ebe69f7c5aad6fa9063a377ea',
            'MASK upstream revision differs')
    require(run_meta['rows_per_archetype'] == MASK_ROWS, 'MASK row panel differs')
    require(set(result['per_archetype']) == set(MASK_ROWS), 'MASK archetypes differ')
    for archetype, expected in MASK_ROWS.items():
        cell = result['per_archetype'][archetype]
        require(cell['n_rows'] == cell['n_responses'] == expected, f'MASK count differs: {archetype}')
        ids = []
        for path in (root / f'rollouts/{archetype}.csv', root / f'results/{archetype}_evaluated.csv'):
            with path.open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            require(len(rows) == expected, f'MASK CSV incomplete: {path.name}')
            require(all('task_id' in row for row in rows), f'MASK task ids missing: {path.name}')
            row_ids = [row['task_id'] for row in rows]
            require(len(set(row_ids)) == expected and all(row_ids), f'MASK duplicate/empty task ids: {path.name}')
            ids.append(set(row_ids))
        require(ids[0] == ids[1], f'MASK rollout/evaluation rows differ: {archetype}')
        require((root / f'results/{archetype}_metrics.csv').is_file(), f'MASK missing metrics: {archetype}')
    errors = generation_errors(root / 'rollouts')
    require(errors['generations'] == 4438, f'MASK generation coverage differs: {errors}')
    actual_rate = errors['errors'] / errors['generations']
    require(actual_rate <= 0.05, f'MASK actual error rate exceeds 5%: {actual_rate}')
    require(len(result['per_pass']) == 1 and result['per_pass'][0]['pass'] == 1, 'MASK pass coverage differs')
    for report in (result, result['per_pass'][0], run_meta):
        require(report['generation_errors'] == errors
                and report['generation_error_rate'] == round(actual_rate, 4), 'MASK error accounting differs')
    require(result['empty_content_policy'] == 'evasion', 'MASK empty-content policy differs')
    empty = result['empty_content']
    require(empty['total'] == sum(empty['by_type'].values()) and 0 <= empty['total'] <= 4438,
            'MASK empty-content counts invalid')
    require(result['empty_content_rate'] == round(empty['total'] / 4438, 4), 'MASK empty-content rate differs')
    require(result['per_pass'][0]['empty_content'] == empty, 'MASK per-pass empty counts differ')
    require(math.isfinite(result['overall_honesty_score']) and 0 <= result['overall_honesty_score'] <= 100,
            'MASK invalid honesty score')
    require((root / 'results/results.md').is_file(), 'MASK missing published results summary')
    return {'verified': True, 'rows': 1000, 'passes': 1, 'honesty': result['overall_honesty_score'],
            'generation_errors': errors, 'actual_generation_error_rate': actual_rate,
            'allowed_generation_error_rate': 0.05, 'empty_content': empty,
            'empty_content_rate': result['empty_content_rate'], 'empty_content_policy': 'evasion',
            'resumed_from': result.get('resumed_from'),
            'regenerated_archetypes': result.get('regenerated_archetypes')}



def audit_template_pin(source, meta):
    """Check the pinned source behavior, including the optional SWE-bench override."""
    require(meta['config'].get('serving', {}).get('preserve_thinking', False) is False,
            'Evaluation requests preserve_thinking=true')
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('pin_template', 'pin_prefix')]
    require(len(functions) == 2, 'Missing pinned template functions')
    # Execute only these two pure functions from the recorded, local Git source.
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<pinned-template>', 'exec'), namespace)
    expected = '{%- set enable_thinking = true -%}\n{%- set preserve_thinking = false -%}\n'
    require(namespace['pin_prefix']('think') == expected,
            'Source default does not pin preserve_thinking=false')
    require(namespace['pin_template']('BODY', 'think') == expected + 'BODY',
            'Source template does not use the verified prefix')
    if 'preserve_thinking: bool' in source:
        require('requirements.get("preserve_thinking", False)' in source,
                'Serving default is not explicitly false')
        require("pin_template(template, mode, preserve_thinking=preserve_thinking)" in source,
                'Remote template does not propagate serving preserve_thinking')
        require("preserve_thinking=plan['preserve_thinking']" in source,
                'Serving does not pass the resolved preserve_thinking value')


def audit_entry(entry):
    global TARGET, TARGET_REVISION, MODEL_KEY, KIND
    KIND, TARGET, TARGET_REVISION = entry['eval'], entry['target'], entry['target_revision']
    require(KIND in ('odcv', 'mask'), 'Unknown evaluation')
    require(re.fullmatch('[0-9a-f]{40}', TARGET_REVISION) is not None, 'Require exact target revision')
    require(bool(entry.get('hf_repo')) == bool(entry.get('hf_revision')), 'Provide both HF repo and revision')
    root = Path(entry['run_dir']).resolve(strict=True)
    meta = read_json(root / 'metadata/run_meta.json')
    require(re.fullmatch('[0-9a-f]{40}', meta['git_sha']) is not None, 'Missing exact source commit')
    source = subprocess.check_output(['git', 'show', f"{meta['git_sha']}:src/infra/endpoints/vllm.py"],
                                     cwd=REPO, text=True, encoding='utf-8')
    audit_template_pin(source, meta)
    from src.naming import undated
    MODEL_KEY = undated(TARGET).replace('-', '_')
    report = (audit_odcv if KIND == 'odcv' else audit_mask)(root)
    report.update(eval=KIND, target=TARGET, target_revision=TARGET_REVISION,
                  base_revision=BASE_REVISION, git_sha=meta['git_sha'], run_dir=str(root),
                  reference_runs=REFERENCES[KIND], audited_utc=datetime.now(timezone.utc).isoformat())
    if entry.get('hf_repo'):
        report['publication'] = audit_hub(root, entry['hf_repo'], entry['hf_revision'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--eval', choices=['odcv', 'mask'])
    parser.add_argument('--run-dir')
    parser.add_argument('--target')
    parser.add_argument('--target-revision')
    parser.add_argument('--hf-repo')
    parser.add_argument('--hf-revision')
    args = parser.parse_args()
    if args.manifest:
        require(not any((args.eval, args.run_dir, args.target, args.target_revision, args.hf_repo, args.hf_revision)),
                'Do not mix manifest with single-run options')
        entries = read_json(args.manifest)
        require(isinstance(entries, list) and bool(entries), 'Manifest must be a nonempty entry list')
    else:
        require(all((args.eval, args.run_dir, args.target, args.target_revision)), 'Missing required run options')
        entries = [vars(args)]
    keys = [(e['eval'], e['target'], e['target_revision']) for e in entries]
    require(len(set(keys)) == len(keys), 'Duplicate run identities in manifest')
    receipts = []
    for entry in entries:
        try:
            receipts.append(audit_entry(entry))
        except Exception as exc:
            receipts.append({'verified': False, 'eval': entry.get('eval'), 'target': entry.get('target'),
                             'error': f'{type(exc).__name__}: {exc}'})
    verified = all(r['verified'] for r in receipts)
    print(json.dumps({'verified': verified, 'receipts': receipts}, indent=2))
    return 0 if verified else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({'verified': False, 'error': f'{type(exc).__name__}: {exc}'}))
        raise SystemExit(1)


