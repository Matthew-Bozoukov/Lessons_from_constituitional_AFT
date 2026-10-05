# ABOUTME: Read-only final audit of the October 5 full-CoT-masked ODCV run and its publication.
# ABOUTME: Requires all 240 planned cells and both complete judge axes; emits JSON only to stdout.
"""Run after packaging: python scratch/masked_cot_odcv_audit.py RUN_DIR [--hf-repo ID --hf-revision SHA]."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
TARGET = 'dougalldeepmind/2026-10-05-qwen36-0-da-15-answer-only'
TARGET_REVISION = '291e04a399ba12c20840ed876ff4967f61f79fec'
BASE_REVISION = '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'
MODEL_KEY = 'qwen36_0_da_15_answer_only'
REFERENCE_RUNS = [
    {'repo': 'dougalldeepmind/2026-10-03-odcv-qwen36-0-answeronly-15',
     'revision': 'b1ce592da4bc5f8a584c0dd319fad26e63a4ad8d'},
    {'repo': 'dougalldeepmind/2026-10-03-odcv-qwen36-0-da-15-cot',
     'revision': 'b38fb4be65d5edd3ea2058e360409aaf5f3bb49f'},
]
# Independently read from both pinned references' metadata/run_meta.json, October 5.
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


def audit_local(root):
    meta = read_json(root / 'metadata/run_meta.json')
    require(meta['target'] == TARGET and meta['target_revision'] == TARGET_REVISION,
            'Wrong target or target revision')
    require(meta['base_model'] == 'Qwen/Qwen3.6-27B'
            and meta['base_model_revision'] == BASE_REVISION and meta['mode'] == 'think',
            'Wrong base model, base revision or serving mode')
    require(meta['model_key'] == MODEL_KEY, 'Wrong model key')
    for key, expected in REFERENCE_CONFIG.items():
        require(meta['config'].get(key) == expected, f'Reference config differs: {key}')
    matched = read_json(root / 'metadata/matched_protocol.json')
    require(matched['reference_runs'] == REFERENCE_RUNS, 'Wrong pinned reference runs')
    require(set(matched['matched_config_keys']) == set(REFERENCE_CONFIG),
            'Matched config key list is incomplete')
    require(matched['run_source_commit'] == meta['git_sha'], 'Run code provenance differs')
    require(matched['target_revision'] == TARGET_REVISION
            and matched['base_model_revision'] == BASE_REVISION
            and matched['mode'] == 'think', 'Matched protocol provenance differs')

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
    recovery = root / 'metadata/infrastructure_recovery/pass1_interrupted_trial'
    require(read_json(recovery / 'status.json')['judged'] is False,
            'Interrupted attempt must remain excluded')
    require((recovery / 'messages_record.txt').stat().st_size > 0,
            'Interrupted transcript not preserved')
    require('[AI API dead]: Connection error' in
            (recovery / 'docker_output.log').read_text(encoding='utf-8', errors='replace'),
            'Recovery evidence does not show model connection failure')
    return {'verified': True, 'target': TARGET, 'target_revision': TARGET_REVISION,
            'rollouts': 240, 'passes': 3, 'misconduct_judged': 240, 'progress_judged': 240,
            'misconduct': result['ours']['overall'],
            'progress': result['progress']['ours']['overall'],
            'submission': result.get('submission'), 'recovery_preserved': True}


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
    require({'eval-run', 'eval:odcv', 'mode:think', f'model:{MODEL_KEY}'} <= set(info.tags or []),
            'Required HF discovery tags missing')
    return {'repo': repo_id, 'revision': revision, 'verified_files': len(local), 'bytes': total}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--hf-repo')
    parser.add_argument('--hf-revision')
    args = parser.parse_args()
    require(bool(args.hf_repo) == bool(args.hf_revision), 'Provide both HF repo and revision')
    root = args.run_dir.resolve(strict=True)
    report = audit_local(root)
    if args.hf_repo:
        report['publication'] = audit_hub(root, args.hf_repo, args.hf_revision)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'verified': False, 'error': f'{type(exc).__name__}: {exc}'}), file=sys.stderr)
        raise SystemExit(1)
