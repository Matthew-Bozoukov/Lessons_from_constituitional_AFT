# ABOUTME: Generate one pinned Qwen base answer set, then compare both saved SFT arms concurrently.
# ABOUTME: Uses the standard eval lifecycle with isolated workers and one serialized Hub publication.
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

from dotenv import dotenv_values, load_dotenv
from omegaconf import OmegaConf

from src.eval.layout import publish_layout, run_tags
from src.eval.run_eval import _card_fields, _credentials_preflight, _publish
from src.infra import runpod
from src.infra.huggingface import hf_api, hf_download, hf_repo_id
from src.naming import eval_name
from src.utils import read_jsonl, write_run_meta

ROOT = Path('output/arena_qwen_base_20261013').resolve()
BASE = 'Qwen/Qwen3.6-27B'
BASE_SHA = '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'
ARMS = {
    'control': ('dougalldeepmind/2026-10-12-ah-qwen36-0-plain',
                'bc2fb7d0d17202357d8f04c3849354fee76ec0a4', 'qwen36_0_plain'),
    'da15': ('dougalldeepmind/2026-10-12-ah-qwen36-0-da-15',
             'e15e030aaaf72fc05cad06bead91fa7284082553', 'qwen36_0_da_15'),
}
DATE = '2026-10-13'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def config():
    cfg = OmegaConf.load('output/arena_qwen_da15_20261012/full.yaml')
    cfg.mode = 'think'  # Explicit full-model template pin, matching both saved adapters.
    cfg.target_revisions = {BASE: BASE_SHA}
    cfg.campaign_dir = str(ROOT / 'base')
    cfg.output_root = str(ROOT / 'serving')
    return cfg


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    _credentials_preflight(push=True)
    cfg = config()
    evidence = {}
    for label, (repo, revision, key) in ARMS.items():
        meta = read(hf_download(repo, 'metadata/run_meta.json', repo_type='dataset', revision=revision))
        protocol = read(hf_download(repo, 'metadata/generation_protocol.json', repo_type='dataset', revision=revision))
        path = Path(hf_download(repo, 'rollouts/answers.jsonl', repo_type='dataset', revision=revision))
        rows = read_jsonl(path)
        assert len(rows) == len({r['uid'] for r in rows}) == 750
        assert meta['base_model_revision'] == BASE_SHA and meta['mode'] == 'think'
        assert protocol['settings']['max_tokens'] == 12000
        evidence[label] = {'repo': repo, 'revision': revision, 'model_key': key,
                           'target': meta['target'], 'target_revision': meta['target_revision'],
                           'answers_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'generation_protocol': protocol}
    assert evidence['control']['generation_protocol'] == evidence['da15']['generation_protocol']
    # Refuse changing the scientific settings of the prior comparison.
    prior = OmegaConf.load('output/arena_qwen_da15_20261012/full.yaml')
    for field in ('generation', 'judge', 'judge_validation', 'statistics', 'thresholds',
                  'serving', 'slices', 'arm_defaults', 'seed', 'bench_name'):
        assert OmegaConf.to_container(cfg, resolve=True)[field] == OmegaConf.to_container(prior, resolve=True)[field]
    OmegaConf.save(cfg, ROOT / 'base.yaml')
    save(ROOT / 'input_verification.json', {'verified_utc': now(), 'base': BASE,
         'base_revision': BASE_SHA, 'saved_arms': evidence,
         'parallelism': 'Two isolated judge workers; shared immutable base answers; single publication writer'})
    print('Verified both immutable 750-answer sets and unchanged scientific settings.', flush=True)


def execute(command, label):
    save(ROOT / f'{label}_launch.json', {'command': command, 'started_utc': now()})
    with (ROOT / f'{label}.stdout.log').open('ab') as stdout, (ROOT / f'{label}.stderr.log').open('ab') as stderr:
        code = subprocess.call(command, stdout=stdout, stderr=stderr,
                               env=os.environ | {'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1'})
    save(ROOT / f'{label}_exit.json', {'exit_code': code, 'finished_utc': now()})
    if code:
        raise RuntimeError(f'{label} exited {code}; retained logs and checkpoints under {ROOT}')


def generate():
    if (ROOT / 'base/qwen36.json').exists():
        print('Base receipt exists; validate before reusing with compare action.', flush=True)
        return
    if (ROOT / 'pod.json').exists():
        raise RuntimeError('Pod receipt already exists; inspect before allocating another GPU')
    def registered(pod_id):
        save(ROOT / 'pod.json', {'id': pod_id, 'created_utc': now(), 'gpu': 'NVIDIA H200', 'max_hours': 4})
    try:
        result = runpod.up(name='nika-arena-qwen-base-20261013', eval='arena_hard',
                           target=BASE, gpu='NVIDIA H200', count=1,
                           max_hours=4, on_provisioned=registered)
        print(result, flush=True)
        receipt = read(ROOT / 'pod.json')
        receipt['host'] = re.search(r'^host:\s+(\S+)', result, re.M).group(1)
        receipt['hourly_usd'] = runpod.call('GET', '/pods/' + receipt['id']).get('costPerHr')
        save(ROOT / 'pod.json', receipt)
        command = [sys.executable, '-u', 'scripts/run_eval.py', '--name', 'arena_hard',
                   '--config', str(ROOT / 'base.yaml'), '--target', BASE, '--reference', BASE,
                   '--server', receipt['host'], '--port', '18953', '--terminate-pod']
        execute(command, 'base')
    finally:
        if (ROOT / 'pod.json').exists():
            receipt = read(ROOT / 'pod.json')
            if any(p['id'] == receipt['id'] for p in runpod.active_pods()):
                runpod.teardown(receipt['id'])
            pods = runpod.active_pods()
            assert all(p['id'] != receipt['id'] for p in pods)
            save(ROOT / 'teardown_verification.json', {'verified_utc': now(), 'own_pod_absent': True,
                 'owned_pod': receipt['id'], 'other_pods': [{k: p.get(k) for k in ('id', 'name')} for p in pods],
                 'account': runpod.graphql('query { myself { clientBalance currentSpendPerHr } }')})


def verify_base():
    receipt = read(ROOT / 'base/qwen36.json')
    directory = Path(receipt['published']['out_dir'])
    for name, digest in receipt['sha256'].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    repo = hf_repo_id(eval_name('arena_hard', 'qwen36', date=DATE))
    revision = hf_api().dataset_info(repo).sha
    remote = Path(hf_download(repo, 'rollouts/answers.jsonl', repo_type='dataset', revision=revision))
    assert remote.read_bytes() == (directory / 'rollouts/answers.jsonl').read_bytes()
    protocol = read(directory / 'metadata/generation_protocol.json')
    inputs = read(ROOT / 'input_verification.json')
    for arm in inputs['saved_arms'].values():
        assert protocol == arm['generation_protocol'], 'Base and saved-arm protocols differ; no paid judging allowed'
    save(ROOT / 'base_verified.json', {'repo': repo, 'revision': revision, 'verified_utc': now(),
         'answers_sha256': hashlib.sha256(remote.read_bytes()).hexdigest(), 'n_answers': 750})
    return repo, revision


def compare():
    base_repo, base_revision = verify_base()
    workers = []
    for index, (label, (repo, revision, key)) in enumerate(ARMS.items()):
        if (ROOT / label / 'comparison/results/results.json').exists():
            continue
        cfg = config()
        cfg.target_revisions = {base_repo: base_revision, repo: revision}
        cfg.campaign_dir = str(ROOT / label)
        cfg.output_root = str(ROOT / label / 'serving')
        OmegaConf.save(cfg, ROOT / f'{label}.yaml')
        command = [sys.executable, '-u', 'scripts/run_eval.py', '--name', 'arena_hard',
                   '--config', str(ROOT / f'{label}.yaml'), '--target', repo,
                   '--reference', base_repo, '--port', str(18954 + index), '--no-push']
        stdout = (ROOT / f'{label}.stdout.log').open('ab')
        stderr = (ROOT / f'{label}.stderr.log').open('ab')
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                   env=os.environ | {'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1'})
        save(ROOT / f'{label}_launch.json', {'pid': process.pid, 'command': command, 'started_utc': now()})
        workers.append((label, process, stdout, stderr))
    print('Concurrent comparison workers:', [(label, p.pid) for label, p, _, _ in workers], flush=True)
    failed = []
    # Both have already started. Waiting for one never blocks the other's execution.
    for label, process, stdout, stderr in workers:
        code = process.wait()
        stdout.close()
        stderr.close()
        save(ROOT / f'{label}_exit.json', {'exit_code': code, 'finished_utc': now()})
        if code:
            failed.append(label)
    if failed:
        raise RuntimeError(f'Incomplete workers {failed}; preserve valid judgments and inspect failed orderings')


def publish():
    directory = ROOT / 'published'
    rollouts, results, metadata = publish_layout(directory)
    summaries = {}
    for label, (_, _, key) in ARMS.items():
        source = ROOT / label / 'comparison'
        summary = read(source / 'results/results.json')
        assert len(summary['leaderboard']) == 1
        assert summary['leaderboard'][0]['model_key'] == key
        assert summary['leaderboard'][0]['n_prompts'] == 500
        assert summary['leaderboard'][0]['by_slice']['creative_writing']['controlled']['n'] == 250
        assert summary['judge_validation']['primary_complete']
        summaries[label] = summary
        # Distinct arm filenames make merging rollouts safe; metadata/results stay separately inspectable.
        shutil.copytree(source / 'rollouts', rollouts, dirs_exist_ok=True)
        shutil.copytree(source / 'results', results / 'comparisons' / label, dirs_exist_ok=True)
        shutil.copytree(source / 'metadata', metadata / 'comparisons' / label, dirs_exist_ok=True)
    first, second = summaries.values()
    for field in ('baseline', 'judge', 'mode', 'judge_policy', 'primary_slice', 'metric'):
        assert first[field] == second[field]
    merged = dict(first)
    merged['n_arms'] = 2
    merged['leaderboard'] = sorted(first['leaderboard'] + second['leaderboard'], key=lambda x: x['win_rate'], reverse=True)
    merged['judge_validation'] = {'policy': 'diagnostic', 'by_arm': {ARMS[k][2]: v['judge_validation'] for k, v in summaries.items()}}
    merged['generation_protocol_validation'] = {'status': 'matched', 'n_questions': 750,
         'by_comparison': {k: v['generation_protocol_validation'] for k, v in summaries.items()}}
    merged['generation_health'] = first['generation_health'] | second['generation_health']
    merged['pooled_from'] = list({r['model_key']: r for s in summaries.values() for r in s['pooled_from']}.values())
    for source in merged['pooled_from']:
        source['repo'] = 'https://huggingface.co/datasets/' + source['target']
    merged['execution'] = {'independent_parallel_comparisons': True, 'shared_frozen_base_answers': True,
                            'source_code_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    save(results / 'leaderboard.json', merged)
    save(metadata / 'sources.json', merged['pooled_from'])
    for path in ROOT.glob('*.json'):
        shutil.copy2(path, metadata / path.name)
    for path in ROOT.glob('*.yaml'):
        shutil.copy2(path, metadata / path.name)
    lines = ['# Qwen control and DA15 versus original Qwen3.6', '',
             'Two independent comparisons using the same 750 frozen base answers and the October 12 protocol.', '',
             '| Target | Slice | Controlled win rate | 95% interval | Raw win rate |', '|---|---|---:|---|---:|']
    for row in merged['leaderboard']:
        for category, block in row['by_slice'].items():
            c, u = block['controlled'], block['uncontrolled']
            lines.append(f"| {row['model_key']} | {category} | {100*c['mean']:.2f}% | {100*c['ci_lower']:.2f}–{100*c['ci_upper']:.2f}% | {100*u['mean']:.2f}% |")
    lines += ['', 'Intervals bootstrap prompts, not training seeds. GPT-4.1 is primary; Gemini is diagnostic.',
              'All empty and truncated model answers are retained. Model reasoning is saved but hidden from judges.',
              'The two comparisons share answers and are statistically dependent; do not infer DA15 versus control from their score difference.']
    (results / 'campaign_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # Scan published bytes against actual configured secret values without printing them.
    secrets = [v.encode() for k, v in dotenv_values('.env').items() if v and len(v) >= 16 and any(s in k for s in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD'))]
    for path in directory.rglob('*'):
        if path.is_file() and any(secret in path.read_bytes() for secret in secrets):
            raise RuntimeError(f'Secret scan blocked publication: {path.relative_to(directory)}')
    cfg = config()
    command = 'uv run python -X utf8 scratch/arena_base_campaign.py all'
    write_run_meta(directory, OmegaConf.to_container(cfg, resolve=True), extra={'command': command, 'pooled_from': merged['pooled_from']})
    inputs = read(ROOT / 'input_verification.json')
    models = {BASE: BASE_SHA} | {x['target']: x['target_revision'] for x in inputs['saved_arms'].values()}
    url = _publish(directory, name='arena_hard', model_key='vs_qwen36', mode='think',
                   target='pooled: control and DA15 versus Qwen/Qwen3.6-27B', summary=merged, push=True,
                   card=_card_fields('arena_hard', cfg, command,
                         experiment='Arena-Hard paired comparisons against original Qwen3.6; prompt-cluster bootstrap intervals',
                         models=json.dumps(models)),
                   tags=run_tags('arena_hard', 'vs_qwen36', 'think', pooled=True))
    repo = hf_repo_id(eval_name('arena_hard', 'vs_qwen36', date=DATE))
    revision = hf_api().dataset_info(repo).sha
    checked = {}
    for path in directory.rglob('*'):
        if path.is_file():
            relative = path.relative_to(directory).as_posix()
            remote = Path(hf_download(repo, relative, repo_type='dataset', revision=revision))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert hashlib.sha256(remote.read_bytes()).hexdigest() == digest, relative
            checked[relative] = digest
    save(ROOT / 'publication_verification.json', {'repo': repo, 'url': url, 'revision': revision,
         'verified_utc': now(), 'sha256': checked})
    print('Published and hash-verified', url, revision, flush=True)


def main(action):
    load_dotenv('.env')
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if action == 'prepare':
        prepare()
    elif action == 'all':
        prepare()
        generate()
        compare()
        publish()
    elif action == 'compare':
        compare()
        publish()
    elif action == 'publish':
        publish()
    else:
        raise ValueError(action)


if __name__ == '__main__':
    main(sys.argv[1])
