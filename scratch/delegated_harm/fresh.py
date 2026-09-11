# ABOUTME: Initialize a fresh adapter evaluation using the shared live GPU episode queue.
# ABOUTME: Run: uv run scratch/delegated_harm/fresh.py --config scratch/delegated_harm/configs/nonmoral.yaml --root <path>.
import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from omegaconf import OmegaConf
from openai import OpenAI

from scratch.delegated_harm.episode_queue import EpisodeQueue
from scratch.delegated_harm.recovery import AuthorValidator, read
from src.eval.misalignment.delegated_harm.runner import author_one, cells
from src.eval.misalignment.delegated_harm.rubrics import COMPONENTS
from src.eval.misalignment.delegated_harm.source import digest, prepare, save
from src.infra.endpoints.vllm import resolve_target
from src.naming import eval_name, run_dir
from src.utils import write_run_meta


def initialize(root, override):
    assert not (root / 'metadata/handoff.json').exists(), 'Already initialized'
    cfg = OmegaConf.merge(OmegaConf.load('configs/eval/delegated_harm.yaml'), OmegaConf.load(override))
    target = str(cfg.orchestration.target)
    spec = resolve_target(target)
    assert spec.revision == cfg.expected_revisions[target], 'Adapter revision drift'
    assert spec.base_revision == cfg.expected_base_revision, 'Base revision drift'
    cfg.source.path = str((root / 'upstream').resolve())
    scenarios, humans, _, manifest = prepare(cfg)
    assert set(scenarios) == set(COMPONENTS)
    arm = str(cfg.orchestration.arm)
    assert arm.replace('-', '').isalnum()
    schedule = cells(list(scenarios), int(cfg.author.variants), int(cfg.generation.repetitions), int(cfg.seed))
    queue = EpisodeQueue(root / 'queue.sqlite')
    assert not queue.snapshot()['jobs']
    queue.enqueue(arm, schedule)
    manifest.update(target=asdict(spec), components=COMPONENTS,
                    protocol=OmegaConf.to_container(cfg, resolve=True), protocol_kind='fresh')
    manifest['implementation_sha256'] = {n: digest(Path('src/eval/misalignment/delegated_harm', n).read_text(encoding='utf-8'))
        for n in ('source.py', 'runner.py', 'judging.py', 'rubrics.py')}
    save(root / 'metadata/protocol.json', manifest)
    save(root / 'metadata/schedule.json', schedule)
    save(root / 'metadata/human_requests.json', {k: humans[k] for k in scenarios})
    OmegaConf.save(cfg, root / f'metadata/config-{arm}.yaml')
    save(root / 'metadata/handoff.json', {'created': time.time(), 'protocol_kind': 'fresh',
        'arms': {}, 'arm': arm, 'target': asdict(spec),
        'policy': 'Fresh same-checkpoint authors frozen before episodes; atomic claims; no outcome-based reassignment.'})
    write_run_meta(root, config=OmegaConf.to_container(cfg, resolve=True))
    print(f'Initialized {len(schedule)} episodes for {target} at {spec.revision}', flush=True)


def await_inputs(target, cfg):
    """One endpoint writes requests; every subject waits for the immutable bundle."""
    root = Path(cfg.scaling.root)
    path = Path(cfg.scaling.inputs)
    pinned = read(root / 'metadata/handoff.json')['target']
    assert target.spec.revision == pinned['revision'], 'Adapter changed since initialization'
    assert target.spec.base_revision == pinned['base_revision'], 'Base changed since initialization'
    if cfg.scaling.author_leader and not path.exists():
        scenarios, _, notes, _ = prepare(cfg)
        client = OpenAI(base_url=target.base_url, api_key=target.api_key,
                        timeout=float(cfg.generation.request_timeout), max_retries=2)
        validator = AuthorValidator(cfg.judge, root / 'metadata/author_validation')
        jobs = [(key, variant) for key in scenarios for variant in range(int(cfg.author.variants))]
        def make(job):
            key, variant = job
            ident = f'{key}__v{variant}'
            record = author_one(client, target.model_name, scenarios[key], notes[key]['clear'],
                variant, cfg, validator, root / 'metadata/authors' / f'{ident}.json')
            print(f'AUTHOR {ident}: {record["accepted"]}', flush=True)
            return ident, record
        with ThreadPoolExecutor(max_workers=int(cfg.generation.parallel)) as pool:
            authors = dict(pool.map(make, jobs))
        save(root / 'metadata/authorship_frozen.json', {
            key: {'accepted': r['accepted'], 'sha256': r.get('sha256')} for key, r in authors.items()})
        save(root / 'metadata/judge_ledger.json', {
            'charged_or_reserved_usd': sum(v.ledger['charged_or_reserved_usd'] for v in (validator.primary, validator.secondary)),
            'calls': [], 'detail': 'author_validation/{primary,secondary}/metadata/judge_ledger.json'})
        save(path, {'target': asdict(target.spec), 'authors': authors, 'protocol_kind': 'fresh', 'frozen_at': time.time()})
    deadline = time.time() + float(cfg.orchestration.author_wait_seconds)
    while not path.exists():
        if time.time() > deadline:
            raise TimeoutError('Author bundle never became available; no episodes were claimed')
        leaders = list((root / 'workers').glob('*-h200-1/controller.json'))
        if leaders and read(leaders[0]).get('status') in ('failed', 'cleanup_failed'):
            raise RuntimeError('Author worker failed before freezing requests')
        time.sleep(5)


def merge(root):
    handoff = read(root / 'metadata/handoff.json')
    arm = handoff['arm']
    snapshot = EpisodeQueue(root / 'queue.sqlite').snapshot()
    schedule = read(root / 'metadata/schedule.json')
    jobs = [j for j in snapshot['jobs'] if j['arm'] == arm]
    assert len(jobs) == len(schedule) and {j['id'] for j in jobs} == {c['id'] for c in schedule}
    assert all(j['state'] == 'done' for j in jobs)
    controllers = [read(p) for p in (root / 'workers').glob('*/controller.json')]
    assert controllers and all(c.get('terminated') for c in controllers), 'GPU cleanup must precede scoring'
    base = root / 'combined' / arm
    destination = run_dir(base / 'runs', handoff['target']['model_key'])
    assert not destination.exists(), 'Refuse to overwrite a merged run'
    (destination / 'rollouts').mkdir(parents=True)
    shutil.copytree(root / 'metadata', destination / 'metadata')
    for job in jobs:
        result = Path(job['result_path'])
        rec = read(result)
        assert digest(rec) == job['result_hash'], f'Result changed: {job["id"]}'
        assert rec['id'] == job['id']
        save(destination / 'results/episodes' / result.name, rec)
        for ext in ('json', 'md'):
            trace = result.parents[2] / 'rollouts' / f'{job["id"]}.{ext}'
            if trace.exists():
                shutil.copy2(trace, destination / 'rollouts' / trace.name)
    worker_metadata = {}
    for worker in snapshot['workers']:
        folder = Path(json.loads(worker['metadata'])['out_dir'])
        for candidate in (folder / 'metadata/run_meta.json', folder / 'run_meta.json'):
            if candidate.exists():
                worker_metadata[worker['id']] = read(candidate)
                break
    save(destination / 'metadata/horizontal_scaling.json', {
        'queue': snapshot, 'worker_controllers': controllers, 'worker_run_metadata': worker_metadata})
    write_run_meta(destination, config=read(root / 'metadata/protocol.json')['protocol'])
    (destination / 'run_meta.json').replace(destination / 'metadata/run_meta.json')
    save(base / 'controller.json', {'arm': arm, 'terminated': True, 'status': 'scoring', 'run_dir': str(destination)})
    return destination


def merge_and_score(root):
    handoff = read(root / 'metadata/handoff.json')
    arm = handoff['arm']
    cfg = OmegaConf.load(root / f'metadata/config-{arm}.yaml')
    controller = root / 'combined' / arm / 'controller.json'
    destination = Path(read(controller)['run_dir']) if controller.exists() else merge(root)
    for item in cfg.orchestration.outcome_judges:
        command = [sys.executable, 'scratch/delegated_harm/rescore.py', '--run-dir', str(destination),
            '--ignore-spending-cap', '--workers', str(cfg.orchestration.judge_workers),
            '--judge-max-tokens', str(cfg.orchestration.judge_max_tokens), '--judge-model', str(item.model)]
        if item.json_mode:
            command.append('--json-mode')
        subprocess.run(command, check=True)
        summary = read(destination / 'results/results.json')
        if summary['unjudged_completed_episodes'] == 0:
            break
    assert summary['unjudged_completed_episodes'] == 0, 'Saved completions still require scoring'
    from src.infra.huggingface import hf_api, hf_org
    repo = f'{hf_org()}/{eval_name("delegated_harm", handoff["target"]["model_key"])}'
    revision = hf_api().dataset_info(repo).sha
    save(controller, {'arm': arm, 'terminated': True, 'status': 'finished', 'run_dir': str(destination),
                      'repo': repo, 'revision': revision, 'completed': summary['completed_episodes']})
    from scratch.delegated_harm.compare import plot_multiple
    sources = OmegaConf.to_container(cfg.orchestration.comparisons, resolve=True)
    sources[arm] = {'repo': repo, 'revision': revision}
    comparison = plot_multiple(sources, OmegaConf.to_container(cfg.orchestration.labels),
        'delegated-harm-nonmoral-comparison',
        'Control and DA include recovery attempts; nonmoral uses uniform limits.')
    subprocess.run([sys.executable, 'scratch/delegated_harm/plot_by_world.py', '--comparison', str(comparison),
        '--artifact-subject', 'delegated-harm-nonmoral-complied-silent-by-world'], check=True)
    save(root / 'metadata/published.json', {'repo': repo, 'revision': revision, 'comparison': str(comparison)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    initialize(args.root.resolve(), args.config)


if __name__ == '__main__':
    main()
