# ABOUTME: Bind GPU-free capacity, regression and real-agent integration evidence to the reusable Lite recipe.
# ABOUTME: Run on the CPU: python -m scratch.swebench_qualify_recipe --load-dir PATH --smoke-root PATH; never rents GPUs.
import argparse
import hashlib
from pathlib import Path
import re
import shutil
import statistics
import time

from dotenv import load_dotenv
from omegaconf import OmegaConf
from src.eval.capabilities.swebench_mini import fleet
from src.eval.capabilities.swebench_mini.fleet_state import atomic, digest, read
from src.eval.capabilities.swebench_mini.fleet_task import resource_shell
from src.infra.huggingface import hf_api, hf_download, hf_repo_id, push_run_dir
from src.naming import artifact_name, today


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', type=Path, required=True)
    parser.add_argument('--smoke-root', type=Path, required=True)
    parser.add_argument('--test-log', type=Path, required=True)
    parser.add_argument('--transport-log', type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load('configs/eval/swebench_mini/lite.yaml')
    load_dotenv(cfg.credentials)
    load = read(args.load_dir/'results.json')
    assert load['status'] == 'finished' and load['model_inference'] is False
    assert load['memory_gib'] >= 190 and load['cpu_count'] >= 60, 'Host below 80-conversation sizing floor'
    assert [(p['agents'], p['tool_limit']) for p in load['phases']] == [(6, 1), (40, 40), (60, 60), (80, 80), (80, 32)]
    phases = []
    for phase in load['phases']:
        assert all(not t.get('error') and t['resources']['memory.events']['oom_kill'] == 0 for t in phase['tasks'])
        commands = [c for t in phase['tasks'] for c in t['commands']]
        assert all(c['returncode'] == 0 for c in commands)
        assert len(commands) == phase['agents'] * phase['rounds']
        assert min(s['available_gib'] for s in phase['samples']) > 32
        phases.append({'agents': phase['agents'], 'tools': phase['tool_limit'], 'commands': len(commands),
                       'seconds': phase['seconds'], 'median_command_seconds': statistics.median(c['execution_seconds'] for c in commands),
                       'min_available_gib': min(s['available_gib'] for s in phase['samples'])})
    smoke = read(args.smoke_root/'results/infrastructure.json')
    assert smoke['status'] == 'passed' and smoke['synthetic'] and not smoke['model_evaluation']
    test_log = args.test_log
    transport_log = args.transport_log
    assert re.search(r'\d+ passed', test_log.read_text()) and ' failed' not in test_log.read_text()
    assert '\nOK' in transport_log.read_text()
    proof = {'status': 'passed', 'created': time.time(), 'qualified_workers': 80, 'tool_concurrency': 32,
             'cpu_count': load['cpu_count'], 'min_host_memory_gib': 190,
             'agent_cpus': 2, 'agent_memory': '4g', 'agent_pids': 512,
             'dataset_revision': cfg.dataset_revision,
             'images_sha256': digest(Path(cfg.readiness)/'metadata/images.json'),
             'resource_policy_sha256': hashlib.sha256(resource_shell(2, 120).encode()).hexdigest(),
             'load_test_sha256': digest(args.load_dir/'results.json'), 'phases': phases,
             'limitations': 'Six representative test workloads, not arbitrary future commands. No GPU performance qualification.'}
    atomic(cfg.cpu_qualification_path, proof)
    recipe = {'settings': fleet.recipe_settings(cfg), 'source_hashes': fleet.sources(),
              'validated_full_run': False,
              'qualification': {'protocol_reviewed': True, 'cpu_qualified': True, 'recovery_tests_passed': True,
                                'performance_benchmark_validated': False},
              'cpu_qualification_sha256': digest(cfg.cpu_qualification_path),
              'calibration': {'gpu_seconds_per_task': None, 'workers_per_replica': 4,
                              'limitation': 'No throughput measurement on this CPU; first requested model run supplies evidence'},
              'source_hf_repo': None}
    atomic(cfg.recipe_path, recipe)
    fleet.validate_recipe(cfg, recipe)
    evidence = args.smoke_root/'metadata'
    shutil.copytree(args.load_dir, evidence/'cpu-qualification', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('.tool-slots'))
    for path in (test_log, transport_log, Path(cfg.recipe_path), Path(cfg.cpu_qualification_path)):
        shutil.copy2(path, evidence/path.name)
    for name in recipe['source_hashes']:
        dest = evidence/'qualified-source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, dest)
    atomic(evidence/'qualification-summary.json', {'status': 'passed', 'model_evaluation': False,
           'commands': sum(p['commands'] for p in phases), 'cpu_workers': 80, 'tool_limit': 32,
           'tests': test_log.read_text().splitlines()[-1],
           'excluded_test': 'All-eval registry import needs unrelated plotting dependencies absent in the CPU-only driver',
           'real_agent_and_official_grader': smoke})
    repo = hf_repo_id(artifact_name('swebench-lite-infrastructure'))
    fields = {'experiment': 'GPU-free Lite CPU, recovery and end-to-end infrastructure qualification',
              'date_generated': today(), 'constitution': 'none', 'source_repo': 'teaching_claude_why_replication; exact files in metadata/qualified-source',
              'models': 'none; synthetic endpoint and a public reference patch, not model output',
              'generation_config': '4 conversations per GPU fixed; tested CPU counts 40/60/80; 32 tool slots',
              'schema': 'rollouts: synthetic only; results: synthetic official grading; metadata: capacity, source and regression evidence',
              'provenance': 'CPU-only qualification; no inference GPUs rented; historical serving counters labelled separately'}
    push_run_dir(args.smoke_root, repo, fields, atomic_commit=True,
                 front_matter={'tags': ['infrastructure-check', 'swebench-lite', 'synthetic']})
    revision = hf_api().dataset_info(repo).sha
    assert read(hf_download(repo, 'metadata/qualification-summary.json', repo_type='dataset', revision=revision)) == read(evidence/'qualification-summary.json')
    # Exercise the production verifier against a real Hub tree as well as unit mocks.
    manifest = read(evidence/'manifest.json')
    manifest['repo'] = repo
    atomic(evidence/'manifest.json', manifest)
    check_cfg = OmegaConf.create({'root': str(args.smoke_root)})
    count = fleet.verify_final_files(check_cfg, revision)
    atomic(evidence/'qualification-hf-verification.json', {'revision': revision, 'rollout_result_files_verified': count,
                                                          'verified_at': time.time()})
    push_run_dir(args.smoke_root, repo, fields, atomic_commit=True,
                 front_matter={'tags': ['infrastructure-check', 'swebench-lite', 'synthetic']})
    print('QUALIFIED', repo, hf_api().dataset_info(repo).sha, 'verified files', count)


if __name__ == '__main__':
    main()
