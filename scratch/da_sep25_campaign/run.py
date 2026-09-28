# ABOUTME: Owns only this campaign's datasets, single training attempts and parallel eval pods.
# ABOUTME: Enforces reserved rental exposure plus API headroom; records durable per-job receipts.
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from dotenv import load_dotenv
load_dotenv(ROOT / '.env', override=True)
os.environ.update(HF_ORG='dougalldeepmind', PYTHONUTF8='1', PYTHONUNBUFFERED='1')
from filelock import FileLock
from omegaconf import OmegaConf
from src.infra import runpod
from src.infra.huggingface import hf_api, hf_download
from src.infra.endpoints.vllm import SshExec, POD_VENV, resolve_target
from src.naming import eval_name
from scratch.nonmoral.result_backup import fetch_training_outputs

CFG = OmegaConf.load(ROOT / 'scratch/da_sep25_campaign/campaign.yaml')
OUT = ROOT / CFG.output_root
FLAGS = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding='utf-8')
    temp.replace(path)


def spawn(argv, log):
    with Path(log).open('ab') as stream:
        return subprocess.Popen(argv, cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT, creationflags=FLAGS)


def reserve(key):
    with FileLock(str(OUT / 'budget.lock')):
        p = OUT / 'budget.json'
        budget = read(p) if p.exists() else {'limit': float(CFG.budget_usd), 'api_reserved': float(CFG.api_reserve_usd), 'jobs': {}}
        assert key not in budget['jobs'], 'Duplicate job reservation'
        amount = float(CFG.pod_max_hours * CFG.pod_hourly_ceiling_usd)
        exposure = budget['api_reserved'] + sum(j.get('settled_usd', j['reserved_usd']) for j in budget['jobs'].values())
        assert exposure + amount <= budget['limit'], 'Campaign reservation cap reached'
        budget['jobs'][key] = {'reserved_usd': amount, 'reserved_epoch': time.time()}
        save(p, budget)


def settle(key, state):
    with FileLock(str(OUT / 'budget.lock')):
        p = OUT / 'budget.json'
        budget = read(p)
        job = budget['jobs'][key]
        if state.get('terminated'):
            job['settled_usd'] = (state['terminated_epoch'] - state['created_epoch']) / 3600 * state['hourly_usd']
        elif not state.get('owned_pod'):
            job['settled_usd'] = 0
        job['pod_id'] = state.get('owned_pod')
        save(p, budget)


def job(kind, pct, attempt=1):
    key = f'{kind}{pct}-attempt{attempt}'
    dest = OUT / key
    dest.mkdir(exist_ok=False)
    arm = read(OUT / 'campaign.json')['arms'][str(pct)]
    audit = read(OUT / f'audit{pct}.json')
    assert audit['passed'] and arm['data_repo'] == audit['data_repo']
    state = {'key': key, 'kind': kind, 'pct': pct, 'pid': os.getpid(), 'phase': 'preflight', 'owned_pod': None}
    def update(**values):
        state.update(values, updated_epoch=time.time())
        save(dest / 'status.json', state)
    update()
    reserve(key)
    remote = None
    child = None
    def owned(pod):
        update(owned_pod=pod, created_epoch=time.time(), phase='provisioning')
        guard = runpod.start_watchdog(pod, int(CFG.pod_max_hours * 3600), dest / 'watchdog.log')
        info = runpod.call('GET', '/pods/' + pod)
        price = float(info['costPerHr']) + 0.10
        update(hourly_usd=price, watchdog_pid=guard.pid)
        assert price <= float(CFG.pod_hourly_ceiling_usd), 'Rental quote exceeds reserved rate'
    try:
        api = hf_api()
        if kind == 'train':
            assert attempt == 1, 'Training retries require explicit checkpoint recovery, never another training'
            assert not api.repo_exists(arm['organism'], repo_type='model'), 'Adapter already exists'
            assert api.dataset_info(arm['data_repo'], revision=audit['data_revision']).sha == audit['data_revision']
            result = runpod.up(name=f'nika-da-sep25-{pct}-train', train='configs/train/sft.yaml', model='qwen36',
                               count=1, push_env=True, max_hours=float(CFG.pod_max_hours), on_provisioned=owned)
        else:
            trained = read(OUT / f'train{pct}-attempt1/status.json')
            assert trained.get('adapter_revision'), 'No verified adapter'
            spec = resolve_target(arm['organism'])
            assert spec.revision == trained['adapter_revision'] and spec.base_revision == CFG.base_model_revision
            assert spec.mode == 'think'
            name = eval_name(kind, arm['organism'])
            assert not api.repo_exists('dougalldeepmind/' + name, repo_type='dataset'), 'Eval repo already exists'
            port = int(CFG.ports[f'{kind}{pct}'])
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', port))
            result = runpod.up(name=f'nika-da-sep25-{pct}-{kind}', eval=kind, target=arm['organism'], count=1,
                               push_env=True, max_hours=float(CFG.pod_max_hours), on_provisioned=owned)
        (dest / 'provision.txt').write_text(result, encoding='utf-8')
        host = re.search(r'^host:\s+(\S+)', result, re.M).group(1)
        remote = SshExec(host, port=8000, workdir='/root/work')
        update(host=host, phase='bootstrap')
        assert runpod.wait_bootstrapped(state['owned_pod'], timeout_s=1800), 'Bootstrap timeout'
        py = '/root/work/.venv/bin/python' if kind == 'train' else f'{POD_VENV}/bin/python'
        probe = "import torch;assert torch.cuda.is_available();assert torch.cuda.device_count()==1;print(torch.cuda.get_device_name(),torch.ones(1,device='cuda').item())"
        update(cuda_check=remote._ssh(py + ' -c ' + shlex.quote(probe), timeout=120))
        if kind == 'train':
            rd = '/root/work/output/da-sep25-campaign'
            command = ['uv', 'run', '--frozen', 'train', '--config', 'configs/train/sft.yaml', 'model=qwen36',
                       'seed=0', 'wandb=true', 'data_repo=' + arm['data_repo'], 'data_revision=' + audit['data_revision'],
                       'base_model_revision=' + CFG.base_model_revision, 'hf_repo=' + arm['organism'].split('/')[1],
                       'train.loss_agg=token_mean', 'train.packing=true', 'train.token_budget=8000', 'train.epochs=1']
            script = '\n'.join(['#!/bin/bash', 'set -u', 'cd /root/work', 'export HF_HOME=/workspace/hf',
                                'export PYTHONUNBUFFERED=1', 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
                                f'mkdir -p {rd}', shlex.join(command) + f' > {rd}/train.log 2>&1',
                                'rc=$?', f'printf "%s" "$rc" > {rd}/train.exit', 'exit "$rc"', ''])
            remote._ssh(f'mkdir -p {rd} && cat > {rd}/run.sh', stdin_text=script)
            remote._ssh(f'bash -n {rd}/run.sh')
            remote._ssh(f'nohup setsid bash {rd}/run.sh > {rd}/driver.log 2>&1 </dev/null & echo $! > {rd}/driver.pid')
            update(phase='training', training_started=True, command=shlex.join(command))
            last_change = time.time()
            prior_size = None
            checkpoint_worker = None
            while time.time() - state['created_epoch'] < float(CFG.pod_max_hours) * 3600 - 900:
                probe = f"""import json,subprocess
from pathlib import Path
p=Path({rd!r});f=p/'train.log';e=p/'train.exit'
text=f.read_text(errors='replace') if f.exists() else ''
c=list(Path('/root/work/output/train').glob('*/checkpoint-*/trainer_state.json'))
print(json.dumps(dict(bytes=len(text),tail=text[-10000:],exit=int(e.read_text()) if e.exists() else None,checkpoints=[str(x.parent.relative_to('/root/work')) for x in c],gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True))))
"""
                try:
                    progress = json.loads(remote._ssh('python3 -c ' + shlex.quote(probe), timeout=90))
                except Exception as exc:
                    update(monitor_error=type(exc).__name__)
                    if time.time() - last_change > 1200:
                        raise RuntimeError('Training monitor unavailable for 20 minutes') from exc
                    time.sleep(30)
                    continue
                (dest / 'train_tail.log').write_text(progress['tail'], encoding='utf-8')
                if progress['bytes'] != prior_size:
                    last_change = time.time()
                    prior_size = progress['bytes']
                assert not re.search(r"(?:'loss'|'grad_norm'):\s*(?:nan|inf)\b", progress['tail'], re.I), 'Non-finite metric'
                update(progress=progress, estimated_usd=(time.time()-state['created_epoch'])/3600*state['hourly_usd'])
                if progress['checkpoints'] and checkpoint_worker is None:
                    checkpoint_worker = spawn([sys.executable, '-m', 'scratch.nonmoral.result_backup', '--host', host,
                                               '--checkpoint', sorted(progress['checkpoints'])[0], '--out', str(dest/'checkpoint_backup')], dest/'checkpoint_backup.log')
                    update(checkpoint_backup_pid=checkpoint_worker.pid)
                if progress['exit'] is not None:
                    assert progress['exit'] == 0, 'Training failed; inspect saved log'
                    info = api.model_info(arm['organism'])
                    meta = read(hf_download(arm['organism'], 'training_meta.json', revision=info.sha))
                    assert meta['dataset']['revision'] == audit['data_revision']
                    assert meta['base_model_revision'] == CFG.base_model_revision and meta['thinking'] is True
                    config = meta['train_config']
                    assert config['seed'] == 0 and config['train']['loss_agg'] == 'token_mean' and config['train']['epochs'] == 1
                    assert any(f.rfilename.endswith('.safetensors') for f in info.siblings)
                    update(phase='published', adapter_revision=info.sha, training_complete=True)
                    break
                assert time.time() - last_change < 1200, 'Training stalled for 20 minutes'
                time.sleep(30)
            else:
                raise TimeoutError('Training deadline reached')
            update(phase='preserving')
            logs = remote._ssh('cat ' + rd + '/train.log', timeout=60)
            (dest/'train.log').write_text(logs, encoding='utf-8')
            receipt = fetch_training_outputs(remote, dest, timeout=720, include_roots=['output/train'],
                                             archive_name='da-sep25-final-backup.tar', exclude_checkpoints=True)
            save(dest/'backup.json', receipt)
            api.upload_file(path_or_fileobj=str(dest/'train.log'), path_in_repo='training_logs/train.log', repo_id=arm['organism'], commit_message='Preserve complete training log')
            update(adapter_revision=api.model_info(arm['organism']).sha, backup=receipt)
        else:
            module = 'scratch.da_supervision.odcv_eval' if kind == 'odcv' else 'src.eval.run_eval'
            cmd = [sys.executable, '-m', module, '--name', kind, '--target', arm['organism'], '--server', host,
                   '--port', str(port), '--terminate-pod', 'output_root=' + (dest/'eval').as_posix()]
            if kind == 'odcv':
                cmd += ['--server-bind', '0.0.0.0', f'concurrency={CFG.odcv_concurrency}', 'prune_images=false']
            child = spawn(cmd, dest/'eval.log')
            update(phase='evaluating', eval_pid=child.pid, target_revision=trained['adapter_revision'])
            deadline = state['created_epoch'] + float(CFG.pod_max_hours)*3600 - 120
            while child.poll() is None and time.time() < deadline:
                update(estimated_usd=(time.time()-state['created_epoch'])/3600*state['hourly_usd'])
                time.sleep(30)
            assert child.poll() == 0, 'Eval failed or exceeded deadline; preserve local work'
            repo = 'dougalldeepmind/' + eval_name(kind, arm['organism'])
            info = api.dataset_info(repo)
            meta = read(hf_download(repo, 'metadata/run_meta.json', repo_type='dataset', revision=info.sha))
            assert meta['target_revision'] == trained['adapter_revision'], 'Published eval target revision mismatch'
            results = read(hf_download(repo, 'results/results.json', repo_type='dataset', revision=info.sha))
            if kind == 'odcv':
                summary = read(hf_download(repo, 'metadata/pass_summary.json', repo_type='dataset', revision=info.sha))
                assert summary['requested_passes'] == 3 and summary['kept_passes'] == 3
                assert sum(a['transcripts_nonempty'] for a in summary['audits']) == 240
            else:
                assert results['n_rows'] == 1000 and results['passes'] == 1
            update(eval_repo=repo, eval_revision=info.sha, results=results, eval_complete=True)
        update(phase='complete')
    except BaseException as exc:
        update(phase='failed', error=f'{type(exc).__name__}: {exc}')
        print(state['error'], flush=True)
        if kind == 'train' and remote and state.get('training_started'):
            try:
                # Preserve the exact failed attempt for checkpoint recovery, never retrain blindly.
                remaining = int(state['created_epoch'] + float(CFG.pod_max_hours)*3600 - time.time() - 60)
                assert remaining > 60, 'No preservation time remains before hard deadline'
                recovery = fetch_training_outputs(remote, dest, timeout=min(remaining, 720),
                                                  include_roots=['output/train'], archive_name='da-sep25-recovery.tar')
                update(recovery_backup=recovery)
            except BaseException as backup_exc:
                update(recovery_error=type(backup_exc).__name__)
    finally:
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
        if state.get('owned_pod'):
            try:
                runpod.teardown(state['owned_pod'])
                update(terminated=True, terminated_epoch=time.time())
            except BaseException as exc:
                update(teardown_error=type(exc).__name__)
        settle(key, state)
    return 0 if state['phase'] == 'complete' else 1


def api_usage():
    import requests
    r = requests.get('https://openrouter.ai/api/v1/credits', headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']}, timeout=20)
    r.raise_for_status()
    return float(r.json()['data']['total_usage'])


def controller():
    receipt = OUT/'controller.json'
    assert not receipt.exists(), 'Controller already registered'
    usage_start = api_usage()
    state = {'pid': os.getpid(), 'started_epoch':time.time(), 'api_usage_start':usage_start, 'jobs':{}}
    save(receipt,state)
    processes = {}
    try:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        for pct in [5,25]:
            assert read(OUT/f'audit{pct}.json')['passed']
            key=f'train{pct}-attempt1'
            p=spawn([sys.executable,__file__,'train',str(pct)],OUT/(key+'.log'))
            processes[key]=p;state['jobs'][key]=p.pid
        while True:
            try:
                state['api_usage_delta_upper_bound']=max(0,api_usage()-usage_start)
                state['api_checked_epoch']=time.time()
            except Exception as exc:
                state['api_check_error']=type(exc).__name__
                assert time.time()-state.get('api_checked_epoch',state['started_epoch'])<600,'API accounting unavailable for 10 minutes'
            state.setdefault('api_usage_delta_upper_bound',0)
            assert state['api_usage_delta_upper_bound']<float(CFG.api_stop_delta_usd),'API budget guard reached'
            for pct in [5,25]:
                path=OUT/f'train{pct}-attempt1/status.json'
                trained=read(path) if path.exists() else {}
                # Wait for the final model revision, including its training-log upload.
                if trained.get('phase')=='complete':
                    for kind in ['odcv','mask']:
                        key=f'{kind}{pct}-attempt1'
                        if key not in processes:
                            p=spawn([sys.executable,__file__,kind,str(pct)],OUT/(key+'.log'))
                            processes[key]=p;state['jobs'][key]=p.pid
            state['updated_epoch']=time.time()
            state['exit_codes']={k:p.poll() for k,p in processes.items()}
            save(receipt,state)
            if all(p.poll() is not None for p in processes.values()):
                break
            time.sleep(30)
        state['phase']='finished' if len(processes)==6 and all(p.returncode==0 for p in processes.values()) else 'needs_recovery'
        save(receipt,state)
    except BaseException as exc:
        state.update(phase='stopped',error=f'{type(exc).__name__}: {exc}')
        save(receipt,state)
        for p in processes.values():
            if p.poll() is None:
                p.terminate()
        # Per-job and deadline watchdogs retain independent cleanup responsibility.
        for path in OUT.glob('*-attempt*/status.json'):
            owned=read(path).get('owned_pod')
            if owned:
                try:
                    runpod.teardown(owned)
                except Exception:
                    pass
        raise
    finally:
        if os.name=='nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('kind',choices=['controller','train','odcv','mask'])
    parser.add_argument('pct',nargs='?',type=int,choices=[5,25])
    args=parser.parse_args()
    if args.kind=='controller':
        controller()
    else:
        raise SystemExit(job(args.kind,args.pct))
