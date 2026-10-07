# ABOUTME: Runs the authorized original-nonmoral plain-base training then matched ODCV and MASK.
# ABOUTME: Uses existing guarded owners, immutable publication audits and a fixed resource envelope; never retries outcomes.
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scratch.nonmoral.account_snapshot import snapshot
from scratch.plain_dose.audit_evals import audit_entry
from src.infra import runpod
from src.infra.huggingface import hf_api
from src.naming import eval_name, undated

OUT = ROOT / 'output/nonmoral_plain'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def launch(argv, log):
    with log.open('wb') as stream:
        return subprocess.Popen([sys.executable, *argv], cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1'},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)


def funds(required):
    account = snapshot()
    assert account['runpod_http_status'] == account['openrouter_http_status'] == 200
    assert account['runpod']['clientBalance'] >= 50 + required, 'Hold rentals: insufficient unreserved balance'
    assert account['openrouter']['total_credits'] - account['openrouter']['total_usage'] >= 10
    return account


def run():
    plan = read(OUT / 'training_plan.json')
    status = OUT / 'campaign_status.json'
    with status.open('x', encoding='utf-8') as stream:
        json.dump({'phase': 'claimed', 'pid': os.getpid()}, stream)
    state = {'phase': 'preflight', 'pid': os.getpid(), 'started_epoch': time.time(), 'gpu_limit_usd': 40,
             'training_plan': plan, 'owners': {}}
    owner_processes = []

    def save(**updates):
        state.update(updates, updated_epoch=time.time())
        write(status, state)

    try:
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == plan['source_commit']
        assert plan['pod']['max_hours'] * plan['pod']['hourly_ceiling_usd'] == 9.6
        assert len(plan['arms']) == 1
        arm = plan['arms'][0]
        assert not hf_api().repo_exists(arm['organism'], repo_type='model'), 'Refuse duplicate adapter training'
        from src.eval.docker import docker_preflight, require_lf_shell_scripts
        docker_preflight()
        require_lf_shell_scripts(ROOT / 'src/eval/misalignment/odcv/third_party/odcv-bench')
        account = funds(34)
        save(account_before=account)
        keeper = launch(['scratch/nonmoral/keep_awake.py', str(OUT)], OUT / 'keep_awake.log')
        save(keep_awake_pid=keeper.pid)
        train_out = OUT / 'train'
        train = launch(['scratch/da_supervision/owner.py', str(OUT / 'training_plan.json'), arm['key'], '--out', str(train_out)], OUT / 'train_owner.log')
        owner_processes.append(train)
        save(phase='training', train_owner_pid=train.pid)
        while train.poll() is None:
            if (train_out / 'status.json').exists():
                state['owners']['train'] = read(train_out / 'status.json')
                save()
            time.sleep(20)
        trained = read(train_out / 'status.json')
        state['owners']['train'] = trained
        save()
        assert train.returncode == 0 and trained['phase'] == 'complete' and trained.get('terminated')
        publication = read(train_out / 'publication_verified.json')
        assert publication['verified'] and publication['revision'] == trained['adapter_revision']
        assert trained['owned_pod'] not in {p['id'] for p in runpod.active_pods()}
        save(phase='trained_verified', adapter=publication)
        # Total admitted upper bound: training 9.6 + MASK 9.6 + ODCV 14.8 = $34 GPU.
        account = funds(24.4)
        save(account_before_evals=account)
        children = {}
        for kind, port, hours, ceiling in [('odcv', 18115, 4, 3.7), ('mask', 18116, 2, 4.8)]:
            alias = Path('C:/Users/nikak/npc')
            assert alias.resolve() == OUT.resolve(), 'Short-path junction must resolve to this campaign'
            ep = {'eval': kind, 'target': arm['organism'], 'target_revision': publication['revision'],
                  'base_revision': plan['base_model_revision'], 'pod_name': 'nika-nonmoral-plain-' + kind,
                  'port': port, 'output_dir': str(alias / kind), 'max_hours': hours,
                  'hourly_ceiling_usd': ceiling, 'cloud': 'SECURE', 'boot_timeout_s': 2400, 'idle_timeout_s': 3600}
            pp = OUT / (kind + '_plan.json')
            write(pp, ep)
            child = launch(['scratch/plain_dose/eval_owner.py', str(pp)], OUT / (kind + '_owner.log'))
            owner_processes.append(child)
            children[kind] = child
            state['owners'][kind] = {'pid': child.pid, 'phase': 'starting', 'plan': ep}
            save(phase='evaluating')
        pending = set(children)
        while pending:
            for kind in list(pending):
                sp = OUT / kind / 'status.json'
                if sp.exists():
                    state['owners'][kind] = read(sp)
                if children[kind].poll() is not None:
                    pending.remove(kind)
                    state['owners'][kind]['owner_exit'] = children[kind].returncode
                save()
            time.sleep(20)
        receipts = []
        for kind in ('odcv', 'mask'):
            owner = read(OUT / kind / 'status.json')
            assert children[kind].returncode == 0 and owner.get('terminated') and owner.get('eval_exit') == 0, kind
            root = Path(owner['packaged_run'])
            # Names are minted using the actual run metadata date, not the campaign start date.
            meta = read(root / 'metadata/run_meta.json')
            dataset = meta.get('hf_repo') or ('dougalldeepmind/' + eval_name(kind, undated(arm['organism'].split('/')[1]), date=root.name[:10]))
            info = hf_api().dataset_info(dataset)
            entry = {'eval': kind, 'run_dir': str(root), 'target': arm['organism'], 'target_revision': publication['revision'],
                     'hf_repo': dataset, 'hf_revision': info.sha}
            receipts.append(audit_entry(entry))
            save(phase='verifying', eval_receipts=receipts)
        ids = {v['owned_pod'] for v in state['owners'].values() if v.get('owned_pod')}
        active = runpod.active_pods()
        assert not ids.intersection(p['id'] for p in active), 'Owned GPU remains active'
        save(phase='complete', eval_receipts=receipts, account_after=snapshot(), zero_owned_gpus=True)
    except BaseException as exc:
        save(phase='needs_attention', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if all(child.poll() is not None for child in owner_processes):
            (OUT / 'keep_awake.stop').write_text('campaign stopped\n', encoding='utf-8')


if __name__ == '__main__':
    run()
