# ABOUTME: Complete preservation for the already-running original-low-stakes owner without interrupting training.
# ABOUTME: Correct its loaded verifier's metadata lookup, then verify backups/publication before owned teardown.
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.infra import runpod
from src.infra.endpoints.vllm import SshExec
from scratch.nonmoral.result_backup import fetch_training_outputs, verify_publication
from scratch.nonmoral.train_pair import dump
from scratch.nonmoral.account_snapshot import snapshot


def main(campaign):
    root = Path(campaign)
    out = root / 'completion_preserver'
    out.mkdir(exist_ok=True)
    assert not (out / 'status.json').exists(), 'Do not duplicate completion preservation'
    progress = {'phase': 'waiting_for_completed_adapter', 'pid': os.getpid(),
                'reason': 'Loaded owner looks up organism in run_meta rather than adapter training_meta. Training itself is unchanged.'}
    dump(out / 'status.json', progress)
    while True:
        state = json.loads((root / 'train_attempt1/status.json').read_text())
        if state.get('adapter_backup', {}).get('verified') and state.get('completed_arms') == [0]:
            break
        if state.get('terminated'):
            progress.update(phase='owner_terminated_before_completion')
            dump(out / 'status.json', progress)
            (root / 'keep_awake.stop').touch()
            return
        if time.time() > state['created_epoch'] + 17280:
            progress.update(phase='watchdog_deadline_reached')
            dump(out / 'status.json', progress)
            return
        time.sleep(30)
    plan = state['plan']
    assert plan['run_name'] == 'nika-low-stakes-original-train' and plan['gpu_budget_usd'] == 50
    expected = [dict(a, base_model_revision=plan['base_model_revision']) for a in plan['arms']]
    try:
        progress['phase'] = 'verifying_publication'
        dump(out / 'status.json', progress)
        progress['publication'] = verify_publication(state['adapter_backup']['archive'], expected, steps=625, world_size=2)
        dump(out / 'publication.json', progress['publication'])
        remaining = int(state['created_epoch'] + 17280 - time.time() - 120)
        assert remaining > 120, 'Insufficient time for output recovery'
        remote = SshExec(state['host'], port=8000, workdir='/root/work')
        assert remote._ssh('cat /root/work/output/nonmoral-paired-supervision/arm_0.exit', timeout=30).strip() == '0'
        progress['phase'] = 'preserving_complete_outputs'
        dump(out / 'status.json', progress)
        progress['local_backup'] = fetch_training_outputs(remote, out, expected,
            timeout=remaining, archive_name='completion-preserver-full.tar')
        dump(out / 'local_backup.json', progress['local_backup'])
        assert runpod.terminate(state['owned_pod'])
        assert not any(p['id'] == state['owned_pod'] for p in runpod.active_pods())
        progress.update(phase='complete', owned_pod=state['owned_pod'], terminated=True,
                        estimated_gpu_storage_usd=(time.time()-state['created_epoch'])/3600*state['budget_hourly_usd'])
        assert progress['estimated_gpu_storage_usd'] <= 50
        # The original owner cannot load the verifier correction. Stop it only AFTER
        # every artifact is preserved and its pod is independently verified absent.
        import psutil
        match = re.search(r'parent (\d+) cap', (root / 'train_attempt1/watchdog.log').read_text())
        if match:
            pid = int(match.group(1))
            try:
                process = psutil.Process(pid)
                command = ' '.join(process.cmdline())
                assert 'train_pair.py' in command and 'lowstakes_original_reuse_20260916' in command
                os.kill(pid, signal.SIGTERM)
                progress['retired_owner_pid'] = pid
            except psutil.NoSuchProcess:
                pass
        (root / 'keep_awake.stop').touch()
        progress['accounts_after'] = snapshot()
        dump(out / 'status.json', progress)
        dump(root / 'completion.json', progress)
    except BaseException as exc:
        progress.update(phase='attention', error=f'{type(exc).__name__}: {exc}')
        dump(out / 'status.json', progress)
        raise


if __name__ == '__main__':
    main(sys.argv[1])
