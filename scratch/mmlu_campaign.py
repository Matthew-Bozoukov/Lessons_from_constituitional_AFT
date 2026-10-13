# ABOUTME: Bounded three-H200 MMLU campaign with persisted receipts and exact targets.
# ABOUTME: Run via uv run python scratch/mmlu_campaign.py prepare|launch|worker|status|report.
from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from dotenv import load_dotenv
load_dotenv(Path(r'C:\Users\nikak\source\repos\LASR\teaching_claude_why_replication\.env'))
os.environ['HF_ORG'] = 'dougalldeepmind'
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUNBUFFERED'] = '1'
from omegaconf import OmegaConf
from src.infra import runpod
from src.eval.capabilities.mmlu import runner
from src.eval.capabilities.mmlu.mmlu import build_subset, load_split, build_prompt, prompt_hash, score_records, paired_diff

BASE = 'Qwen/Qwen3.6-27B'
BASE_REV = '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'
TARGETS = {
    'base': (BASE, BASE_REV),
    'plain': ('dougalldeepmind/2026-10-05-qwen36-0-plain', '1565d4f3356aa1b3bea1c503700164faa240ef3a'),
    'da15': ('dougalldeepmind/2026-10-05-qwen36-0-da-15', 'ce773a6b87db66bd569dd13071ec5e27201462f1'),
}
PORTS = {'base': 18201, 'plain': 18202, 'da15': 18203}
OUT = ROOT / 'output/mmlu_campaign_2026-10-14'
LOCK = threading.RLock()
STATE = {}
ABORT = threading.Event()
PROCESSES = {}
GUARDS = {}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    for attempt in range(10):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.1)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def update(arm=None, **fields):
    with LOCK:
        if arm:
            STATE['arms'][arm].update(fields)
        else:
            STATE.update(fields)
        STATE['updated_at'] = time.time()
        save(OUT / 'state.json', STATE)


def prepare():
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi
    from src.infra.endpoints.vllm import resolve_target, plan_serving, serving_params, native_context_window
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'state.json').exists(), 'Existing campaign; do not overwrite its receipt'
    keypair = runpod.default_keypair()
    assert keypair, 'No SSH keypair'
    api = HfApi(token=os.environ.get('HF_TOKEN'))
    print('HF authentication:', api.whoami()['name'])
    runpod.validate_scheduled_provision()
    price = runpod.gpu_price('NVIDIA H200', 'SECURE')
    assert price is not None and price <= 5.4, f'Unexpected GPU rate: {price}'
    account = runpod.graphql('query { myself { clientBalance currentSpendPerHr } }')['myself']
    assert account['clientBalance'] >= 60, 'Insufficient credit plus shared-account reserve'
    cfg = OmegaConf.load('configs/eval/mmlu.yaml')
    cfg.mode = 'think'
    cfg.generation.extra_body = {'top_k': 20, 'min_p': 0.0, 'repetition_penalty': 1.0}
    cfg.generation.sampling_seed = 0
    rows = load_split('test', cfg.subset.dataset, cfg.subset.name, revision=cfg.subset.revision)
    questions = build_subset(rows, 10, seed=0, shuffle_choices=True)
    shots = runner._shots_by_subject(cfg, 0)
    dev = load_split('dev', cfg.subset.dataset, cfg.subset.name, revision=cfg.subset.revision)
    smoke = build_subset(dev, 1, seed=0, subjects=['anatomy', 'business_ethics', 'high_school_biology', 'professional_medicine'])
    # Smoke deliberately has no few-shot examples: no test item is spent on qualification.
    for q in smoke:
        q['uid'] = 'smoke-dev:' + q['uid']
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REV)
    manifest = []
    for q in questions:
        prompt = build_prompt(q, shots.get(q['subject'], []), instruction=str(cfg.prompt.instruction), cue=str(cfg.prompt.cue))
        rendered = tok.apply_chat_template([{'role':'user','content':prompt}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=True)
        n = len(tok.encode(rendered, add_special_tokens=False))
        assert n > 20, 'Prompt tokenization returned an unexpected shape'
        assert n + int(cfg.generation.max_tokens) <= int(cfg.serving.context_window)
        manifest.append({**q, 'prompt': prompt, 'prompt_hash': prompt_hash(prompt), 'prompt_tokens': n})
    assert len(questions) == 570 and len(smoke) == 4
    save(OUT / 'manifest.json', manifest)
    save(OUT / 'smoke.json', smoke)
    pins = {BASE: BASE_REV}
    pins.update({t:r for t,r in TARGETS.values()})
    for arm, (target, rev) in TARGETS.items():
        spec = resolve_target(target, revision=rev)
        assert spec.base_revision == BASE_REV and spec.mode != 'nothink'
        plan_serving(dict(serving_params(BASE), native_context_window=native_context_window(BASE, BASE_REV)),
                     OmegaConf.to_container(cfg.serving), BASE, spec.mode)
        armcfg = OmegaConf.merge(cfg, {'target_revision': rev, 'output_root': str(OUT / arm / 'eval')})
        (OUT / arm).mkdir(exist_ok=True)
        OmegaConf.save(armcfg, OUT / arm / 'config.yaml')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', PORTS[arm]))
    # Validate bootstrap syntax before any paid action.
    for arm, (target, rev) in TARGETS.items():
        _, weights, _, _, _ = runpod.plan_eval_pod(target, eval='mmlu', revisions=pins)
        runpod._check_bash(runpod._bootstrap(None, weights, weight_revisions=pins))
    save(OUT / 'preflight.json', {'timestamp': time.time(), 'gpu_price':price, 'account':account,
         'key_public_path':keypair[0], 'key_private_path':keypair[1], 'pins':pins,
         'questions':570, 'max_prompt_tokens':max(x['prompt_tokens'] for x in manifest),
         'manifest_sha256':hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest(),
         'budget_usd':50, 'startup_smoke_budget_usd':8, 'max_hours':3,
         'other_pods':[{k:p.get(k) for k in ['id','name','costPerHr']} for p in runpod.active_pods()]})
    print('PREPARED', OUT, '570 questions; max prompt tokens', max(x['prompt_tokens'] for x in manifest))


def worker(arm, host, identity):
    from src.eval.run_eval import main
    target, revision = TARGETS[arm]
    def qualified_run(served, cfg, out_dir):
        save(OUT / arm / 'phase.json', {'phase':'smoke', 'time':time.time(), 'out_dir':str(out_dir),
                                      'target':served.spec.hf_path, 'revision':served.spec.revision})
        assert served.spec.hf_path == target and served.spec.revision == revision
        arm_info = {'name':served.spec.model_key, 'served':served.model_name, 'target':target,
                    'revision':revision, 'base_revision':BASE_REV,
                    'adapter':target if arm != 'base' else None, 'synthetic_fraction':None,
                    'role':'target', 'trained':True}
        cfg.generation.enable_thinking = True
        runner.run_arm(arm_info, read(OUT/'smoke.json'), {}, cfg, served.base_url,
                       OUT/arm/'smoke', api_key=served.api_key)
        save(OUT/arm/'phase.json', {'phase':'smoke_passed','time':time.time(),'out_dir':str(out_dir)})
        while not (OUT/'GO').exists():
            if (OUT/'ABORT').exists():
                raise RuntimeError('Campaign aborted before scored inference')
            time.sleep(1)
        save(OUT/arm/'phase.json', {'phase':'evaluation','time':time.time(),'out_dir':str(out_dir)})
        result = runner.run(served, cfg, out_dir)
        records = [json.loads(x) for x in (out_dir/'rollouts/records.jsonl').read_text(encoding='utf-8').split('\n') if x]
        expected = {q['uid']:q for q in read(OUT/'manifest.json')}
        assert len(records) == 570 and len({r['uid'] for r in records}) == 570
        for r in records:
            q = expected[r['uid']]
            assert r['prompt'] == q['prompt'] and r['answer_letter'] == q['answer_letter']
        save(out_dir/'metadata/campaign_manifest.json', read(OUT/'manifest.json'))
        save(OUT/arm/'result.json', {'scores':result,'out_dir':str(out_dir), 'finished':time.time()})
        return result
    main(['--target',target,'--name','mmlu','--config',str(OUT/arm/'config.yaml'),
          '--server',host,'--ssh-key',identity,'--port',str(PORTS[arm])],runner=qualified_run)
    save(OUT/arm/'phase.json', {'phase':'published','time':time.time()})


def attributable_cost():
    now = time.time()
    return sum(max(0, a.get('terminated_at',now)-a['rented_at'])/3600*a['hourly_rate']
               for a in STATE['arms'].values() if 'rented_at' in a)


def stop_pod(arm):
    with LOCK:
        a = dict(STATE['arms'][arm])
    if a.get('pod_id') and not a.get('terminated_at'):
        runpod.teardown(a['pod_id'])
        update(arm, terminated_at=time.time())
        guard = GUARDS.get(arm)
        if guard:
            guard.terminate()


def run_arm_process(arm, preflight):
    from src.infra.endpoints.vllm import SshExec
    start = time.time()
    target, rev = TARGETS[arm]
    def allocated(pod_id):
        update(arm, pod_id=pod_id, rented_at=time.time(), hourly_rate=preflight['gpu_price']+.05,
               status='starting')
        GUARDS[arm] = runpod.start_watchdog(pod_id, 10800, OUT/arm/'watchdog.log')
    try:
        deadline = start + 10800
        pod = runpod.provision_eval_pod(target, name=f'nika-mmlu-20261014-{arm}', gpu='NVIDIA H200',
            count=1, eval='mmlu', revisions=preflight['pins'], cloud='SECURE',
            pubkey_path=preflight['key_public_path'], identity=preflight['key_private_path'],
            terminate_at=datetime.fromtimestamp(deadline,timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            boot_deadline=start+1200, on_provisioned=allocated,
            env={'LASR_POD_OWNER':runpod.POD_OWNER,'LASR_POD_DEADLINE':str(deadline)})
        update(arm, host=pod.host, reachable=pod.reachable)
        ex = SshExec(pod.host, port=PORTS[arm], identity=preflight['key_private_path'])
        shard = f'https://huggingface.co/{BASE}/resolve/{BASE_REV}/model-00001-of-00012.safetensors'
        # Exact weight file selected from the checked model file listing by prepare/metadata below.
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(BASE, revision=BASE_REV)
        shard = f'https://huggingface.co/{BASE}/resolve/{BASE_REV}/'+next(f for f in files if f.endswith('.safetensors'))
        speed = ex._ssh(f"curl -sL --range 0-16777215 -o /dev/null -m 15 -w '%{{speed_download}}' '{shard}' || true", timeout=25)
        try:
            speed_num = float(speed.strip().splitlines()[-1])
        except (ValueError,IndexError):
            speed_num = 0
        update(arm, download_bytes_per_second=speed_num)
        if speed_num < 1_000_000:
            raise RuntimeError(f'Download qualification failed: {speed_num:.0f} bytes/s')
        while not ABORT.is_set():
            log = ex._ssh('tail -n 60 /workspace/boot.log', timeout=30)
            (OUT/arm/'boot_tail.log').write_text(log,encoding='utf-8')
            if 'READY vllm venv' in log:
                break
            if time.time()-start > 1200:
                raise TimeoutError('20-minute startup cap')
            time.sleep(10)
        if ABORT.is_set():
            return
        # Prove CUDA works, then invoke the maintained server/eval/publish lifecycle.
        check = ex._ssh("/workspace/vllmenv/bin/python -c \"import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())\"",timeout=30)
        update(arm, cuda_check=check.strip(), status='serving')
        with (OUT/arm/'driver.log').open('w',encoding='utf-8') as log:
            proc = subprocess.Popen([sys.executable,'-u',str(Path(__file__)), 'worker','--arm',arm,
                                     '--host',pod.host,'--identity',preflight['key_private_path']],
                                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            PROCESSES[arm] = proc
            update(arm, worker_pid=proc.pid)
            while proc.poll() is None:
                if ABORT.is_set():
                    proc.terminate()
                    break
                time.sleep(2)
            code = proc.wait(timeout=30)
        if code:
            raise RuntimeError(f'Eval worker exited {code}; see driver.log')
        update(arm, status='completed')
    except BaseException as exc:
        update(arm, status='failed', error=f'{type(exc).__name__}: {exc}')
        (OUT/arm/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        ABORT.set()
    finally:
        try:
            stop_pod(arm)
        except BaseException:
            (OUT/arm/'teardown_error.txt').write_text(traceback.format_exc(),encoding='utf-8')


def launch():
    global STATE
    preflight = read(OUT/'preflight.json')
    assert not (OUT/'state.json').exists(), 'Refusing duplicate launch'
    runpod.validate_scheduled_provision()
    assert runpod.gpu_price('NVIDIA H200','SECURE') <= 5.4
    STATE = {'started_at':time.time(),'controller_pid':os.getpid(),'budget_usd':50,'status':'starting',
             'arms':{a:{'target':t,'revision':r,'port':PORTS[a],'status':'pending'} for a,(t,r) in TARGETS.items()}}
    update()
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    futures = [pool.submit(run_arm_process,a,preflight) for a in TARGETS]
    try:
        while not all(f.done() for f in futures):
            phases = {}
            for a in TARGETS:
                p = OUT/a/'phase.json'
                if p.exists():
                    phases[a] = read(p)
            cost = attributable_cost()
            update(estimated_cost_usd=cost)
            if cost >= 49 or time.time()-STATE['started_at'] >= 10800:
                update(abort_reason='Campaign dollar/lifetime cap')
                ABORT.set()
            if not (OUT/'GO').exists():
                for a,p in phases.items():
                    if p['phase']=='smoke' and time.time()-p['time']>600:
                        update(abort_reason=f'{a}: 10-minute smoke cap')
                        ABORT.set()
                for a,s in STATE['arms'].items():
                    if s.get('rented_at') and a not in phases and time.time()-s['rented_at']>1200:
                        update(abort_reason=f'{a}: 20-minute startup cap')
                        ABORT.set()
                if cost>=8 or time.time()-STATE['started_at']>1800:
                    update(abort_reason='Startup/smoke $8 or 30-minute cap')
                    ABORT.set()
                if len(phases)==3 and all(p['phase']=='smoke_passed' for p in phases.values()) and not ABORT.is_set():
                    (OUT/'GO').write_text(str(time.time()))
                    update(status='evaluating',evaluation_started_at=time.time())
            if ABORT.is_set():
                (OUT/'ABORT').write_text(str(time.time()))
                # Also end pods stuck inside a blocking startup call immediately.
                for a in TARGETS:
                    try:
                        stop_pod(a)
                    except Exception:
                        pass
            time.sleep(5)
    finally:
        ABORT.set()
        for proc in PROCESSES.values():
            if proc.poll() is None:
                proc.terminate()
        pool.shutdown(wait=True)
        for a in TARGETS:
            stop_pod(a)
        update(status='completed' if all(a['status']=='completed' for a in STATE['arms'].values()) else 'failed',
               finished_at=time.time(),estimated_cost_usd=attributable_cost())
        if os.name=='nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    if STATE['status']=='completed':
        report()


def status():
    state = read(OUT/'state.json') if (OUT/'state.json').exists() else {'status':'not launched'}
    for arm in TARGETS:
        paths = list((OUT/arm/'eval').glob('**/records.jsonl'))
        recs = []
        if paths:
            for line in max(paths,key=lambda p:p.stat().st_mtime).read_text(encoding='utf-8').split('\n'):
                if line:
                    try: recs.append(json.loads(line))
                    except json.JSONDecodeError: pass
        if arm in state.get('arms',{}):
            state['arms'][arm]['progress'] = {'completed':len({r['uid'] for r in recs}),
                'truncated':sum(r.get('finish_reason')=='length' for r in recs),
                'errors':sum(r.get('finish_reason')=='timeout' for r in recs),
                'output_tokens':sum((r.get('usage') or {}).get('completion_tokens',0) for r in recs)}
            if (OUT/arm/'phase.json').exists(): state['arms'][arm]['phase']=read(OUT/arm/'phase.json')
    print(json.dumps(state,indent=2))


def report():
    records = {}
    summaries = {}
    for arm in TARGETS:
        r = read(OUT/arm/'result.json')
        path = Path(r['out_dir'])/'rollouts/records.jsonl'
        records[arm] = sorted([json.loads(s) for s in path.read_text(encoding='utf-8').split('\n') if s],key=lambda r:r['uid'])
        summaries[arm] = score_records(records[arm])
    contrasts = {}
    for reference, arm in [('plain','da15'),('base','plain'),('base','da15')]:
        d = paired_diff(records[reference],records[arm])
        d['non_inferior_3pp'] = d['ci_lower'] >= -.03
        contrasts[f'{arm}_minus_{reference}'] = d
    save(OUT/'comparison.json', {'arms':summaries,'contrasts':contrasts,'state':read(OUT/'state.json')})
    lines = ['# MMLU matched comparison, 2026-10-14','', '| Arm | Correct / 570 | Accuracy | 95% CI |','|---|---:|---:|---|']
    for a,s in summaries.items(): lines.append(f"| {a} | {s['n_correct']} / {s['n']} | {s['mean']:.2%} | [{s['ci_lower']:.2%}, {s['ci_upper']:.2%}] |")
    for name,d in contrasts.items(): lines += ['',f"{name}: {100*d['diff']:+.2f} pp, 95% CI [{100*d['ci_lower']:+.2f}, {100*d['ci_upper']:+.2f}]; non-inferior within 3 pp: {d['non_inferior_3pp']}."]
    (OUT/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','launch','worker','status','report'])
    p.add_argument('--arm');p.add_argument('--host');p.add_argument('--identity')
    args=p.parse_args()
    if args.action=='worker': worker(args.arm,args.host,args.identity)
    else: globals()[args.action]()
