# ABOUTME: Freeze historical looping prefixes and run a bounded single-GPU paired sampling investigation.
# ABOUTME: Saves exact requests/responses locally, preserves old outcomes, and owns watchdog-protected teardown.
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import requests
import yaml
from dotenv import load_dotenv

CONFIG = Path('scratch/swebench_loop_probe/config.yaml')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def config():
    cfg = yaml.safe_load(CONFIG.read_text())
    load_dotenv(cfg['credentials'])
    return cfg, Path(cfg['root'])


def prepare():
    import copy
    import http.server
    import threading
    import tempfile
    os.environ['MSWEA_GLOBAL_CONFIG_DIR'] = tempfile.mkdtemp(prefix='loop-probe-')
    os.environ['MSWEA_SILENT_STARTUP'] = '1'
    os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
    from minisweagent.models.litellm_model import LitellmModel
    from src.infra.huggingface import hf_download
    cfg, root = config()
    assert not (root/'pod.json').exists(), 'Never change inputs after allocation'
    root.mkdir(parents=True, exist_ok=True)
    examples = json.loads(Path('output/2026-09-24_swebench_da5_format_audit/loop-examples.json').read_text())
    by_id = {x['task']: x for x in examples}
    captured = []

    class Capture(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            captured.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body = json.dumps({'id': 'prepare-only', 'object': 'chat.completion', 'created': 1,
                'model': 'qwen36_0_da_5', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': 'prepare-only'}}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Capture)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    manifest = []
    for iid in cfg['tasks']:
        example = by_id[iid]
        remote = f"rollouts/{iid}/{example['attempt']}/checkpoint.traj.json"
        source = Path(hf_download(cfg['source_repo'], remote, repo_type='dataset', revision=cfg['source_revision']))
        trajectory = json.loads(source.read_text(encoding='utf-8'))
        destination = root/'sources'/f'{iid}.traj.json'
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
        messages = trajectory['messages']
        matches = [i for i,m in enumerate(messages) if m.get('extra', {}).get('response', {}).get('usage', {}).get('prompt_tokens') == example['prompt_tokens'] and m.get('extra', {}).get('response', {}).get('usage', {}).get('completion_tokens') == 65536]
        assert len(matches) == 1, (iid, matches)
        index = matches[0]
        prefix = copy.deepcopy(messages[:index])
        model_cfg = copy.deepcopy(trajectory['info']['config']['model'])
        model_cfg['litellm_model_registry'] = None
        model_cfg['model_kwargs'].update(api_base=f'http://127.0.0.1:{server.server_port}/v1', api_key='local-only', timeout=20, num_retries=0)
        model = LitellmModel(**model_cfg)
        model._query(model._prepare_messages_for_api(copy.deepcopy(prefix)))
        wire = captured.pop()
        assert len(wire['messages']) == len(prefix)
        for original, actual in zip(prefix, wire['messages']):
            for key in ('role','content','tool_calls','tool_call_id','reasoning_content'):
                assert original.get(key) == actual.get(key), (iid,key)
        save(root/'inputs'/f'{iid}.json', wire)
        row = dict(example, message_index=index, source_path=remote,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            request_sha256=digest(wire), prefix_sha256=digest(wire['messages']))
        manifest.append(row)
        print('prepared', iid, 'prompt_tokens', example['prompt_tokens'], flush=True)
    server.shutdown()
    for file in ['generation_config.json','config.json','tokenizer_config.json']:
        source = Path(hf_download(cfg['base'], file, revision=cfg['base_revision']))
        shutil.copy2(source, root/file)
    save(root/'manifest.json', {'config': cfg, 'selection': 'Ten preselected confirmed loops spanning four repositories and 7k-62k prompt tokens; no selection based on replay outcome.', 'cases': manifest})
    save(root/'run_meta.json', {'created_at': datetime.now(timezone.utc).isoformat(), 'git_sha': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), 'config': cfg, 'purpose': 'Diagnostic prefix replay, not a SWE-bench score; no repository tools executed.'})
    shutil.copy2(CONFIG, root/'config.yaml')


def one_request(cfg, root, endpoint, model, case, condition, phase, max_tokens, seed):
    from src.eval.capabilities.swebench_mini.browser import repeated_span
    iid = case['task']; dest = root/'responses'/phase/condition['name']/iid
    if (dest/'summary.json').exists(): return json.loads((dest/'summary.json').read_text())
    body = json.loads((root/'inputs'/f'{iid}.json').read_text())
    body.update(cfg['sampling']); body.update({k:v for k,v in condition.items() if k!='name'})
    body.update(model=model,max_tokens=max_tokens,seed=seed,stream=False)
    assert digest(body['messages']) == case['prefix_sha256']
    save(dest/'request.json', body)
    token_body = {'model':model,'messages': json.loads(json.dumps(body['messages'])),'tools':body['tools'],'add_generation_prompt':True}
    for m in token_body['messages']:
        if m.get('reasoning_content') is not None: m['reasoning'] = m['reasoning_content']
    token = requests.post(endpoint.removesuffix('/v1')+'/tokenize',json=token_body,timeout=120)
    token.raise_for_status(); token = token.json()
    save(dest/'tokenization.json', {k:v for k,v in token.items() if k!='tokens'} | {'token_ids_sha256':digest(token['tokens']) if 'tokens' in token else None})
    assert token['count'] == case['prompt_tokens'], (iid, token['count'], case['prompt_tokens'])
    start = time.time()
    response = requests.post(endpoint+'/chat/completions',json=body,timeout=(30,7200))
    (dest/'response.raw.json').write_bytes(response.content)
    response.raise_for_status(); data = response.json()
    choice = data['choices'][0]; message = choice['message']
    assert data['usage']['prompt_tokens'] == token['count']
    texts = {key:message.get(key) or '' for key in ('reasoning','reasoning_content','content')}
    loops = {key:span for key,value in texts.items() if (span:=repeated_span(value))}
    tools = message.get('tool_calls') or []
    valid_tools = []
    for call in tools:
        try:
            args = json.loads(call['function']['arguments'])
            valid_tools.append(call['function']['name']=='bash' and isinstance(args.get('command'),str))
        except (ValueError,KeyError,AttributeError): valid_tools.append(False)
    summary = dict(task=iid,phase=phase,condition=condition['name'],seed=seed,
        seconds=time.time()-start,usage=data['usage'],finish_reason=choice['finish_reason'],
        loop=bool(loops),loop_evidence=loops,valid_bash=bool(valid_tools) and all(valid_tools),
        historical_motif_occurrences=max((v.count(case['motif']) for v in texts.values()),default=0))
    save(dest/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='loop_evidence'}),flush=True)
    return summary


def run():
    from src.infra import runpod
    from src.infra.endpoints.vllm import resolve_target, VllmServer, SshExec
    cfg,root=config(); manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['config']==cfg
    assert not (root/'pod.json').exists(), 'Existing receipt: diagnose, never rent a second GPU automatically'
    before=runpod.active_pods();save(root/'inventory-before.json',[{k:p.get(k) for k in ('id','name','desiredStatus','costPerHr')} for p in before])
    save(root/'balance-before.json',runpod.graphql('query { myself { currentSpendPerHr clientBalance } }'))
    spec=resolve_target(cfg['target'],revision=cfg['target_revision'])
    assert spec.base_revision==cfg['base_revision']
    gpu=cfg['gpu']; price=runpod.gpu_price(gpu)
    previous_cost=sum(json.loads(p.read_text())['quoted_cost_upper_estimate'] for p in (root/'previous-allocations').glob('*/cleanup.json'))
    assert price and price*1.05*cfg['max_hours']+previous_cost <= cfg['budget_usd']
    start=time.time();deadline=start+cfg['max_hours']*3600
    owned=[];server=None
    def allocated(pod_id):
        owned.append(pod_id)
        save(root/'pod.json',dict(id=pod_id,gpu=gpu,quoted_hourly=price,created_epoch=time.time(),deadline_epoch=deadline,budget_usd=cfg['budget_usd']))
        runpod.start_watchdog(pod_id,int(deadline-time.time()),root/'watchdog.log')
    try:
        pod=runpod.provision_eval_pod(cfg['target'],name=os.environ['USER_PREFIX']+'-swe-loop-probe',gpu=gpu,count=1,
            revisions={cfg['target']:cfg['target_revision'],cfg['base']:cfg['base_revision']},
            terminate_at=datetime.fromtimestamp(deadline,timezone.utc).isoformat(),
            pubkey_path=cfg['ssh_key']+'.pub',identity=cfg['ssh_key'],eval='swebench_mini',
            cuda_versions='13.0,13.1,13.2',on_provisioned=allocated)
        save(root/'connection.json',dict(id=pod.id,ip=pod.ip,port=pod.port,reachable=pod.reachable))
        assert pod.reachable
        executor=SshExec(f'root@{pod.ip}:{pod.port}',cfg['port'],identity=cfg['ssh_key'])
        # Wait for bootstrap installation AND weights before starting serving.
        while time.time()<min(deadline-900,start+1800):
            status=executor._ssh('tail -n 6 /workspace/boot.log',timeout=30)
            if 'READY vllm venv' in status: break
            print('bootstrap',status[-550:],flush=True);time.sleep(20)
        else: raise TimeoutError('Bootstrap failed to finish within 30 minutes')
        server=VllmServer(root/'serve',port=cfg['port'],executor=executor,serve_requirements=cfg['serving'])
        endpoint=server.serve(spec)
        (root/'vllm-startup.log').write_text(executor.tail_log(180),encoding='utf-8')
        save(root/'endpoint.json',dict(base_url=endpoint,model=spec.model_key))
        save(root/'server-facts.json',dict(gpu=executor._ssh('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv'),packages=executor._ssh('/workspace/vllm-env/bin/python -m pip list --format=json') if False else 'vLLM pinned by repository bootstrap'))
        all_rows=[]
        for condition in cfg['conditions']:
            if time.time()>deadline-1200: raise TimeoutError('Insufficient time remaining for next condition')
            with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
                futures=[pool.submit(one_request,cfg,root,endpoint,spec.model_key,c,condition,'screen',cfg['screen_tokens'],cfg['seed']) for c in manifest['cases']]
                for f in as_completed(futures): all_rows.append(f.result());save(root/'screen-results.json',all_rows)
        candidates=[]
        for condition in cfg['conditions'][1:]:
            rows=[r for r in all_rows if r['condition']==condition['name']]
            candidates.append((sum(r['loop'] for r in rows),-sum(r['valid_bash'] for r in rows),condition['presence_penalty'],condition))
        chosen=min(candidates,key=lambda x:x[:3])[-1]
        save(root/'selection.json',dict(rule='Fewest exact loops, then most valid bash responses, then lowest presence penalty.',chosen=chosen))
        validation=[]
        with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
            futures=[pool.submit(one_request,cfg,root,endpoint,spec.model_key,c,chosen,'validation',cfg['validation_tokens'],cfg['seed']+1) for c in manifest['cases']]
            for f in as_completed(futures):validation.append(f.result());save(root/'validation-results.json',validation)
        save(root/'completed.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),screen=len(all_rows),validation=len(validation)))
    except BaseException as exc:
        save(root/'failure.json',dict(type=type(exc).__name__,message=str(exc)[:2000],at=datetime.now(timezone.utc).isoformat()))
        raise
    finally:
        if server:
            try:(root/'vllm-final.log').write_text(server.executor.tail_log(250),encoding='utf-8')
            except Exception:pass
            try:server.stop()
            except Exception:pass
        for pod_id in owned:
            runpod.teardown(pod_id)
        cost=(time.time()-start)*price/3600*1.05
        save(root/'cleanup.json',dict(owned_ids=owned,remaining_ids=[p['id'] for p in runpod.active_pods()],elapsed_seconds=time.time()-start,quoted_cost_upper_estimate=cost,cumulative_quoted_cost_upper_estimate=previous_cost+cost,balance=runpod.graphql('query { myself { currentSpendPerHr clientBalance } }')))


if __name__=='__main__':
    # Windows redirected stdout otherwise uses cp1252; boot progress contains Unicode.
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    sys.stderr.reconfigure(encoding='utf-8',errors='replace')
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','run'])
    args=parser.parse_args();globals()[args.phase]()
