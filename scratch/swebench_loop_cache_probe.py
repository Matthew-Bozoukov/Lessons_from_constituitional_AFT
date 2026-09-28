# ABOUTME: Test historical-prefix cache warming on the already-owned looping-probe GPU.
# ABOUTME: Replays saved prior-turn prefixes with one-token outputs, preserving raw evidence and never renting resources.
import json
from pathlib import Path
import time
import requests
from scratch.swebench_loop_probe import config, save, one_request, digest


def idle(endpoint):
    text=requests.get(endpoint.removesuffix('/v1')+'/metrics',timeout=20).text
    running=[float(x.rsplit(' ',1)[1]) for x in text.splitlines() if x.startswith('vllm:num_requests_running{') or x.startswith('vllm:num_requests_waiting{')]
    assert running and all(x==0 for x in running),'Never clear cache with requests still running'
    return text


def main():
    cfg,root=config(); endpoint=json.loads((root/'endpoint.json').read_text());url=endpoint['base_url'];model=endpoint['model']
    cases=json.loads((root/'manifest.json').read_text())['cases']
    assert (root/'driver-pause.json').exists(),'Outer driver must be paused to prevent teardown or concurrent work'
    for _ in range(240):
        try:idle(url);break
        except AssertionError:time.sleep(5)
    else:raise TimeoutError('Prior requests did not drain')
    selected=['django__django-13757','django__django-11964','scikit-learn__scikit-learn-10949']
    for case in cases:
        iid=case['task']
        if iid not in selected:continue
        idle(url)
        # The production server has no dev-only reset endpoint. Salt isolates a fresh
        # cache namespace without restarting, changing tokens, or affecting the main probe.
        salt='loop-probe-warm-history-v1-'+iid
        prefix=json.loads((root/'inputs'/f'{iid}.json').read_text())
        directory=root/'cache-warmup'/iid;directory.mkdir(parents=True,exist_ok=True)
        save(directory/'cache-isolation.json',{'cache_salt':salt,'mechanism':'vLLM request prefix-cache namespace'})
        indices=[i for i,m in enumerate(prefix['messages']) if m['role']=='assistant']
        for turn,index in enumerate(indices):
            body=dict(prefix);body.update(cfg['sampling']);body.update(cfg['conditions'][0]);body.pop('name')
            body.update(messages=prefix['messages'][:index],model=model,max_tokens=1,seed=42,stream=False,cache_salt=salt)
            save(directory/f'{turn:03d}-request.json',body)
            response=requests.post(url+'/chat/completions',json=body,timeout=(30,300))
            (directory/f'{turn:03d}-response.json').write_bytes(response.content);response.raise_for_status()
            print('warmed',iid,turn+1,'/',len(indices),response.json()['usage'],flush=True)
        idle(url)
        result=one_request(cfg,root,url,model,case,cfg['conditions'][0] | {'cache_salt':salt},'warm_history',cfg['screen_tokens'],cfg['seed'])
        (directory/'metrics-after.txt').write_text(idle(url),encoding='utf-8')
        print('warm result',json.dumps(result),flush=True)
    idle(url)
    save(root/'cache-probe-completed.json',{'cases':selected,'finished_epoch':time.time(),'outer_driver_still_paused':True})


if __name__=='__main__': main()
