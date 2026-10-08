# ABOUTME: Downloads pinned public ODCV transcripts and measures consecutive malformed-JSON turns.
# ABOUTME: Read-only artifact analysis; never samples models or executes recorded commands.
import ast
import collections
import concurrent.futures
import hashlib
import json
import pathlib
import re
import urllib.request

REPO='dougalldeepmind/2026-10-06-odcv-gptoss120b-0-plain'
REV='8e80b855661ea1cb114ad639021e4099ac660c47'
ROOT=pathlib.Path('output/gptoss_swe_analysis/teammate_odcv')
info=json.load(urllib.request.urlopen(f'https://huggingface.co/api/datasets/{REPO}/revision/{REV}'))
files=[r['rfilename'] for r in info['siblings'] if r['rfilename'].startswith('rollouts/') and r['rfilename'].endswith(('messages_record.txt','cell_meta.json'))]
def fetch(name):
    path=ROOT/name
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(urllib.request.urlopen(f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{name}',timeout=60).read())
    return name,hashlib.sha256(path.read_bytes()).hexdigest()
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
    hashes=dict(pool.map(fetch,files))
(ROOT/'transcript_download_receipt.json').write_text(json.dumps({'repo':REPO,'revision':REV,'sha256':hashes},indent=2),encoding='utf-8')
results=[]
for name in files:
    if not name.endswith('messages_record.txt'):continue
    text=(ROOT/name).read_text(encoding='utf-8')
    turns=[]
    for block in re.split(r'^== Step \d+ ==\s*$',text,flags=re.M):
        if not re.search(r'^role: assistant\s*$',block,flags=re.M):continue
        match=re.search(r'^call: (.*)$',block,flags=re.M)
        calls=ast.literal_eval(match.group(1)) if match else []
        bad=[];names=[];valid=[]
        for call in calls:
            fn=call['function'];names.append(fn['name']);args=fn['arguments']
            try:
                obj=json.loads(args) if isinstance(args,str) else args
                if not isinstance(obj,dict):bad.append(args)
                else:valid.append(fn['name'])
            except json.JSONDecodeError:bad.append(args)
        turns.append({'bad_json':bool(bad),'bad_calls':len(bad),'names':names,'valid_names':valid})
    streaks=[];i=0
    while i<len(turns):
        if not turns[i]['bad_json']:i+=1;continue
        start=i
        while i<len(turns) and turns[i]['bad_json']:i+=1
        streaks.append({'start':start+1,'length':i-start,'next_valid_tool':i<len(turns) and bool(turns[i]['valid_names'])})
    meta=json.loads((ROOT/name.replace('messages_record.txt','cell_meta.json')).read_text())
    results.append({'path':name,'turns':len(turns),'bad_turns':sum(t['bad_json'] for t in turns),'bad_calls':sum(t['bad_calls'] for t in turns),
                    'streaks':streaks,'max_streak':max([s['length'] for s in streaks],default=0),
                    'token_limit_hit':meta.get('token_limit_hit'),'meta':meta,
                    'last_names':turns[-1]['names'] if turns else [],'last_bad_json':turns[-1]['bad_json'] if turns else False,
                    'tail':text[-1500:]})
summary={'rollouts':len(results),'json_affected_rollouts':sum(r['bad_turns']>0 for r in results),'bad_turns':sum(r['bad_turns'] for r in results),
         'bad_calls':sum(r['bad_calls'] for r in results),'streak_histogram':dict(collections.Counter(s['length'] for r in results for s in r['streaks'])),
         'token_limit_rollouts':sum(bool(r['token_limit_hit']) for r in results),
         'json_and_token_limit':sum(bool(r['token_limit_hit']) and r['bad_turns']>0 for r in results),
         'rollouts_50plus_assistant_turns':sum(r['turns']>=50 for r in results),
         'worst':sorted([{k:v for k,v in r.items() if k not in ('tail','meta')} for r in results],key=lambda r:r['max_streak'],reverse=True)[:12]}
(ROOT/'streak_audit.json').write_text(json.dumps({'summary':summary,'rollouts':results},indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
