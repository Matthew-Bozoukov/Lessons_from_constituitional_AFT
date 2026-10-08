# ABOUTME: Offline census of saved stopped-campaign trajectories, format errors and recovery.
# ABOUTME: Reads the verified archive without changing or executing any model output.
import ast
import collections
import hashlib
import json
import pathlib
import re
import tarfile
import gzip

ROOT = pathlib.Path('output/gptoss_swe_shutdown')
OUT = pathlib.Path('output/gptoss_swe_analysis')
OUT.mkdir(exist_ok=True)
archive = ROOT / 'gptoss-three-20261008-stopped.tar.gz'
best, diagnostics, states = {}, {}, {}
with tarfile.open(archive, 'r|gz') as tf:
    for member in tf:
        if not member.isfile():
            continue
        parts = member.name.split('/')
        if len(parts) < 4 or parts[1] not in ('base-parallel', 'control-parallel', 'da15-parallel'):
            continue
        arm = parts[1].removesuffix('-parallel')
        if member.name.endswith('/metadata/state.json'):
            states[arm] = json.load(tf.extractfile(member))
        if '/rollouts/' not in member.name:
            continue
        j = parts.index('rollouts')
        if len(parts) <= j+3:
            continue
        key = (arm, parts[j+1], parts[j+2])
        if member.name.endswith('.traj.json'):
            data = json.load(tf.extractfile(member))
            if key not in best or len(data.get('messages', [])) > len(best[key].get('messages', [])):
                best[key] = data
        elif parts[-1] in ('diagnostics.json', 'forced-submission.json', 'done.json'):
            diagnostics.setdefault(key, {})[parts[-1]] = json.load(tf.extractfile(member))
print('loaded', len(best), 'attempts', flush=True)

def unpack_error(s):
    try:
        call = ast.parse(s, mode='eval').body
        return {x.arg: ast.literal_eval(x.value) for x in call.keywords}
    except Exception:
        return {'raw_text': '', 'error': s}

def bracket_check(raw):
    if '<|message|>' not in raw:
        return None
    arg = raw.rsplit('<|message|>', 1)[1]
    arg = re.split(r'<\|(?:call|return|end|start)\|>', arg)[0].strip()
    try:
        json.loads(arg)
        return None
    except json.JSONDecodeError as e:
        if e.pos < len(arg) and arg[e.pos] == ']':
            try:
                fixed = json.loads(arg[:e.pos]+arg[e.pos+1:])
                if isinstance(fixed, dict):
                    return {'literal_ending':bool(re.fullmatch(r'\s*}?\s*',arg[e.pos+1:])),
                            'argument_tail':arg[-220:], 'diagnostic_command':fixed.get('command')}
            except json.JSONDecodeError:
                pass
    return None

rows, tasks, long_outputs, examples = [], [], [], []
for key, data in sorted(best.items()):
    arm, iid, attempt = key
    ms = data.get('messages', [])
    indices = [i for i,m in enumerate(ms) if m.get('role') == 'assistant' and m.get('extra',{}).get('response',{}).get('tinker_metadata')]
    taskrows = []
    for turn,i in enumerate(indices):
        m=ms[i];resp=m['extra']['response'];meta=resp['tinker_metadata'];choice=resp['choices'][0]
        end=indices[turn+1] if turn+1<len(indices) else len(ms)
        feedback=ms[i+1:end]
        errors=[x for x in feedback if x.get('extra',{}).get('interrupt_type')=='FormatError' or 'Tool call error:' in (x.get('content') or '')]
        raw=meta.get('raw_completion','');parse=[unpack_error(x) for x in meta.get('unparsed_tool_calls',[])]
        brackets=[b for x in parse if (b:=bracket_check(x.get('raw_text','')))]
        categories=[]
        for x in parse:
            error=x.get('error','')
            if 'Invalid JSON' in error:cat='invalid_json'
            elif 'unterminated tool block' in error:cat='unparseable_tool_header_or_body'
            else:cat='other_parser_error'
            categories.append(cat)
        calls=m.get('tool_calls') or []
        row={'arm':arm,'iid':iid,'attempt':attempt,'turn':turn+1,'id':resp.get('id'),'finish':choice.get('finish_reason'),
             'tokens':resp.get('usage',{}),'sampling':meta.get('sampling'), 'errors':len(errors),'error_text':[x.get('content') for x in errors],
             'parse_categories':categories,'parse_errors':[x.get('error') for x in parse],
             'brackets':brackets,'tool_calls':len(calls),'tool_names':[c.get('function',{}).get('name') for c in calls],
             'tool_feedback':sum(x.get('role')=='tool' for x in feedback),'raw_chars':len(raw),
             'raw_tail':raw[-700:],'raw_head':raw[:450], 'raw_sha256':hashlib.sha256(raw.encode()).hexdigest(),
             'raw_has_call':raw.endswith('<|call|>'),'raw_has_return':raw.endswith('<|return|>'),
             'content_contains_harmony':any(t in (m.get('content') or '') for t in ('<|channel|>','<|start|>','<|call|>')),
             'exact_raw_fallback_duplicate':bool(calls and (m.get('content') or '')==re.sub(r'<\|(?:call|return)\|>$','',raw)),
             'cached_prompt_tokens':meta.get('prompt_cache_hit_tokens',0),
             'reasoning_contains_malformed_header':'<commentary to=functions.' in (m.get('reasoning_content') or ''),
             'response_metadata_has_reasoning':bool(m.get('reasoning_content'))}
        row['accepted_tool']=bool(calls and row['tool_feedback'] and not errors)
        row['accepted_commands']=[]
        if row['accepted_tool']:
            for call in calls:
                try:
                    args=json.loads(call.get('function',{}).get('arguments','{}'))
                    if isinstance(args,dict) and isinstance(args.get('command'),str):
                        row['accepted_commands'].append(args['command'])
                except (TypeError,json.JSONDecodeError):
                    pass
        row['multiple_calls']=len(calls)>1
        row['reasoning_chars']=len(m.get('reasoning_content') or '')
        if choice.get('finish_reason')=='length':
            long_outputs.append({'arm':arm,'iid':iid,'turn':turn+1,'tokens':resp.get('usage'), 'raw':raw})
        if row['content_contains_harmony'] and calls and not any(x['arm']==arm for x in examples):
            examples.append({'arm':arm,'iid':iid,'messages':ms[:i+1], 'response':resp})
        if errors and not categories:
            row['error_kind']='no_structured_tool_call' if not calls else 'other_executor_format_error'
        else:row['error_kind']='parser_rejection' if errors else None
        taskrows.append(row)
    episodes=[];idx=0
    while idx<len(taskrows):
        if not taskrows[idx]['errors']:idx+=1;continue
        start=idx
        while idx<len(taskrows) and taskrows[idx]['errors']:idx+=1
        episodes.append({'start_turn':start+1,'length':idx-start,'recovered_next':idx<len(taskrows) and taskrows[idx]['accepted_tool'],
                         'later_tool':any(x['accepted_tool'] for x in taskrows[idx:]),'censored':idx==len(taskrows)})
    for i,row in enumerate(taskrows):
        if row['errors']:
            row['next_turn']='unobserved' if i+1==len(taskrows) else ('accepted_tool' if taskrows[i+1]['accepted_tool'] else 'format_error' if taskrows[i+1]['errors'] else 'other_no_tool')
            row['later_tool']=any(x['accepted_tool'] for x in taskrows[i+1:])
            row['next_exact_bracket_command']=i+1<len(taskrows) and any(
                isinstance(b.get('diagnostic_command'),str) and b['diagnostic_command'] in taskrows[i+1]['accepted_commands']
                for b in row['brackets'])
    state=states.get(arm,{}).get('tasks',{}).get(iid,{})
    tasks.append({'arm':arm,'iid':iid,'attempt':attempt,'state':state.get('status'),'exit':(state.get('attempts') or [{}])[-1].get('exit_status'),
                  'turns':len(taskrows),'error_turns':sum(bool(x['errors']) for x in taskrows),'bracket_calls':sum(len(x['brackets']) for x in taskrows),
                  'accepted_turns':sum(x['accepted_tool'] for x in taskrows),'episodes':episodes,'diagnostics':diagnostics.get(key,{}),
                  'traj_info':data.get('info',{})})
    rows.extend(taskrows)

summary={}
for arm in ('base','control','da15'):
    rr=[x for x in rows if x['arm']==arm];tt=[x for x in tasks if x['arm']==arm]
    er=[x for x in rr if x['errors']];br=[x for x in rr if x['brackets']]
    eps=[e for t in tt for e in t['episodes']]
    summary[arm]={'attempts':len(tt),'responses':len(rr),'error_responses':len(er),'error_tasks':sum(t['error_turns']>0 for t in tt),
                  'bracket_calls':sum(len(x['brackets']) for x in br),'bracket_responses':len(br),'bracket_tasks':len({x['iid'] for x in br}),
                  'literal_ending_calls':sum(b['literal_ending'] for x in br for b in x['brackets']),
                  'next_after_error':dict(collections.Counter(x['next_turn'] for x in er)),
                  'next_after_bracket':dict(collections.Counter(x.get('next_turn','no_format_feedback') for x in br)),
                  'next_exact_bracket_command':sum(x.get('next_exact_bracket_command',False) for x in br),
                  'error_kinds':dict(collections.Counter(x['error_kind'] for x in er)),
                  'parser_errors':dict(collections.Counter(c for x in rr for c in x['parse_categories'])),
                  'finish':dict(collections.Counter(x['finish'] for x in rr)),
                  'sampling':dict(collections.Counter(json.dumps({k:x['sampling'].get(k) for k in ('temperature','top_p','top_k','stop')},sort_keys=True) for x in rr)),
                  'episodes':len(eps),'episodes_recovered_next':sum(e['recovered_next'] for e in eps),'episodes_later_tool':sum(e['later_tool'] for e in eps),
                  'longest_error_streak':max([e['length'] for e in eps],default=0),
                  'raw_harmony_in_content':sum(x['content_contains_harmony'] for x in rr),
                  'exact_raw_fallback_duplicates':sum(x['exact_raw_fallback_duplicate'] for x in rr),
                  'malformed_header_in_reasoning':sum(x['reasoning_contains_malformed_header'] for x in rr),
                  'total_completion_tokens':sum(x['tokens'].get('completion_tokens',0) for x in rr),
                  'error_completion_tokens':sum(x['tokens'].get('completion_tokens',0) for x in er)}
(OUT/'calls.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
(OUT/'tasks.json').write_text(json.dumps(tasks,ensure_ascii=False),encoding='utf-8')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
with gzip.open(OUT/'length_outputs.json.gz','wt',encoding='utf-8') as f:json.dump(long_outputs,f,ensure_ascii=False)
with gzip.open(OUT/'renderer_examples.json.gz','wt',encoding='utf-8') as f:json.dump(examples,f,ensure_ascii=False)
print(json.dumps(summary,indent=2))
