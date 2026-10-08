# ABOUTME: Summarize saved base-only smoke evidence without generating or judging new outcomes.
# ABOUTME: Distinguishes tool completion, protocol failures, numerical limits and official SWE grades.
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT=Path('output/gptoss_shared_smoke')

def key(messages):
    return hashlib.sha256(json.dumps([(m['role'],m.get('content')) for m in messages[:2]],sort_keys=True).encode()).hexdigest()

def main():
    result={}
    for kind in ('odcv','swe'):
        groups={}
        for path in sorted((ROOT/(kind+'-traces')).glob('*/request.json'),key=lambda p:p.stat().st_mtime):
            req=json.loads(path.read_text())
            response=path.with_name('response.json')
            group=groups.setdefault(key(req['request']['messages']),dict(requests=0,responses=0,boundaries=Counter(),
                invalid_json=0,invalid_json_next_turn_corrected=0,validation_feedback_ids=set(),sampling=[],pending_invalid=False,
                invalid_streak=0,max_consecutive_invalid_json=0,stray_closing_bracket=0,
                no_tool_responses=0,no_tool_streak=0,max_consecutive_no_tool=0,visible_text_counts=Counter()))
            if kind=='swe' and 'instance_id' not in group:
                text=str(req['request']['messages'][1].get('content'))
                rows=json.loads((ROOT/'cache/metadata/swebench_lite_test.json').read_text())
                ids=[row['instance_id'] for row in rows if row['problem_statement'].strip() in text]
                if len(ids)==1: group['instance_id']=ids[0]
            group['requests']+=1
            params=req['sampling']
            if params not in group['sampling']: group['sampling'].append(params)
            for message in req['request']['messages']:
                if message['role']=='tool' and ('tool_validation_error' in str(message.get('content')) or 'Tool validation error' in str(message.get('content'))):
                    group['validation_feedback_ids'].add(message.get('tool_call_id'))
            if not response.exists(): continue
            res=json.loads(response.read_text())
            group['responses']+=1
            group['boundaries'][res['tinker_metadata']['boundary']]+=1
            invalid=False
            calls=res['choices'][0]['message'].get('tool_calls',[])
            group['no_tool_responses']+=int(not calls)
            group['no_tool_streak']=0 if calls else group['no_tool_streak']+1
            group['max_consecutive_no_tool']=max(group['max_consecutive_no_tool'],group['no_tool_streak'])
            if not calls:
                visible=res['choices'][0]['message'].get('content','') or ''
                group['visible_text_counts'][visible]+=1
            for call in calls:
                try: json.loads(call['function']['arguments'])
                except ValueError:
                    invalid=True
                    group['invalid_json']+=1
                    if re.search(r'"\s*\]\s*\}\s*$',call['function']['arguments']): group['stray_closing_bracket']+=1
            if group['pending_invalid'] and calls and not invalid:
                group['invalid_json_next_turn_corrected']+=1
            group['pending_invalid']=invalid
            group['invalid_streak']=group['invalid_streak']+1 if invalid else 0
            group['max_consecutive_invalid_json']=max(group['max_consecutive_invalid_json'],group['invalid_streak'])
        for group in groups.values():
            group['validation_feedback_count']=len(group.pop('validation_feedback_ids'))
            group.pop('pending_invalid')
            group.pop('invalid_streak')
            group.pop('no_tool_streak')
            group['most_repeated_no_tool_text']=group.pop('visible_text_counts').most_common(5)
        if kind=='odcv':
            sentence=' Use the bash tool to complete the task and use the task_complete tool to end the task.'
            for prompt in (ROOT/'odcv').glob('**/workspaces/*/*/orchestrator_api/prompt.json'):
                data=json.loads(prompt.read_text())
                identity=key([dict(role='system',content=data['system_prompt']+sentence),dict(role='user',content=data['user_prompt'])])
                if identity in groups:
                    groups[identity]['scenario']=prompt.parent.parent.name
                    groups[identity]['variant']=prompt.parent.parent.parent.name
            # Completed ODCV workspaces are removed by the ordinary driver.
            # This smoke is strictly serial and recorded the selected order.
            from omegaconf import OmegaConf
            cfg=OmegaConf.load(ROOT/'odcv.yaml')
            assert cfg.concurrency==1
            order=[(v,s) for v in ('mandated','incentivized') for s in cfg.include_scenarios[v]]
            for group,(variant,scenario) in zip(groups.values(),order):
                assert group.get('scenario',scenario)==scenario
                assert group.get('variant',variant)==variant
                group.update(variant=variant,scenario=scenario)
        result[kind]=dict(conversations=groups)
        ledger=ROOT/(kind+'-budget.json')
        if ledger.exists():
            data=json.loads(ledger.read_text())
            result[kind]['budget']=dict(ceiling=data['ceiling_usd'],accounted=sum(x['upper_usd'] for x in data['requests'].values()),
                                      request_states=dict(Counter(x['state'] for x in data['requests'].values())),invoice=False)
    odcv=[]
    for file in sorted((ROOT/'odcv').glob('**/experiments/*/messages_record.txt')):
        text=file.read_text(encoding='utf-8')
        log=file.with_name('docker_output.log')
        logtext=log.read_text(encoding='utf-8',errors='replace') if log.exists() else ''
        completed='"action": "task_complete"' in text
        odcv.append(dict(scenario=file.parent.name,variant=file.parent.parent.parent.name.rsplit('-',1)[1],
            tool_completed=completed,step_limit='Reached cycle limit' in logtext,
            response_limit='finish_reason' in text and 'length' in text,
            budget_notes=[s for s in text.splitlines() if 'token_limit' in s or 'budget exceeded' in s.lower()],
            cycles=max([int(n) for n in re.findall(r'CYCLE (\d+) /',logtext)] or [0])))
    result['odcv']['cells']=odcv
    result['odcv']['initial_version_cells']=['mandated/Academic-Research-Integrity-Finding','mandated/Ai-Education-Assistant','mandated/Ai-Healthcare-Administrator']
    state=ROOT/'swe/metadata/state.json'
    if state.exists():
        data=json.loads(state.read_text())
        result['swe']['task_statuses']=dict(Counter(v['status'] for v in data['tasks'].values()))
        result['swe']['halt']=data.get('halt')
        result['swe']['ended_statuses']=dict(Counter(t['attempts'][-1].get('exit_status','unknown') for t in data['tasks'].values() if t['status']=='valid'))
        outcomes={}
        for iid,task in data['tasks'].items():
            if task['status']!='valid': continue
            attempt=task['attempts'][-1]
            out=ROOT/'swe/rollouts'/iid/attempt['id']
            traj=out/iid/(iid+'.traj.json')
            trajectory=json.loads(traj.read_text()) if traj.exists() else {}
            limits=[m.get('extra',{}).get('limit_reason') for m in trajectory.get('messages',[]) if m.get('extra',{}).get('limit_reason')]
            outcomes[iid]=dict(exit_status=attempt.get('exit_status'),limit_reasons=sorted(set(limits)))
        result['swe']['outcomes']=outcomes
    (ROOT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(odcv_completed=sum(x['tool_completed'] for x in odcv),odcv_recorded=len(odcv),
        budgets={k:v.get('budget') for k,v in result.items()},swe=result['swe'].get('task_statuses')),indent=2))

if __name__=='__main__': main()
