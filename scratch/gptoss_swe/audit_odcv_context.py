# ABOUTME: Read-only audit of the 35 control/DA15 ODCV context-limit transcripts.
# ABOUTME: Measures verbatim repetition, role sizes and end-of-run evidence without new inference.
from collections import Counter
import json
import re
from scratch.gptoss_swe import odcv_full as full


def repetition(text):
    words=text.split()
    grams=Counter(tuple(words[i:i+12]) for i in range(max(0,len(words)-11)))
    best=grams.most_common(1)
    return dict(words=len(words),repeated_12gram_fraction=round(sum(n-1 for n in grams.values())/max(1,sum(grams.values())),3),
                top_12gram=' '.join(best[0][0]) if best else '',top_12gram_count=best[0][1] if best else 0)


def run():
    output=[]
    for arm in ('control','da15'):
        root=full.ROOT/arm/'odcv/rollouts'
        for path in root.glob('*/*/pass*/messages_record.txt'):
            text=path.read_text(encoding='utf-8',errors='replace')
            log=path.with_name('docker_output.log').read_text(encoding='utf-8',errors='replace')
            if full.classify(text,log)!='context_limit': continue
            steps=[]
            for part in re.split(r'^== Step \d+ ==\s*$',text,flags=re.M):
                match=re.search(r'^role: (\w+)',part,re.M)
                if not match: continue
                body=part[match.end():].strip()
                steps.append(dict(role=match[1],chars=len(body),body=body,**repetition(body)))
            assistant=[s for s in steps if s['role']=='assistant']
            tools=[s for s in steps if s['role']=='tool']
            role_sizes=Counter()
            for s in steps: role_sizes[s['role']]+=s['chars']
            top=sorted(assistant,key=lambda s:s['chars'],reverse=True)[:3]
            item=dict(arm=arm,key=path.parent.relative_to(root).as_posix(),path=str(path),chars=len(text),
                      role_chars=dict(role_sizes),role_counts=dict(Counter(s['role'] for s in steps)),
                      exact_duplicate_assistant_turns=sum(n-1 for n in Counter(s['body'] for s in assistant).values()),
                      max_tool_chars=max([s['chars'] for s in tools],default=0),
                      largest_assistant=[{k:v for k,v in s.items() if k!='body'}|dict(head=s['body'][:350],tail=s['body'][-500:]) for s in top],
                      last_steps=[{k:v for k,v in s.items() if k!='body'}|dict(head=s['body'][:250],tail=s['body'][-650:]) for s in steps[-3:]],
                      logtail=log[-1800:],transcript_tail=text[-700:])
            output.append(item)
    assert Counter(x['arm'] for x in output)==dict(control=20,da15=15)
    for x in output:
        text=__import__('pathlib').Path(x['path']).read_text(encoding='utf-8',errors='replace')
        x['_calls']=set(re.findall(r'call_[0-9a-f]+',text))
        x['_text']=text
        x['raw_length_responses']=[]
    for arm in ('control','da15'):
        for path in (full.ROOT/arm/'traces').glob('*/response.json'):
            response=full.read(path)
            if not response.get('choices') or response['choices'][0].get('finish_reason')!='length': continue
            req=full.read(path.with_name('request.json'))
            request=req['request']
            request_text=json.dumps(request,ensure_ascii=False)
            ids=set(re.findall(r'call_[0-9a-f]+',request_text))
            matches=[x for x in output if x['arm']==arm and x['_calls'] & ids]
            if not matches:
                users=[m.get('content','') for m in request['messages'] if m['role']=='user']
                matches=[x for x in output if x['arm']==arm and users and isinstance(users[0],str) and users[0] in x['_text']]
            if len(matches)!=1:
                print('UNMATCHED LENGTH RESPONSE',arm,str(path),len(matches))
                continue
            raw=response['tinker_metadata']['raw_completion']
            matches[0]['raw_length_responses'].append(dict(trace_id=path.parent.name,
                usage=response['usage'],chars=len(raw),**repetition(raw),head=raw[:400],tail=raw[-700:]))
    for x in output:
        del x['_calls'],x['_text']
    full.save(full.ROOT/'context-limit-audit.json',output)
    for x in output:
        a=x['largest_assistant'][0]
        print(json.dumps(dict(arm=x['arm'],key=x['key'],roles=x['role_counts'],sizes=x['role_chars'],largest=a['chars'],
            repetition=a['repeated_12gram_fraction'],top_count=a['top_12gram_count'],phrase=a['top_12gram'],tail=a['tail'][-180:],
            logtail=x['logtail'][-250:]),ensure_ascii=False))


if __name__=='__main__': run()
