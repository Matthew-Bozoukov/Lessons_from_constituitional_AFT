"""Read-only corpus audit; deterministic flags are candidates, not semantic verdicts.

Run with .venv/bin/python scratch/audit_nosynth_instructions.py.
Original rows are never modified. Indices in outputs are zero-based.
"""
import collections
import hashlib
import json
import re
from pathlib import Path

INPUT = Path('/Users/jamie/.cache/huggingface/hub/datasets--LASR-Callum--2026-09-05-nosynth-mix/snapshots/a517e99ba4cfd45189ef2fafc7653ad6162a1ebe/mixture.jsonl')
OUT = Path('output/nosynth_instruction_audit')
NUMBERS = dict(zip('one two three four five six seven eight nine ten'.split(), range(1, 11)))
N = r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten)'
def number(s):
    return int(s) if s.isdigit() else NUMBERS[s.lower()]

def words(s):
    return re.findall(r"\b\w+(?:['’\-]\w+)*\b", s)

def sentences(s):
    # Deliberately conservative candidate detector, not a linguistic tokenizer.
    s = re.sub(r'\b(?:Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr)\.', 'ABBR', s)
    s = re.sub(r'\b(?:[A-Za-z]\.){2,}', 'ABBR', s)
    return len(re.split(r'[.!?]+[\"”\')]*\s+(?=[A-Z“\"])', s.strip()))

def schema_errors(v, spec, path='arguments'):
    errors = []
    t = spec.get('type')
    if v is None and spec.get('nullable'):
        return []
    if isinstance(t,list):
        variants=[schema_errors(v,{**spec,'type':alt},path) for alt in t]
        return [] if any(not e for e in variants) else [f'{path}: does not match {t}']
    types = {'string': str, 'object': dict, 'array': list, 'boolean': bool,
             'integer': int, 'number': (int, float), 'null': type(None)}
    if t in types and (not isinstance(v, types[t]) or t in ('integer','number') and isinstance(v, bool)):
        return [f'{path}: expected {t}, got {type(v).__name__}']
    if 'enum' in spec and v not in spec['enum']:
        errors.append(f'{path}: not in enum')
    if isinstance(v, dict):
        errors += [f'{path}: missing {k}' for k in spec.get('required', []) if k not in v]
        for k, value in v.items():
            if k in spec.get('properties', {}):
                errors += schema_errors(value, spec['properties'][k], f'{path}.{k}')
            elif spec.get('additionalProperties') is False:
                errors.append(f'{path}: unexpected {k}')
    if isinstance(v, list) and isinstance(spec.get('items'), dict):
        for i, value in enumerate(v):
            errors += schema_errors(value, spec['items'], f'{path}[{i}]')
    return errors

def main():
    rows = [json.loads(s) for s in INPUT.open()]
    flags, checked = [], collections.defaultdict(set)
    pronouns = collections.Counter()
    def check(i, j, kind, condition, evidence):
        checked[kind].add(i)
        if condition:
            flags.append(dict(i=i, turn=j, source=rows[i]['source'], kind=kind, evidence=evidence))
    for i, row in enumerate(rows):
        source = row['source']
        system = '\n'.join(m['content'] for m in row['messages'] if m['role']=='system')
        for j, msg in enumerate(row['messages']):
            if msg['role'] != 'assistant':
                continue
            a = msg['content']
            p = system + '\n' + '\n'.join(m['content'] for m in row['messages'][:j] if m['role']=='user')
            check(i,j,'empty_answer',not a.strip(),'Empty assistant content')
            if source == 'smol_summarize':
                if 'without using second or third person pronouns' in system:
                    personal = re.findall(r'\b(?:you|your|yours|yourself|yourselves|he|him|his|himself|she|her|hers|herself|they|them|their|theirs|themselves)\b',a,re.I)
                    its = re.findall(r'\b(?:it|its|itself)\b',a,re.I)
                    pronouns['instructed'] += 1
                    pronouns['personal'] += bool(personal)
                    pronouns['it_only'] += bool(its) and not personal
                    check(i,j,'pronoun_ban',bool(personal or its),sorted(set(personal+its)))
                limit = 3 if 'up to three sentences' in system else 1
                check(i,j,'summary_sentences_candidate',sentences(a)>limit,dict(limit=limit,detected=sentences(a)))
                continue
            if source == 'apigen_function_calling':
                try:
                    tools = json.loads(re.search(r'<tools>(.*?)</tools>',system,re.S)[1])
                    normalized = [t.get('function',t) for t in tools]
                    specs = {t['name']:t['parameters'] for t in normalized}
                    m = re.fullmatch(r'\s*<tool_call>(.*?)</tool_call>\s*',a,re.S)
                    calls = json.loads(m[1]) if m else None
                    check(i,j,'tool_format',not isinstance(calls,list),'Expected tagged JSON list')
                    if isinstance(calls,list):
                        for call in calls:
                            name = call.get('name')
                            check(i,j,'tool_name',name not in specs,name)
                            if name in specs and specs[name].get('type')=='object':
                                errors = schema_errors(call.get('arguments'),specs[name])
                                check(i,j,'tool_schema',bool(errors),errors)
                except (ValueError,TypeError,KeyError,AttributeError) as e:
                    check(i,j,'tool_parse',True,str(e))
                continue
            # These lexical extractors target explicit wording. Review flags for
            # quotation/scope/contradictory instructions before calling them errors.
            clean = re.sub(r'\*\*','',p)
            for m in re.finditer(r'(exactly|at least|less than|at most|no more than) '+N+r' (words|bullet points|placeholders)',clean,re.I):
                op,n,unit=m.groups();n=number(n)
                context=clean[max(0,m.start()-140):m.end()+100].lower()
                if unit.lower()=='words':
                    if re.search(r'each|per (?:sentence|paragraph|line)|capital|contain.{0,20}letter|words.{0,20}(?:starting|ending)|(?:first|second|third) (?:paragraph|section)|one using|quote|after the summary',context):
                        continue
                    if op.lower()=='less than' and clean[max(0,m.start()-3):m.start()].lower()=='no ':
                        continue
                    counts=[len(a.split()),len(words(a))]
                elif unit.lower()=='bullet points':
                    if re.search(r'each|or numbered',context) or 'json' in clean.lower():
                        continue
                    counts=[len(re.findall(r'^\s*[-*+]\s+\S',a,re.M))]
                else:
                    if 'square brackets' not in clean and not re.search(r'such as \[',clean):
                        continue
                    if re.search(r'(?:first|second|third) paragraph',context):
                        continue
                    counts=[len(re.findall(r'\[[^\[\]\n]+\]',a))]
                bad={'exactly':lambda x:x!=n,'at least':lambda x:x<n,'less than':lambda x:x>=n,'at most':lambda x:x>n,'no more than':lambda x:x>n}[op.lower()]
                check(i,j,unit.lower().replace(' ','_'),all(bad(x) for x in counts),dict(instruction=m[0],counts=counts))
            if re.search(r'(?:response|answer) (?:should |must )?be (?:entirely )?in (?:all )?(?:lowercase|lower case)|no capital letters are allowed',clean,re.I):
                check(i,j,'lowercase',any(c.isupper() for c in a),'Uppercase letters: '+''.join(c for c in a if c.isupper())[:100])
            if 'wrapped in double angular brackets' in clean:
                check(i,j,'title',not re.search(r'<<[^<>]+>>',a),'Missing <<title>>')
            if re.search(r'(?:use|uses|using) no commas|(?:do not|don\x27t) use (?:any )?commas',clean,re.I):
                check(i,j,'no_commas',',' in a,dict(count=a.count(',')))
            if re.search(r'(?:use|uses|using) no periods',clean,re.I):
                check(i,j,'no_periods','.' in a,dict(count=a.count('.')))
            for m in re.finditer(r'word\s+["\[]([\w-]+)["\]]\s+(?:should (?:appear|be used) |(?:appears? )?)(at least|exactly|less than|at most) '+N+r' times',clean,re.I):
                word,op,n=m.groups();n=number(n);count=len(re.findall(r'\b'+re.escape(word)+r'\b',a,re.I))
                if word.lower()=='keyword' or re.search(r'each|every',clean[max(0,m.start()-100):m.end()+50],re.I):
                    continue
                bad={'exactly':count!=n,'at least':count<n,'less than':count>=n,'at most':count>n}[op.lower()]
                check(i,j,'word_frequency',bad,dict(instruction=m[0],count=count))
            for m in re.finditer(r'(?:avoid using|do not (?:use|mention)|don\x27t (?:use|mention)) the word ["\[]([\w-]+)\.?["\]]',clean,re.I):
                word=m[1];count=len(re.findall(r'\b'+re.escape(word)+r'\b',a,re.I))
                check(i,j,'forbidden_word',count>0,dict(instruction=m[0],count=count))
            if re.search(r'(?:entire response|response must|response should).{0,25}(?:JSON format|valid JSON)',clean,re.I):
                content=re.sub(r'^```(?:json)?\s*|\s*```$','',a.strip())
                try:
                    json.loads(content);bad=False
                except ValueError:
                    bad=True
                check(i,j,'json',bad,'Not JSON even allowing enclosing code fence')
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'flags.jsonl').open('w') as f:
        for flag in flags:
            f.write(json.dumps(flag,ensure_ascii=False)+'\n')
    report = dict(input=str(INPUT),sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),rows=len(rows),
                  sources=dict(collections.Counter(r['source'] for r in rows)),pronouns=dict(pronouns),
                  checks={k:dict(checked_rows=len(v),flagged_rows=len({f['i'] for f in flags if f['kind']==k})) for k,v in checked.items()},
                  flagged_rows_by_source={s:len({f['i'] for f in flags if f['source']==s}) for s in sorted({r['source'] for r in rows})})
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
