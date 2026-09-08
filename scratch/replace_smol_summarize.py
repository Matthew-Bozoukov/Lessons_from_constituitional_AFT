# ABOUTME: Prepare and directly review unchanged smol-summarize replacements.
# ABOUTME: Validate exact upstream provenance and preserve all other mixture rows.
"""Prepare direct-Codex review candidates and build a non-destructive replacement.

No model clients. Review decisions must be written by the reviewing assistant.
"""
import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path('output/smol_summarize_replacement')
BASE = Path('output/reasoning_backfill/enrich_20260908_114055/mixture.jsonl')
RAW = ROOT/'upstream/data/smol-summarize/train-00000-of-00001.parquet'
DECISIONS = Path('scratch/smol_summarize_review_decisions.json')
QUOTAS = {'three':644, 'one':340}
BAN = re.compile(r'\b(?:you|your|yours|yourself|yourselves|he|him|his|himself|she|her|hers|herself|they|them|their|theirs|themselves|it|its|itself)\b',re.I)
DIVERSITY_REVIEWS = Path('scratch/smol_summarize_diversity_reviews.json')
EMAIL = re.compile(r'(?im)^(?:hi|hey|dear|hello)\b|^[A-Z][\w .\x27-]{0,45}[,!]\s*\n|\n\s*(?:best(?: regards| wishes)?|kind regards|warm(?:est)?(?: regards|ly)?|cheers|sincerely|all (?:the )?best|take care|talk (?:to you )?soon)[,!]?\s*\n|\n[A-Z][a-z]+(?: [A-Z][a-z]+){0,2}\s*$')

def sentence_count(answer):
    text=re.sub(r'\b(?:Dr|Mr|Mrs|Ms|Prof|U\.S|U\.K|St|Maj|Gen|Lt|Col|Sen|Rep|Dec|Nov|Oct|Sept|Aug|Jul|Jun|Apr|Mar|Feb|Jan)\.(?=\s+\S)','ABBR',answer)
    return len(re.findall(r'[.!?]+[\"”\x27’]?(?:\s|$)',text))

def broad_sentence_count(answer):
    """Discovery only: over-admit acronym/abbreviation boundaries for direct review."""
    text=re.sub(r'\b(?:[A-Za-z]\.){2,}','ACRONYM',answer)
    text=re.sub(r'\b(?:Dr|Mr|Mrs|Ms|Prof|St|Mt|Jr|Sr|Gen|Lt|Col|Sen|Rep|Rev|Capt|Sgt|Maj|Gov|Hon|No|vs|etc|Dec|Nov|Oct|Sept|Sep|Aug|Jul|Jun|Apr|Mar|Feb|Jan)\.','ABBR',text)
    return len(re.findall(r'[.!?]+[\"”\x27’]?(?:\s|$)',text))

def inspect_openorca(path):
    """Discover constrained upstream summaries; this does not judge or select rows."""
    import pyarrow.parquet as pq
    import subprocess
    systems=collections.Counter();patterns=collections.Counter();candidates=[]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
        for row in batch.to_pylist():
            question=row['question']
            if not re.search(r'summar(?:ize|ise|y).*?(?:one|two|three|[123]) sentence',question,re.I|re.S):continue
            systems[row['system_prompt']]+=1
            match=re.search(r'.{0,30}summar(?:ize|ise|y).{0,150}',question,re.I)
            patterns[match.group(0)[:160]]+=1
            candidates.append(row)
    target=path.parent
    (target/'candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in candidates))
    meta={'dataset':'Open-Orca/OpenOrca','revision':'e9c87b4abb2609913751f9b26553fdb9c061796c',
          'file':'1M-GPT4-Augmented.parquet','file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'config':{'target_news':492,'target_emails':492,'rewrite_allowed':False},
          'candidate_count':len(candidates),'accepted_count':0,'training_mixture_changed':False,
          'system_prompts':dict(systems),'status':'Discovery only; no acceptance inferred from screening.'}
    (target/'run_meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({'candidate_count':len(candidates),'systems':dict(systems),'example_patterns':patterns.most_common(15)},indent=2))

def report_news_sources(out_root):
    """Persist directly reviewed P3 examples, with no changes to the mixture."""
    import pyarrow.parquet as pq
    import subprocess
    reviews=json.loads(Path('scratch/news_summarize_source_reviews.json').read_text())
    accepted=[];sources=[]
    for name,review in [('p3-xsum-train-0.parquet',reviews),('p3-cnn-train-0.parquet',reviews['cnn_review'])]:
        path=out_root/name
        rows=pq.read_table(path,columns=['inputs_pretokenized','targets_pretokenized']).to_pylist()
        source={'dataset':reviews['dataset'],'revision':reviews['revision'],'subset':review['subset'],
                'file':review['file'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'downloaded_rows':len(rows),'direct_reviews':len(review['reviews'])}
        sources.append(source)
        for decision in review['reviews']:
            if decision['verdict']!='accept':continue
            row=rows[decision['index']]
            messages=[{'role':'user','content':row['inputs_pretokenized']},
                      {'role':'assistant','content':row['targets_pretokenized']}]
            accepted.append({'messages':messages,'source':source,'upstream_index':decision['index'],
                             'review':decision,'messages_sha256':signature(messages)})
    report={'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'config':{'news_target':492,'email_target':492,'rewrite_allowed':False},'sources':sources,
            'directly_accepted_p3_news':len(accepted),'target_achieved':False,'mixture_changed':False,
            'scope':'Source eligibility investigation, not a completed replacement. Exact released input/target strings; no invented system prompts.',
            'argilla':{'dataset':'argilla/ifeval-like-data','revision':'56505d1771e36fa5aa07aa9a1b39008fe60a8487',
                       'subset':'filtered','summarisation_mention_rows':49,'inspected_examples':10,
                       'finding':'Inspected examples mostly lack a supplied news article; not a suitable quota source.'}}
    (out_root/'p3_accepted_examples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted))
    (out_root/'p3_source_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

def balance_news_audit():
    """Record the broader discovery pass without changing any training rows."""
    import pyarrow.parquet as pq
    import subprocess
    from src.naming import artifact_name
    raw=pq.read_table(RAW).to_pylist()
    prior=json.loads(DIVERSITY_REVIEWS.read_text())
    new_reviews=[
        {'upstream_index':499,'verdict':'skip','reason':'Student asks further civics questions; correspondence, not news.'},
        {'upstream_index':609,'verdict':'reject','reason':'Navigation stub; answer invents actual news coverage of Iraq and Hawaii.'},
        {'upstream_index':2682,'verdict':'reject','reason':'Manager confirmed interest, not the signing as the answer attributes to the manager. No. 39 explains the prior sentence-screen false rejection, but attribution remains an issue.'},
        {'upstream_index':27544,'verdict':'reject','reason':'Invents a convention 91 years ago by confusing the time since a hurricane with convention history; long bullet list beyond the requested concise prose.'},
        {'upstream_index':64925,'verdict':'reject','reason':'Switches into Chinese for multiple sentences and ends in a truncated meta-note.'},
        {'upstream_index':74822,'verdict':'reject','reason':'Four real sentences; D.C. at sentence end fools the broad screen. Navigation stub does not supply actual news segments.'},
        {'upstream_index':74907,'verdict':'reject','reason':'Long structured reproduction of fixture information, truncated midway through West Ham team news; not a concise up-to-three-sentence summary.'},
    ]
    prior_ids={r['upstream_index'] for r in prior['reviews']}
    candidates=[];edge_cases=[]
    for i,row in enumerate(raw):
        msgs=row['messages'];answer=msgs[-1]['content'];prompt=msgs[1]['content'];k=kind(msgs)
        if k=='three' and any(hit!='IT' for hit in BAN.findall(answer)):continue
        if EMAIL.search(prompt):
            if '\n' not in prompt or re.match(r'(?i)^(?:By\s*\.|\(?CNN\)?|PUBLISHED:|UPDATED:)',prompt.strip()) or not re.search(r'\b(?:I|we|We|my|My|our|Our|you|You|your|Your)\b',prompt):edge_cases.append(i)
            continue
        if broad_sentence_count(answer)<=(3 if k=='three' else 1):candidates.append(i)
    assert set(candidates)==prior_ids|{r['upstream_index'] for r in new_reviews}
    assert edge_cases==[7313,42540,54016]
    report={'target':{'total':984,'news':492,'email':492},'achieved':False,'mixture_changed':False,
            'upstream_rows':len(raw),'upstream_revision':prior['upstream_revision'],
            'upstream_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),
            'candidate_indices':candidates,'candidate_count':len(candidates),
            'previously_reviewed':len(prior_ids),'new_direct_reviews':new_reviews,
            'email_classifier_edge_cases':{'indices':edge_cases,'direct_review':'German, Turkish and French correspondence, not news.'},
            'previously_accepted_news':4,'newly_accepted_news':0,
            'discovery_changes':['Expanded abbreviation handling','No original-input exclusion','No word-count cap','No sentence minimum or complete-ending requirement','IT acronym not automatically treated as a pronoun'],
            'limitation':'Candidate discovery still uses heuristics; this is not a proof of a universal four-row ceiling. No evidence supports a 492-row clean news selection under unchanged instructions.'}
    target=Path('output')/artifact_name('smol-summarize-news-balance-audit')
    target.mkdir(parents=True,exist_ok=False)
    (target/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (target/'prior_reviews.json').write_bytes(DIVERSITY_REVIEWS.read_bytes())
    (target/'candidates.jsonl').write_text(''.join(json.dumps({'upstream_index':i,'messages':raw[i]['messages']},ensure_ascii=False)+'\n' for i in candidates))
    (target/'run_meta.json').write_text(json.dumps({'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'command':'uv run --no-sync python -m scratch.replace_smol_summarize balance-news-audit','config':report['target'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print(json.dumps({'output':str(target),'candidate_count':len(candidates),'newly_accepted_news':0,'target_achieved':False},indent=2))

def diversity_pool():
    """Discovery only: passing a heuristic never constitutes judge acceptance."""
    import pyarrow.parquet as pq
    original={m['content'].strip() for r in read(BASE) for m in r['messages'] if m['role']=='user'}
    candidates=[]
    for i,r in enumerate(pq.read_table(RAW).to_pylist()):
        msgs=[{'role':m['role'],'content':m['content']} for m in r['messages']]
        prompt=msgs[1]['content'];answer=msgs[-1]['content'];k=kind(msgs)
        if EMAIL.search(prompt) or prompt.strip() in original:continue
        if k=='three' and BAN.search(answer):continue
        if not 1<=sentence_count(answer)<=(3 if k=='three' else 1):continue
        if k=='one' and len(answer.split())>30:continue
        if not re.search(r'[.!?][\"”\x27’]?$',answer.strip()):continue
        candidates.append({'upstream_index':i,'kind':k,'sha256':signature(msgs),'messages':msgs})
    return candidates

def diversity_review(count,offset):
    pool=diversity_pool()
    print('HEURISTIC CANDIDATES',len(pool),'OFFSET',offset)
    for r in pool[offset:offset+count]:
        print('\nINDEX',r['upstream_index'],'KIND',r['kind'],'INPUT',r['messages'][1]['content'],'\nANSWER',r['messages'][-1]['content'])
    print('IDS',[r['upstream_index'] for r in pool[offset:offset+count]])

def random_audit(seed, count, offset, batch_size):
    """Unfiltered uniform sample, fixed before reading any sampled answers."""
    import pyarrow.parquet as pq
    import subprocess
    from src.naming import artifact_name
    raw=pq.read_table(RAW).to_pylist()
    indices=random.Random(seed).sample(range(len(raw)),count)
    target=Path('output')/artifact_name('smol-summarize-random-audit')
    sample=[{'sample_number':n+1,'upstream_index':i,'messages':raw[i]['messages']} for n,i in enumerate(indices)]
    serialized=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sample)
    target.mkdir(parents=True,exist_ok=True)
    if (target/'sample.jsonl').exists():
        assert (target/'sample.jsonl').read_text()==serialized, 'Do not replace the audit sample'
    else:
        (target/'sample.jsonl').write_text(serialized)
        meta={'sampling':'uniform without replacement; no content filters or exclusions',
              'seed':seed,'count':count,'population':len(raw),'upstream_revision':'5feaf2fd3ffca7c237fc38d1861bc30365d48ffa',
              'upstream_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),
              'sample_sha256':hashlib.sha256(serialized.encode()).hexdigest(),
              'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'command':f'uv run --no-sync python -m scratch.replace_smol_summarize random-audit --seed {seed} --count {count}',
              'config':{'seed':seed,'count':count,'filters':None},'judge':'Codex direct reading; no API calls'}
        (target/'run_meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('AUDIT',target,'POPULATION',len(raw),'SEED',seed,'SAMPLE',count)
    for r in sample[offset:offset+batch_size]:
        print('\nSAMPLE',r['sample_number'],'UPSTREAM',r['upstream_index'])
        for m in r['messages']: print(m['role'].upper()+':',m['content'])

def random_audit_report(out_root):
    """Aggregate explicit manual judgments; never manufacture missing reviews."""
    import pyarrow.parquet as pq
    source=Path('scratch/smol_summarize_random_audit_reviews.json')
    data=json.loads(source.read_text())
    reviews=[dict(zip(data['columns'],r,strict=True)) for r in data['reviews']]
    sample=read(out_root/'sample.jsonl')
    meta=json.loads((out_root/'run_meta.json').read_text())
    assert len(sample)==len(reviews)==meta['count']==100
    assert [r['sample_number'] for r in reviews]==list(range(1,101))
    assert hashlib.sha256(RAW.read_bytes()).hexdigest()==meta['upstream_sha256']
    assert hashlib.sha256((out_root/'sample.jsonl').read_bytes()).hexdigest()==meta['sample_sha256']
    raw=pq.read_table(RAW).to_pylist()
    assert len(raw)==meta['population']
    assert [r['upstream_index'] for r in sample]==random.Random(meta['seed']).sample(range(len(raw)),100)
    for row,review in zip(sample,reviews,strict=True):
        assert row['upstream_index']==review['upstream_index']
        assert row['messages']==raw[row['upstream_index']]['messages']
        assert review['instruction'] in {'pass','fail','borderline'}
        assert type(review['faithful']) is bool and type(review['other_error']) is bool
    groups={}
    for label in ['correspondence','non_correspondence','all']:
        selected=[r for r in reviews if label=='all' or (r['genre']=='correspondence')==(label=='correspondence')]
        groups[label]={'sampled':len(selected),'source_faithful':sum(r['faithful'] for r in selected),
                       'source_faithful_and_complete':sum(r['faithful'] and not r['other_error'] for r in selected),
                       'instruction':dict(collections.Counter(r['instruction'] for r in selected)),
                       'passes_all_checks':sum(r['faithful'] and r['instruction']=='pass' and not r['other_error'] for r in selected)}
    result={'sampling_validated':True,'full_upstream_messages_exact_match':True,'all_100_directly_reviewed':True,
            'groups':groups,'genres':dict(collections.Counter(r['genre'] for r in reviews)),
            'rubric':data['rubric'],'mixture_changed':False,'published':False,
            'limitations':['One 100-row random sample does not establish a full-dataset ceiling.',
                           'Faithfulness is judged against supplied source text, not independent real-world fact-checking.',
                           'All 27 non-correspondence answers exceed three sentences on direct reading; 26 contain prohibited pronouns.',
                           'The old sentence regex also overcounts U.N. and H.W. abbreviations; do not treat it as a definitive judge.'],
            'reviews_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    (out_root/'reviews.json').write_text(json.dumps({'rubric':data['rubric'],'reviews':reviews},indent=2)+'\n')
    (out_root/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

def read(path):
    return [json.loads(line) for line in path.open()]

def signature(messages):
    return hashlib.sha256(json.dumps([(m['role'],m['content']) for m in messages],ensure_ascii=False).encode()).hexdigest()

def kind(messages):
    return 'three' if 'without using second or third person pronouns' in messages[0]['content'] else 'one'

def prepare():
    import pyarrow.parquet as pq
    base=read(BASE)
    seen={signature(r['messages']) for r in base}
    seen_inputs={m['content'].strip() for r in base for m in r['messages'] if m['role']=='user'}
    raw=pq.read_table(RAW).to_pylist()
    order=list(range(len(raw)));random.Random(20260908).shuffle(order)
    pool=[];stats=collections.Counter()
    for idx in order:
        messages=[{'role':m['role'],'content':m['content']} for m in raw[idx]['messages']]
        sig=signature(messages);k=kind(messages)
        a=messages[-1]['content'];p=messages[1]['content']
        if sig in seen or p.strip() in seen_inputs:
            stats['duplicate_or_existing']+=1;continue
        if k=='three' and BAN.search(a):
            stats['banned_pronoun']+=1;continue
        if len(re.findall(r'[.!?]+(?:\s|$)',a))>(3 if k=='three' else 1):
            stats['sentence_screen']+=1;continue
        if not re.search(r'[.!?][\"”\x27’]?$',a.strip()):
            stats['incomplete_ending_screen']+=1;continue
        if k=='one' and len(a.split())>30:
            stats['not_very_short_screen']+=1;continue
        seen.add(sig);seen_inputs.add(p.strip())
        pool.append({'id':len(pool),'upstream_index':idx,'kind':k,'sha256':sig,'messages':messages})
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'candidates.jsonl').open('w') as f:
        for row in pool:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    (ROOT/'preparation.json').write_text(json.dumps({'base':str(BASE),'base_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),'upstream_revision':'5feaf2fd3ffca7c237fc38d1861bc30365d48ffa','upstream_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),'seed':20260908,'upstream_rows':len(raw),'screen_rejections':dict(stats),'eligible':dict(collections.Counter(r['kind'] for r in pool))},indent=2)+'\n')
    print((ROOT/'preparation.json').read_text())

def decisions():
    batches=json.loads(DECISIONS.read_text()) if DECISIONS.exists() else []
    result={}
    for b in batches:
        for i in b['reviewed']:
            assert i not in result, f'duplicate review {i}'
            reason=b.get('reject',{}).get(str(i))
            result[i]={'verdict':'reject' if reason else 'accept','reason':reason or 'Directly reviewed full input and answer: faithful, concise, and compliant with the supplied system instruction.'}
    return result

def review(count):
    pool=read(ROOT/'candidates.jsonl');d=decisions()
    accepted=collections.Counter(r['kind'] for r in pool if d.get(r['id'],{}).get('verdict')=='accept')
    pending=[];reserved=accepted.copy()
    for r in pool:
        if r['id'] not in d and reserved[r['kind']]<QUOTAS[r['kind']]:
            pending.append(r);reserved[r['kind']]+=1
            if len(pending)==count:break
    print('ACCEPTED',dict(accepted),'REVIEWED',len(d))
    print('SYSTEM three: Provide a concise, objective summary of the input text in up to three sentences, focusing on key actions and intentions without using second or third person pronouns.')
    print('SYSTEM one: Extract and present the main key point of the input text in one very short sentence, including essential details like dates or locations if necessary.')
    for r in pending:
        print(f"\nID {r['id']} ({r['kind']}) INPUT: {r['messages'][1]['content']}\nANSWER: {r['messages'][-1]['content']}")
    print('REVIEW_IDS', [r['id'] for r in pending])

def build():
    from transformers import AutoTokenizer
    from src.model_profile import render_chat, model_profile
    pool=read(ROOT/'candidates.jsonl');d=decisions();base=read(BASE)
    prep=json.loads((ROOT/'preparation.json').read_text())
    assert hashlib.sha256(BASE.read_bytes()).hexdigest()==prep['base_sha256'],'Base changed during review'
    selected={k:[r for r in pool if r['kind']==k and d.get(r['id'],{}).get('verdict')=='accept'] for k in QUOTAS}
    assert {k:len(v) for k,v in selected.items()}==QUOTAS
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B',local_files_only=True)
    profile=model_profile('Qwen/Qwen3.6-27B')
    replacements=[];max_tokens=0
    for k,rs in selected.items():
        for r in rs:
            n=len(render_chat(tok,r['messages'],render_kwargs=profile.render_kwargs,tokenize=True,return_dict=True)['input_ids'])
            assert n<=8192,(r['id'],n)
            max_tokens=max(max_tokens,n)
            if k=='three':assert not BAN.search(r['messages'][-1]['content'])
    queues={k:iter(v) for k,v in selected.items()}
    lines=BASE.read_text().splitlines(keepends=True);out=[]
    for i,(line,row) in enumerate(zip(lines,base)):
        if row['source']!='smol_summarize':out.append(line);continue
        replacement=next(queues[kind(row['messages'])])
        new={'messages':replacement['messages'],'source':'smol_summarize'}
        out.append(json.dumps(new,ensure_ascii=False)+'\n')
        replacements.append({'mixture_index':i,'candidate_id':replacement['id'],'upstream_index':replacement['upstream_index'],'old_sha256':signature(row['messages']),'new_sha256':replacement['sha256'],'kind':replacement['kind'],**d[replacement['id']]})
    assert len(replacements)==984 and len(out)==10000
    assert all(a==b for a,b,r in zip(lines,out,base) if r['source']!='smol_summarize')
    target=ROOT/'mixture.jsonl'
    target.write_text(''.join(out))
    (ROOT/'replacements.json').write_text(json.dumps(replacements,indent=2)+'\n')
    with (ROOT/'smol_summarize.jsonl').open('w') as f:
        for k in QUOTAS:
            for r in selected[k]:f.write(json.dumps({'messages':r['messages'],'source':'smol_summarize'},ensure_ascii=False)+'\n')
    (ROOT/'report.json').write_text(json.dumps({**prep,'output':str(target),'output_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'replaced':984,'unchanged_rows':9016,'quotas':QUOTAS,'reviewed':len(d),'rejected':sum(x['verdict']=='reject' for x in d.values()),'max_replacement_tokens':max_tokens,'judge':'Codex main conversation; direct review, no model API calls','published':False},indent=2)+'\n')
    print((ROOT/'report.json').read_text())

def diversify(news_only=False):
    """Select only directly accepted, verbatim upstream rows into a new artifact."""
    import subprocess
    from datetime import datetime, timezone
    from src.naming import artifact_name
    from transformers import AutoTokenizer
    from src.model_profile import render_chat, model_profile

    validate()
    pool=diversity_pool()
    reviews=json.loads(DIVERSITY_REVIEWS.read_text())
    judged={r['upstream_index']:r for r in reviews['reviews']}
    assert len(judged)==len(reviews['reviews'])==len(pool)
    assert set(judged)=={r['upstream_index'] for r in pool}
    report=json.loads((ROOT/'report.json').read_text())
    assert reviews['upstream_revision']==report['upstream_revision']
    selected=[r for r in pool if judged[r['upstream_index']]['verdict']=='accept']
    if news_only:
        selected=[r for r in selected if judged[r['upstream_index']].get('genre')=='news_report']
        assert {r['upstream_index'] for r in selected}=={19277,30325,44281,47831}
    assert selected, 'No directly accepted diversity candidates'
    base=read(BASE)
    lines=(ROOT/'mixture.jsonl').read_text().splitlines(keepends=True)
    manifest=json.loads((ROOT/'replacements.json').read_text())
    available={k:iter([e for e in manifest if e['kind']==k and not EMAIL.search(base[e['mixture_index']]['messages'][1]['content'])]) for k in QUOTAS}
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B',local_files_only=True)
    profile=model_profile('Qwen/Qwen3.6-27B')
    max_tokens=report['max_replacement_tokens']
    changes=[]
    for row in selected:
        entry=next(available[row['kind']])
        i=entry['mixture_index']
        n=len(render_chat(tok,row['messages'],render_kwargs=profile.render_kwargs,tokenize=True,return_dict=True)['input_ids'])
        assert n<=8192
        max_tokens=max(max_tokens,n)
        changes.append({'mixture_index':i,'removed_upstream_index':entry['upstream_index'],'added_upstream_index':row['upstream_index']})
        entry.pop('candidate_id')
        entry.update(upstream_index=row['upstream_index'],new_sha256=row['sha256'],
                     review_source='diversity',**{k:v for k,v in judged[row['upstream_index']].items() if k!='upstream_index'})
        lines[i]=json.dumps({'messages':row['messages'],'source':'smol_summarize'},ensure_ascii=False)+'\n'
    target=Path('output')/artifact_name('smol-summarize-final' if news_only else 'smol-summarize-diversity')
    target.mkdir(parents=True,exist_ok=False)
    (target/'mixture.jsonl').write_text(''.join(lines))
    (target/'smol_summarize.jsonl').write_text(''.join(lines[e['mixture_index']] for e in manifest))
    (target/'replacements.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report.update(output=str(target/'mixture.jsonl'),output_sha256=hashlib.sha256((target/'mixture.jsonl').read_bytes()).hexdigest(),
                  previous_output=str(ROOT/'mixture.jsonl'),previous_output_sha256=hashlib.sha256((ROOT/'mixture.jsonl').read_bytes()).hexdigest(),
                  max_replacement_tokens=max_tokens,diversity_changes=changes,
                  diversity_reviewed=len(judged),diversity_verdicts=dict(collections.Counter(r['verdict'] for r in judged.values())),
                  selected_genres={'correspondence':984-len(selected),**dict(collections.Counter(judged[r['upstream_index']]['genre'] for r in selected))},
                  diversity_limitation='This remains predominantly correspondence, not a balanced news/email selection. Heuristic discovery is not a proof that no further suitable rows exist.',
                  selection_policy='980 reviewed correspondence rows plus four reviewed news reports; Smol only, as requested.' if news_only else 'All five directly accepted diversity candidates.')
    (target/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (target/'diversity_reviews.json').write_bytes(DIVERSITY_REVIEWS.read_bytes())
    (target/'review_decisions.json').write_bytes(DECISIONS.read_bytes())
    meta={'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'created_at':datetime.now(timezone.utc).isoformat(),'command':'uv run --no-sync python -m scratch.replace_smol_summarize diversify'+(' --news-only' if news_only else ''),
          'config':{'instruction_quotas':QUOTAS,'max_tokens':8192,'one_sentence_max_words':30,'rewrite':False,'judge':'Codex direct review; no API','news_only':news_only},
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'diversity_reviews_sha256':hashlib.sha256(DIVERSITY_REVIEWS.read_bytes()).hexdigest(),
          'review_decisions_sha256':hashlib.sha256(DECISIONS.read_bytes()).hexdigest()}
    (target/'run_meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    validate(target)
    print(json.dumps({'output':str(target),'genres':report['selected_genres'],'diversity_changes':changes},indent=2))

def validate(out_root=ROOT):
    """Independently check built rows against the pinned upstream and original bytes."""
    import pyarrow.parquet as pq
    base=read(BASE);built=read(out_root/'mixture.jsonl')
    raw=pq.read_table(RAW).to_pylist()
    manifest=json.loads((out_root/'replacements.json').read_text())
    report=json.loads((out_root/'report.json').read_text())
    d=decisions()
    candidates={r['id']:r for r in read(ROOT/'candidates.jsonl')}
    diversity={r['upstream_index']:r for r in json.loads((out_root/'diversity_reviews.json').read_text())['reviews']} if (out_root/'diversity_reviews.json').exists() else {}
    assert hashlib.sha256(BASE.read_bytes()).hexdigest()==report['base_sha256']
    assert hashlib.sha256(RAW.read_bytes()).hexdigest()==report['upstream_sha256']
    assert hashlib.sha256((out_root/'mixture.jsonl').read_bytes()).hexdigest()==report['output_sha256']
    assert len(base)==len(built)==10000
    assert collections.Counter(r['source'] for r in base)==collections.Counter(r['source'] for r in built)
    before=BASE.read_bytes().splitlines(keepends=True)
    after=(out_root/'mixture.jsonl').read_bytes().splitlines(keepends=True)
    indices={i for i,r in enumerate(base) if r['source']=='smol_summarize'}
    assert indices=={r['mixture_index'] for r in manifest}
    assert len(indices)==len(manifest)==984
    assert {i for i,(a,b) in enumerate(zip(before,after)) if a!=b}==indices
    old_inputs={m['content'].strip() for r in base for m in r['messages'] if m['role']=='user'}
    new_inputs=[];counts=collections.Counter();sentence_counts=collections.Counter()
    for entry in manifest:
        i=entry['mixture_index'];row=built[i];msgs=row['messages']
        upstream=[{'role':m['role'],'content':m['content']} for m in raw[entry['upstream_index']]['messages']]
        assert msgs==upstream
        assert [m['role'] for m in msgs]==['system','user','assistant']
        assert kind(msgs)==kind(base[i]['messages'])==entry['kind']
        if entry.get('review_source')=='diversity':
            assert diversity[entry['upstream_index']]['verdict']==entry['verdict']=='accept'
        else:
            assert d[entry['candidate_id']]['verdict']==entry['verdict']=='accept'
            assert candidates[entry['candidate_id']]['messages']==msgs
            assert candidates[entry['candidate_id']]['upstream_index']==entry['upstream_index']
        assert signature(msgs)==entry['new_sha256']
        assert signature(base[i]['messages'])==entry['old_sha256']
        prompt=msgs[1]['content'].strip();assert prompt not in old_inputs
        new_inputs.append(prompt);counts[kind(msgs)]+=1
        answer=msgs[-1]['content']
        # Honorifics are not sentence boundaries; punctuation inside a closing
        # quotation mark still ends a sentence (e.g. a quoted article title).
        sentences=sentence_count(answer)
        assert 1<=sentences<=(3 if kind(msgs)=='three' else 1),(entry['upstream_index'],sentences,answer)
        sentence_counts[sentences]+=1
        if kind(msgs)=='three':assert not BAN.search(answer)
        else:assert len(answer.split())<=30
    assert dict(counts)==QUOTAS
    assert len(set(new_inputs))==984
    standalone=read(out_root/'smol_summarize.jsonl')
    assert len(standalone)==984
    assert {signature(r['messages']) for r in standalone}=={signature(built[i]['messages']) for i in indices}
    def reasoning_count(rows):
        return sum(bool(m.get('reasoning_content')) for r in rows for m in r['messages'])
    assert reasoning_count(base)==reasoning_count(built)
    result={'validated':True,'total_rows':10000,'changed_rows':984,'byte_identical_non_smol_rows':9016,
            'fresh_unique_upstream_inputs':984,'instruction_split':dict(counts),'banned_pronoun_violations':0,
            'sentence_count_histogram':dict(sentence_counts),'preserved_reasoning_messages':reasoning_count(built),
            'upstream_rows_exact_match':True,'all_replacements_directly_accepted':True}
    (out_root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

def publish_revision(out_root,repo_name,parent_revision):
    """Atomically revise an existing Hub mix, preserving its newer non-Smol rows."""
    import subprocess
    from datetime import datetime, timezone
    from huggingface_hub import CommitOperationAdd
    from src.infra.huggingface import hf_api,hf_download,hf_repo_id,gate_push,REQUIRED_FIELDS
    assert repo_name and parent_revision
    validate(out_root)
    repo=hf_repo_id(repo_name);api=hf_api()
    info=api.dataset_info(repo)
    assert info.sha==parent_revision,'Hub changed since inspection; inspect again before publishing'
    remote_path=Path(hf_download(repo,'mixture.jsonl',repo_type='dataset',revision=parent_revision))
    card_path=Path(hf_download(repo,'README.md',repo_type='dataset',revision=parent_revision))
    remote_lines=remote_path.read_bytes().splitlines(keepends=True)
    local_lines=(out_root/'mixture.jsonl').read_bytes().splitlines(keepends=True)
    base=read(BASE);remote=[json.loads(s) for s in remote_lines]
    assert len(remote)==len(base)==len(local_lines)==10000
    assert [r['source'] for r in remote]==[r['source'] for r in base]
    indices={i for i,r in enumerate(base) if r['source']=='smol_summarize'}
    assert len(indices)==984
    for i in indices:assert remote[i]==base[i],'Remote Smol rows changed; do not overwrite unseen changes'
    merged=[local_lines[i] if i in indices else line for i,line in enumerate(remote_lines)]
    assert sum(a!=b for a,b in zip(remote_lines,merged))==984
    assert all(merged[i]==remote_lines[i] for i in range(10000) if i not in indices)
    rows=[json.loads(s) for s in merged]
    traces=lambda rs:sum(bool(m.get('reasoning_content')) for r in rs for m in r['messages'])
    assert traces(rows)==traces(remote)
    assert all(rows[i]==json.loads(local_lines[i]) for i in indices)
    manifest=json.loads((out_root/'replacements.json').read_text())
    assert {r['upstream_index'] for r in manifest if r.get('review_source')=='diversity'}=={19277,30325,44281,47831}
    staging=out_root/'hf-revision';staging.mkdir(exist_ok=False)
    mixture=staging/'mixture.jsonl';mixture.write_bytes(b''.join(merged))
    digest=hashlib.sha256(mixture.read_bytes()).hexdigest()
    report={'validated':True,'repo':repo,'parent_revision':parent_revision,
            'parent_mixture_sha256':hashlib.sha256(remote_path.read_bytes()).hexdigest(),
            'mixture_sha256':digest,'total_rows':10000,'replaced_smol_rows':984,
            'selected_genres':{'correspondence':980,'news_report':4},
            'byte_identical_non_smol_rows':9016,'preserved_reasoning_messages':traces(rows),
            'unchanged_upstream_content':True,'all_replacements_directly_accepted':True,
            'instruction_split':QUOTAS,'rewritten_rows':0,
            'local_reviewed_mixture':str(out_root/'mixture.jsonl'),
            'note':'Rebased reviewed Smol replacements onto the pinned Hub mixture; newer non-Smol reasoning is preserved.'}
    (staging/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    meta={'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'created_at':datetime.now(timezone.utc).isoformat(),'config':report,
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'command':f'uv run --no-sync python -m scratch.replace_smol_summarize publish-revision --out-root {out_root} --repo-name {repo_name} --parent-revision {parent_revision}'}
    (staging/'run_meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    card=card_path.read_text()
    fields={name:re.search(r'\| `'+re.escape(name)+r'` \| (.*?) \|',card).group(1) for name in REQUIRED_FIELDS}
    gate_push(repo,fields,what='existing mixture revision')
    marker='| field | value |'
    assert card.count(marker)==1
    update=(f'## Current revision: reviewed Smol replacement\n\n'
            f'Replaces only the 984 `smol_summarize` rows: 980 correspondence summaries and four news summaries, '
            f'all directly reviewed and selected unchanged from the original upstream dataset. No prompts, '
            f'reference answers, or reasoning were rewritten, and no alternative summarisation dataset was added.\n\n'
            f'The mixture still contains 10,000 rows. All 9,016 non-Smol rows are byte-identical to '
            f'parent revision `{parent_revision}`, preserving all {traces(rows):,} reasoning-bearing messages. '
            f'The Smol instruction split remains 644 up-to-three-sentence/pronoun-constrained and 340 one-sentence rows. '
            f'This is intentionally email-heavy, not a balanced news/email sample. Direct review is not a guarantee '
            f'against every subtle error.\n\n'
            f'Checks, upstream row mapping, direct review decisions, and the publishing script are under '
            f'`smol_replacement/`. The default training file remains `mixture.jsonl`. '
            f'SHA-256: `{digest}`.\n\n'
            f'## Historical reasoning-enrichment provenance\n\n'
            f'The table below describes the preceding enrichment run; its unchanged-answer statement applies '
            f'to that run, before the Smol replacement described above.\n\n')
    (staging/'README.md').write_text(card.replace(marker,update+marker))
    uploads={'mixture.jsonl':mixture,'README.md':staging/'README.md',
             'smol_replacement/validation.json':staging/'validation.json',
             'smol_replacement/run_meta.json':staging/'run_meta.json',
             'smol_replacement/replacements.json':out_root/'replacements.json',
             'smol_replacement/review_decisions.json':out_root/'review_decisions.json',
             'smol_replacement/diversity_reviews.json':out_root/'diversity_reviews.json',
             'smol_replacement/local_selection_report.json':out_root/'report.json',
             'smol_replacement/replace_smol_summarize.py':Path(__file__)}
    result=api.create_commit(repo,repo_type='dataset',parent_commit=parent_revision,
            operations=[CommitOperationAdd(path_in_repo=k,path_or_fileobj=str(v)) for k,v in uploads.items()],
            commit_message='Replace 984 Smol summaries with reviewed unchanged rows; preserve all reasoning')
    receipt={'repo':repo,'revision':result.oid,'url':str(result.commit_url),'mixture_sha256':digest,'verified':False}
    (staging/'push_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    check=Path(hf_download(repo,'mixture.jsonl',repo_type='dataset',revision=result.oid))
    assert hashlib.sha256(check.read_bytes()).hexdigest()==digest
    receipt['verified']=True
    (staging/'push_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','review','build','validate','diversify','random-audit','audit-report','balance-news-audit','inspect-openorca','report-news-sources','publish-revision']);parser.add_argument('--count',type=int,default=30);parser.add_argument('--out-root',type=Path,default=ROOT)
    parser.add_argument('--repo-name');parser.add_argument('--parent-revision')
    parser.add_argument('--news-only',action='store_true')
    parser.add_argument('--seed',type=int,default=20260909);parser.add_argument('--offset',type=int,default=0);parser.add_argument('--batch-size',type=int,default=10);args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='review':review(args.count)
    elif args.action=='build':build()
    elif args.action=='diversify':diversify(args.news_only)
    elif args.action=='random-audit':random_audit(args.seed,args.count,args.offset,args.batch_size)
    elif args.action=='audit-report':random_audit_report(args.out_root)
    elif args.action=='balance-news-audit':balance_news_audit()
    elif args.action=='inspect-openorca':inspect_openorca(args.out_root/'openorca-gpt4.parquet')
    elif args.action=='report-news-sources':report_news_sources(args.out_root)
    elif args.action=='publish-revision':publish_revision(args.out_root,args.repo_name,args.parent_revision)
    else:validate(args.out_root)
