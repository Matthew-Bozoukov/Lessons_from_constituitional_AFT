# ABOUTME: Matched minimal stakes edits of historical684 using the shared SynthDoc engine.
# ABOUTME: Keeps literal source provenance, bounded paid calls, paired review, and public artifacts.
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time

from omegaconf import OmegaConf
from tenacity import stop_after_attempt
from scratch.nonmoral.pilot import CappedClient, file_sha256, read_rows, verify_live_prices
from scratch.build_t2_9284_da716_mixture import render
from src.data.synth.ours.pipeline import run
from src.data.synth.ours.hf_cache import write_jsonl
from src.infra.endpoints.openrouter import OpenRouterClient
from src.infra.huggingface import hf_download
from src.utils import git_sha

CONFIG = Path('configs/data/synth/nonmoral-stakes.yaml')
ROOT = Path('output/nonmoral_stakes/20260910')
MIX_REPO = 'LASR-Callum/2026-09-02-table2-9284-nonmoral-deliberation-684-train-mixture'
MIX_REV = '6364505df02b0020b030bf379bd42285a14de6a5'
MIX_FILE = 't2_9284_nonmoral_684.jsonl'
MIX_SHA = '0517ef85d288f14e42bc77f371d3e2e48866879b5824984feaf4a7451ce60561'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    for attempt in range(20):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt==19: raise
            time.sleep(0.05)


def messages(r):
    out = [{'role':'system','content':r['system']}] if r['system'].strip() else []
    return out + [{'role':'user','content':r['user']},
                  {'role':'assistant','reasoning_content':r['reasoning'],'content':r['response']}]


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    dest = ROOT/'original_mixture.jsonl'
    if not dest.exists():
        shutil.copyfile(hf_download(MIX_REPO, MIX_FILE, repo_type='dataset', revision=MIX_REV), dest)
    assert file_sha256(dest) == MIX_SHA
    old = read_rows(dest)
    corpus = read_rows('output/nonmoral_deliberation/20260902_013651/stage_7_revise_responses.jsonl')
    by_id = {r['scenario_id']:r for r in corpus}
    selected = []
    for row in old:
        if row['source'] != 'nonmoral_deliberation':
            continue
        r = by_id[row['scenario_id']]
        assert render(messages(r)) == row['text'], row['scenario_id']
        selected.append({**r, 'feedback':'', 'parent_text_sha256':hashlib.sha256(row['text'].encode()).hexdigest()})
    assert len(selected) == len({r['scenario_id'] for r in selected}) == 684
    path = ROOT/'originals.jsonl'
    if path.exists():
        assert read_rows(path) == selected
    else:
        write_jsonl(path, selected)
    if not (ROOT/'authorization.json').exists():
        write_json(ROOT/'authorization.json', dict(approval='User: no more for you to inspect; go ahead and do it',
          scope='Create and publish both matched original684 stakes corpora and mixtures; no GPUs in this driver',
          prior_exposure_usd=167.894061131299, data_cap_usd=60, project_cap_usd=300,
          source_repo=MIX_REPO,source_revision=MIX_REV,source_sha256=MIX_SHA,
          count=684,git_sha=git_sha(),config_sha256=file_sha256(CONFIG),
          created_utc=datetime.now(timezone.utc).isoformat()))
    return selected


def apply_edits(row, edits):
    """Apply exact, nonoverlapping edits against original coordinates, never fuzzy match."""
    if not isinstance(edits, list) or not edits:
        raise ValueError('Missing stakes edits')
    result = {k:row[k] for k in ('system','user','reasoning','response')}
    for field in ('user','reasoning','response'):
        text = row[field]
        spans = []
        append = None
        for edit in edits:
            if not isinstance(edit, dict) or set(edit) != {'field','old','new'}:
                raise ValueError('Malformed edit')
            if edit['field'] not in ('user','reasoning','response'):
                raise ValueError('Forbidden field')
            if not isinstance(edit['old'],str) or not isinstance(edit['new'],str):
                raise ValueError('Non-string replacement')
            if edit['field'] != field:
                continue
            old, new = edit['old'], edit['new']
            if old == '':
                if append is not None:
                    raise ValueError('Multiple appends')
                append = new
            else:
                if text.count(old) != 1:
                    raise ValueError(f'{field}: old span occurs {text.count(old)} times: {old[:100]!r}')
                start = text.index(old)
                spans.append((start, start+len(old), new))
        spans.sort()
        for left, right in zip(spans, spans[1:]):
            if left[1] > right[0]:
                raise ValueError('Overlapping replacements')
        for start, end, new in reversed(spans):
            text = text[:start]+new+text[end:]
        if append is not None:
            text += append
        if not text.strip():
            raise ValueError('Empty conversation field')
        if any(tag in text for tag in ('<|im_start|>','<|im_end|>')):
            raise ValueError('Injected turn boundary')
        result[field] = text
    if result['user'] == row['user']:
        raise ValueError('No stakes manipulation in user request')
    return result


def materialize(row):
    pair = json.loads(row['pair_json'])
    if 'edits' in pair:
        values=[pair['low_value'],pair['high_value']]
        if any(not isinstance(x,str) or not re.fullmatch(r'\d+(?:\.\d+)?',x) for x in values):
            raise ValueError('Stakes values must be positive numeric strings')
        if not (0<float(values[0]) and float(values[1])>=10*float(values[0])):
            raise ValueError('Need a substantial low/high magnitude contrast')
        for field in ('user','reasoning'):
            if not any(e['field']==field and '__STAKE__' in e['new'] for e in pair['edits']):
                raise ValueError(f'Missing shared stake placeholder in {field}')
        def expand(value):
            return [{**e,'new':e['new'].replace('__STAKE__',value)} for e in pair['edits']]
        low,high=apply_edits(row,expand(values[0])),apply_edits(row,expand(values[1]))
        pair={**pair,'low_loss':values[0]+' '+pair.get('unit',''),
              'high_loss':values[1]+' '+pair.get('unit','')}
    else:
        low, high = apply_edits(row,pair['low']), apply_edits(row,pair['high'])
    for key in ('system','user','reasoning','response'):
        # Historical retained pairs may format dollar amounts with commas.
        normalize=lambda text: re.sub(r'\d[\d,]*(?:\.\d+)?','<STAKE_NUMBER>',text)
        if normalize(low[key]) != normalize(high[key]):
            raise ValueError(f'Non-numeric difference between stakes arms: {key}')
    if low['user'] == high['user']:
        raise ValueError('Identical low/high prompts')
    return pair, low, high


class StakesClient(CappedClient):
    def save(self):
        write_json(self.path,self.entries)

    def __init__(self, *args, deadline, phase, **kwargs):
        super().__init__(*args, **kwargs)
        self.deadline, self.phase = deadline, phase
        self.attempt_path = self.path.with_name('attempts.json')
        self.attempts = json.loads(self.attempt_path.read_text()) if self.attempt_path.exists() else {}
        self.attempt_lock = threading.Lock()

    def chat(self, model, messages, **kwargs):
        sid = re.match(r'Record ID: ([a-zA-Z0-9_]+)\n',messages[-1]['content']).group(1)
        key = self.phase+':'+sid
        with self.attempt_lock:
            if time.time()>self.deadline or (ROOT/'STOP_DISPATCH').exists():
                raise RuntimeError('Dispatch stopped by deadline or marker')
            if self.attempts.get(key,0)>=2:
                raise RuntimeError('Two paid format attempts exhausted')
            self.attempts[key]=self.attempts.get(key,0)+1
            write_json(self.attempt_path,self.attempts)
        result=super().chat(model,messages,**kwargs)
        # Lossless format normalization: raw provider text remains in raw_calls.
        # A valid bare JSON object needs no paid retry merely for omitted XML framing.
        tag='pair' if self.phase.endswith('_author') else 'review'
        content=result.content.strip()
        if content.startswith('```'):
            content=re.sub(r'^```(?:json)?\s*|\s*```$','',content)
        if not content.startswith('<'+tag+'>'):
            try:
                parsed=json.loads(content)
                required={'decision','issues'} if tag=='review' else {'edits','low_value','high_value'}
                if isinstance(parsed,dict) and (required<=parsed.keys() or
                    tag=='pair' and {'low','high'}<=parsed.keys()):
                    result.content='<'+tag+'>'+content+'</'+tag+'>'
            except ValueError:
                pass
        return result


def phase(cfg, rows, label, stage_name, deadline):
    if (ROOT/'STOP_DISPATCH').exists() or time.time()>deadline:
        raise RuntimeError('Stopped before next phase')
    d = ROOT/'batches'/label
    d.mkdir(parents=True,exist_ok=True)
    result = d/'run'/'dataset.jsonl'
    source = d/'inputs.jsonl'
    if source.exists():
        assert read_rows(source) == rows, 'Inputs changed on resume'
    else:
        write_jsonl(source,rows)
    if result.exists():
        return read_rows(result)
    eff = copy.deepcopy(cfg)
    stage = next(s for s in cfg['stages'] if s['name']==stage_name)
    eff.update(source={'local_dir':str(d),'snapshot':'inputs.jsonl'},total_scenarios=len(rows),
               stages=[cfg['stages'][0],stage])
    write_json(d/'config.json',eff)
    (d/'run').mkdir(exist_ok=True)
    client = OpenRouterClient()
    single = client.chat.retry_with(stop=stop_after_attempt(1))
    capped = StakesClient(lambda **kw:single(client,**kw),ROOT/'spend.json',cfg['budget_usd'],
                {cfg['models'][stage['model']]['model']},allow_reasoning_off=True,
                deadline=deadline,phase=label)
    done = threading.Event()
    def heartbeat():
        while not done.wait(25):
            with capped.lock:
                health=dict(phase=label,pid=os.getpid(),calls=len(capped.entries),
                  settled=sum(e['status']=='settled' for e in capped.entries),
                  exposure_usd=sum(e['charged_or_reserved_usd'] for e in capped.entries),epoch=time.time())
            write_json(ROOT/'heartbeat.json',health)
            print(json.dumps(health),flush=True)
    thread=threading.Thread(target=heartbeat,daemon=True)
    thread.start()
    try:
        run(eff,resume=str(d/'run'),client=capped)
    finally:
        done.set()
        thread.join(timeout=2)
    return read_rows(result) if result.exists() else []


def process(cfg, rows, label, deadline):
    recovered_path=ROOT/'recovered.jsonl'
    recovered={r['scenario_id']:r for r in read_rows(recovered_path)} if recovered_path.exists() and not label.startswith('repair') else {}
    pending=[r for r in rows if r['scenario_id'] not in recovered]
    authored=phase(cfg,pending,label+'_author','stakes_edits',deadline) if pending else []
    authored += [recovered[r['scenario_id']] for r in rows if r['scenario_id'] in recovered]
    by_id={r['scenario_id']:r for r in authored}
    valid, failures=[],[]
    for source in rows:
        r=by_id.get(source['scenario_id'])
        try:
            if r is None:
                raise ValueError('No author output; retry once')
            materialize(r)
            valid.append(r)
        except (ValueError,KeyError,TypeError) as exc:
            failures.append({**source,'feedback':f'Fix literal-patch/structure issue: {exc}. Prior JSON: '+(r or {}).get('pair_json','')})
    reviewed=[r for r in valid if 'review_json' in r and r['scenario_id'] in recovered]
    needs_review=[r for r in valid if r not in reviewed]
    reviewed += phase(cfg,needs_review,label+'_review','stakes_review',deadline) if needs_review else []
    reviews={r['scenario_id']:r for r in reviewed}
    accepted=[]
    for r in valid:
        rr=reviews.get(r['scenario_id'])
        try:
            verdict=json.loads(rr['review_json'])
            if not (verdict['decision']=='accept' and verdict['same_task'] is True and
                    verdict['nonmoral'] is True and verdict['low_stakes']=='low' and verdict['high_stakes']=='high'):
                raise ValueError(json.dumps(verdict,ensure_ascii=False))
            accepted.append(rr)
        except (ValueError,KeyError,TypeError) as exc:
            failures.append({**r,'feedback':f'Repair these specific intervention defects: {exc}. Prior JSON: '+r['pair_json']})
    write_jsonl(ROOT/'batches'/f'{label}_accepted.jsonl',accepted)
    write_jsonl(ROOT/'batches'/f'{label}_failures.jsonl',failures)
    print(json.dumps(dict(batch=label,accepted=len(accepted),repair_needed=len(failures))),flush=True)
    return accepted,failures


def execute(cfg, limit):
    originals=prepare()
    write_json(ROOT/'live_prices.json',verify_live_prices({m['model'] for m in cfg['models'].values()}))
    # Round robin across original craft traits for early coverage; original mixture order is restored at export.
    groups={}
    for r in originals:
        groups.setdefault(r['trait_id'],[]).append(r)
    order=[]
    while any(groups.values()):
        for key in sorted(groups):
            if groups[key]: order.append(groups[key].pop(0))
    order=order[:limit]
    deadline_path=ROOT/'deadline.json'
    if not deadline_path.exists():
        write_json(deadline_path,dict(epoch=time.time()+3*3600))
    deadline=json.loads(deadline_path.read_text())['epoch']
    accepted, failures=[],[]
    for i in range(0,len(order),36):
        a,f=process(cfg,order[i:i+36],f'shared{i//36:02d}',deadline)
        accepted+=a; failures+=f
        write_json(ROOT/'status.json',dict(status='generating',processed=min(i+36,len(order)),
                    accepted=len(accepted),repair_needed=len(failures),target=684))
    if limit==684 and failures:
        repaired=[]
        for i in range(0,len(failures),36):
            a,f=process(cfg,failures[i:i+36],f'repair_shared{i//36:02d}',deadline)
            accepted+=a; repaired+=f
        failures=repaired
    write_jsonl(ROOT/'accepted.jsonl',accepted)
    write_jsonl(ROOT/'unresolved.jsonl',failures)
    write_json(ROOT/'status.json',dict(status='generated_pending_local_audit',accepted=len(accepted),
               unresolved=len(failures),target=684,processed=len(order)))


def assemble():
    from collections import Counter
    from src.data.synth.ours.hf_cache import StageCache
    from src.infra.huggingface import card_markdown,training_data_tags
    from src.naming import synth_name,mix_name
    from src.utils import origin_url
    from scratch.nonmoral.publish_broader_mixture import token_mask_checks,BASE_REVISION
    originals=prepare()
    accepted=read_rows(ROOT/'accepted.jsonl')
    by_id={r['scenario_id']:r for r in accepted}
    assert len(by_id)==len(accepted)==684, 'Do not silently publish incomplete matched corpus'
    assert set(by_id)=={r['scenario_id'] for r in originals}
    local=json.loads((ROOT/'local_audit.json').read_text(encoding='utf-8'))
    assert local['accepted_sha256']==file_sha256(ROOT/'accepted.jsonl') and local['status']=='passed'
    date=datetime.now(timezone.utc).date().isoformat()
    tokenizer=Path(hf_download('Qwen/Qwen3.6-27B','tokenizer.json',revision=BASE_REVISION,local_files_only=True)).parent
    plan={}
    for arm in ('low','high'):
        config_path=Path(f'configs/data/synth/nonmoral-stakes-{arm}.yaml')
        arm_cfg=OmegaConf.to_container(OmegaConf.load(config_path),resolve=True)
        assert arm_cfg['arm']==arm and Path(arm_cfg['pair_config'])==CONFIG
        style=config_path.stem
        dest=ROOT/'publications'/arm
        dest.mkdir(parents=True,exist_ok=True)
        final=[]
        stats=[]
        for original in originals:
            sid=original['scenario_id']; r=by_id[sid]
            assert all(r[k]==original[k] for k in ('system','user','reasoning','response'))
            pair,low,high=materialize(r)
            row=low if arm=='low' else high
            final.append(dict(scenario_id=sid,trait_id=original['trait_id'],trait_name=original['trait_name'],
                trait_text=original['trait_text'],domain=original['domain'],messages=messages(row),
                stakes=arm,parent_text_sha256=original['parent_text_sha256'],
                source='nonmoral_stakes_'+arm,supervise='all'))
            stats.append(dict(scenario_id=sid,mechanism=pair.get('mechanism'),low_loss=pair.get('low_loss'),
                high_loss=pair.get('high_loss'),caveat=pair.get('caveat'),
                user_added_chars=len(row['user'])-len(original['user']),
                reasoning_added_chars=len(row['reasoning'])-len(original['reasoning']),
                final_unchanged=row['response']==original['response']))
        cache=StageCache(dest,None)
        stage1=cache.save(1,'historical_selected',originals)
        stage2=cache.save(2,'paired_edits_and_reviews',accepted)
        stage3=cache.save(3,'materialized',final)
        (dest/'stages').mkdir(exist_ok=True)
        for p in (stage1,stage2,stage3):
            p.replace(dest/'stages'/p.name)
        cache.publish_final(final)
        shutil.copyfile(CONFIG,dest/'pair_generation_config.yaml')
        shutil.copyfile(config_path,dest/'generation_config.yaml')
        shutil.copyfile(ROOT/'local_audit.json',dest/'local_audit.json')
        shutil.copyfile('output/nonmoral_deliberation/20260902_013651/manifest.json',dest/'historical_generation_manifest.json')
        write_json(dest/'edit_summary.json',stats)
        fields=dict(experiment=f'Matched {arm} stakes version of original nonmoral684',date_generated=date,
          constitution='preferences/craft_tensions_09/preferences.md; historical craft tensions preserved',
          source_repo=f'{origin_url()} @ {git_sha()}',
          models=dict(editor='anthropic/claude-opus-4.8',reviewer='anthropic/claude-sonnet-5',
                      revision='API model IDs; immutable provider weights unavailable',
                      local_review='Codex audit and exact corrections; see local_audit.json for review coverage and missing provider reviews',
                      historical_author='See historical_generation_manifest.json and pinned parent corpus; original source fields retained in stages'),
          generation_config='pair_generation_config.yaml; per-phase frozen configs in paired audit artifact',
          schema='dataset.jsonl: 684 full single-turn conversations with reasoning_content, matched scenario_id and stakes',
          provenance=f'uv run --no-sync python scratch/nonmoral/stakes.py --execute; --assemble; --publish. Parent {MIX_REPO}@{MIX_REV}; {MIX_SHA}',
          limitations='Post-generation stakes edits with shared wording and numeric-only low/high contrast; substantive reasoning held fixed. Low/high are relative loss magnitudes, not validated perceived-stakes labels. Inherited source imperfections retained. One pair per original ID. No ODCV selection or training/evaluation by data driver. Editing changes token lengths; no exact token matching.',
          intervention='Scale a task-linked nonmoral loss while retaining task, craft preference and full answer. See paired edits and review at stages/.',
          paired_arm='high' if arm=='low' else 'low')
        front=dict(configs=[dict(config_name='dataset',data_files='dataset.jsonl',default=True)]+
            [dict(config_name=p.stem,data_files='stages/'+p.name) for p in sorted((dest/'stages').glob('*.jsonl'))],
            tags=training_data_tags('synth',style,'craft_tensions_09',extra=['stakes:'+arm,'stage:final']))
        (dest/'README.md').write_text(card_markdown(fields,front),encoding='utf-8')
        write_json(dest/'manifest.json',dict(arm=arm,count=len(final),parent_revision=MIX_REV,parent_sha256=MIX_SHA,
            accepted_sha256=file_sha256(ROOT/'accepted.jsonl'),dataset_sha256=file_sha256(dest/'dataset.jsonl'),
            pair_config_sha256=file_sha256(CONFIG),git_sha=git_sha(),by_trait=dict(Counter(r['trait_id'] for r in final))))
        plan[arm]=dict(path=str(dest),name=synth_name(style,date=date),fields=fields,front=front)
        mixdest=ROOT/'publications'/(arm+'_mix'); mixdest.mkdir(exist_ok=True)
        selected={r['scenario_id']:r for r in final}
        original_lines=(ROOT/'original_mixture.jsonl').read_bytes().splitlines(keepends=True)
        mixed=[]; replay_count=0
        for line in original_lines:
            old=json.loads(line)
            if old['source']=='nonmoral_deliberation':
                item=selected[old['scenario_id']]
                new={**old,'source':'nonmoral_stakes_'+arm,'text':render(item['messages'])}
                mixed.append((json.dumps(new,ensure_ascii=False)+'\n').encode('utf-8'))
            else:
                mixed.append(line); replay_count+=1
        assert replay_count==9284 and len(mixed)==9968
        (mixdest/'mixture.jsonl').write_bytes(b''.join(mixed))
        loaded=read_rows(mixdest/'mixture.jsonl')
        prior=json.loads((mixdest/'manifest.json').read_text()) if (mixdest/'manifest.json').exists() else {}
        if (prior.get('mixture_sha256')==file_sha256(mixdest/'mixture.jsonl') and
            prior.get('train_config_sha256')==file_sha256('configs/train/sft.yaml') and
            prior.get('token_mask_checks',{}).get('status')=='passed'):
            checks=prior['token_mask_checks']
        else:
            checks=token_mask_checks(loaded,tokenizer,Path('configs/train/sft.yaml'),{'nonmoral_stakes_'+arm})
        write_json(mixdest/'token_mask_checks.json',checks)
        assert checks['status']=='passed' and checks['full_synthetic_masks_checked']==684, checks
        for before,after in zip(original_lines,(mixdest/'mixture.jsonl').read_bytes().splitlines(keepends=True)):
            if json.loads(before)['source']!='nonmoral_deliberation': assert before==after
        write_json(mixdest/'manifest.json',dict(status='validated_training_input',count=9968,synthetic=684,replay=9284,
           source_corpus=plan[arm]['name'],parent_mixture_revision=MIX_REV,parent_mixture_sha256=MIX_SHA,
           accepted_sha256=file_sha256(ROOT/'accepted.jsonl'),train_config_sha256=file_sha256('configs/train/sft.yaml'),
           replay_bytes_and_positions_preserved=True,mixture_sha256=file_sha256(mixdest/'mixture.jsonl'),
           token_mask_checks=checks,arm=arm))
        shutil.copyfile(config_path,mixdest/'generation_config.yaml')
        shutil.copyfile('configs/train/sft.yaml',mixdest/'train_recipe.yaml')
        mixfields={**fields,'experiment':f'Matched {arm} stakes nonmoral684 + exact historical9284 replay',
            'schema':'mixture.jsonl: 9968 pre-rendered Qwen ChatML text/source rows; full reasoning and final supervision',
            'generation_config':'generation_config.yaml selects arm of pair_generation_config.yaml in source corpus; train_recipe.yaml records proposed training settings',
            'source_corpus':plan[arm]['name']}
        mixfront=dict(configs=[dict(config_name='default',data_files='mixture.jsonl',default=True)],
                      tags=training_data_tags('mixture',style,'craft_tensions_09',extra=['stakes:'+arm,'stage:final']))
        (mixdest/'README.md').write_text(card_markdown(mixfields,mixfront),encoding='utf-8')
        plan[arm+'_mix']=dict(path=str(mixdest),name=mix_name(style,7,date=date),fields=mixfields,front=mixfront)
    write_json(ROOT/'publication_plan.json',plan)
    print(json.dumps(dict(assembled=list(plan),counts='684 per corpus; 9968 per mixture')),flush=True)


def audit_corrections():
    """Stage exact auditor-proposed corrections without touching historical fields or originals."""
    source=ROOT/'pre_local_audit_candidates.jsonl'
    if not source.exists():
        source=ROOT/'accepted.jsonl'
    rows=read_rows(source)
    by_id={r['scenario_id']:r for r in rows}
    changes=[]; coverage={}; missing=[]
    audit_names=['independent_audit.json','audit_middle.json','audit_later.json']
    if (ROOT/'root_audit.json').exists():
        audit_names.append('root_audit.json')
    if (ROOT/'correction_followups.json').exists():
        audit_names.append('correction_followups.json')
    for name in audit_names:
        path=ROOT/name
        audit=json.loads(path.read_text(encoding='utf-8'))
        findings=audit.get('findings',audit.get('dispositions',{}))
        for sid,item in findings.items():
            if sid not in by_id:
                missing.append(sid); continue
            r=by_id[sid]
            expected=item.get('pair_json_sha256',item.get('pair_sha256'))
            actual=hashlib.sha256(r['pair_json'].encode()).hexdigest()
            if expected and actual!=expected:
                missing.append(sid); continue
            if sid not in coverage:
                coverage[sid]=dict(audit=name,disposition=item.get('status',item.get('decision')))
            else:
                coverage[sid].setdefault('followup_audits',[]).append(name)
            replacements=item.get('proposed_pair_replacements',item.get('proposed_generated_text_replacements',item.get('replacements',[])))
            if not replacements:
                continue
            pair=json.loads(r['pair_json']); before=copy.deepcopy(pair)
            for edit in replacements:
                path_parts=edit.get('path')
                if path_parts is None:
                    target=edit['target'].removeprefix('pair_json.')
                    path_parts=[int(x) if x.isdigit() else x for x in re.findall(r'[A-Za-z_]+|\d+',target)]
                assert path_parts[0] in ('edits','low','high','low_value','high_value','unit','mechanism','low_loss','high_loss','caveat')
                cursor=pair
                for part in path_parts[:-1]: cursor=cursor[part]
                leaf=path_parts[-1]; old,new=edit['old'],edit['new']
                assert isinstance(cursor[leaf],str) and cursor[leaf].count(old)==edit.get('count',1),(sid,path_parts,old)
                cursor[leaf]=cursor[leaf].replace(old,new)
            changed={**r,'pair_json':json.dumps(pair,ensure_ascii=False)}
            materialize(changed)
            for key in ('system','user','reasoning','response'):
                assert changed[key]==r[key]
            changed['pre_local_correction_review_json']=r.get('pre_local_correction_review_json',r.get('review_json',''))
            changed.pop('review_json',None)
            changed['local_correction_json']=json.dumps(dict(audit=name,source_pair_sha256=actual,
                 reason=item.get('issue',item.get('reason',item.get('rationale',''))),replacements=replacements,
                 previous_correction_json=r.get('local_correction_json','')),ensure_ascii=False)
            by_id[sid]=changed
            changes.append(dict(scenario_id=sid,audit=name,before=before,after=pair,
                  reason=item.get('issue',item.get('reason',item.get('rationale','')))))
    ordered=[by_id[r['scenario_id']] for r in rows]
    write_jsonl(ROOT/'locally_corrected_candidates.jsonl',ordered)
    write_json(ROOT/'local_corrections.json',dict(parent_accepted_sha256=file_sha256(source),
         corrected=len({r['scenario_id'] for r in changes}),correction_passes=len(changes),records=changes,coverage=coverage,
         not_yet_audited=sorted(set(by_id)-set(coverage)),stale_or_missing=sorted(set(missing))))
    print(json.dumps(dict(corrected=len({r['scenario_id'] for r in changes}),covered=len(coverage),not_yet_audited=len(set(by_id)-set(coverage)))),flush=True)


def finalize():
    """Freeze locally reviewed pairs; preserve absent/stale provider reviews explicitly."""
    from collections import Counter
    originals={r['scenario_id']:r for r in prepare()}
    corrections=json.loads((ROOT/'local_corrections.json').read_text(encoding='utf-8'))
    assert not corrections['not_yet_audited'] and not corrections['stale_or_missing']
    assert set(corrections['coverage'])==set(originals)
    # Cross-checks must explicitly clear concrete findings before publication.
    clearance=json.loads((ROOT/'correction_clearance.json').read_text(encoding='utf-8'))
    assert clearance['status']=='passed'
    assert clearance['corrected_candidates_sha256']==file_sha256(ROOT/'locally_corrected_candidates.jsonl')
    rows=read_rows(ROOT/'locally_corrected_candidates.jsonl')
    assert len(rows)==len({r['scenario_id'] for r in rows})==684
    final=[]; counts=Counter(); answer_changed=0
    for r in rows:
        sid=r['scenario_id']; original=originals[sid]
        assert all(r[k]==original[k] for k in ('system','user','reasoning','response'))
        pair,low,high=materialize(r)
        answer_changed+=low['response']!=original['response']
        local_corrected=bool(r.get('local_correction_json'))
        review_status=('local_correction_after_model_review' if r.get('pre_local_correction_review_json') else
                       'provider_review_unavailable_local_review') if not r.get('review_json') else 'model_review_plus_local_audit'
        counts[review_status]+=1
        final.append({**r,'final_review_json':json.dumps(dict(decision='accept',
            review_source='Codex local audit and independent correction cross-check',
            model_review_status=review_status,locally_corrected=local_corrected,
            audit=corrections['coverage'][sid]['audit']),ensure_ascii=False)})
    archive=ROOT/'pre_local_audit_candidates.jsonl'
    if not archive.exists():
        assert file_sha256(ROOT/'accepted.jsonl')==corrections['parent_accepted_sha256']
        shutil.copyfile(ROOT/'accepted.jsonl',archive)
    # HF/Arrow infers columns chunk by chunk. Optional review provenance must be
    # present from the first chunk, including the18 provider-unreviewed cases.
    columns=sorted({key for r in final for key in r})
    final=[{key:r.get(key) for key in columns} for r in final]
    write_jsonl(ROOT/'accepted.jsonl',final)
    write_json(ROOT/'local_audit.json',dict(status='passed',count=684,
        accepted_sha256=file_sha256(ROOT/'accepted.jsonl'),originals_sha256=file_sha256(ROOT/'originals.jsonl'),
        audited_pairs=684,locally_corrected_pairs=corrections['corrected'],review_status_counts=dict(counts),
        coverage=corrections['coverage'],correction_clearance=clearance,
        checks=dict(original_fields_preserved=True,all_four_arm_fields_numeric_only=True,
                    matching_ids=True,full_source_answers_retained=True),
        final_answers_with_stakes_consistency_edits=answer_changed,
        limitations='Local semantic review is fallible. Read scope is recorded per auditor: original user, generated edits and affected source passages; some full-conversation reads. Provider review applies to pre-correction text when archived; no claim of rejudging corrected rows. Inherited original errors and incomplete prompts are retained to preserve the original684 intervention.',
        source_mixture=dict(repo=MIX_REPO,revision=MIX_REV,sha256=MIX_SHA)))
    write_json(ROOT/'status.json',dict(status='finalized_pending_publication',accepted=684,unresolved=0,
          previous_model_review_unavailable=counts['provider_review_unavailable_local_review'],local_corrections=corrections['corrected']))
    print(json.dumps(dict(finalized=684,review_status_counts=dict(counts),answer_consistency_edits=answer_changed)),flush=True)


def publish():
    from src.infra.huggingface import hf_org,hf_api,push_run_dir
    from src.naming import artifact_name
    from src.utils import origin_url
    from scratch.nonmoral.publish_invalid_baseline import scan,secret_values
    import zipfile
    assert hf_org()=='dougalldeepmind'
    plan=json.loads((ROOT/'publication_plan.json').read_text(encoding='utf-8'))
    accepted_sha=file_sha256(ROOT/'accepted.jsonl')
    local=json.loads((ROOT/'local_audit.json').read_text(encoding='utf-8'))
    assert local['status']=='passed' and local['accepted_sha256']==accepted_sha
    assert set(plan)=={'low','high','low_mix','high_mix'}
    # Refuse stale assembled inputs before creating or modifying any HF repository.
    for key,entry in plan.items():
        path=Path(entry['path'])
        manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
        assert manifest['accepted_sha256']==accepted_sha,(key,'stale accepted snapshot')
        is_mix=key.endswith('_mix')
        stem='mixture' if is_mix else 'dataset'
        assert manifest[stem+'_sha256']==file_sha256(path/(stem+'.jsonl'))
        if is_mix:
            assert manifest['token_mask_checks']['status']=='passed'
            assert manifest['token_mask_checks']['full_synthetic_masks_checked']==684
            assert manifest['train_config_sha256']==file_sha256('configs/train/sft.yaml')
        else:
            assert manifest['pair_config_sha256']==file_sha256(CONFIG)
            assert file_sha256(path/'local_audit.json')==file_sha256(ROOT/'local_audit.json')
    receipts={}
    secrets=secret_values()
    audit=ROOT/'publications'/'pair_audit'; audit.mkdir(exist_ok=True)
    shutil.copyfile(ROOT/'accepted.jsonl',audit/'pairs.jsonl')
    shutil.copyfile(ROOT/'local_audit.json',audit/'local_audit.json')
    shutil.copyfile(CONFIG,audit/'generation_config.yaml')
    shutil.copyfile(__file__,audit/'stakes.py')
    shutil.copyfile('output/nonmoral_deliberation/20260902_013651/manifest.json',audit/'historical_generation_manifest.json')
    ledger=json.loads((ROOT/'spend.json').read_text())
    accounting=dict(calls=len(ledger),exposure_usd=sum(e['charged_or_reserved_usd'] for e in ledger),
         prior_exposure_usd=167.894061131299,project_ceiling_usd=300,
         unsettled_or_retained=[dict(index=i,**e) for i,e in enumerate(ledger) if not e['status'].startswith('settled')])
    accounting['project_exposure_usd']=accounting['prior_exposure_usd']+accounting['exposure_usd']
    write_json(audit/'cost_accounting.json',accounting)
    files=[p for p in ROOT.glob('*.json*') if p.name!='original_mixture.jsonl']
    files += list(ROOT.glob('*audit*.md')) + list(ROOT.glob('*crosscheck*.md'))
    if (ROOT/'root_review.py').exists():
        files.append(ROOT/'root_review.py')
    for dirname in ('raw_calls','batches','bootstrap_archive'):
        files += [p for p in (ROOT/dirname).rglob('*') if p.is_file()]
    with zipfile.ZipFile(audit/'generation_audit.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            scan(f.read_bytes(),str(f),secrets)
            z.write(f,f.relative_to(ROOT).as_posix())
    date=datetime.now(timezone.utc).date().isoformat()
    audit_fields=dict(experiment='Original684 matched nonmoral stakes: paired edits, source, reviews and complete generation audit',
      date_generated=date,constitution='preferences/craft_tensions_09/preferences.md',
      source_repo=f'{origin_url()} @ {git_sha()}',models='anthropic/claude-opus-4.8 editor; anthropic/claude-sonnet-5 initial reviewer; Codex local audit and corrections; API revision pins unavailable',
      generation_config='generation_config.yaml; all prior/final per-phase configs in generation_audit.zip',
      schema='pairs.jsonl: final paired edit records; generation_audit.zip: all raw requests/responses, stage snapshots, failures and ledger',
      provenance='uv run --no-sync python scratch/nonmoral/stakes.py --execute; --assemble; --publish',
      limitations='Auxiliary audit, not SFT input. Failed earlier formulations are archived separately from accepted final pairs. No training or ODCV result.')
    audit_name=artifact_name('nonmoral-stakes-pair-audit',date=date)
    audit_url=push_run_dir(audit,audit_name,audit_fields,private=False,
          front_matter=dict(configs=[dict(config_name='pairs',data_files='pairs.jsonl',default=True)],tags=['nonmoral-stakes','paired-data-audit']))
    audit_repo=hf_org()+'/'+audit_name; audit_info=hf_api().dataset_info(audit_repo)
    assert not audit_info.private
    for filename in ('pairs.jsonl','generation_audit.zip','cost_accounting.json'):
        saved=hf_download(audit_repo,filename,repo_type='dataset',revision=audit_info.sha)
        assert file_sha256(saved)==file_sha256(audit/filename)
    receipts['audit']=dict(repo=audit_repo,url=audit_url,revision=audit_info.sha,public=True)
    write_json(ROOT/'publication.json',receipts)
    for key,entry in plan.items():
        path=Path(entry['path'])
        entry['fields']['paired_audit']=audit_repo+' @ '+audit_info.sha
        write_json(path/'paired_audit.json',receipts['audit'])
        for f in path.rglob('*'):
            if f.is_file(): scan(f.read_bytes(),str(f),secrets)
        url=push_run_dir(path,entry['name'],entry['fields'],private=False,front_matter=entry['front'])
        repo=hf_org()+'/'+entry['name']; info=hf_api().dataset_info(repo)
        assert not info.private
        filename='mixture.jsonl' if key.endswith('_mix') else 'dataset.jsonl'
        remote=hf_download(repo,filename,repo_type='dataset',revision=info.sha)
        assert file_sha256(remote)==file_sha256(path/filename)
        receipts[key]=dict(repo=repo,url=url,revision=info.sha,sha256=file_sha256(remote),public=True)
        write_json(ROOT/'publication.json',receipts)
        print(json.dumps(receipts[key]),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--limit',type=int,default=684)
    parser.add_argument('--assemble',action='store_true')
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--apply-audit',action='store_true')
    parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    cfg=OmegaConf.to_container(OmegaConf.load(CONFIG),resolve=True)
    if args.apply_audit:
        audit_corrections()
    elif args.finalize:
        finalize()
    elif args.assemble:
        assemble()
    elif args.publish:
        publish()
    elif args.execute:
        ROOT.mkdir(parents=True,exist_ok=True)
        lock=ROOT/'dispatch.lock'
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        try:
            os.write(fd,str(os.getpid()).encode())
            execute(cfg,args.limit)
        finally:
            os.close(fd); lock.unlink()
    else:
        print(json.dumps(dict(prepared=len(prepare()),paid=False)))


if __name__=='__main__':
    main()
