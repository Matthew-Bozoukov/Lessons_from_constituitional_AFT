# ABOUTME: Runs standard synth operators with a durable budget and one physical call per stage/item.
# ABOUTME: Freezes provenance, calibrates the reviewer, preserves every candidate, and never auto-repairs.
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import threading

from omegaconf import OmegaConf
from src.data.synth.ours import pipeline
from src.data.synth.ours.constitution import units_from_config
from src.data.synth.ours.stage_runtime import _parse_json
from src.naming import artifact_name, to_local
from src.utils import git_sha, timestamp
from scratch.dataset_refresh.run import BudgetClient, write_json


class SingleAttemptClient:
    def __init__(self, root, cfg, send=None):
        self.root, self.cfg = Path(root), cfg
        self.budget = BudgetClient(Path(cfg.get('campaign_budget_root', self.root / 'budget')), cfg['budget_usd'],
                                   [m['model'] for m in cfg['models'].values()], send=send)
        self.lock = threading.Lock()
        self.stopped = False
        self.claims = self.root / 'attempts.json'

    def chat(self, **kw):
        match = re.match(r'\[record_id=([^;]+); stage=([^\]]+)\]', kw['messages'][-1]['content'])
        if not match:
            raise ValueError('Every physical request needs an explicit record and stage')
        rid, stage = match.groups()
        key = f'{stage}/{rid}'
        with self.lock:
            previous = json.loads(self.claims.read_text()) if self.claims.exists() else []
            if self.stopped or key in previous or len(previous) >= self.cfg['smoke_contract']['max_physical_calls']:
                raise RuntimeError('No second attempt, resumed dispatch, or call beyond the smoke limit')
            previous.append(key)
            write_json(self.claims, previous)
        self.budget.local.candidate_id = rid
        self.budget.local.stage = stage
        self.budget.local.arm = self.cfg['pipeline']
        try:
            result = self.budget.chat(**kw)
        except BaseException:
            with self.lock:
                self.stopped = True
            raise
        if result.response_model and result.response_model != kw['model']:
            with self.lock:
                self.stopped = True
            raise RuntimeError('Unexpected response model; inspect preserved raw response')
        return result


def validate_review(value, extra_fields=()):
    required = {'verdict', 'stakes', 'decisive_fact_check', 'competing_considerations', 'mechanism', 'domain', 'findings'}
    required.update(extra_fields)
    if not isinstance(value, dict) or not required <= value.keys():
        raise ValueError('Incomplete review schema')
    if value['verdict'] not in ('pass', 'fail') or type(value['stakes']) is not int or value['stakes'] not in range(4):
        raise ValueError('Invalid review verdict or stakes')
    if not isinstance(value['findings'], list):
        raise ValueError('Findings must be an array')
    for finding in value['findings']:
        if not all(isinstance(finding.get(k), str) and finding[k].strip() for k in ('code', 'quote', 'why')):
            raise ValueError('A finding requires a code, quote and explanation')
    if (value['verdict'] == 'pass') != (not value['findings'] and value['stakes'] <= 1):
        raise ValueError('Verdict inconsistent with findings or stakes')
    return value


def anchored(quote, text):
    # Accept separately quoted spans joined by an ellipsis only when ALL pieces occur
    # in their given order. This verifies presence, not that the inference is sound.
    norm = lambda s: ' '.join(s.split()).casefold()
    text = norm(text)
    pos = 0
    seen = False
    for part in re.split(r'\.\.\.|…', quote):
        part = norm(part)
        if not part:
            continue
        seen = True
        index = text.find(part, pos)
        if index < 0:
            return False
        pos = index + len(part)
    return seen


def leakage_findings(system, user, reasoning, response):
    supplied_draft = bool(re.search(r"(?:here(?: is|'s) (?:my |the )?draft|my draft (?:is|reads)|draft:)\s*\S", user, re.I))
    findings = []
    for block, text in [('reasoning', reasoning), ('response', response)]:
        for match in re.finditer(r"\b(?:previous|earlier|prior) draft\b|\b(?:my|the) rewritten answer\b", text, re.I):
            if not supplied_draft and match.group(0).lower() not in (system+' '+user).lower():
                findings.append(dict(code='process_narration', quote=match.group(0), why=f'{block} refers to a draft absent from the conversation', detector='deterministic'))
    return findings


def anchor_errors(review, system, user, reasoning, response, flexible=False):
    errors = []
    source = system + '\n' + user
    for item in review.get('claim_audit', []):
        blocks = {'reasoning': reasoning, 'response': response}
        if flexible:
            blocks.update(user=user, system=system)
        block = blocks.get(item.get('block'))
        matches = anchored if flexible else lambda q,t: bool(q) and q in t
        claim = item.get('claim','')
        # Reasoning/reply labels are metadata: verify the actual answer quotation
        # even when the reviewer names the wrong one of its two visible blocks.
        in_other_answer_block = flexible and any(matches(claim,t) for t in (reasoning,response))
        if (block is None or not matches(claim, block)) and not in_other_answer_block:
            errors.append('Claim quotation absent from named block')
        quotes = [v for k,v in item.items() if re.fullmatch(r'source_quote\d*', k) and v]
        for quote in quotes:
            if not isinstance(quote,str) or not matches(quote, source):
                errors.append('Source quotation absent from original conversation')
        # The semantic judgment is in assessment + findings. A missing redundant
        # classification label must not turn a correct judgment into a content defect.
        status_optional = flexible and 'status' not in item and bool(item.get('assessment'))
        if not status_optional and item.get('status') not in {'supported', 'inference', 'proposal', 'unsupported'}:
            errors.append('Invalid claim status')
        if item.get('status') == 'supported' and not quotes:
            errors.append('Supported claim has no original-source quotation')
    if 'claim_audit' in review and not review['claim_audit']:
        errors.append('Empty claim audit')
    return errors


def effective_review(review, system, user, reasoning, response, guarded=False):
    findings = list(review['findings'])
    if guarded:
        findings += leakage_findings(system,user,reasoning,response)
    errors = anchor_errors(review,system,user,reasoning,response,flexible=guarded)
    return dict(verdict='pass' if review['verdict']=='pass' and not findings and not errors else 'fail', findings=findings, anchor_errors=errors)


def answer_gate(sc, cfg):
    from src.data.synth.ours.stage_runtime import Stage
    def gate(ctx, records, ckpt):
        decisions = []
        for record in records:
            try:
                validate_review(record['review'],cfg['smoke_contract'].get('review_fields',[]))
                quality = effective_review(record['review'],record['system'],record['user'],record['reasoning'],record['response'],guarded=True)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                quality = dict(verdict='fail',findings=[],anchor_errors=[],technical_error=str(exc))
            decisions.append({**record,'quality':quality})
        write_json(ctx.run_dir / 'answer_decisions.json',decisions)
        kept=[r for r in decisions if r['quality']['verdict']=='pass']
        if not kept:
            ctx.stop='No answer passed; diagnostic records saved'
        return kept
    return Stage(sc['name'],gate)


def source_errors(record):
    audit = record.get('source_record', {})
    errors = []
    if audit.get('verdict') not in {'pass', 'fail'} or type(audit.get('stakes')) is not int:
        errors.append('Invalid admission verdict/stakes')
    if audit.get('verdict') == 'pass' and (audit.get('findings') or audit.get('stakes', 9) > 1):
        errors.append('Inconsistent passing admission')
    source = record['system'] + '\n' + record['user']
    if not audit.get('facts'):
        errors.append('No source facts')
    for fact in audit.get('facts', []):
        if not fact.get('quote') or not anchored(fact['quote'], source):
            errors.append('Fact quote absent from original conversation')
    return errors


def source_gate(sc, cfg):
    from src.data.synth.ours.stage_runtime import Stage
    def gate(ctx, records, ckpt):
        kept, rejected = [], []
        for record in records:
            try:
                errors = source_errors(record)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                errors = ['Malformed source audit: '+str(exc)]
            if errors or record['source_record']['verdict'] != 'pass':
                rejected.append({**record, 'mechanical_errors': errors})
            else:
                kept.append(record)
        write_json(ctx.run_dir / 'scenario_rejections.json', rejected)
        ctx.manifest_extra['source_admission'] = {'submitted': len(records), 'admitted': len(kept), 'rejected': len(rejected)}
        if not kept:
            ctx.stop = 'No scenario passed source admission; no answers generated'
        elif cfg['smoke_contract'].get('stop_on_unreachable_yield') and len(kept)<cfg['smoke_contract']['minimum_independently_acceptable']:
            ctx.stop = 'Admission yield cannot reach the frozen smoke threshold; stop before paying authors'
        return kept
    return Stage(sc['name'], gate)


def focus_clause(sc, cfg):
    from src.data.synth.ours.stage_runtime import Stage
    def focus(ctx, records, ckpt):
        for record in records:
            focus = cfg['coverage_focus'][record['trait_id']]
            if focus['clause'] not in record['trait_text']:
                raise ValueError('Coverage clause must be an exact constitution excerpt')
            record['focus_clause'] = focus['clause']
            record['focus_scope'] = focus['scope']
        return records
    return Stage(sc['name'], focus)


def bounded_llm(sc, cfg):
    """Use standard parsers, but one malformed record does not abort its peers.

    Transport/budget failures still abort. No conversion of provider reasoning to
    authored reasoning, no model repair, and no exception-induced second call.
    """
    from src.data.synth.ours.stage_runtime import Stage, _parse_tagged, model_cfg
    def execute(ctx, records, ckpt):
        m = model_cfg(cfg, sc['model'])
        lock = threading.Lock()
        failures = []
        def one(record):
            fields = {**ctx.vars, **record}
            messages = [{'role': role, 'content': sc['prompts'][role].format(**fields)} for role in ('system','user')]
            result = ctx.client.chat(model=m['model'], messages=messages, temperature=m['temperature'],
                                     max_tokens=m['max_tokens'], extra_body=m.get('extra_body', {}))
            with lock:
                ctx.usage.add(m['model'], result, sc['name'])
            try:
                if result.finish_reason == 'length':
                    raise ValueError('Truncated output')
                parsed = _parse_json(result.content) if sc['kind']=='smoke_json' else _parse_tagged(result.content,tuple(sc['tags']))
                output = {**record, **{dest:parsed[source] for dest,source in sc['save'].items()}}
            except (ValueError, KeyError, TypeError) as exc:
                with lock:
                    failures.append(dict(scenario_id=record['scenario_id'],error=str(exc),raw_preserved=True))
                    write_json(ctx.run_dir/(sc['name']+'_technical_rejections.json'),failures)
                return None
            with lock:
                if ckpt:
                    ckpt.record(output)
            return output
        with ThreadPoolExecutor(max_workers=ctx.workers) as pool:
            results = list(pool.map(one,records))
        ctx.manifest_extra.setdefault('technical_rejections',{})[sc['name']] = failures
        kept = [r for r in results if r is not None]
        if not kept:
            ctx.stop = 'Every record failed format checks; raw outputs preserved'
        elif cfg['smoke_contract'].get('stop_on_unreachable_yield') and len(kept)<cfg['smoke_contract']['minimum_independently_acceptable']:
            ctx.stop = 'Format yield cannot reach the frozen smoke threshold; stop before further paid stages'
        return kept
    return Stage(sc['name'],execute,paid=True,checkpoint_key='scenario_id')


def conversation_context(sc, cfg):
    """Build the conversation input without revealing its training target."""
    from src.data.synth.ours.stage_runtime import Stage
    def build(ctx, records, ckpt):
        output=[]
        for record in records:
            r=dict(record)
            r['authored_system']=r['system']
            r['system']=r['system'] if r['trait_id'] in sc['operator_traits'] else sc['neutral_system']
            output.append(r)
        return output
    return Stage(sc['name'],build)


def calibrate(cfg, root, client, fixture_path):
    cases = OmegaConf.to_container(OmegaConf.load(fixture_path), resolve=True)['cases']
    traits = {t.as_trait().trait_id: t.as_trait().text for t in units_from_config(cfg)}
    cached = {}
    if cfg.get('calibration_cache'):
        oldroot=Path(cfg['calibration_cache'])
        oldcfg=json.loads((oldroot/'run_meta.json').read_text(encoding='utf-8'))['config']
        reusable_stages = {}
        for stage_name in ('review_responses','audit_scenarios'):
            old=next(s for s in oldcfg['stages'] if s['name']==stage_name)
            new=next(s for s in cfg['stages'] if s['name']==stage_name)
            reusable_stages[stage_name] = old['prompts']==new['prompts'] and oldcfg['models'][old['model']]==cfg['models'][new['model']]
        assert oldcfg.get('coverage_focus')==cfg.get('coverage_focus')
        assert (oldroot/'constitution.md').read_bytes()==Path(cfg['constitution']).read_bytes()
        cached={x['case']['id']:x for x in json.loads((oldroot/'calibration_results.json').read_text(encoding='utf-8'))}
        shutil.copy2(oldroot/'calibration_results.json',root/'cached_calibration_original.json')
    def one(case):
        old=cached.get(case['id'])
        stage_name='audit_scenarios' if case.get('judge_stage')=='source' else 'review_responses'
        if old and old['case']==case and old['correct'] and reusable_stages[stage_name]:
            value={**old,'reused_from':cfg['calibration_cache']}
            write_json(root/'calibration'/(case['id']+'.json'),value)
            return value
        source_test = case.get('judge_stage') == 'source'
        spec = next(s for s in cfg['stages'] if s['name'] == ('audit_scenarios' if source_test else 'review_responses'))
        model = cfg['models'][spec['model']]
        focus = cfg.get('coverage_focus', {}).get(case['trait_id'], {})
        fields = {**case, 'scenario_id': 'calibration_' + case['id'], 'trait_text': traits[case['trait_id']],
                  'focus_clause': focus.get('clause', ''), 'focus_scope': focus.get('scope', '')}
        result = client.chat(model=model['model'], temperature=model['temperature'], max_tokens=model['max_tokens'],
            extra_body=model['extra_body'], messages=[{'role': 'system', 'content': spec['prompts']['system'].format(**fields)},
            {'role': 'user', 'content': spec['prompts']['user'].format(**fields)}])
        if source_test:
            review = _parse_json(result.content)['source_record']
            errors = source_errors({**case,'source_record':review})
            quality = dict(verdict=review['verdict'] if not errors else 'fail',findings=review.get('findings',[]),anchor_errors=errors)
        else:
            review = validate_review(_parse_json(result.content)['review'], cfg['smoke_contract'].get('review_fields', []))
            quality = effective_review(review,case['system'],case['user'],case['reasoning'],case['response'],
                                       guarded=cfg['smoke_contract'].get('deterministic_guards', False))
        correct = quality['verdict'] == case['expected'] and (not case.get('expected_code') or
            case['expected_code'] in [f['code'] for f in quality['findings']])
        forbidden = set(case.get('forbidden_codes', [])) & {f['code'] for f in review['findings']}
        anchors = quality['anchor_errors']
        correct = correct and not forbidden and not anchors
        value = {'case': case, 'review': review, 'effective_review': quality, 'correct': correct, 'forbidden_codes_found': sorted(forbidden), 'anchor_errors': anchors}
        write_json(root / 'calibration' / (case['id'] + '.json'), value)
        return value
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        outcomes = list(pool.map(one, cases))
    write_json(root / 'calibration_results.json', outcomes)
    if not all(o['correct'] for o in outcomes):
        raise RuntimeError('Reviewer calibration failed; stop without generating or changing the rubric')
    return len(outcomes)


def replay_calibration(cfg, root):
    original = Path(cfg['calibration_replay'])
    meta = json.loads((original/'run_meta.json').read_text(encoding='utf-8'))
    oldspec = next(s for s in meta['config']['stages'] if s['name']=='review_responses')
    newspec = next(s for s in cfg['stages'] if s['name']=='review_responses')
    assert oldspec['prompts']==newspec['prompts'] and meta['config']['models']['review']==cfg['models']['review'], 'Cannot reuse verdicts after reviewer changes'
    assert (original/'constitution.md').read_bytes()==Path(cfg['constitution']).read_bytes()
    outcomes=json.loads((original/'calibration_results.json').read_text(encoding='utf-8'))
    if any(x['case'].get('judge_stage')=='source' for x in outcomes):
        oldsource=next(s for s in meta['config']['stages'] if s['name']=='audit_scenarios')
        newsource=next(s for s in cfg['stages'] if s['name']=='audit_scenarios')
        assert oldsource['prompts']==newsource['prompts'] and meta['config']['models']['source']==cfg['models']['source'], 'Cannot reuse changed source judgments'
    for x in outcomes:
        c=x['case']
        x['original_correct']=x['correct']
        if c.get('judge_stage')=='source':
            errors=source_errors({**c,'source_record':x['review']})
            q=dict(verdict=x['review']['verdict'] if not errors else 'fail',findings=x['review'].get('findings',[]),anchor_errors=errors)
        else:
            q=effective_review(x['review'],c['system'],c['user'],c['reasoning'],c['response'],guarded=True)
        x['effective_review']=q
        x['correct']=q['verdict']==c['expected'] and not q['anchor_errors'] and (not c.get('expected_code') or c['expected_code'] in [f['code'] for f in q['findings']]) and not (set(c.get('forbidden_codes', [])) & {f['code'] for f in q['findings']})
    write_json(root/'calibration_results.json',outcomes)
    write_json(root/'calibration_replay.json',dict(source=str(original),source_sha256=hashlib.sha256((original/'calibration_results.json').read_bytes()).hexdigest(),paid_calls=0,description='Same frozen model judgments with deterministic leakage guard and verified multi-span citation handling; not a fresh model validation'))
    if not all(x['correct'] for x in outcomes):
        raise RuntimeError('Guarded saved calibration failed')
    return len(outcomes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    path = Path(args.config)
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if cfg['pipeline'] != path.stem or cfg['total_scenarios'] != 18 or cfg['budget_usd'] > 8:
        raise ValueError('This dispatch is authorized only for the fixed 18-row, $8 smoke')
    if any('sonnet' not in m['model'] for m in cfg['models'].values()):
        raise ValueError('Sonnet only')
    root = Path('output') / to_local(artifact_name(cfg['pipeline'] + ' smoke')) / timestamp()
    root.mkdir(parents=True, exist_ok=False)
    fixture = path.with_name(cfg.get('calibration_file', 'constitution_smoke_calibration.yaml'))
    if len(OmegaConf.load(fixture)['cases']) != cfg['smoke_contract']['calibration_calls']:
        raise ValueError('Calibration count does not match frozen call budget')
    for source in (path, Path(__file__), fixture, Path(cfg['constitution']), Path('configs/endpoints/providers.yaml')):
        shutil.copy2(source, root / source.name)
    write_json(root / 'run_meta.json', {'git_sha': git_sha(), 'config': cfg, 'status': 'prepared',
        'only_generation_content_input': cfg['constitution'], 'calibration_not_author_input': True,
        'hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (path, Path(__file__), fixture, Path(cfg['constitution']), Path('configs/endpoints/providers.yaml'))}})
    print('SMOKE_ROOT=' + str(root.resolve()), flush=True)
    client = SingleAttemptClient(root, cfg)
    initial_entries = client.budget.entries()
    try:
        count = replay_calibration(cfg, root) if cfg.get('calibration_replay') else calibrate(cfg, root, client, fixture)
        print(f'Calibration passed {count}/{count}. Starting 18 candidates.', flush=True)
        generation = root / 'generation'
        generation.mkdir()
        if cfg['smoke_contract'].get('source_first'):
            from src.data.synth.ours.stage_operators import OPERATORS
            OPERATORS['smoke_source_gate'] = source_gate
            OPERATORS['smoke_answer_gate'] = answer_gate
            OPERATORS['smoke_focus_clause'] = focus_clause
            OPERATORS['smoke_json'] = bounded_llm
            OPERATORS['smoke_tagged'] = bounded_llm
            OPERATORS['smoke_context'] = conversation_context
        manifest = pipeline.run(cfg, smoke=True, resume=str(generation), client=client)
        if manifest.get('halted'):
            write_json(root / 'automatic_summary.json', {'completed': 0, 'model_pass': 0, 'manifest': manifest})
            return
        rows = [json.loads(line) for line in (generation / 'dataset.jsonl').read_text(encoding='utf-8').splitlines()]
        for row in rows:
            validate_review(row['metadata']['review'], cfg['smoke_contract'].get('review_fields', []))
        mechanical = {r['metadata']['scenario_id']: anchor_errors(r['metadata']['review'], r['messages'][0]['content'], r['messages'][1]['content'], r['messages'][2]['reasoning_content'], r['messages'][2]['content'], flexible=cfg['smoke_contract'].get('deterministic_guards', False)) for r in rows}
        write_json(root / 'answer_anchor_checks.json', mechanical)
        write_json(root / 'automatic_summary.json', {'completed': len(rows),
            'model_pass': sum(r['metadata']['review']['verdict'] == 'pass' for r in rows),
            'independent_review': 'pending; this is not a training-ready release', 'manifest': manifest})
    except BaseException as exc:
        write_json(root / 'stopped.json', {'exception': type(exc).__name__, 'message': str(exc),
            'retry_allowed': False})
        raise
    finally:
        all_entries = client.budget.entries()
        entries = all_entries[len(initial_entries):]
        if cfg.get('campaign_budget_root'):
            shutil.copytree(client.budget.root, root / 'campaign_budget_snapshot', dirs_exist_ok=True)
        write_json(root / 'cost_summary.json', {'campaign_total_usd':sum(e['charged_or_reserved_usd'] for e in all_entries), ** {'physical_calls': len(entries),
            'charged_or_reserved_usd': sum(e['charged_or_reserved_usd'] for e in entries),
            'statuses': {s: sum(e['status'] == s for e in entries) for s in {e['status'] for e in entries}}}})
        print('SMOKE_ROOT=' + str(root.resolve()), flush=True)


if __name__ == '__main__':
    main()
