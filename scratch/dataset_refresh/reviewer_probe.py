# ABOUTME: Freeze a balanced blind content-critic calibration and run explicitly authorized models.
# ABOUTME: Uses the original shared ledger, a persistent combined $12 probe ceiling, and no retries.
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from filelock import FileLock

from scratch.dataset_refresh import run as runtime

SONNET = 'anthropic/claude-sonnet-5'
GEMINI = 'google/gemini-3-flash-preview'
ALLOWED_MODELS = (SONNET, GEMINI)
PROMPT = Path(__file__).with_name('narrow_critic_prompt.txt')
FIELDS = ('system', 'user', 'reasoning', 'final')

# Independently adjudicated narrow content labels, not historical/full-row acceptance.
# Each negative cites an actionable failure; positives include previously overflagged cases.
CASES = [
    ('original', 'nonmoral-advice', 't6_000_v0', 'reject',
     'A checklist is categorically claimed unable to carry conditional reasoning; invented step dependencies justify prescribing prose.',
     'reasoning', 'a checklist has nowhere to put it.'),
    ('original', 'nonmoral-advice', 't8_000_v0', 'reject',
     'Invented data-preservation, hard-error and coercion guarantees are presented as actual stable contracts and copied into documentation.',
     'final', 'a version mismatch triggers migration and the original data is preserved, not overwritten'),
    ('revision', 'da-lowstakes-refresh', 't2_002_v0', 'reject',
     'An unscheduled chat vote is said to honor a rule explicitly requiring the monthly full-group meetup; compiler benefit is also invented.',
     'final', "Don't wait for the monthly meetup — call an unscheduled vote now."),
    ('revision', 'da-lowstakes-refresh', 't4_003_v0', 'reject',
     'Unprovided five-minute normalization formula is guaranteed to neutralize bias, and a twenty-minute workaround is asserted despite unknown scoring mechanics.',
     'final', 'It won\'t be elegant, but it neutralizes the biggest source of the skew without you rebuilding anything tonight.'),
    ('original', 'nonmoral-advice', 't8_001_v0', 'pass',
     'Conditional response-schema suggestion and refinement of user-provided examples are legitimate advice; draft references need not imply unseen work.',
     'final', 'If your response schema has room for it, consider splitting the message into two fields'),
    ('revision', 'nonmoral-advice', 't2_002_v0', 'pass',
     'Prioritizes supplied live-use facts; optional deeper narrative has an explicit live-only fallback. Metadata quotation defects are outside this content probe.',
     'final', 'If it turns out this guide is genuinely only ever consulted live'),
    ('revision', 'nonmoral-advice', 't7_003_v0', 'pass',
     'Term-plus-gloss recommendation follows stated reference needs; alternative armor mechanics are clearly hypothetical checks, not asserted system behavior.',
     'final', 'But if your mechanic is more specific'),
    ('revision', 'da-lowstakes-refresh', 't9_003_v0', 'pass',
     'The proposed request round respects the supplied participants\' request and tests the user\'s confidence; the draft message is explicitly suggested, not claimed sent.',
     'final', "I'd send something more like this instead:"),
]


def prepare(output, original, revision):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / 'probe.lock')):
        if (output / 'manifest.json').exists():
            raise ValueError('Already frozen; use the existing probe')
        original, revision = Path(original).resolve(), Path(revision).resolve()
        meta = json.loads((original / 'run_meta.json').read_text(encoding='utf-8'))
        budget = Path(meta.get('budget_root', original / 'budget')).resolve()
        if not (budget / 'spend.json').is_file():
            raise ValueError('Original shared spend ledger must already exist')
        cases = []
        for origin, arm, cid, expected, rationale, field, quote in CASES:
            source = (original if origin == 'original' else revision) / arm / 'records' / cid / 'result.json'
            raw = source.read_bytes()
            row = json.loads(raw)['record']
            conversation = {k: row['response' if k == 'final' else k] for k in FIELDS}
            if not all(isinstance(v, str) and v.strip() for v in conversation.values()):
                raise ValueError('Missing complete training conversation: ' + str(source))
            if quote not in conversation[field]:
                raise ValueError('Calibration evidence no longer matches: ' + str(source))
            cases.append(dict(id=f'{origin}_{arm}_{cid}', conversation=conversation,
                              expected=expected, rationale=rationale, evidence_field=field,
                              evidence_quote=quote, source_path=str(source),
                              source_sha256=runtime.digest(raw)))
        manifest = dict(schema=1, models=list(ALLOWED_MODELS), budget_root=str(budget),
                        allocation_key='narrow_critic_probe_allocation.json', max_probe_usd=12,
                        prompt=PROMPT.read_text(encoding='utf-8'), cases=cases,
                        code_sha256=runtime.digest(Path(__file__).read_bytes()),
                        scope='Eight purposively selected narrow content cases, four each label; no sensitivity or full-row quality estimate.')
        runtime.save_checkpoint(output / 'manifest.json', manifest)
        return manifest


def messages(manifest, case):
    return [{'role': 'system', 'content': manifest['prompt']},
            {'role': 'user', 'content': json.dumps(case['conversation'], ensure_ascii=False)}]


def validate_verdict(verdict, conversation):
    if set(verdict) != {'accepted', 'issues', 'assessment'} or type(verdict['accepted']) is not bool:
        raise ValueError('Invalid critic schema/decision')
    findings = verdict['issues']
    if not isinstance(findings, list) or len(findings) > 3 or not isinstance(verdict['assessment'], str):
        raise ValueError('Invalid findings/assessment')
    if (not verdict['accepted']) != bool(findings):
        raise ValueError('Reject must have findings; pass must have none')
    keys = {'kind', 'answer_field', 'answer_quote', 'premise_field', 'premise_quote',
            'material_consequence', 'alternative_reading'}
    kinds = {'changed_premise', 'unsupported_decisive_fact', 'constraint_violation',
             'categorical_false_claim', 'contradiction'}
    for finding in findings:
        if set(finding) != keys or finding['kind'] not in kinds:
            raise ValueError('Invalid finding schema')
        af, pf = finding['answer_field'], finding['premise_field']
        if af not in ('reasoning', 'final') or pf not in ('system', 'user', 'absent'):
            raise ValueError('Invalid quote field')
        if not isinstance(finding['answer_quote'], str) or not finding['answer_quote'] or finding['answer_quote'] not in conversation[af]:
            raise ValueError('Answer evidence is not an exact quote')
        pq = finding['premise_quote']
        if (pf == 'absent' and pq != '') or (pf != 'absent' and (not isinstance(pq, str) or not pq or pq not in conversation[pf])):
            raise ValueError('Premise evidence is not an exact quote')
        if any(not isinstance(finding[k], str) or not finding[k].strip() for k in ('material_consequence', 'alternative_reading')):
            raise ValueError('Missing materiality/alternative interpretation')
    return verdict


def allocation(manifest):
    budget = Path(manifest['budget_root'])
    # Persist at the shared budget root, so another model/output cannot reset the allowance.
    with FileLock(str(budget / 'spend.lock'), timeout=120):
        path = budget / manifest['allocation_key']
        if path.exists():
            value = runtime.load_checkpoint(path)
        else:
            entries = json.loads((budget / 'spend.json').read_text(encoding='utf-8'))
            exposure = sum(e['charged_or_reserved_usd'] for e in entries)
            if not math.isfinite(exposure) or exposure < 0:
                raise ValueError('Invalid existing shared exposure')
            value = {'starting_exposure_usd': exposure, 'allowance_usd': 12,
                     'absolute_ceiling_usd': min(250, exposure + 12)}
            runtime.save_checkpoint(path, value)
        if value['allowance_usd'] > 12 or value['absolute_ceiling_usd'] > min(250, value['starting_exposure_usd'] + 12):
            raise ValueError('Invalid probe budget allocation')
        return value


def run_probe(output, model=SONNET, send=None):
    output = Path(output)
    if model not in ALLOWED_MODELS:
        raise ValueError('Only Sonnet 5 or explicitly selected Gemini 3 Flash allowed; no Haiku')
    with FileLock(str(output / 'probe.lock'), timeout=1):
        manifest = runtime.load_checkpoint(output / 'manifest.json')
        if runtime.digest(Path(__file__).read_bytes()) != manifest['code_sha256']:
            raise ValueError('Probe code changed after freezing')
        if model not in manifest['models']:
            raise ValueError('Model absent from frozen plan')
        cap = allocation(manifest)
        client = runtime.BudgetClient(manifest['budget_root'], cap['absolute_ceiling_usd'], {model}, send=send)
        client.local.arm = 'narrow_critic_probe'
        client.local.run_root = str(output.resolve())
        client.local.stage = 'blind_content_critic'
        results = []
        for case in manifest['cases']:
            path = output / model.replace('/', '__') / (case['id'] + '.json')
            if path.exists():
                result = runtime.load_checkpoint(path)
                # A started/failed record is retained, never silently repeated.
                results.append(result)
                continue
            client.local.candidate_id = case['id']
            request = messages(manifest, case)
            result = dict(id=case['id'], expected=case['expected'], status='started', model=model,
                          request_sha256=runtime.digest(request))
            runtime.save_checkpoint(path, result)
            try:
                response = client.chat(model=model, messages=request, temperature=0, max_tokens=2400,
                                       extra_body={'reasoning': {'enabled': False}})
                result['response'] = asdict(response)
                verdict = validate_verdict(runtime._parse_json(response.content), case['conversation'])
                result.update(status='complete', verdict=verdict,
                              matches_expected=verdict['accepted'] == (case['expected'] == 'pass'))
            except BaseException as exc:
                result.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:1500])
                runtime.save_checkpoint(path, result)
                if isinstance(exc, (runtime.BudgetStop, KeyboardInterrupt, SystemExit)):
                    raise
            runtime.save_checkpoint(path, result)
            results.append(result)
        summary = dict(model=model, allocation=cap, cases=results,
                       complete=sum(r['status'] == 'complete' for r in results),
                       correct=sum(r.get('matches_expected', False) for r in results),
                       false_rejects=[r['id'] for r in results if r['status'] == 'complete' and r['expected'] == 'pass' and not r['matches_expected']],
                       missed_defects=[r['id'] for r in results if r['status'] == 'complete' and r['expected'] == 'reject' and not r['matches_expected']],
                       scope=manifest['scope'])
        runtime.save_checkpoint(output / model.replace('/', '__') / 'summary.json', summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--output', required=True)
    prep.add_argument('--original', default='output/2026-09-14_dataset_refresh')
    prep.add_argument('--revision', default='output/2026-09-14_dataset_refresh_revision')
    run = sub.add_parser('run')
    run.add_argument('--output', required=True)
    run.add_argument('--model', choices=ALLOWED_MODELS, default=SONNET)
    run.add_argument('--execute', action='store_true', help='Explicitly dispatch paid calls after authorization')
    args = parser.parse_args()
    if args.command == 'prepare':
        manifest = prepare(args.output, args.original, args.revision)
        print(json.dumps({'frozen_cases': len(manifest['cases']), 'budget_root': manifest['budget_root']}))
    else:
        if not args.execute:
            parser.error('Paid dispatch requires --execute')
        summary = run_probe(args.output, args.model)
        print(json.dumps({k: v for k, v in summary.items() if k != 'cases'}))


if __name__ == '__main__':
    main()
