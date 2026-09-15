# ABOUTME: Source-first critic comparison on unchanged frozen examples, with default model reasoning.
# ABOUTME: Preserves v1 evidence and caps this Sonnet-only revision at $2 within the shared allocation.
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from filelock import FileLock

from scratch.dataset_refresh import reviewer_probe as base

runtime = base.runtime
PROMPT = Path(__file__).with_name('narrow_critic_prompt_v2.txt')


def prepare(output, baseline):
    output, baseline = Path(output), Path(baseline)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / 'probe.lock')):
        if (output / 'manifest.json').exists():
            raise ValueError('V2 already frozen')
        manifest = runtime.load_checkpoint(baseline / 'manifest.json')
        manifest.update(prompt=PROMPT.read_text(encoding='utf-8'),
                        code_sha256=runtime.digest(Path(__file__).read_bytes()),
                        baseline_manifest_sha256=runtime.digest((baseline / 'manifest.json').read_bytes()),
                        helper_sha256=runtime.digest(Path(base.__file__).read_bytes()),
                        models=[base.SONNET], reasoning='provider_default_no_override')
        cap = base.allocation(manifest)
        budget = Path(manifest['budget_root'])
        with FileLock(str(budget / 'spend.lock')):
            path = budget / 'narrow_critic_probe_v2_allocation.json'
            if path.exists():
                subcap = runtime.load_checkpoint(path)
            else:
                exposure = sum(e['charged_or_reserved_usd'] for e in json.loads((budget / 'spend.json').read_text()))
                subcap = dict(starting_exposure_usd=exposure, allowance_usd=2,
                              absolute_ceiling_usd=min(cap['absolute_ceiling_usd'], exposure + 2, 250))
                runtime.save_checkpoint(path, subcap)
        manifest['v2_allocation'] = subcap
        runtime.save_checkpoint(output / 'manifest.json', manifest)
        return manifest


def run_probe(output, send=None):
    output = Path(output)
    with FileLock(str(output / 'probe.lock'), timeout=1):
        manifest = runtime.load_checkpoint(output / 'manifest.json')
        if manifest['code_sha256'] != runtime.digest(Path(__file__).read_bytes()) or manifest['helper_sha256'] != runtime.digest(Path(base.__file__).read_bytes()):
            raise ValueError('Frozen code changed')
        cap = manifest['v2_allocation']
        if cap['allowance_usd'] != 2 or cap['absolute_ceiling_usd'] > min(cap['starting_exposure_usd'] + 2, 250, base.allocation(manifest)['absolute_ceiling_usd']):
            raise ValueError('Invalid v2 budget')
        client = runtime.BudgetClient(manifest['budget_root'], cap['absolute_ceiling_usd'], {base.SONNET}, send=send)
        client.local.arm, client.local.stage = 'narrow_critic_probe_v2', 'source_first_content_critic'
        client.local.run_root = str(output.resolve())
        results = []
        for case in manifest['cases']:
            path = output / (case['id'] + '.json')
            if path.exists():
                results.append(runtime.load_checkpoint(path))
                continue
            client.local.candidate_id = case['id']
            request = base.messages(manifest, case)
            result = dict(id=case['id'], expected=case['expected'], status='started',
                          model=base.SONNET, request_sha256=runtime.digest(request))
            runtime.save_checkpoint(path, result)
            try:
                response = client.chat(model=base.SONNET, messages=request, temperature=0, max_tokens=3200)
                result['response'] = asdict(response)
                verdict = base.validate_verdict(runtime._parse_json(response.content), case['conversation'])
                result.update(status='complete', verdict=verdict,
                              matches_expected=verdict['accepted'] == (case['expected'] == 'pass'))
            except BaseException as exc:
                result.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:1500])
                runtime.save_checkpoint(path, result)
                if isinstance(exc, (runtime.BudgetStop, KeyboardInterrupt, SystemExit)):
                    raise
            runtime.save_checkpoint(path, result)
            results.append(result)
        summary = dict(model=base.SONNET, reasoning=manifest['reasoning'], cases=results,
                       complete=sum(r['status'] == 'complete' for r in results),
                       correct=sum(r.get('matches_expected', False) for r in results),
                       false_rejects=[r['id'] for r in results if r['status'] == 'complete' and r['expected'] == 'pass' and not r['matches_expected']],
                       missed_defects=[r['id'] for r in results if r['status'] == 'complete' and r['expected'] == 'reject' and not r['matches_expected']],
                       scope=manifest['scope'])
        runtime.save_checkpoint(output / 'summary.json', summary)
        return summary


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--output', required=True)
    prep.add_argument('--baseline', required=True)
    run = sub.add_parser('run')
    run.add_argument('--output', required=True)
    run.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        m = prepare(args.output, args.baseline)
        print(json.dumps({'cases': len(m['cases']), 'allocation': m['v2_allocation']}))
    else:
        if not args.execute:
            parser.error('Paid dispatch requires --execute')
        summary = run_probe(args.output)
        print(json.dumps({k: v for k, v in summary.items() if k != 'cases'}))


if __name__ == '__main__':
    main()
