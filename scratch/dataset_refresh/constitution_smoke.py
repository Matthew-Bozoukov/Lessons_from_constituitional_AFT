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
        self.budget = BudgetClient(self.root / 'budget', cfg['budget_usd'],
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


def calibrate(cfg, root, client, fixture_path):
    cases = OmegaConf.to_container(OmegaConf.load(fixture_path), resolve=True)['cases']
    traits = {t.as_trait().trait_id: t.as_trait().text for t in units_from_config(cfg)}
    spec = next(s for s in cfg['stages'] if s['name'] == 'review_responses')
    model = cfg['models'][spec['model']]
    def one(case):
        fields = {**case, 'scenario_id': 'calibration_' + case['id'], 'trait_text': traits[case['trait_id']]}
        result = client.chat(model=model['model'], temperature=model['temperature'], max_tokens=model['max_tokens'],
            extra_body=model['extra_body'], messages=[{'role': 'system', 'content': spec['prompts']['system']},
            {'role': 'user', 'content': spec['prompts']['user'].format(**fields)}])
        review = validate_review(_parse_json(result.content)['review'], cfg['smoke_contract'].get('review_fields', []))
        correct = review['verdict'] == case['expected'] and (not case.get('expected_code') or
            case['expected_code'] in [f['code'] for f in review['findings']])
        value = {'case': case, 'review': review, 'correct': correct}
        write_json(root / 'calibration' / (case['id'] + '.json'), value)
        return value
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        outcomes = list(pool.map(one, cases))
    write_json(root / 'calibration_results.json', outcomes)
    if not all(o['correct'] for o in outcomes):
        raise RuntimeError('Reviewer calibration failed; stop without generating or changing the rubric')
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
    fixture = path.with_name('constitution_smoke_calibration.yaml')
    for source in (path, Path(__file__), fixture, Path(cfg['constitution']), Path('configs/endpoints/providers.yaml')):
        shutil.copy2(source, root / source.name)
    write_json(root / 'run_meta.json', {'git_sha': git_sha(), 'config': cfg, 'status': 'prepared',
        'only_generation_content_input': cfg['constitution'], 'calibration_not_author_input': True,
        'hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (path, Path(__file__), fixture, Path(cfg['constitution']), Path('configs/endpoints/providers.yaml'))}})
    print('SMOKE_ROOT=' + str(root.resolve()), flush=True)
    client = SingleAttemptClient(root, cfg)
    try:
        count = calibrate(cfg, root, client, fixture)
        print(f'Calibration passed {count}/{count}. Starting 18 candidates.', flush=True)
        generation = root / 'generation'
        generation.mkdir()
        manifest = pipeline.run(cfg, smoke=True, resume=str(generation), client=client)
        rows = [json.loads(line) for line in (generation / 'dataset.jsonl').read_text(encoding='utf-8').splitlines()]
        for row in rows:
            validate_review(row['metadata']['review'], cfg['smoke_contract'].get('review_fields', []))
        write_json(root / 'automatic_summary.json', {'completed': len(rows),
            'model_pass': sum(r['metadata']['review']['verdict'] == 'pass' for r in rows),
            'independent_review': 'pending; this is not a training-ready release', 'manifest': manifest})
    except BaseException as exc:
        write_json(root / 'stopped.json', {'exception': type(exc).__name__, 'message': str(exc),
            'retry_allowed': False})
        raise
    finally:
        entries = client.budget.entries()
        write_json(root / 'cost_summary.json', {'physical_calls': len(entries),
            'charged_or_reserved_usd': sum(e['charged_or_reserved_usd'] for e in entries),
            'statuses': {s: sum(e['status'] == s for e in entries) for s in {e['status'] for e in entries}}})
        print('SMOKE_ROOT=' + str(root.resolve()), flush=True)


if __name__ == '__main__':
    main()
