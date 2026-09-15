# ABOUTME: Execute an explicit per-trait candidate plan through the unchanged imported generation pipeline.
# ABOUTME: Prioritizes valid saved stages, freezes selected identities and invocation, and preserves shared spending limits.
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import re
import subprocess
import threading
import uuid

from filelock import FileLock

from scratch.dataset_refresh import run as runtime, per_row


def resolve(plan):
    if not isinstance(plan, dict) or set(plan) - {'root', 'arm', 'ceiling', 'workers', 'candidate_counts', 'candidate_ids'}:
        raise ValueError('Unexpected targeted plan fields')
    if ('candidate_counts' in plan) == ('candidate_ids' in plan):
        raise ValueError('Supply exactly one of candidate_counts or candidate_ids')
    root, arm = Path(plan['root']).resolve(), plan['arm']
    ceiling, workers = plan['ceiling'], plan.get('workers', 8)
    if type(ceiling) not in (int, float) or not math.isfinite(ceiling) or not 0 < ceiling <= 250:
        raise ValueError('Ceiling must remain within the cumulative250 cap')
    if type(workers) is not int or workers <= 0:
        raise ValueError('Positive integer worker count required')
    meta = runtime.load_checkpoint(root / 'run_meta.json')
    if arm not in meta['arms']:
        raise ValueError('Arm absent from frozen root')
    cfg = runtime.validate_arm(root, arm)
    if cfg.get('per_row_regime') is not True:
        raise ValueError('Targeted execution requires the frozen per-row regime')
    if per_row.base is not runtime:
        raise ValueError('Imported runner module identity differs')
    for name, field in [('run.py', 'code_sha256'), ('per_row.py', 'code_per_row_sha256'), ('reviewer_probe.py', 'critic_validator_sha256')]:
        if meta.get(field) != runtime.digest(Path(__file__).with_name(name).read_bytes()):
            raise ValueError('Frozen implementation changed: ' + name)
    per_row.assert_models(cfg)
    candidates = runtime.read_rows(root / arm / 'candidates.jsonl')
    by_id = {c['candidate_id']: c for c in candidates}
    if len(by_id) != len(candidates):
        raise ValueError('Duplicate frozen candidate IDs')
    accepted = {t: 0 for t in runtime.quotas()}
    used_sources = set()
    for path in (root / arm / 'records').glob('*/result.json'):
        result = runtime.load_result(path)
        if result['status'] == 'accepted':
            accepted[result['trait_id']] += 1
            used_sources.add(result['record']['source_id'])
    saved = {}
    available = []
    for index, candidate in enumerate(candidates):
        cid = candidate['candidate_id']
        row = root / arm / 'records' / cid
        if (row / 'result.json').exists() or (row / 'independent_exclusion.json').exists():
            continue
        if cfg.get('source', {}).get('unique_parent_ids') and candidate['source_id'] in used_sources:
            continue
        if (row / 'identity.json').exists() and runtime.load_checkpoint(row / 'identity.json') != {
                'candidate_sha256': runtime.digest(candidate), 'config_sha256': runtime.digest(cfg)}:
            raise ValueError('Saved partial candidate identity differs: ' + cid)
        checkpoints = {}
        for path in row.glob('*.json'):
            stage = path.stem
            if stage in {'scenario', 'preflight', *(s['name'] for s in cfg['response_stages'])} or re.fullmatch(r'(grounding|review|repair)_\d+', stage):
                runtime.load_checkpoint(path)
                checkpoints[stage] = runtime.digest(path.read_bytes())
        saved[cid] = checkpoints
        available.append((index, candidate))
    if 'candidate_ids' in plan:
        ids = plan['candidate_ids']
        if not isinstance(ids, list) or not all(isinstance(cid, str) for cid in ids) or len(set(ids)) != len(ids):
            raise ValueError('Explicit candidate_ids must be a unique list')
        usable = {c['candidate_id'] for _, c in available}
        if set(ids) - usable:
            raise ValueError('Explicit plan includes unknown, completed, excluded or unavailable candidates')
        chosen = [by_id[cid] for cid in ids]
    else:
        counts = plan['candidate_counts']
        if not isinstance(counts, dict) or set(counts) != set(runtime.quotas()) or any(type(n) is not int or n < 0 for n in counts.values()):
            raise ValueError('candidate_counts requires all nine traits and nonnegative integers')
        # Prefer the most already-paid stages within each requested trait, with frozen order as tie-break.
        ranked = sorted(available, key=lambda item: (-len(saved[item[1]['candidate_id']]), item[0]))
        chosen, remaining = [], dict(counts)
        for _, candidate in ranked:
            trait = candidate['trait_id']
            if not remaining[trait]:
                continue
            if cfg.get('source', {}).get('unique_parent_ids') and candidate['source_id'] in used_sources:
                continue
            chosen.append(candidate)
            remaining[trait] -= 1
            if cfg.get('source', {}).get('unique_parent_ids'):
                used_sources.add(candidate['source_id'])
        if any(remaining.values()):
            raise ValueError('Not enough eligible pending candidates for exact requested counts: ' + json.dumps(remaining))
    counts = {t: sum(c['trait_id'] == t for c in chosen) for t in runtime.quotas()}
    if any(accepted[t] + counts[t] > runtime.quotas()[t] for t in counts):
        raise ValueError('Candidate plan could exceed this phase trait quota')
    if cfg.get('source', {}).get('unique_parent_ids') and len({c['source_id'] for c in chosen}) != len(chosen):
        raise ValueError('Candidate plan repeats a parent source')
    record = {'phase': 'production', 'kind': 'explicit_targeted_candidate_plan', 'root': str(root),
              'arms': [arm], 'jobs': len(chosen), 'candidate_counts': counts, 'plan': plan,
              'plan_sha256': runtime.digest(plan), 'config_sha256': runtime.digest(cfg),
              'run_meta_sha256': runtime.digest((root / 'run_meta.json').read_bytes()),
              'budget_root': meta['budget_root'], 'ceiling_usd': ceiling, 'workers': workers,
              'code_sha256': runtime.digest(Path(runtime.__file__).read_bytes()),
              'per_row_code_sha256': runtime.digest(Path(per_row.__file__).read_bytes()),
              'wrapper_sha256': runtime.digest(Path(__file__).read_bytes()),
              'selection_rule': 'Requested counts only; most saved paid stages first, frozen candidate order for ties. Existing terminals and independent exclusions never reopened.',
              'candidates': [{'candidate_id': c['candidate_id'], 'trait_id': c['trait_id'],
                              'source_id': c['source_id'], 'candidate_sha256': runtime.digest(c),
                              'saved_stage_sha256': saved[c['candidate_id']]} for c in chosen]}
    return record, cfg, chosen


def execute(plan):
    root = Path(plan['root']).resolve()
    with FileLock(str(root / 'execution.lock'), timeout=1):
        phase, cfg, candidates = resolve(plan)
        phase.update(started_at=runtime.timestamp(), git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
        name = f'targeted_{phase["started_at"]}_{uuid.uuid4().hex[:12]}'
        runtime.save_checkpoint(root / 'phases' / (name + '.json'), phase)
        runtime.write_json(root / 'active_phase.json', phase)
        if not candidates:
            return {'jobs': 0, 'stopped_before_next_dispatch': False}
        client = runtime.BudgetClient(Path(phase['budget_root']), phase['ceiling_usd'], {m['model'] for m in cfg['models'].values()})
        stop = threading.Event()
        errors, completed = [], []
        arm = phase['arms'][0]
        def task(candidate):
            if stop.is_set():
                return None
            try:
                return runtime.generate_one(root, arm, candidate, client)
            except BaseException:
                stop.set()
                raise
        with ThreadPoolExecutor(max_workers=phase['workers']) as pool:
            futures = [pool.submit(task, c) for c in candidates]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        item = {k: result.get(k) for k in ('candidate_id', 'trait_id', 'status', 'error')}
                        completed.append(item)
                        print(json.dumps(item), flush=True)
                except runtime.BudgetStop as exc:
                    errors.append(str(exc))
        status = runtime.status(root)
        runtime.write_json(root / 'status.json', status)
        outcome = {'phase_file': name + '.json', 'jobs': len(candidates), 'completed': completed,
                   'stopped_before_next_dispatch': bool(errors), 'stop_reasons': errors, 'status': status}
        runtime.save_checkpoint(root / 'phases' / (name + '_outcome.json'), outcome)
        return outcome


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--execute', action='store_true', help='Explicitly dispatch planned work; default only resolves the plan')
    parser.add_argument('--output', type=Path, help='Optional read-only preview report outside the source root')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if args.output and args.output.resolve().is_relative_to(Path(plan['root']).resolve()):
        parser.error('--output must be outside the frozen root')
    result = execute(plan) if args.execute else resolve(plan)[0]
    if args.output:
        runtime.write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
    if args.execute and result.get('stopped_before_next_dispatch'):
        raise SystemExit(2)
