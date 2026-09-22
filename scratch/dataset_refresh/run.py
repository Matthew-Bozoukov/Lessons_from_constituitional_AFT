# ABOUTME: Bounded parallel regeneration of moral low-stakes and nonmoral human-advice data.
# ABOUTME: Pins inputs, checkpoints every call and stage, shares a durable budget, and preserves failures.
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import threading
import time

from filelock import FileLock
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from tenacity import stop_after_attempt

from src.data.synth.ours.constitution import full_text
from src.data.synth.ours.stage_runtime import _parse_json, _parse_tagged, lint_problems
from src.infra.endpoints.openrouter import OpenRouterClient, provider_pin, provider_price
from src.infra.huggingface import push_run_dir, training_data_tags
from src.naming import artifact_name, synth_name
from src.utils import timestamp


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    for attempt in range(6):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * 2 ** attempt)


def write_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


class BudgetStop(RuntimeError):
    pass


class BudgetClient:
    """One physical call per reservation; file lock also protects separate processes."""
    def __init__(self, root, ceiling, approved_models, send=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'spend.json'
        self.lock = FileLock(str(self.root / 'spend.lock'), timeout=120)
        self.ceiling = float(ceiling)
        if not math.isfinite(self.ceiling) or not 0 < self.ceiling <= 250:
            raise ValueError('Ceiling must stay inside the user-authorized $250 maximum')
        self.models = set(approved_models)
        self.local = threading.local()
        if send is None:
            api = OpenRouterClient()
            single = api.chat.retry_with(stop=stop_after_attempt(1))
            self.send = lambda **kw: single(api, **kw)
        else:
            self.send = send

    def entries(self):
        return json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else []

    def chat(self, model, messages, temperature, max_tokens, extra_body=None):
        if model not in self.models:
            raise ValueError('Unapproved generator: ' + model)
        price = provider_price(model)
        if (not price or any(not math.isfinite(p) or p <= 0 for p in price.values())
                or type(max_tokens) is not int or max_tokens <= 0):
            raise ValueError('Missing positive model price/output limit')
        if any(not isinstance(m.get('content'), str) for m in messages):
            raise ValueError('Budget requires plain text messages')
        # Byte count upper-bounds text tokens, plus generous framing. Account for cache writes.
        input_bound = len(json.dumps(messages, ensure_ascii=False).encode()) + len(json.dumps(extra_body or {},ensure_ascii=False).encode()) + 2048
        reserve = (1.25 * input_bound * price['in'] + max_tokens * price['out']) / 1e6
        req = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if extra_body:
            reasoning = extra_body.get('reasoning', {})
            if set(extra_body)-{'reasoning','response_format'}:
                raise ValueError('Unbounded or unapproved request extension')
            response_format=extra_body.get('response_format')
            if response_format and not (response_format.get('type')=='json_schema' and response_format.get('json_schema',{}).get('strict') is True and isinstance(response_format['json_schema'].get('schema'),dict)):
                raise ValueError('Only explicit strict JSON schemas are supported')
            bounded_thinking = (isinstance(reasoning, dict)
                and set(reasoning) == {'max_tokens'} and type(reasoning['max_tokens']) is int
                and 1024 <= reasoning['max_tokens'] < max_tokens and model.startswith('anthropic/'))
            if reasoning != {'enabled': False} and not bounded_thinking:
                raise ValueError('Only reasoning-off or a bounded Anthropic thinking allocation is supported')
            req['extra_body'] = extra_body
        with self.lock:
            entries = self.entries()
            if any(not math.isfinite(e['charged_or_reserved_usd']) or e['charged_or_reserved_usd'] < 0 for e in entries):
                raise BudgetStop('Invalid ledger amount; halt without dispatch')
            exposure = sum(e['charged_or_reserved_usd'] for e in entries)
            if any(e['status'] == 'bound_exceeded' for e in entries):
                raise BudgetStop('A previous request invalidated the budget bound')
            if exposure + reserve > self.ceiling:
                raise BudgetStop(f'Budget stop: ${exposure:.3f} exposed, ${reserve:.3f} next reservation, ceiling ${self.ceiling}')
            call_id = len(entries)
            entry = dict(call_id=call_id, model=model, arm=getattr(self.local, 'arm', None),
                         run_root=getattr(self.local, 'run_root', None),
                         candidate_id=getattr(self.local, 'candidate_id', None),
                         stage=getattr(self.local, 'stage', None), status='reserved',
                         charged_or_reserved_usd=reserve, reserve_usd=reserve,
                         input_bound=input_bound, output_bound=max_tokens,
                         request_sha256=digest(req), started_at=timestamp())
            entries.append(entry)
            write_json(self.path, entries)
            raw_path = self.root / 'raw_calls' / f'{call_id:06d}.json'
            write_json(raw_path, {'request': req, 'accounting': entry})
        try:
            result = self.send(**req)
        except BaseException as exc:
            with self.lock:
                entries = self.entries()
                entries[call_id].update(status='uncertain_failure', exception_type=type(exc).__name__,
                                        error=str(exc)[:1500], finished_at=timestamp())
                write_json(self.path, entries)
                write_json(raw_path, {'request': req, 'accounting': entries[call_id],
                                      'diagnostics': getattr(exc, 'diagnostics', None)})
            raise
        observed = result.cost
        valid_usage = (type(result.prompt_tokens) is int and type(result.completion_tokens) is int and
                       0 < result.prompt_tokens <= input_bound and 0 < result.completion_tokens <= max_tokens)
        estimate = ((1.25 * result.prompt_tokens * price['in'] + result.completion_tokens * price['out']) / 1e6
                    if valid_usage else reserve)
        try:
            charge = float(observed) if observed is not None else estimate
        except (TypeError, ValueError):
            charge = float('nan')
        invalid = (not valid_usage or not math.isfinite(charge) or
                   charge > reserve + 1e-9 or charge < 0)
        with self.lock:
            entries = self.entries()
            entries[call_id].update(status='bound_exceeded' if invalid else 'settled',
                                   charged_or_reserved_usd=(max(charge, reserve) if math.isfinite(charge) else reserve) if invalid else charge,
                                   api_reported_cost_usd=observed, conservative_estimate_usd=estimate,
                                   prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
                                   provider=result.provider, finished_at=timestamp())
            write_json(self.path, entries)
            write_json(raw_path, {'request': req, 'response': asdict(result), 'accounting': entries[call_id]})
        if invalid:
            raise BudgetStop('Provider usage exceeded reservation or was missing; preserved and stopped')
        if result.finish_reason in ('length', 'content_filter'):
            raise ValueError('Excluded incomplete/provider-filtered output: ' + result.finish_reason)
        return result


def pin_source(spec):
    if not re.fullmatch('[0-9a-f]{40}', spec['revision']):
        raise ValueError('Every source must pin a full commit')
    path = hf_hub_download(spec['repo'], spec.get('file', 'dataset.jsonl'),
                          repo_type='dataset', revision=spec['revision'])
    sha = digest(Path(path).read_bytes())
    expected = spec.get('sha256') or spec.get('dataset_sha256')
    if expected and expected != sha:
        raise ValueError('Pinned source bytes differ from declared hash')
    return read_rows(path), {**spec, 'sha256': sha}


def flatten(row):
    if 'messages' not in row:
        return dict(row)
    out = dict(row.get('metadata') or {})
    for msg in row['messages']:
        if msg['role'] in ('system', 'user'):
            out[msg['role']] = msg.get('content', '')
        elif msg['role'] == 'assistant':
            out['source_reasoning'] = msg.get('reasoning_content', '')
            out['source_response'] = msg.get('content', '')
    return out


def quotas(n=716):
    return {f't{i+1}': n // 9 + int(i < n % 9) for i in range(9)}


def load_config(path):
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if Path(path).stem != cfg['pipeline']:
        raise ValueError('Config stem must equal pipeline style')
    return cfg


def prepare(config_paths, root, budget_root=None, revision_of=None, pilot_offset=0):
    if config_paths and len({bool(load_config(p).get('per_row_regime')) for p in config_paths}) > 1:
        raise ValueError('Cannot mix legacy and per-row regimes in one run')
    if config_paths and all(load_config(p).get('per_row_regime') for p in config_paths):
        from scratch.dataset_refresh.per_row import prepare as prepare_per_row
        return prepare_per_row(config_paths, root, budget_root)
    if revision_of:
        with FileLock(str(Path(revision_of) / 'recipe_revision.lock'), timeout=1):
            receipt = Path(revision_of) / 'recipe_revision.json'
            if receipt.exists():
                raise ValueError('The single authorized recipe revision has already been prepared')
            result = _prepare(config_paths, root, budget_root, revision_of, pilot_offset)
            save_checkpoint(receipt, {'revision_root': str(Path(root).resolve()),
                                     'run_meta_sha256': digest((Path(root) / 'run_meta.json').read_bytes())})
            return result
    return _prepare(config_paths, root, budget_root, revision_of, pilot_offset)


def _prepare(config_paths, root, budget_root=None, revision_of=None, pilot_offset=0):
    root = Path(root)
    if (root / 'run_meta.json').exists():
        raise ValueError('Already prepared; use the existing frozen run instead of replacing provenance')
    root.mkdir(parents=True, exist_ok=True)
    if revision_of:
        previous = json.loads((Path(revision_of) / 'run_meta.json').read_text(encoding='utf-8'))
        if previous.get('revision_of'):
            raise ValueError('Only one recipe revision is authorized')
        inherited_budget = Path(previous.get('budget_root', str(Path(revision_of).resolve() / 'budget')))
        if budget_root and Path(budget_root).resolve() != inherited_budget.resolve():
            raise ValueError('Recipe revisions must share the original budget ledger')
        budget_root = inherited_budget
        if pilot_offset < 18:
            raise ValueError('Revised pilot must exclude the original first18 candidates')
    elif pilot_offset:
        raise ValueError('Pilot offsets are only for an explicit recipe revision')
    frozen = {}
    for path in config_paths:
        cfg = load_config(path)
        source, pin = pin_source(cfg['source'])
        source = [flatten(r) for r in source]
        joins = []
        for filename in cfg['source'].get('join_files', []):
            joined, join_pin = pin_source({'repo': pin['repo'], 'revision': pin['revision'], 'file': filename})
            joins.append(join_pin)
            by_id = {r['scenario_id']: r for r in joined}
            for r in source:
                if r['scenario_id'] in by_id:
                    r.update(by_id[r['scenario_id']])
        if cfg['source'].get('exported_file'):
            exported, exported_pin = pin_source({'repo': pin['repo'], 'revision': pin['revision'],
                                                'file': cfg['source']['exported_file']})
            joins.append(exported_pin)
            exported_ids = {flatten(r)['scenario_id'] for r in exported}
            for r in source:
                r['parent_exported'] = r['scenario_id'] in exported_ids
        pin['joined_sources'] = joins
        if {r['trait_id'] for r in source} != set(quotas()):
            raise ValueError('Source must contain exactly nine numbered targets')
        cfg['source'] = pin
        actual_constitution = digest(full_text(cfg['constitution']).encode())
        if cfg.get('constitution_sha256') and cfg['constitution_sha256'] != actual_constitution:
            raise ValueError('Declared constitution does not match current generation text')
        cfg['constitution_sha256'] = actual_constitution
        cfg['review_constitution_text'] = full_text(cfg['constitution'])
        if cfg.get('craft_spec'):
            cfg['craft_spec_sha256'] = digest(full_text(cfg['craft_spec']).encode())
            if cfg.get('use_operational_craft_traits'):
                from src.data.synth.ours.constitution import chunk
                cfg['operational_traits'] = {c.parent_id: c.text for c in chunk(cfg['craft_spec'])}
                if set(cfg['operational_traits']) != set(quotas()):
                    raise ValueError('Operational craft preference must retain exactly nine targets')
        if cfg.get('original_craft_spec'):
            cfg['original_craft_spec_sha256'] = digest(full_text(cfg['original_craft_spec']).encode())
        cfg['config_path'] = str(path)
        cfg['source_config_sha256'] = digest(Path(path).read_bytes())
        arm_dir = root / cfg['pipeline']
        arm_dir.mkdir()
        write_json(arm_dir / 'config.json', cfg)
        write_rows(arm_dir / 'source.jsonl', source)
        candidates = []
        # Trait round-robin makes every fixed prefix balanced. Preserve all original seed identities.
        groups = {t: sorted([r for r in source if r['trait_id'] == t],
                            key=lambda r: digest(['refresh-20260914', r['scenario_id']])) for t in quotas()}
        for variant in range(int(cfg.get('selection', {}).get('max_variants_per_seed', 2))):
            for index in range(max(len(g) for g in groups.values())):
                for t in quotas():
                    if index >= len(groups[t]):
                        continue
                    r = groups[t][index]
                    candidates.append({'candidate_id': f'{t}_{index:03d}_v{variant}',
                                       'trait_id': t, 'source_id': r['scenario_id'], 'variant': variant,
                                       'source': r})
        candidates = candidates[:int(cfg.get('selection', {}).get('max_candidate_total', 1200))]
        if revision_of:
            old_candidates = read_rows(Path(revision_of) / cfg['pipeline'] / 'candidates.jsonl')
            original_pilot_ids = {r['source_id'] for r in old_candidates[:18]}
            revised_pilot_ids = {r['source_id'] for r in candidates[pilot_offset:pilot_offset + 18]}
            if len(revised_pilot_ids) != 18 or original_pilot_ids & revised_pilot_ids:
                raise ValueError('Revised pilot must use18 distinct sources disjoint from the original pilot')
        write_rows(arm_dir / 'candidates.jsonl', candidates)
        frozen[cfg['pipeline']] = {'config_sha256': digest(cfg), 'source': pin,
                                   'source_rows': len(source), 'candidate_count': len(candidates),
                                   'source_snapshot_sha256': digest((arm_dir / 'source.jsonl').read_bytes()),
                                   'candidates_sha256': digest((arm_dir / 'candidates.jsonl').read_bytes())}
    write_json(root / 'run_meta.json', {'created_at': timestamp(),
        'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'code_sha256': digest(Path(__file__).read_bytes()), 'arms': frozen,
        'budget_root': str(Path(budget_root or root / 'budget').resolve()),
        'revision_of': str(Path(revision_of).resolve()) if revision_of else None,
        'pilot_offset': pilot_offset,
        'target_per_arm': 716, 'replay_count': 9284, 'target_budget_usd': 200, 'hard_cap_usd': 250,
        'base': {'repo': 'dougalldeepmind/2026-09-08-nosynth-mix', 'file': 'mixture.jsonl',
                 'revision': '7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd'},
        'authorization': 'User approved 2026-09-14: two advice arms, new09; craft generation plus constitutional compatibility review; 716+9284; pilot self-approval; one design revision; publish; no training/eval'})


def render(template, fields):
    return template.format_map(fields)


def simple_checks(record):
    issues = []
    for field in ('system', 'user', 'reasoning', 'response'):
        if not isinstance(record.get(field), str) or not record[field].strip():
            issues.append('missing_' + field)
    text = '\n'.join(str(record.get(k, '')) for k in ('system', 'user', 'reasoning', 'response'))
    if re.search(r'\b(?:Anthropic|Claude)\b', text, re.I):
        issues.append('identity_leakage')
    if re.search(r'\[(?:INSERT|TODO|PLACEHOLDER)\b', text, re.I):
        issues.append('placeholder')
    if any(x in record.get('response', '') for x in ('<reasoning>', '<response>', '</think>')):
        issues.append('unparsed_tags')
    return issues


def acceptance(review, cfg):
    keys = cfg['acceptance']['required_true']
    if not isinstance(review, dict) or not all(review.get(k) is True for k in keys):
        return False
    gates = review.get('gates', {})
    if cfg['acceptance'].get('gates') and not isinstance(gates, dict):
        return False
    if any(gates.get(k) is not True for k in cfg['acceptance'].get('gates', [])):
        return False
    maximum = cfg['acceptance'].get('stakes_max')
    return maximum is None or (type(review.get('stakes')) is int and 0 <= review['stakes'] <= maximum)


def request_options(model):
    return {k: model[k] for k in ('model', 'temperature', 'max_tokens', 'extra_body') if k in model}


def save_checkpoint(path, value):
    write_json(path, value)
    write_json(Path(path).with_suffix('.receipt.json'), {'sha256': digest(Path(path).read_bytes())})


def load_checkpoint(path):
    path = Path(path)
    receipt = path.with_suffix('.receipt.json')
    if not receipt.exists() or json.loads(receipt.read_text())['sha256'] != digest(path.read_bytes()):
        raise BudgetStop('Missing or changed stage receipt: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def load_result(path):
    """Apply a documented independent exclusion without rewriting the original verdict."""
    result = load_checkpoint(path)
    exclusion = Path(path).parent / 'independent_exclusion.json'
    if exclusion.exists():
        note = load_checkpoint(exclusion)
        if note.get('result_sha256') != digest(Path(path).read_bytes()) or not note.get('reason'):
            raise BudgetStop('Unbound or unexplained independent exclusion')
        result = {**result, 'automated_status': result['status'], 'status': 'rejected',
                  'rejection_stage': 'independent_audit', 'independent_exclusion': note}
    return result


def validate_arm(root, arm):
    root = Path(root)
    meta = json.loads((root / 'run_meta.json').read_text(encoding='utf-8'))
    cfg = json.loads((root / arm / 'config.json').read_text(encoding='utf-8'))
    if cfg.get('per_row_regime'):
        meta = load_checkpoint(root / 'run_meta.json')
    if digest(cfg) != meta['arms'][arm]['config_sha256']:
        raise ValueError('Frozen recipe changed; preserve original and create an explicit revision')
    if digest(full_text(cfg['constitution']).encode()) != cfg['constitution_sha256']:
        raise ValueError('Frozen constitution changed')
    if cfg.get('craft_spec') and digest(full_text(cfg['craft_spec']).encode()) != cfg['craft_spec_sha256']:
        raise ValueError('Frozen craft specification changed')
    if cfg.get('original_craft_spec') and digest(full_text(cfg['original_craft_spec']).encode()) != cfg['original_craft_spec_sha256']:
        raise ValueError('Original craft specification changed')
    for filename, key in [('source.jsonl', 'source_snapshot_sha256'), ('candidates.jsonl', 'candidates_sha256')]:
        if digest((root / arm / filename).read_bytes()) != meta['arms'][arm][key]:
            raise ValueError('Frozen input changed: ' + filename)
    return cfg


def generate_one(root, arm, candidate, client):
    cfg = json.loads((root / arm / 'config.json').read_text(encoding='utf-8'))
    if cfg.get('per_row_regime'):
        from scratch.dataset_refresh.per_row import generate_one as generate_per_row
        return generate_per_row(root, arm, candidate, client)
    out_dir = root / arm / 'records' / candidate['candidate_id']
    out_dir.mkdir(parents=True, exist_ok=True)
    terminal = out_dir / 'result.json'
    identity = {'candidate_sha256': digest(candidate), 'config_sha256': digest(cfg)}
    identity_path = out_dir / 'identity.json'
    if identity_path.exists():
        if load_checkpoint(identity_path) != identity:
            raise BudgetStop('Candidate/config changed while resuming')
    else:
        save_checkpoint(identity_path, identity)
    if terminal.exists():
        return load_checkpoint(terminal)
    client.local.arm, client.local.candidate_id = arm, candidate['candidate_id']
    client.local.run_root = str(root.resolve())
    src = candidate['source']
    record = {k: src[k] for k in ('trait_id', 'trait_name', 'trait_text', 'domain') if k in src}
    record.update(scenario_id=candidate['candidate_id'], source_id=candidate['source_id'],
                  variant=candidate['variant'], parent_revision=cfg['source']['revision'])
    record['parent_exported'] = src.get('parent_exported')
    record['adapted_parent_id'] = candidate['source_id'] if not candidate['variant'] else None
    record['lineage_kind'] = 'source_adaptation' if not candidate['variant'] else 'fresh_unpaired_inspiration'
    record['original_context'] = {k: src[k] for k in ('system', 'user', 'situation', 'shortcut', 'domain') if k in src}
    if cfg.get('scenario_domains'):
        index = int(candidate['candidate_id'].split('_')[1])
        domain_index = (index * 7 + (int(candidate['trait_id'][1:]) - 1) * 2 + candidate['variant'] * 5) % len(cfg['scenario_domains'])
        record['assigned_domain_id'] = domain_index
        record['domain'] = cfg['scenario_domains'][domain_index]
    inspiration = src if not candidate['variant'] else {'domain': src['domain'],
        'trait_name': src['trait_name'], 'instruction': 'No matched source: create a fresh independent case in this domain.'}
    if cfg.get('scenario_source_fields') and not candidate['variant']:
        inspiration = {k: src[k] for k in cfg['scenario_source_fields'] if k in src}
    # Only the original artifact's target chunk guides generation; review receives the new constitution separately.
    fields = {**record, 'source_json': json.dumps(inspiration, ensure_ascii=False),
              'adaptation_mode': ('adapt the original mechanism' if not candidate['variant'] else
                'create a genuinely different self-contained situation inspired by this target and domain; no close paraphrase'),
              'style_guidance': cfg.get('style_guidance', '')}
    try:
        scenario_file = out_dir / 'scenario.json'
        if scenario_file.exists():
            generated = load_checkpoint(scenario_file)
        else:
            client.local.stage = 'scenario'
            model = cfg['models']['scenario']
            prompt = cfg['prompts']['scenario_user']
            messages = [{'role': 'system', 'content': render(cfg['prompts']['scenario_system'], fields)},
                        {'role': 'user', 'content': render(prompt, fields) + '\n\nAdaptation mode: ' + fields['adaptation_mode']}]
            result = client.chat(messages=messages, **request_options(model))
            generated = _parse_json(result.content)
            if not isinstance(generated, dict) or any(not generated.get(k) for k in ('system', 'user', 'situation', 'domain')):
                raise ValueError('Scenario JSON missing required fields')
            save_checkpoint(scenario_file, generated)
        protected = {'trait_id', 'trait_text', 'trait_name', 'scenario_id', 'source_id', 'variant', 'parent_revision',
                     'parent_exported', 'adapted_parent_id', 'lineage_kind', 'original_context', 'assigned_domain_id'}
        record.update({k: v for k, v in generated.items() if k not in protected})
        def extra_review(name, full_conversation):
            spec = cfg[name]
            file = out_dir / (name + '.json')
            if file.exists():
                return load_checkpoint(file)
            client.local.stage = name
            conversation = {k: record[k] for k in ('system', 'user', 'reasoning', 'response')
                            if k in record and (full_conversation or k in ('system', 'user'))}
            fields = {**record, 'conversation_json': json.dumps(conversation, ensure_ascii=False),
                      'record_json': json.dumps(record, ensure_ascii=False),
                      'metadata_json': json.dumps({k: v for k, v in record.items() if k not in
                         ('system', 'user', 'reasoning', 'response', 'draft_reasoning', 'draft_response')}, ensure_ascii=False),
                      'source_json': json.dumps(src, ensure_ascii=False),
                      'eligibility_json': json.dumps(eligibility, ensure_ascii=False)}
            if name == 'preflight':
                fields = {k: v for k, v in fields.items() if k in
                          ('system', 'user', 'trait_id', 'trait_name', 'trait_text', 'conversation_json')}
            messages = [{'role': role, 'content': render(spec['prompts'][role], fields)} for role in ('system', 'user')]
            verdict = _parse_json(client.chat(messages=messages, **request_options(cfg['models'][spec['model']])).content)
            save_checkpoint(file, verdict)
            return verdict
        eligibility = {}
        if cfg.get('preflight'):
            eligibility = extra_review('preflight', False)
            if not acceptance(eligibility, cfg['preflight']):
                result = {'candidate_id': candidate['candidate_id'], 'trait_id': candidate['trait_id'],
                          'status': 'rejected', 'rejection_stage': 'preflight', 'record': record, 'review': eligibility}
                save_checkpoint(terminal, result)
                return result
        for stage in cfg['response_stages']:
            client.local.stage = stage['name']
            stage_file = out_dir / (stage['name'] + '.json')
            if stage_file.exists():
                saved = load_checkpoint(stage_file)
            else:
                model = cfg['models'][stage['model']]
                fields = {**record, 'style_guidance': cfg.get('style_guidance', '')}
                messages = [{'role': role, 'content': render(stage['prompts'][role], fields)} for role in ('system', 'user')]
                result = client.chat(messages=messages, **request_options(model))
                parsed = _parse_tagged(result.content, tuple(stage['tags']))
                issues = lint_problems(parsed, stage.get('lint') or {}, record)
                # Fail a thin row instead of asking the generator to pad it until it passes a length floor.
                if issues:
                    write_json(out_dir / (stage['name'] + '_lint.json'), issues)
                    raise ValueError('Stage lint failed: ' + '; '.join(issues))
                saved = {dest: parsed[tag] for dest, tag in stage['save'].items()}
                save_checkpoint(stage_file, saved)
            record.update(saved)
        local_issues = simple_checks(record)
        if local_issues:
            raise ValueError('Local checks failed: ' + ', '.join(local_issues))
        review_file = out_dir / 'review.json'
        if review_file.exists():
            review = load_checkpoint(review_file)
        else:
            client.local.stage = 'review'
            fields = {**record, 'record_json': json.dumps(record, ensure_ascii=False),
                      'conversation_json': json.dumps({k: record[k] for k in ('system', 'user', 'reasoning', 'response')}, ensure_ascii=False),
                      'metadata_json': json.dumps({k: v for k, v in record.items() if k not in ('system', 'user', 'reasoning', 'response', 'draft_reasoning', 'draft_response')}, ensure_ascii=False),
                      'constitution': cfg.get('review_constitution_text') or full_text(cfg['constitution'])}
            fields['eligibility_json'] = json.dumps(eligibility, ensure_ascii=False)
            if cfg.get('review_blind_metadata'):
                fields['record_json'] = fields['conversation_json']
                fields['metadata_json'] = '{}'
                fields = {k: v for k, v in fields.items() if k in
                          ('system', 'user', 'reasoning', 'response', 'trait_id', 'trait_name', 'trait_text',
                           'record_json', 'conversation_json', 'metadata_json', 'constitution', 'eligibility_json')}
            messages = [{'role': role, 'content': render(cfg['prompts']['review_' + role], fields)} for role in ('system', 'user')]
            review = _parse_json(client.chat(messages=messages, **request_options(cfg['models']['review'])).content)
            save_checkpoint(review_file, review)
        result = {'candidate_id': candidate['candidate_id'], 'trait_id': candidate['trait_id'],
                  'status': 'accepted' if acceptance(review, cfg) else 'rejected',
                  'record': record, 'review': review}
        if result['status'] == 'accepted' and cfg.get('lineage_review'):
            lineage = extra_review('lineage_review', True)
            result['lineage_review'] = lineage
            if not acceptance(lineage, cfg['lineage_review']):
                result.update(status='rejected', rejection_stage='lineage_review')
        save_checkpoint(terminal, result)
        return result
    except BudgetStop:
        raise
    except Exception as exc:
        result = {'candidate_id': candidate['candidate_id'], 'trait_id': candidate['trait_id'],
                  'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc), 'record': record}
        save_checkpoint(terminal, result)
        return result


def status(root):
    root = Path(root)
    meta = json.loads((root / 'run_meta.json').read_text(encoding='utf-8'))
    out = {}
    for arm in meta['arms']:
        results = [load_result(p) for p in (root / arm / 'records').glob('*/result.json')]
        accepted = [r for r in results if r['status'] == 'accepted']
        out[arm] = {'attempted': len(results), 'accepted': len(accepted),
                    'by_trait': {t: sum(r['trait_id'] == t for r in accepted) for t in quotas()},
                    'rejected': sum(r['status'] == 'rejected' for r in results),
                    'failed': sum(r['status'] == 'failed' for r in results)}
    ledger = Path(meta.get('budget_root', str(root / 'budget'))) / 'spend.json'
    entries = json.loads(ledger.read_text(encoding='utf-8')) if ledger.exists() else []
    out['budget'] = {'calls': len(entries), 'charged_or_reserved_usd': sum(e['charged_or_reserved_usd'] for e in entries),
                     'api_reported_usd': sum(e.get('api_reported_cost_usd') or 0 for e in entries),
                     'unsettled': sum(e['status'] != 'settled' for e in entries)}
    return out


def execute(root, phase, ceiling, workers=8, arms=None, per_trait=2, batch_limit=180):
    with FileLock(str(Path(root) / 'execution.lock'), timeout=1):
        return _execute(root, phase, ceiling, workers, arms, per_trait, batch_limit)


def _execute(root, phase, ceiling, workers=8, arms=None, per_trait=2, batch_limit=180):
    if phase == 'pilot' and (ceiling > 20 or per_trait != 2):
        raise ValueError('Approved pilot is two per trait with a combined $20 ceiling')
    root = Path(root)
    meta = json.loads((root / 'run_meta.json').read_text(encoding='utf-8'))
    arms = arms or list(meta['arms'])
    jobs, models = [], set()
    current = status(root)
    for arm in arms:
        cfg = validate_arm(root, arm)
        if cfg.get('per_row_regime'):
            from scratch.dataset_refresh.per_row import assert_models
            assert_models(cfg)
            if (meta.get('code_sha256') != digest(Path(__file__).read_bytes()) or
                meta.get('code_per_row_sha256') != digest((Path(__file__).parent / 'per_row.py').read_bytes()) or
                meta.get('critic_validator_sha256') != digest((Path(__file__).parent / 'reviewer_probe.py').read_bytes())):
                raise ValueError('Frozen implementation changed; prepare a new run with preserved provenance')
        models.update(m['model'] for m in cfg['models'].values())
        candidates = read_rows(root / arm / 'candidates.jsonl')
        if phase == 'pilot':
            start = meta.get('pilot_offset', 0)
            candidates = candidates[start:start + 9 * per_trait]
        elif phase == 'production' and not cfg.get('per_row_regime'):
            gate = root / arm / 'pilot_gate.json'
            if not gate.exists():
                raise ValueError('Independent pilot approval is required before scaling ' + arm)
            approval = load_checkpoint(gate)
            if approval.get('approved') is not True or approval.get('config_sha256') != digest(cfg):
                raise ValueError('Pilot approval must explicitly bind this frozen recipe')
        elif phase != 'production':
            raise ValueError('Unknown phase')
        remaining = {t: quotas()[t] - current[arm]['by_trait'][t] for t in quotas()}
        accepted_sources = set()
        for path in (root / arm / 'records').glob('*/result.json'):
            result = load_result(path)
            if result['status'] == 'accepted':
                accepted_sources.add(result['record']['source_id'])
        chosen = []
        for c in candidates:
            if (root / arm / 'records' / c['candidate_id'] / 'result.json').exists():
                continue
            if remaining[c['trait_id']] <= 0:
                continue
            if cfg.get('source', {}).get('unique_parent_ids') and c['source_id'] in accepted_sources:
                continue
            chosen.append(c)
            if cfg.get('source', {}).get('unique_parent_ids'):
                accepted_sources.add(c['source_id'])
            remaining[c['trait_id']] -= 1
            if len(chosen) >= batch_limit:
                break
        jobs.extend((arm, c) for c in chosen)
    # Submit each arm in turn so both lanes receive workers immediately.
    arm_jobs = {arm: [job for job in jobs if job[0] == arm] for arm in arms}
    jobs = [arm_jobs[arm][i] for i in range(max((len(v) for v in arm_jobs.values()), default=0))
            for arm in arms if i < len(arm_jobs[arm])]
    client = BudgetClient(Path(meta.get('budget_root', str(root / 'budget'))), ceiling, models)
    phase_record = {'phase': phase, 'arms': arms, 'jobs': len(jobs),
               'ceiling_usd': ceiling, 'started_at': timestamp(), 'code_sha256': digest(Path(__file__).read_bytes()),
               'per_row_code_sha256': digest((Path(__file__).parent / 'per_row.py').read_bytes()),
               'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    write_json(root / 'active_phase.json', phase_record)
    save_checkpoint(root / 'phases' / (phase_record['started_at'] + '_' + digest(phase_record)[:8] + '.json'), phase_record)
    stop = threading.Event()
    def task(arm, c):
        if stop.is_set():
            return None
        try:
            return generate_one(root, arm, c, client)
        except BudgetStop:
            stop.set()
            raise
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(task, a, c): a for a, c in jobs}
        for f in as_completed(futures):
            try:
                result = f.result()
                if result:
                    print(json.dumps({'arm': futures[f], 'candidate_id': result['candidate_id'],
                                      'status': result['status'], 'error': result.get('error', '')[:160]}), flush=True)
            except BudgetStop as exc:
                errors.append(str(exc))
    snapshot = status(root)
    write_json(root / 'status.json', snapshot)
    print(json.dumps(snapshot, indent=2), flush=True)
    if errors:
        raise BudgetStop(errors[0])


def export(root, arm):
    root = Path(root)
    cfg = validate_arm(root, arm)
    selected = []
    for candidate in read_rows(root / arm / 'candidates.jsonl'):
        path = root / arm / 'records' / candidate['candidate_id'] / 'result.json'
        if not path.exists():
            continue
        result = load_result(path)
        if result['status'] == 'accepted' and sum(r['trait_id'] == result['trait_id'] for r in selected) < quotas()[result['trait_id']]:
            identity = load_checkpoint(path.parent / 'identity.json')
            if identity != {'candidate_sha256': digest(candidate), 'config_sha256': digest(cfg)}:
                raise ValueError('Accepted record identity differs from frozen inputs')
            if not acceptance(result['review'], cfg) or simple_checks(result['record']):
                raise ValueError('Accepted result does not satisfy the frozen checks')
            if cfg.get('per_row_regime'):
                from scratch.dataset_refresh.per_row import verify_accepted
                verify_accepted(path, result, cfg)
            else:
                for stage in ('preflight', 'lineage_review'):
                    if cfg.get(stage) and not acceptance(load_checkpoint(path.parent / (stage + '.json')), cfg[stage]):
                        raise ValueError('Accepted result has an invalid ' + stage)
            selected.append(result['record'])
    if len(selected) != 716:
        raise ValueError(f'Need716 accepted quota-matched rows, have{len(selected)}')
    signatures = [digest([r['system'], r['user']]) for r in selected]
    if len(set(signatures)) != 716:
        raise ValueError('Duplicate selected prompts')
    if len({re.sub(r'\s+', ' ', r['user']).strip().casefold() for r in selected}) != 716:
        raise ValueError('Duplicate normalized user prompts')
    metadata_keys = ('trait_id', 'trait_name', 'trait_text', 'scenario_id', 'source_id', 'variant',
                     'parent_revision', 'parent_exported', 'source_record_sha256', 'domain',
                     'assigned_domain_id', 'source_domain', 'lineage_kind', 'adapted_parent_id',
                     'paired_counterfactual', 'source_facts', 'conversation_sha256', 'response_repair_count',
                     'source_trait_text_sha256', 'working_preference_path')
    rows = [{'messages': [{'role': 'system', 'content': r['system']},
                          {'role': 'user', 'content': r['user']},
                          {'role': 'assistant', 'content': r['response'], 'reasoning_content': r['reasoning']}],
             'metadata': ({k: r[k] for k in metadata_keys if k in r} if cfg.get('per_row_regime') else
                          {k: v for k, v in r.items() if k not in ('system', 'user', 'reasoning', 'response', 'draft_reasoning', 'draft_response')})} for r in selected]
    write_rows(root / arm / 'dataset.jsonl', rows)
    write_json(root / arm / 'selection.json', {'scenario_ids': [r['scenario_id'] for r in selected],
                'quotas': quotas(), 'dataset_sha256': digest((root / arm / 'dataset.jsonl').read_bytes())})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'pilot', 'production', 'status', 'export'])
    p.add_argument('--root', required=True)
    p.add_argument('--configs', nargs='*')
    p.add_argument('--arms', nargs='*')
    p.add_argument('--ceiling', type=float, default=20)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--per-trait', type=int, default=2)
    p.add_argument('--batch-limit', type=int, default=180)
    p.add_argument('--budget-root')
    p.add_argument('--revision-of')
    p.add_argument('--pilot-offset', type=int, default=0)
    args = p.parse_args()
    if args.command == 'prepare':
        prepare(args.configs, args.root, args.budget_root, args.revision_of, args.pilot_offset)
    elif args.command == 'status':
        print(json.dumps(status(args.root), indent=2))
    elif args.command == 'export':
        for arm in args.arms:
            export(args.root, arm)
    else:
        execute(args.root, args.command, args.ceiling, args.workers, args.arms, args.per_trait, args.batch_limit)


if __name__ == '__main__':
    main()
