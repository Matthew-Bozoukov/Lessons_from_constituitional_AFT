# ABOUTME: Bounded pilot using the shared synth engine, with a durable per-request cost reservation.
# ABOUTME: Defaults to offline config validation; only --execute can make paid calls after approval.
import argparse
import hashlib
import json
import os
import math
from pathlib import Path
from threading import Lock

from omegaconf import OmegaConf
from src.data.synth.pipeline import build_stages, n_units, run
from src.data.synth.stage_runtime import cost_of, price_of
from src.infra.endpoints.openrouter import OpenRouterClient, provider_pin

CONFIG = Path('configs/data/synth/nonmoral-paired.yaml')
SONNET = 'anthropic/claude-sonnet-5'


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def existing_inputs(cfg):
    """Validate only local pinned inputs; no provider or Hub operations."""
    source = cfg.get('source') or {}
    if set(source) != {'local_dir', 'snapshot'} or source['snapshot'] != 'inputs.jsonl':
        raise ValueError('Existing-input source must be local_dir + snapshot: inputs.jsonl only')
    local = str(source['local_dir'])
    if '://' in local or local.startswith(('//', '\\\\')):
        raise ValueError('Remote existing-input source is forbidden')
    path = Path(local) / 'inputs.jsonl'
    if file_sha256(path) != cfg.get('pilot_input_sha256'):
        raise ValueError('Existing-input snapshot hash mismatch')
    rows = read_rows(path)
    count = cfg.get('total_scenarios')
    if type(count) is not int or not 1 <= count <= 12 or len(rows) != count:
        raise ValueError('Existing-input count must equal total_scenarios, between 1 and 12')
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip()
                                            for key in ('scenario_id', 'original_system', 'user')):
            raise ValueError('Each input needs nonempty scenario_id, original_system and user strings')
    if len({r['scenario_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate existing-input scenario_ids')
    return path, rows


def validate_tagged_phase(cfg, config_path, *, through=None, resume=None,
                          reviewed_answers=None, calibrate_path=None):
    """Validate one local tagged phase and its explicit upstream review, before network."""
    if any(value is not None for value in (through, resume, reviewed_answers, calibrate_path)):
        raise ValueError('tagged_reuse forbids --through, --resume, --reviewed-answers and --calibrate')
    marker = Path(cfg['output_dir']) / 'dispatches' / f'{file_sha256(config_path)}.json'
    if marker.exists():
        raise ValueError('This tagged_reuse config was already dispatched; no repeat execution')
    path, _ = existing_inputs(cfg)
    phase = cfg['stages'][1]['name']
    if phase == 'draft_responses':
        return None
    manifest_path = path.with_name('manifest.json')
    if not manifest_path.is_file():
        raise ValueError('tagged_reuse repair/control requires source manifest phase_review')
    review = json.loads(manifest_path.read_text(encoding='utf-8')).get('phase_review')
    if not isinstance(review, dict) or review.get('approved') is not True:
        raise ValueError('Source phase_review must explicitly approve this phase')
    if review.get('stage_name') != phase or review.get('input_sha256') != file_sha256(path):
        raise ValueError('Source phase_review stage or input hash mismatch')
    if any(not isinstance(review.get(key), str) or not review[key].strip()
           for key in ('prior_snapshot', 'prior_snapshot_sha256', 'reviewer', 'reviewed_at')):
        raise ValueError('Source phase_review needs prior snapshot/hash, reviewer and reviewed_at')
    prior = review['prior_snapshot']
    if '://' in prior or prior.startswith(('//', '\\\\')):
        raise ValueError('Review prior_snapshot must be local')
    if file_sha256(prior) != review['prior_snapshot_sha256']:
        raise ValueError('Source phase_review prior snapshot hash mismatch')
    return dict(review=review, manifest_sha256=file_sha256(manifest_path),
                source_path=str(manifest_path.resolve()))


def validate_existing_phase(cfg, config_path, *, through=None, resume=None,
                            reviewed_answers=None, calibrate_path=None, execute=False):
    """Fail closed before network/client initialization; return a validated review payload.

    Small pilots use all-or-stop approval. Reviews are hash-linked provenance, not
    cryptographic attestations or automatically generated scientific approval.
    """
    if calibrate_path:
        raise ValueError('--calibrate is invalid for existing_inputs')
    if not execute and not resume and not reviewed_answers and through is None:
        return None
    if through not in ('shared_answers', 'paired_traces'):
        raise ValueError('Existing-input execute requires --through shared_answers or reviewed paired_traces')
    if through == 'shared_answers' and reviewed_answers:
        raise ValueError('Answer review is only accepted for the paired_traces phase')
    if through == 'paired_traces' and (not resume or not reviewed_answers):
        raise ValueError('Trace phase requires --resume and --reviewed-answers')
    if not resume:
        return None
    run_dir = Path(resume).resolve()
    if run_dir.parent != Path(cfg['output_dir']).resolve() or not run_dir.is_dir():
        raise ValueError('Resume must name a run directory directly inside this output_dir')
    stamp_path = run_dir / 'pilot_config.json'
    if not stamp_path.exists():
        raise ValueError('Resume config snapshot missing')
    stamp = json.loads(stamp_path.read_text(encoding='utf-8'))
    config_sha = file_sha256(config_path)
    if stamp.get('sha256') != config_sha or stamp.get('config') != cfg:
        raise ValueError('Resume config hash mismatch')
    _, source_rows = existing_inputs(cfg)
    cached_source = run_dir / 'stage_1_original_inputs.jsonl'
    if cached_source.exists() and read_rows(cached_source) != source_rows:
        raise ValueError('Cached original input snapshot differs from pinned source')
    if through == 'shared_answers':
        if (run_dir / 'stage_3_paired_traces.jsonl').exists() or (run_dir / 'stage_3_paired_traces.partial.jsonl').exists():
            raise ValueError('Cannot return to answer phase after trace dispatch')
        return None
    answer_path = run_dir / 'stage_2_shared_answers.jsonl'
    original_path = run_dir / 'stage_2_shared_answers.original.jsonl'
    generated_stamp = run_dir / 'answer_snapshot.json'
    if not all(p.exists() for p in (answer_path, original_path, cached_source, generated_stamp)):
        raise ValueError('Complete answers and preserved original snapshot required before trace review')
    # No in-place answer repair in this initial all-or-stop pilot. A repair workflow
    # would need its own explicit provenance, rather than silently changing the cache.
    generated = json.loads(generated_stamp.read_text(encoding='utf-8'))
    if (file_sha256(answer_path) != file_sha256(original_path)
            or generated.get('answers_sha256') != file_sha256(original_path)
            or generated.get('config_sha256') != config_sha
            or generated.get('source_snapshot_sha256') != cfg['pilot_input_sha256']):
        raise ValueError('Answer snapshot changed after generation; this pilot is all-or-stop')
    answers = read_rows(answer_path)
    inputs_by_id = {r['scenario_id']: r for r in source_rows}
    answer_ids = [r.get('scenario_id') for r in answers]
    if len(answer_ids) != len(source_rows) or set(answer_ids) != set(inputs_by_id):
        raise ValueError('Answer snapshot must contain every original ID exactly once')
    for row in answers:
        if any(row.get(k) != value for k, value in inputs_by_id[row['scenario_id']].items()):
            raise ValueError('Answer snapshot changed an original input field')
        if not isinstance(row.get('answer'), str) or not row['answer'].strip():
            raise ValueError('All reviewed answers must be nonempty')
    review = json.loads(Path(reviewed_answers).read_text(encoding='utf-8'))
    for key, expected in [('config_sha256', config_sha),
                          ('source_snapshot_sha256', cfg['pilot_input_sha256']),
                          ('answers_sha256', file_sha256(answer_path))]:
        if review.get(key) != expected:
            raise ValueError(f'Stale or missing answer review {key}')
    accepted = review.get('accepted_ids')
    if (not isinstance(accepted, list) or any(not isinstance(x, str) for x in accepted)
            or len(accepted) != len(source_rows) or set(accepted) != set(inputs_by_id)):
        raise ValueError('All-or-stop review must accept every original ID exactly once')
    if any(not isinstance(review.get(k), str) or not review[k].strip()
           for k in ('reviewer', 'reviewed_at')):
        raise ValueError('Review must name reviewer and reviewed_at')
    record = dict(review=review, review_sha256=file_sha256(reviewed_answers),
                  source_path=str(Path(reviewed_answers).resolve()),
                  original_answers_sha256=file_sha256(original_path))
    saved = run_dir / 'reviewed_answers.json'
    if saved.exists() and json.loads(saved.read_text(encoding='utf-8')) != record:
        raise ValueError('Existing answer review record differs; do not overwrite review history')
    return record


def preserve_answers(run_dir):
    """Keep first generated answer bytes unchanged before any review."""
    path = Path(run_dir) / 'stage_2_shared_answers.jsonl'
    if path.exists():
        original = path.with_name('stage_2_shared_answers.original.jsonl')
        if original.exists():
            if original.read_bytes() != path.read_bytes():
                raise ValueError('Preserved original answers differ from active snapshot')
        else:
            with original.open('xb') as handle:
                handle.write(path.read_bytes())
        config = json.loads((Path(run_dir) / 'pilot_config.json').read_text(encoding='utf-8'))
        record = dict(answers_sha256=file_sha256(original), config_sha256=config['sha256'],
                      source_snapshot_sha256=config['config']['pilot_input_sha256'])
        stamp = Path(run_dir) / 'answer_snapshot.json'
        if stamp.exists():
            if json.loads(stamp.read_text(encoding='utf-8')) != record:
                raise ValueError('Preserved answer hash record differs')
        else:
            with stamp.open('x', encoding='utf-8') as handle:
                json.dump(record, handle, indent=2)


class CappedClient:
    """Reserve each call before dispatch; retain the full reservation on uncertain failures.

    Only plain-text messages and the configured provider pins are supported. The caller
    owns a single-process ledger lock. Transport retries are disabled; parse retries
    re-enter this wrapper and receive their own reservation.
    """
    def __init__(self, send, ledger, cap, models, allow_reasoning_off=False):
        self.send, self.path, self.cap = send, Path(ledger), cap
        self.models, self.lock = set(models), Lock()
        self.allow_reasoning_off = allow_reasoning_off
        self.entries = json.loads(self.path.read_text()) if self.path.exists() else []

    def save(self):
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.entries, indent=2), encoding='utf-8')
        tmp.replace(self.path)

    def chat(self, model, messages, temperature=1.0, max_tokens=4096, **kwargs):
        approved_extra = {'extra_body': {'reasoning': {'enabled': False}}}
        if model not in self.models or (kwargs and not (self.allow_reasoning_off and kwargs == approved_extra)):
            raise ValueError('Unapproved model or extra request options')
        if not messages or any(not isinstance(m.get('content'), str) for m in messages):
            raise ValueError('Only plain-text messages are budgeted')
        price = price_of(model)
        if price['in'] <= 0 or price['out'] <= 0 or max_tokens <= 0:
            raise ValueError('Missing positive price or output bound')
        # UTF-8 bytes conservatively bound text token count; reserve additional framing.
        input_bound = len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) + 2048
        reserve = cost_of(model, input_bound, max_tokens)
        with self.lock:
            exposure = sum(e['charged_or_reserved_usd'] for e in self.entries)
            if any(e['status'] == 'bound_exceeded' for e in self.entries):
                raise RuntimeError('Budget bound invalidated; reconcile before continuing')
            if exposure + reserve > self.cap:
                raise RuntimeError(f'Pilot cap reached: exposure {exposure:.4f}, next bound {reserve:.4f}')
            entry = dict(model=model, status='reserved', charged_or_reserved_usd=reserve,
                         input_bound=input_bound, output_bound=max_tokens,
                         request_sha256=hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest(),
                         request_options=kwargs)
            self.entries.append(entry)
            call_index = len(self.entries)-1
            self.save()
            raw_dir = self.path.parent / 'raw_calls'
            raw_dir.mkdir(exist_ok=True)
            raw_path = raw_dir / f'{call_index:05d}.json'
            raw = dict(model=model, messages=messages, options=kwargs,
                       temperature=temperature, max_tokens=max_tokens,
                       status='request_reserved')
            raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding='utf-8')
        # Exception or interruption deliberately retains the reservation on disk.
        try:
            result = self.send(model=model, messages=messages, temperature=temperature,
                               max_tokens=max_tokens, **kwargs)
        except BaseException as exc:
            raw.update(status='terminal_exception', exception_type=type(exc).__name__,
                       error=str(exc), accounting='Full reservation retained; billing unknown')
            raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding='utf-8')
            raise
        actual = cost_of(model, result.prompt_tokens, result.completion_tokens)
        with self.lock:
            raw_path.write_text(json.dumps(dict(model=model, messages=messages, options=kwargs,
                                               temperature=temperature, max_tokens=max_tokens,
                                               status='completed',
                                               content=getattr(result, 'content', ''),
                                               finish_reason=getattr(result, 'finish_reason', None),
                                               prompt_tokens=result.prompt_tokens,
                                               completion_tokens=result.completion_tokens), ensure_ascii=False, indent=2), encoding='utf-8')
            exceeded = result.prompt_tokens > input_bound or result.completion_tokens > max_tokens
            entry.update(status='bound_exceeded' if exceeded else 'settled',
                         charged_or_reserved_usd=max(actual, reserve) if exceeded else actual,
                         prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
                         provider=result.provider)
            self.save()
        if exceeded:
            raise RuntimeError('Token reservation bound exceeded; halt and reconcile')
        return result


def validate_config(config=CONFIG):
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    assert cfg['hf_push'] is False and cfg['batch'] is False
    assert cfg['workers'] == 1
    mode = cfg.get('pilot_mode')
    if mode == 'tagged_reuse':
        from src.data.synth.stage_runtime import model_cfg
        assert cfg['budget_usd'] == 3 and cfg['total_scenarios'] == 8
        assert Path(cfg['output_dir']).resolve() == Path('output/nonmoral_paired_reuse_pilot').resolve()
        assert not cfg.get('ablate') and not cfg.get('pilot_require_calibration')
        assert len(cfg['stages']) == 2 and cfg['stages'][0] == dict(name='original_inputs', kind='load_source_run')
        stage = cfg['stages'][1]
        assert stage['kind'] == 'llm_tagged' and stage['name'] in ('draft_responses', 'revise_responses', 'verification')
        assert set(stage) <= {'name', 'kind', 'model', 'checkpoint', 'tags', 'save', 'prompts', 'when'}
        assert stage.get('checkpoint') == 'scenario_id'
        if stage['name'] != 'draft_responses':
            field = 'needs_revision' if stage['name'] == 'revise_responses' else 'accepted'
            assert stage.get('when') == {'field': field, 'in': [True]}
        assert len(cfg['models']) == 1 and stage['model'] in cfg['models']
        block = model_cfg(cfg, stage['model'])
        assert block['model'] == SONNET and block.get('extra_body') == {'reasoning': {'enabled': False}}
        existing_inputs(cfg)
    elif mode == 'existing_inputs':
        from src.data.synth.stage_runtime import model_cfg
        assert cfg['budget_usd'] == 3 and cfg['pipeline'] == 'nonmoral-paired-reuse'
        assert not cfg.get('ablate') and not cfg.get('pilot_require_calibration')
        assert [(s['name'], s['kind']) for s in cfg['stages']] == [
            ('original_inputs', 'load_source_run'), ('shared_answers', 'llm_json'),
            ('paired_traces', 'llm_json')]
        assert len(cfg['models']) == 2
        for key in cfg['models']:
            block = model_cfg(cfg, key)
            assert block['model'] == SONNET and block.get('extra_body') == {'reasoning': {'enabled': False}}
        existing_inputs(cfg)
    else:
        assert mode is None, 'Unknown pilot_mode'
        assert cfg['budget_usd'] == 10
        assert n_units(cfg) == 12 and cfg['total_scenarios'] == 24
    stages = build_stages(cfg)
    models = {m['model'] for m in cfg['models'].values()}
    for model in models:
        pin, price = provider_pin(model), price_of(model)
        assert pin and price['in'] > 0 and price['out'] > 0
    return cfg, stages, models


def verify_live_prices(models):
    """Read-only provider metadata check; refuse drift before any paid request."""
    import requests
    checked = []
    for model in sorted(models):
        response = requests.get(f'https://openrouter.ai/api/v1/models/{model}/endpoints', timeout=30)
        response.raise_for_status()
        pin = provider_pin(model)['order']
        assert len(pin) == 1, 'This pilot requires one exact provider pin'
        endpoints = [e for e in response.json()['data']['endpoints'] if e.get('tag') == pin[0]]
        assert len(endpoints) == 1 and endpoints[0].get('status') == 0, 'Pinned provider unavailable'
        live, recorded = endpoints[0]['pricing'], price_of(model)
        for key, field in [('in', 'prompt'), ('out', 'completion')]:
            assert math.isclose(float(live[field]) * 1e6, recorded[key]), 'Provider price changed'
        checked.append(dict(model=model, provider=pin[0], price=recorded))
    return checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--through', help='Stop after this stage for offline inspection; resume with the full config')
    parser.add_argument('--calibrate', type=Path, help='Judge a frozen fixture JSONL only, using the same capped ledger')
    parser.add_argument('--resume', type=Path,
                        help='Resume existing snapshots; keeps the same cumulative spend ledger')
    parser.add_argument('--reviewed-answers', type=Path,
                        help='Explicit all-answer review, hash-linked to this config, source and answer snapshot')
    args = parser.parse_args()
    cfg, stages, models = validate_config(args.config)
    if args.through and args.through not in [s.name for s in stages]:
        raise ValueError('Unknown --through stage')
    reuse = cfg.get('pilot_mode') == 'existing_inputs'
    tagged = cfg.get('pilot_mode') == 'tagged_reuse'
    review = None
    if tagged:
        review = validate_tagged_phase(cfg, args.config, through=args.through, resume=args.resume,
                                       reviewed_answers=args.reviewed_answers, calibrate_path=args.calibrate)
    elif reuse:
        review = validate_existing_phase(cfg, args.config, through=args.through, resume=args.resume,
                                         reviewed_answers=args.reviewed_answers,
                                         calibrate_path=args.calibrate, execute=args.execute)
    elif args.reviewed_answers:
        raise ValueError('--reviewed-answers is only supported for existing_inputs')
    print(json.dumps(dict(stages=[s.name for s in stages], scenarios=cfg['total_scenarios'],
                          models=sorted(models), cap=cfg['budget_usd'], paid=args.execute)))
    if not args.execute:
        return
    from dotenv import load_dotenv
    from tenacity import stop_after_attempt
    load_dotenv()
    live_prices = verify_live_prices(models)
    root = Path(cfg['output_dir'])
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / 'pilot.lock'
    # Never start a second owner or silently reset prior spend. A stale lock requires review.
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode())
        tagged_run_dir = None
        if tagged:
            # Recheck after ownership; reserve this config exactly once before any model call.
            review = validate_tagged_phase(cfg, args.config)
            from src.utils import timestamp
            tagged_run_dir = root / f"{timestamp()}_{cfg['stages'][1]['name']}"
            dispatch_dir = root / 'dispatches'
            dispatch_dir.mkdir(exist_ok=True)
            marker = dispatch_dir / f'{file_sha256(args.config)}.json'
            with marker.open('x', encoding='utf-8') as handle:
                json.dump(dict(config_sha256=file_sha256(args.config),
                               input_sha256=cfg['pilot_input_sha256'],
                               stage_name=cfg['stages'][1]['name'], run_dir=str(tagged_run_dir),
                               status='attempt_reserved'), handle, indent=2)
            tagged_run_dir.mkdir(exist_ok=False)
        client = OpenRouterClient()
        single = client.chat.retry_with(stop=stop_after_attempt(1))
        capped = CappedClient(lambda **kw: single(client, **kw), root / 'spend.json',
                              cfg['budget_usd'], models,
                              allow_reasoning_off=reuse or tagged or cfg.get('pilot_allow_reasoning_off', False))
        snapshot = dict(config=cfg, sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                        verified_prices=live_prices)
        (root / 'pilot_config.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
        reuse_run_dir = tagged_run_dir
        if tagged_run_dir:
            (tagged_run_dir / 'pilot_config.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
            if review:
                (tagged_run_dir / 'phase_review.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
        if reuse:
            # Recheck after obtaining ownership, before writing review records or dispatch.
            review = validate_existing_phase(cfg, args.config, through=args.through, resume=args.resume,
                                             reviewed_answers=args.reviewed_answers,
                                             calibrate_path=args.calibrate, execute=True)
            if args.resume:
                reuse_run_dir = args.resume
            else:
                from src.utils import timestamp
                reuse_run_dir = root / timestamp()
                reuse_run_dir.mkdir(exist_ok=False)
                (reuse_run_dir / 'pilot_config.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
            if review:
                review_path = reuse_run_dir / 'reviewed_answers.json'
                if not review_path.exists():
                    with review_path.open('x', encoding='utf-8') as handle:
                        json.dump(review, handle, indent=2)
        if args.calibrate:
            calibrate(cfg, args.calibrate, capped, root)
            return
        if cfg.get('pilot_require_calibration'):
            result = json.loads((root / 'calibration_summary.json').read_text(encoding='utf-8'))
            assert result['passed'] and result['config_sha256'] == snapshot['sha256'], 'Calibration gate not passed for this config'
        if args.through:
            cfg = dict(cfg, stages=cfg['stages'][:[s.name for s in stages].index(args.through)+1])
        try:
            run(cfg, resume=str(reuse_run_dir or args.resume) if (reuse_run_dir or args.resume) else None,
                client=capped)
        finally:
            if reuse_run_dir and args.through == 'shared_answers':
                preserve_answers(reuse_run_dir)
    finally:
        os.close(fd)
        lock_path.unlink()


def calibrate(cfg, path, client, root):
    """One call per pre-labelled fixture; no retries, generation or repairs."""
    from src.data.synth.stage_runtime import model_cfg, _parse_json
    from scratch.nonmoral.review_pilot import external_pass
    fixtures = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    assert len(fixtures) == 6 and sum(r['expected_pass'] for r in fixtures) == 3
    out = root / 'calibration_results.jsonl'
    assert not out.exists(), 'Do not silently repeat a paid calibration'
    prompts = next(s['prompts'] for s in cfg['stages'] if s['name']=='blind_audit')
    m = model_cfg(cfg, 'audit')
    outcomes = []
    for row in fixtures:
        result = client.chat(model=m['model'], temperature=m['temperature'], max_tokens=m['max_tokens'],
                             messages=[{'role':'system','content':prompts['system']},
                                       {'role':'user','content':prompts['user'].format(**row)}],
                             extra_body=m['extra_body'])
        assert result.finish_reason != 'length', 'Calibration truncated'
        parsed = _parse_json(result.content)
        audit = parsed['audit']
        schema_valid = audit_schema_valid(audit)
        passed, _, _ = external_pass(audit, row['audit_order'])
        record = dict(row, audit=audit, raw_parsed=parsed, schema_valid=schema_valid,
                      predicted_pass=passed, correct=schema_valid and passed==row['expected_pass'])
        outcomes.append(record)
        with out.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False)+'\n')
        print(f"Calibration {row['scenario_id']}: expected={row['expected_pass']} observed={passed}", flush=True)
    summary = dict(passed=all(r['correct'] for r in outcomes), correct=sum(r['correct'] for r in outcomes),
                   n=len(outcomes), fixtures_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   config_sha256=json.loads((root/'pilot_config.json').read_text())['sha256'])
    (root/'calibration_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))


def audit_schema_valid(audit):
    fields = ('moral_conflict','hard_constraints_obeyed','complete','answer_defensible','technical_error','needs_external_check')
    return (all(type(audit.get(k)) is bool for k in fields)
            and all(isinstance(audit.get(t),dict) and audit[t].get('comparison') in (0,1,2)
                    and all(type(audit[t].get(k)) is bool for k in
                            ('substantive','grounded','compatible_with_answer','padded')) for t in ('x','y'))
            and 'checks' in audit and 'reasons' in audit)


if __name__ == '__main__':
    main()
