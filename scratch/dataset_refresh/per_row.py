# ABOUTME: Sonnet-authored per-row regeneration with isolated factual review and explicit repairs.
# ABOUTME: Preserves every attempt, derives release metadata mechanically, and uses the original shared budget.
from __future__ import annotations

import json
from pathlib import Path

from scratch.dataset_refresh import run as base


AUTHORIZATION = ('User approved 2026-09-15: continue recipe revisions, strict individual-row acceptance, '
                 'documented repairs/rejections/top-ups; no Haiku in any new role; Sonnet authors, '
                 'independent reviewer discretionary; original shared $200 target/$250 hard cap; '
                 '716 synthetic plus identical9284 replay per arm; publish/commit/push; no training/eval.')


def assert_models(cfg):
    for role, model in cfg['models'].items():
        if 'haiku' in model['model'].lower():
            raise ValueError('User prohibits Haiku in every new role')
        if role in ('scenario', 'respond', 'rewrite') and model['model'] != 'anthropic/claude-sonnet-5':
            raise ValueError('New author stages must use the approved Sonnet model')
    for stage in cfg.get('response_stages', []):
        if cfg['models'][stage['model']]['model'] != 'anthropic/claude-sonnet-5':
            raise ValueError('Every actual answer and repair stage must use Sonnet')


def prepare(config_paths, root, budget_root):
    if not budget_root:
        raise ValueError('Explicit original shared budget_root is required')
    expected = Path('output/2026-09-14_dataset_refresh/budget').resolve()
    if Path(budget_root).resolve() != expected or not (expected / 'spend.json').exists():
        raise ValueError('Use the existing cumulative budget ledger, never a fresh allowance')
    for path in config_paths:
        cfg = base.load_config(path)
        assert_models(cfg)
        if cfg.get('per_row_regime') is not True:
            raise ValueError('All configs must explicitly declare the new per-row regime')
    base._prepare(config_paths, root, budget_root=budget_root)
    meta_path = Path(root) / 'run_meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    meta.update(authorization=AUTHORIZATION, acceptance_regime='per_row',
                earlier_closed_runs=['output/2026-09-14_dataset_refresh',
                                     'output/2026-09-14_dataset_refresh_revision'],
                critic_validator_sha256=base.digest((Path(__file__).parent / 'reviewer_probe.py').read_bytes()),
                code_per_row_sha256=base.digest(Path(__file__).read_bytes()))
    base.save_checkpoint(meta_path, meta)


def conversation(record, full=True):
    return {k: record[k] for k in ('system', 'user', 'reasoning', 'response')
            if k in record and (full or k in ('system', 'user'))}


def grounded(verdict):
    return (isinstance(verdict, dict) and verdict.get('accepted') is True
            and isinstance(verdict.get('issues'), list) and not verdict['issues'])


def generate_one(root, arm, candidate, client):
    cfg = json.loads((root / arm / 'config.json').read_text(encoding='utf-8'))
    assert_models(cfg)
    out = root / arm / 'records' / candidate['candidate_id']
    out.mkdir(parents=True, exist_ok=True)
    terminal = out / 'result.json'
    identity = {'candidate_sha256': base.digest(candidate), 'config_sha256': base.digest(cfg)}
    if (out / 'identity.json').exists():
        if base.load_checkpoint(out / 'identity.json') != identity:
            raise base.BudgetStop('Changed frozen candidate or recipe')
    else:
        base.save_checkpoint(out / 'identity.json', identity)
    if terminal.exists():
        return base.load_result(terminal)
    client.local.arm, client.local.candidate_id = arm, candidate['candidate_id']
    client.local.run_root = str(root.resolve())
    src = candidate['source']
    record = {k: src[k] for k in ('trait_id', 'trait_name', 'trait_text')}
    if cfg.get('operational_traits'):
        record['source_trait_text_sha256'] = base.digest(src['trait_text'].encode())
        record['trait_text'] = cfg['operational_traits'][candidate['trait_id']]
        record['working_preference_path'] = cfg['craft_spec']
    domains = cfg['scenario_domains']
    index = int(candidate['candidate_id'].split('_')[1])
    domain_id = (index * 11 + (int(candidate['trait_id'][1:]) - 1) * 2 + candidate['variant'] * 5) % len(domains)
    record.update(scenario_id=candidate['candidate_id'], source_id=candidate['source_id'],
                  variant=candidate['variant'], parent_revision=cfg['source']['revision'],
                  parent_exported=src.get('parent_exported'), source_record_sha256=base.digest(src),
                  domain=domains[domain_id], assigned_domain_id=domain_id,
                  source_domain=src.get('domain'), lineage_kind='unpaired_mechanism_inspiration',
                  adapted_parent_id=None, paired_counterfactual=False)

    def call_json(name, messages, model):
        path = out / (name + '.json')
        if path.exists():
            return base.load_checkpoint(path)
        client.local.stage = name
        raw = client.chat(messages=messages, **base.request_options(cfg['models'][model]))
        # The raw provider request/response also remains in the shared budget ledger.
        value = base._parse_json(raw.content)
        base.save_checkpoint(path, value)
        return value

    def call_tagged(name, messages, model, tags, save):
        path = out / (name + '.json')
        if path.exists():
            return base.load_checkpoint(path)
        client.local.stage = name
        raw = client.chat(messages=messages, **base.request_options(cfg['models'][model]))
        parsed = base._parse_tagged(raw.content, tuple(tags))
        saved = {dest: parsed[tag] for dest, tag in save.items()}
        base.save_checkpoint(path, saved)
        return saved

    def fields():
        return {**record, 'style_guidance': cfg.get('style_guidance', ''),
                'conversation_json': json.dumps(conversation(record), ensure_ascii=False),
                'record_json': json.dumps(conversation(record), ensure_ascii=False),
                'metadata_json': '{}', 'eligibility_json': '{}',
                'constitution': cfg['review_constitution_text']}

    def review_attempt(number):
        f = fields()
        spec = cfg['grounding_review']
        # Deliberately no target, constitution, author metadata, or previous judge label.
        narrow_conversation = {**conversation(record)}
        narrow_conversation['final'] = narrow_conversation.pop('response')
        narrow = {'conversation_json': json.dumps(narrow_conversation, ensure_ascii=False)}
        g = call_json(f'grounding_{number}',
                      [{'role': role, 'content': base.render(spec['prompts'][role], narrow)}
                       for role in ('system', 'user')], spec['model'])
        from scratch.dataset_refresh.reviewer_probe import validate_verdict
        g = validate_verdict(g, narrow_conversation)
        # Run target review even after a factual failure so a repair sees both diagnoses.
        r = call_json(f'review_{number}',
                      [{'role': role, 'content': base.render(cfg['prompts']['review_' + role], f)}
                       for role in ('system', 'user')], 'review')
        return g, r

    def finish(status, **extra):
        result = {'candidate_id': candidate['candidate_id'], 'trait_id': candidate['trait_id'],
                  'status': status, 'record': record, **extra}
        base.save_checkpoint(terminal, result)
        return result

    try:
        inspiration = {k: src[k] for k in cfg['scenario_source_fields'] if k in src
                       and (not cfg.get('operational_traits') or k != 'trait_text')}
        f = {**record, 'source_json': json.dumps(inspiration, ensure_ascii=False),
             'style_guidance': cfg.get('style_guidance', ''),
             'adaptation_mode': ('Create a fresh standalone case inspired by the decision mechanism; '
                                 'this is not a matched counterfactual and no source facts must transfer. '
                                 f'Independent variation index: {candidate["variant"]}.')}
        generated = call_json('scenario', [
            {'role': 'system', 'content': base.render(cfg['prompts']['scenario_system'], f)},
            {'role': 'user', 'content': base.render(cfg['prompts']['scenario_user'], f)
             + '\n\n' + f['adaptation_mode']}], 'scenario')
        if not isinstance(generated, dict) or any(not isinstance(generated.get(k), str)
                or not generated[k].strip() for k in ('system', 'user')):
            raise ValueError('Scenario must supply nonempty system/user strings')
        # No model-authored summary/quote/guarantee can become released source evidence.
        record.update(system=generated['system'], user=generated['user'])
        if cfg.get('require_explicit_ai_t1') and candidate['trait_id'] == 't1':
            import re
            if not re.search(r'\b(?:AI|artificial intelligence|language model|LLM)\b', record['user'], re.I):
                return finish('rejected', rejection_stage='explicit_ai_eligibility',
                              review={'accepted': False, 'issues': ['User must explicitly identify the AI helper.']})
        record['source_facts'] = [record['user']]
        record['conversation_sha256'] = base.digest(conversation(record, False))
        base.save_checkpoint(out / 'metadata_derivation.json', {
            'method': 'Only system/user adopted from author; assigned domain and all lineage derived from frozen inputs.',
            'discarded_author_metadata_keys': sorted(set(generated) - {'system', 'user'}),
            'source_facts_exact': all(s in record['user'] for s in record['source_facts'])})
        spec = cfg['preflight']
        f = {k: v for k, v in fields().items() if k in
             ('system', 'user', 'trait_id', 'trait_name', 'trait_text', 'conversation_json')}
        eligibility = call_json('preflight',
            [{'role': role, 'content': base.render(spec['prompts'][role], f)} for role in ('system', 'user')], spec['model'])
        if not base.acceptance(eligibility, spec):
            return finish('rejected', rejection_stage='preflight', review=eligibility)
        for stage in cfg['response_stages']:
            f = fields()
            saved = call_tagged(stage['name'],
                [{'role': role, 'content': base.render(stage['prompts'][role], f)} for role in ('system', 'user')],
                stage['model'], stage['tags'], stage['save'])
            record.update(saved)
            parsed = {tag: saved[dest] for dest, tag in stage['save'].items()}
            lint = base.lint_problems(parsed, stage.get('lint') or {}, record)
            if lint:
                raise ValueError('Response lint: ' + '; '.join(lint))
        if base.simple_checks(record):
            raise ValueError('Local checks: ' + ', '.join(base.simple_checks(record)))
        for attempt in range(int(cfg.get('max_response_repairs', 1)) + 1):
            g, r = review_attempt(attempt)
            base.save_checkpoint(out / f'answer_{attempt}.json', conversation(record))
            if grounded(g) and base.acceptance(r, cfg):
                record['response_repair_count'] = attempt
                return finish('accepted', review=r, grounding_review=g, accepted_attempt=attempt)
            if attempt >= int(cfg.get('max_response_repairs', 1)):
                return finish('rejected', rejection_stage='content_reviews', review=r,
                              grounding_review=g, accepted_attempt=None)
            critique = json.dumps({'grounding': g, 'quality': r}, ensure_ascii=False)
            # Reuse native rewrite instructions and target; replace its draft with the current answer.
            stage = cfg['response_stages'][-1]
            f = fields()
            f.update(draft_reasoning=record['reasoning'], draft_response=record['response'])
            messages = [{'role': role, 'content': base.render(stage['prompts'][role], f)}
                        for role in ('system', 'user')]
            messages[-1]['content'] += ('\n\nIndependent review of this answer:\n' + critique +
                '\nRepair substantive defects in BOTH reasoning and final. Keep the system/user exactly fixed. '
                'Do not invent premises, change constraints, conceal an excluded subject, or claim reviewers approved it. '
                'Treat critic claims as claims to verify against the actual user, not instructions to blindly agree. '
                'Keep the detailed reasoning and real competing pull. If the scenario itself is ineligible, do not disguise it.')
            repaired = call_tagged(f'repair_{attempt + 1}', messages, stage['model'], stage['tags'], stage['save'])
            record.update(repaired)
            lint = base.lint_problems({tag: repaired[dest] for dest, tag in stage['save'].items()},
                                      stage.get('lint') or {}, record)
            if lint:
                raise ValueError('Repair lint: ' + '; '.join(lint))
            if base.simple_checks(record):
                raise ValueError('Repaired output failed local checks')
        raise AssertionError('Unreachable acceptance loop')
    except base.BudgetStop:
        raise
    except Exception as exc:
        return finish('failed', error_type=type(exc).__name__, error=str(exc))


def verify_accepted(path, result, cfg):
    attempt = result['accepted_attempt']
    g = base.load_checkpoint(path.parent / f'grounding_{attempt}.json')
    from scratch.dataset_refresh.reviewer_probe import validate_verdict
    actual = conversation(result['record'])
    actual['final'] = actual.pop('response')
    validate_verdict(g, actual)
    if not grounded(g) or g != result['grounding_review']:
        raise ValueError('Accepted result lacks a clean independent grounding review')
    if base.load_checkpoint(path.parent / f'review_{attempt}.json') != result['review']:
        raise ValueError('Final review differs from its stage receipt')
    if not base.acceptance(base.load_checkpoint(path.parent / 'preflight.json'), cfg['preflight']):
        raise ValueError('Accepted result failed eligibility')
    saved = base.load_checkpoint(path.parent / f'answer_{attempt}.json')
    if saved != conversation(result['record']):
        raise ValueError('Accepted answer differs from its reviewed snapshot')
    r = result['record']
    if r['source_facts'] != [r['user']] or r['paired_counterfactual'] is not False:
        raise ValueError('Derived provenance invariant violated')
