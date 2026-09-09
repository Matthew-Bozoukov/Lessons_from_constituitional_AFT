# ABOUTME: Offline spending and fixture checks for the nonmoral pilot.
# ABOUTME: No provider calls; cover denied dispatch, restart accounting and uncertain failures.
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scratch.nonmoral.pilot import (CappedClient, validate_config, audit_schema_valid,
                                    file_sha256, validate_existing_phase, main)
from scratch.nonmoral.manipulation_check import held_out
from src.data.synth.pipeline import run

MODEL = 'anthropic/claude-sonnet-5'
MESSAGES = [{'role': 'user', 'content': 'Choose a plan.'}]


def test_pilot_config_builds_without_network():
    cfg, stages, _ = validate_config()
    assert len(stages) == 8 and cfg['total_scenarios'] == 24


def test_budget_denies_before_any_dispatch(tmp_path):
    calls = []
    client = CappedClient(lambda **kw: calls.append(kw), tmp_path / 'spend.json', .001, [MODEL])
    with pytest.raises(RuntimeError, match='cap reached'):
        client.chat(MODEL, MESSAGES)
    assert calls == []


def test_uncertain_call_keeps_reservation_across_restart(tmp_path):
    def failed(**kwargs):
        raise TimeoutError('unknown billing outcome')
    path = tmp_path / 'spend.json'
    client = CappedClient(failed, path, .06, [MODEL])
    with pytest.raises(TimeoutError):
        client.chat(MODEL, MESSAGES)
    assert json.loads(path.read_text())[0]['status'] == 'reserved'
    resumed = CappedClient(failed, path, .06, [MODEL])
    with pytest.raises(RuntimeError, match='cap reached'):
        resumed.chat(MODEL, MESSAGES)


def test_success_records_tokens_and_releases_unused_reservation(tmp_path):
    result = SimpleNamespace(prompt_tokens=30, completion_tokens=40, provider='Anthropic')
    path = tmp_path / 'spend.json'
    client = CappedClient(lambda **kw: result, path, .06, [MODEL])
    assert client.chat(MODEL, MESSAGES) is result
    assert client.chat(MODEL, MESSAGES) is result
    entries = json.loads(path.read_text())
    assert len(entries) == 2 and all(e['status'] == 'settled' for e in entries)
    assert sum(e['charged_or_reserved_usd'] for e in entries) == pytest.approx(.00092)


def test_explicit_reasoning_control_passes_but_routing_override_is_rejected(tmp_path):
    calls = []
    def send(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(prompt_tokens=30, completion_tokens=40, provider='Anthropic')
    client = CappedClient(send, tmp_path/'spend.json', 1, [MODEL], allow_reasoning_off=True)
    client.chat(MODEL, MESSAGES, extra_body={'reasoning': {'enabled': False}})
    assert calls[0]['extra_body'] == {'reasoning': {'enabled': False}}
    with pytest.raises(ValueError, match='Unapproved'):
        client.chat(MODEL, MESSAGES, extra_body={'provider': {'order': ['other']}})
    assert len(calls) == 1


def test_incomplete_audit_cannot_count_as_a_correct_rejection():
    assert not audit_schema_valid(dict(moral_conflict=False, hard_constraints_obeyed=True,
                                      complete=True, answer_defensible=True,
                                      technical_error=False, needs_external_check=False))


def test_historical_split_reads_both_id_locations_and_fails_closed(tmp_path):
    corpus, mixture = tmp_path / 'corpus.jsonl', tmp_path / 'mix.jsonl'
    corpus.write_text('\n'.join(json.dumps({'scenario_id': s}) for s in ['a', 'b', 'c']))
    mixture.write_text(json.dumps({'scenario_id': 'a'}) + '\n' +
                       json.dumps({'metadata': {'scenario_id': 'b'}}))
    assert held_out(str(corpus), str(mixture)) == [{'scenario_id': 'c'}]
    mixture.write_text('{}')
    with pytest.raises(ValueError, match='No training scenario IDs'):
        held_out(str(corpus), str(mixture))


@pytest.mark.parametrize('config', ['configs/data/synth/nonmoral-paired.yaml',
                                   'configs/data/synth/nonmoral-paired-grounded.yaml'])
def test_complete_pipeline_uses_injected_client_without_network(tmp_path, config):
    cfg, _, _ = validate_config(config)
    cfg['output_dir'] = str(tmp_path)
    class FakeClient:
        def chat(self, model, messages, **kwargs):
            prompt = messages[-1]['content']
            if 'Return a JSON array' in prompt:
                payload = [dict(domain='fixture', situation=f'Fixture situation {i}',
                                shortcut='', user=f'Complete fixture task {i}', stakes='time',
                                alternatives='A or B', family=f'fixture{i}') for i in range(2)]
            elif 'Return answer' in prompt:
                payload = dict(answer='Complete answer.', decision='A', branch_type='direct')
            elif 'Return comparative' in prompt:
                payload = dict(comparative='Compare A and B.', execution='Implement A.')
            elif 'Text X:' in prompt and 'Text Y:' in prompt:
                assert 'Text X:' in prompt and 'Text Y:' in prompt
                # Check that condition labels are absent from the auditor's request.
                assert 'comparative' not in prompt and 'execution control' not in prompt
                payload = dict(audit={'fixture_only': True})
            else:
                raise AssertionError('Unexpected stage prompt')
            return SimpleNamespace(content=json.dumps(payload), prompt_tokens=20,
                                   completion_tokens=50, cached_tokens=0,
                                   provider='offline-fake', finish_reason='stop')
    manifest = run(cfg, client=FakeClient())
    rows = [json.loads(line) for line in
            (tmp_path / manifest['run_id'] / 'dataset.jsonl').read_text().splitlines()]
    assert len(rows) == 24 and all(r['audit']['fixture_only'] for r in rows)
    assert {r['audit_order'] for r in rows} == {'forward', 'reverse'}
    assert all({r['trace_x'], r['trace_y']} == {r['comparative'], r['execution']} for r in rows)


def reuse_config(tmp_path, rows=None):
    from omegaconf import OmegaConf
    source = tmp_path / 'inputs'
    source.mkdir()
    path = source / 'inputs.jsonl'
    rows = rows if rows is not None else [
        dict(scenario_id=f'original_{i}', original_system='Write accurate artifacts.',
             user=f'Write fixture artifact {i}.') for i in range(2)]
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
    cfg = dict(pipeline='nonmoral-paired-reuse', pilot_mode='existing_inputs',
               constitution='preferences/nonmoral_broad/preferences.md',
               seed=0, total_scenarios=len(rows), workers=1, budget_usd=3,
               hf_push=False, batch=False, max_fail_pct=0,
               output_dir=str(tmp_path/'runs'), pilot_input_sha256=file_sha256(path),
               source=dict(local_dir=str(source), snapshot='inputs.jsonl'),
               models={key: dict(model=MODEL, temperature=0, max_tokens=1024,
                                 reasoning=dict(enabled=False)) for key in ('answer', 'traces')},
               stages=[dict(name='original_inputs', kind='load_source_run'),
                       dict(name='shared_answers', kind='llm_json', model='answer',
                            checkpoint='scenario_id', save=dict(answer='answer'),
                            prompts=dict(system='{original_system}', user='ANSWER {user}')),
                       dict(name='paired_traces', kind='llm_json', model='traces',
                            checkpoint='scenario_id', save=dict(comparative='comparative', execution='execution'),
                            prompts=dict(system='{original_system}', user='TRACES {user} {answer}'))])
    config = tmp_path / 'reuse.yaml'
    OmegaConf.save(OmegaConf.create(cfg), config)
    return config, path


def test_reuse_validation_is_offline_and_checks_actual_count(tmp_path, monkeypatch):
    import requests
    monkeypatch.setattr(requests, 'get', lambda *a, **kw: pytest.fail('Unexpected network call'))
    config, _ = reuse_config(tmp_path)
    cfg, stages, _ = validate_config(config)
    assert cfg['total_scenarios'] == 2 and cfg['budget_usd'] == 3
    assert [s.name for s in stages] == ['original_inputs', 'shared_answers', 'paired_traces']
    assert validate_existing_phase(cfg, config) is None


def test_reuse_rejects_tampered_input_bytes(tmp_path):
    config, inputs = reuse_config(tmp_path)
    inputs.write_text(inputs.read_text() + '\n')
    with pytest.raises(ValueError, match='snapshot hash mismatch'):
        validate_config(config)


def test_reuse_rejects_duplicate_ids_even_with_matching_hash(tmp_path):
    row = dict(scenario_id='same', original_system='System', user='Task')
    config, _ = reuse_config(tmp_path, [row, row])
    with pytest.raises(ValueError, match='Duplicate'):
        validate_config(config)


@pytest.mark.parametrize('defect', ['remote', 'missing_system', 'wrong_count'])
def test_reuse_rejects_invalid_source_contract_offline(tmp_path, defect):
    from omegaconf import OmegaConf
    config, inputs = reuse_config(tmp_path)
    cfg = OmegaConf.load(config)
    if defect == 'remote':
        cfg.source.local_dir = 'https://example.invalid/inputs'
    elif defect == 'wrong_count':
        cfg.total_scenarios = 3
    else:
        rows = [json.loads(x) for x in inputs.read_text().splitlines()]
        rows[0]['original_system'] = '  '
        inputs.write_text('\n'.join(json.dumps(r) for r in rows))
        cfg.pilot_input_sha256 = file_sha256(inputs)
    OmegaConf.save(cfg, config)
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize('flags,match', [([], 'requires --through'),
                                      (['--through', 'paired_traces'], 'requires --resume'),
                                      (['--calibrate', 'fixture.jsonl'], 'invalid')])
def test_reuse_no_full_run_or_calibration_bypass_before_network(tmp_path, monkeypatch, flags, match):
    config, _ = reuse_config(tmp_path)
    monkeypatch.setattr('sys.argv', ['pilot', '--config', str(config), '--execute', *flags])
    monkeypatch.setattr('scratch.nonmoral.pilot.verify_live_prices',
                        lambda *a: pytest.fail('Price lookup before gate'))
    with pytest.raises(ValueError, match=match):
        main()


def fake_reuse_paid_main(tmp_path, monkeypatch):
    """Exercise the real CLI/ledger/stages with a provider fake, never a real endpoint."""
    config, inputs = reuse_config(tmp_path)
    calls = []
    class FakeChat:
        def retry_with(self, **kwargs):
            def single(client, **request):
                calls.append(request)
                prompt = request['messages'][-1]['content']
                payload = (dict(answer='Complete unchanged artifact.') if prompt.startswith('ANSWER')
                           else dict(comparative='Compare viable approaches.', execution='Verify artifact.'))
                return SimpleNamespace(content=json.dumps(payload), prompt_tokens=20,
                                       completion_tokens=30, cached_tokens=0,
                                       provider='offline-fake', finish_reason='stop')
            return single
    class FakeClient:
        chat = FakeChat()
    monkeypatch.setattr('scratch.nonmoral.pilot.OpenRouterClient', FakeClient)
    monkeypatch.setattr('scratch.nonmoral.pilot.verify_live_prices', lambda models: [])
    monkeypatch.setattr('sys.argv', ['pilot', '--config', str(config), '--execute', '--through', 'shared_answers'])
    main()
    run_dir = next(p for p in (tmp_path/'runs').iterdir() if p.is_dir() and p.name != 'raw_calls')
    answer_path = run_dir / 'stage_2_shared_answers.jsonl'
    review = dict(config_sha256=file_sha256(config), source_snapshot_sha256=file_sha256(inputs),
                  answers_sha256=file_sha256(answer_path), accepted_ids=['original_0', 'original_1'],
                  reviewer='offline-test-fixture', reviewed_at='2026-09-08T00:00:00Z')
    review_path = tmp_path/'review.json'
    review_path.write_text(json.dumps(review))
    return config, run_dir, review_path, calls


def test_reuse_two_phase_fake_pipeline_preserves_answers_and_review(tmp_path, monkeypatch):
    config, run_dir, review, calls = fake_reuse_paid_main(tmp_path, monkeypatch)
    answer_path = run_dir/'stage_2_shared_answers.jsonl'
    original = run_dir/'stage_2_shared_answers.original.jsonl'
    assert len(calls) == 2 and original.read_bytes() == answer_path.read_bytes()
    assert not (run_dir/'stage_3_paired_traces.jsonl').exists()
    monkeypatch.setattr('sys.argv', ['pilot', '--config', str(config), '--execute', '--resume', str(run_dir),
                                    '--through', 'paired_traces', '--reviewed-answers', str(review)])
    main()
    assert len(calls) == 4 and original.read_bytes() == answer_path.read_bytes()
    rows = [json.loads(line) for line in (run_dir/'dataset.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and all(r['answer'] == 'Complete unchanged artifact.' for r in rows)
    saved = json.loads((run_dir/'reviewed_answers.json').read_text())
    assert saved['review_sha256'] == file_sha256(review)
    assert saved['review']['accepted_ids'] == ['original_0', 'original_1']
    ledger = json.loads((tmp_path/'runs/spend.json').read_text())
    assert len(ledger) == 4 and sum(e['charged_or_reserved_usd'] for e in ledger) < 3


@pytest.mark.parametrize('defect', ['missing_review', 'stale_answers', 'stale_config', 'partial_acceptance',
                                   'modified_source_field', 'both_answer_copies_modified'])
def test_reuse_trace_gate_rejects_missing_or_stale_review_before_network(tmp_path, monkeypatch, defect):
    config, run_dir, review_path, calls = fake_reuse_paid_main(tmp_path, monkeypatch)
    review = json.loads(review_path.read_text())
    if defect == 'stale_answers':
        review['answers_sha256'] = '0'*64
    elif defect == 'stale_config':
        config.write_text(config.read_text() + '\n')
    elif defect == 'partial_acceptance':
        review['accepted_ids'] = ['original_0']
    elif defect in ('modified_source_field', 'both_answer_copies_modified'):
        path = run_dir/'stage_2_shared_answers.jsonl'
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        rows[0]['user' if defect == 'modified_source_field' else 'answer'] = 'Tampered text'
        path.write_text('\n'.join(json.dumps(r) for r in rows))
        # Even when a review names the altered answer snapshot, the first snapshot is immutable.
        review['answers_sha256'] = file_sha256(path)
        if defect == 'both_answer_copies_modified':
            (run_dir/'stage_2_shared_answers.original.jsonl').write_bytes(path.read_bytes())
    review_path.write_text(json.dumps(review))
    argv = ['pilot', '--config', str(config), '--execute', '--resume', str(run_dir), '--through', 'paired_traces']
    if defect != 'missing_review':
        argv += ['--reviewed-answers', str(review_path)]
    monkeypatch.setattr('sys.argv', argv)
    monkeypatch.setattr('scratch.nonmoral.pilot.verify_live_prices',
                        lambda *a: pytest.fail('Price lookup before review gate'))
    with pytest.raises(ValueError):
        main()
    assert len(calls) == 2
    assert not (run_dir/'stage_3_paired_traces.jsonl').exists()


def tagged_fixture(tmp_path, monkeypatch, phase='draft_responses'):
    """Use the real tagged operator and cumulative ledger with only a fake provider."""
    from omegaconf import OmegaConf
    config, _ = reuse_config(tmp_path, [dict(scenario_id=f'original_{i}',
                                           original_system='Write accurate artifacts.',
                                           user=f'Write fixture artifact {i}.',
                                           needs_revision=(i < 2), accepted=(i < 2))
                                      for i in range(8)])
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    cfg.update(pilot_mode='tagged_reuse', pipeline='nonmoral-paired-draft', max_fail_pct=100,
               constitution=str(Path(cfg['constitution']).resolve()),
               output_dir='output/nonmoral_paired_reuse_pilot')
    cfg['models'] = {'respond': cfg['models']['answer']}
    stage = dict(name=phase, kind='llm_tagged', model='respond', checkpoint='scenario_id',
                 tags=['reasoning', 'response'], save=dict(comparative='reasoning', answer='response'),
                 prompts=dict(system='{original_system}', user='{user}'))
    if phase != 'draft_responses':
        stage['when'] = {'field': 'needs_revision' if phase == 'revise_responses' else 'accepted', 'in': [True]}
    cfg['stages'] = [dict(name='original_inputs', kind='load_source_run'), stage]
    OmegaConf.save(OmegaConf.create(cfg), config)
    monkeypatch.chdir(tmp_path)
    calls = []
    class FakeChat:
        def retry_with(self, **kwargs):
            def single(client, **request):
                calls.append(request)
                return SimpleNamespace(content='<reasoning>Compare viable options.</reasoning>'
                                               '<response>Complete fixture artifact.</response>',
                                       prompt_tokens=20, completion_tokens=30, cached_tokens=0,
                                       provider='offline-fake', finish_reason='stop')
            return single
    class FakeClient:
        chat = FakeChat()
    monkeypatch.setattr('scratch.nonmoral.pilot.OpenRouterClient', FakeClient)
    monkeypatch.setattr('scratch.nonmoral.pilot.verify_live_prices', lambda models: [])
    monkeypatch.setattr('sys.argv', ['pilot', '--config', str(config), '--execute'])
    return config, tmp_path / cfg['output_dir'], calls


def test_tagged_shared_engine_preserves_prior_spend_and_reviews_filtered_phase(tmp_path, monkeypatch):
    config, root, calls = tagged_fixture(tmp_path, monkeypatch, 'revise_responses')
    cfg, _, _ = validate_config(config)
    source = Path(cfg['source']['local_dir'])
    prior = tmp_path/'prior_snapshot.jsonl'
    prior.write_bytes((source/'inputs.jsonl').read_bytes())
    review = dict(approved=True, stage_name='revise_responses', input_sha256=cfg['pilot_input_sha256'],
                  prior_snapshot=str(prior), prior_snapshot_sha256=file_sha256(prior),
                  reviewer='offline-test', reviewed_at='2026-09-08T00:00:00Z')
    (source/'manifest.json').write_text(json.dumps({'phase_review': review}))
    root.mkdir(parents=True)
    previous = dict(status='settled', charged_or_reserved_usd=.187348)
    (root/'spend.json').write_text(json.dumps([previous]))
    main()
    run_dir = next(root.glob('*_revise_responses'))
    rows = [json.loads(s) for s in (run_dir/'dataset.jsonl').read_text().splitlines()]
    assert len(rows) == 8 and len(calls) == 2
    assert sum('answer' in r for r in rows) == 2
    assert all(c['extra_body'] == {'reasoning': {'enabled': False}} for c in calls)
    assert json.loads((run_dir/'phase_review.json').read_text())['review'] == review
    ledger = json.loads((root/'spend.json').read_text())
    assert ledger[0] == previous and len(ledger) == 3
    assert sum(e['charged_or_reserved_usd'] for e in ledger) == pytest.approx(.188028)


def test_tagged_duplicate_config_cannot_dispatch_again(tmp_path, monkeypatch):
    config, root, calls = tagged_fixture(tmp_path, monkeypatch)
    main()
    assert len(calls) == 8 and (root/'dispatches'/f'{file_sha256(config)}.json').exists()
    monkeypatch.setattr('scratch.nonmoral.pilot.verify_live_prices',
                        lambda *a: pytest.fail('Duplicate reached network boundary'))
    with pytest.raises(ValueError, match='already dispatched'):
        main()
    assert len(calls) == 8


def test_tagged_missing_review_fails_before_network(tmp_path, monkeypatch):
    _, root, calls = tagged_fixture(tmp_path, monkeypatch, 'verification')
    monkeypatch.setattr('scratch.nonmoral.pilot.verify_live_prices',
                        lambda *a: pytest.fail('Unreviewed source reached network boundary'))
    with pytest.raises(ValueError, match='requires source manifest phase_review'):
        main()
    assert calls == [] and not root.exists()
