# ABOUTME: Verify independent single-arm release keeps the exact paired replay, dosage and native training checks.
# ABOUTME: All fixtures are temporary and all tokenizer/model work is mocked except existing builder sampling.
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from scratch.dataset_refresh import prepare_mixtures as prep, validate_mixtures as audit
from scratch.dataset_refresh.run import quotas, write_rows
from src.data.mixture import build_mixture as builder


@pytest.fixture
def sample(tmp_path):
    base = []
    for source, count in audit.BASE_COUNTS.items():
        for i in range(count):
            base.append({'source': source, 'messages': [{'role': 'user', 'content': f'{source}:{i}'},
                          {'role': 'assistant', 'content': 'A.'}]})
    random.Random(42).shuffle(base)
    basepath = tmp_path / 'base.jsonl'
    write_rows(basepath, base)
    refpath = tmp_path / 'shared_replay.json'
    audit.freeze_replay_reference(basepath, refpath)
    reference = audit.load_replay_reference(refpath)
    released = []
    for trait, count in quotas().items():
        for i in range(count):
            released.append({'metadata': {'trait_id': trait, 'scenario_id': f'{trait}_{i}'},
                'messages': [{'role': 'system', 'content': 'Give advice.'},
                    {'role': 'user', 'content': f'{trait}:decision{i}'},
                    {'role': 'assistant', 'content': 'Supported advice.', 'reasoning_content': 'Weigh facts.'}]})
    sourcepath = tmp_path / 'dataset.jsonl'
    write_rows(sourcepath, released)
    source = {'path': sourcepath, 'style': 'da-lowstakes-refresh',
              'repo': 'test/2026-09-15-da-lowstakes-refresh-synth', 'revision': '0123456789abcdef'*2+'01234567'}
    synth = iter([{'messages': r['messages'], 'source': source['style'], 'supervise': 'all'} for r in released])
    slots = {i: base[index] for i, index, _ in reference['positions']}
    mixed = [slots[i] if i in slots else next(synth) for i in range(10000)]
    mixture = tmp_path / 'mixture.jsonl'
    write_rows(mixture, mixed)
    return SimpleNamespace(base=base, basepath=basepath, refpath=refpath, reference=reference,
                           source=source, mixed=mixed, mixture=mixture)


def fake_tokens(monkeypatch):
    calls = []
    def count(row, *args):
        calls.append(row)
        return {'training_tokens': 10, 'supervised_tokens': 4, 'raw_reasoning_tokens': 1,
                'raw_assistant_content_tokens': 3, 'reasoning_turns': 1}
    monkeypatch.setattr(audit, 'token_audit', count)
    tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda: 'mock-tokenizer'))
    return tokenizer, calls


def run_single(sample, tokenizer, **kwargs):
    return audit.audit([sample.mixture], sample.basepath, tokenizer, single_arm=True,
        synthetic_sources=[sample.source], replay_reference_path=sample.refpath, **kwargs)


def test_reference_matches_existing_builder_sampling_and_both_shuffles(sample, monkeypatch, tmp_path):
    # Exercise the real loader and exact blend, not a second copy of our derivation.
    import src.infra.huggingface as hf
    monkeypatch.setattr(hf, 'resolve_dataset', lambda *args: (sample.basepath, {'revision': audit.BASE_REVISION}))
    monkeypatch.setattr(builder, 'render_chat', lambda *args, **kwargs: {'input_ids': [1]})
    proportions = tmp_path / 'proportions.yaml'
    OmegaConf.save(OmegaConf.create({'sources': {k: {'examples': n} for k,n in prep.BASE_COUNTS.items()}}), proportions)
    cfg = OmegaConf.create({'base': str(proportions), 'base_mixture': {'repo': audit.BASE_REPO,
        'revision': audit.BASE_REVISION, 'file': 'mixture.jsonl'}, 'max_seq_len': 8192})
    specs = builder.blend({k: {'examples': n} for k,n in prep.BASE_COUNTS.items()},
                         {'synthetic': {'examples': 1}}, 7, 10000, synthetic_examples=716)
    specs = {k:v for k,v in specs.items() if k != 'synthetic'}
    replay, _ = builder._load_published_base(None, cfg, specs, 1, 0, {})
    replay = [{k:v for k,v in r.items() if k!='n_tokens'} for r in replay]
    random.Random(0).shuffle(replay)
    merged = replay + [None]*716
    random.Random(0).shuffle(merged)
    assert audit.digest([[i,r] for i,r in enumerate(merged) if r is not None]) == sample.reference['replay_positions_sha256']


def test_single_arm_audits_every_row_without_claiming_observed_pair(sample, monkeypatch):
    tokenizer, calls = fake_tokens(monkeypatch)
    result = run_single(sample, tokenizer)
    assert len(calls) == 10000
    assert result['status'] == 'passed'
    assert result['replay_payloads_and_positions_equal'] is False
    assert result['replay_matches_frozen_reference'] is True
    assert result['audit_mode'] == 'single_arm_replay_reference'
    assert result['synthetic_payloads_equal_pinned_releases'] is True
    assert result['mixtures'][0]['synthetic_rows'] == 716
    assert result['mixtures'][0]['replay_rows'] == 9284


@pytest.mark.parametrize('change', ['swap_positions', 'change_subset', 'forge_reference'])
def test_single_arm_rejects_changed_replay_even_with_right_counts(sample, monkeypatch, change):
    tokenizer, calls = fake_tokens(monkeypatch)
    if change == 'forge_reference':
        value = deepcopy(sample.reference)
        value['positions'][0][0] += 1
        sample.refpath.write_text(json.dumps(value), encoding='utf-8')
    else:
        indices = [i for i,r in enumerate(sample.mixed) if r['source']=='no_robots']
        if change == 'swap_positions':
            a,b = indices[:2]
            sample.mixed[a], sample.mixed[b] = sample.mixed[b], sample.mixed[a]
        else:
            selected = {audit.canonical(r) for r in sample.mixed}
            unused = next(r for r in sample.base if r['source']=='no_robots' and audit.canonical(r) not in selected)
            sample.mixed[indices[0]] = unused
        write_rows(sample.mixture, sample.mixed)
    with pytest.raises(ValueError, match='replay|Replay'):
        run_single(sample, tokenizer)
    assert not calls


def test_single_arm_propagates_token_or_mask_failure(sample, monkeypatch):
    def fail(*args):
        raise ValueError('No assistant tokens supervised')
    monkeypatch.setattr(audit, 'token_audit', fail)
    with pytest.raises(ValueError, match='No assistant tokens supervised'):
        run_single(sample, None)


def test_old_paired_default_still_rejects_one_file(sample):
    with pytest.raises(ValueError, match='exactly two'):
        audit.audit([sample.mixture], sample.basepath, None, synthetic_sources=[sample.source])


def test_single_arm_requires_reference_pin_and_exact_cap(sample):
    for args in [{'synthetic_sources': [sample.source]}, {'replay_reference_path': sample.refpath},
                 {'synthetic_sources': [sample.source], 'replay_reference_path': sample.refpath, 'max_length': 16384}]:
        with pytest.raises(ValueError, match='requires a frozen replay reference'):
            audit.audit([sample.mixture], sample.basepath, None, single_arm=True, **args)


def test_reference_cannot_be_overwritten(sample):
    before=sample.refpath.read_bytes()
    with pytest.raises(ValueError, match='immutable'):
        audit.freeze_replay_reference(sample.basepath, sample.refpath)
    assert sample.refpath.read_bytes()==before


def test_single_prepare_keeps_exact_old_config_and_copies_shared_reference(sample, tmp_path):
    pin={sample.source['style']: (sample.source['repo'], sample.source['revision'])}
    manifest=prep.prepare(tmp_path/'prepared', pin, single_arm=True, replay_reference=sample.refpath)
    assert set(manifest['configs'])=={'da-lowstakes-refresh'}
    assert manifest['single_arm'] is True
    assert Path(manifest['replay_reference']['path']).read_bytes()==sample.refpath.read_bytes()
    cfg=OmegaConf.to_container(OmegaConf.load(manifest['configs']['da-lowstakes-refresh']['path']),resolve=True)
    assert cfg['synthetic_examples']==716 and cfg['total_examples']==10000 and cfg['seed']==0
    assert not any(k in cfg for k in ('hf','filter','reasoning_backfill'))
    # Later nonmoral prep consumes the same bytes; it does not resample.
    other=prep.prepare(tmp_path/'later', {'nonmoral-advice': ('test/2026-09-15-nonmoral-advice-synth', sample.source['revision'])},
                       single_arm=True, replay_reference=sample.refpath)
    assert other['replay_reference']['sha256']==manifest['replay_reference']['sha256']


def single_card_fixture(sample, tmp_path, monkeypatch):
    tokenizer,_=fake_tokens(monkeypatch)
    report=run_single(sample,tokenizer)
    auditpath=tmp_path/'audit.json';prep.write_json(auditpath,report)
    manifest=prep.prepare(tmp_path/'prepared', {sample.source['style']:(sample.source['repo'],sample.source['revision'])},
                          single_arm=True,replay_reference=sample.refpath)
    cfgpath=Path(manifest['configs'][sample.source['style']]['path'])
    cfg=OmegaConf.to_container(OmegaConf.load(cfgpath),resolve=True)
    # The card checks this exact local mixture and its bound report.
    prep.write_json(tmp_path/'run_meta.json',{'smoke':False,'config':cfg,'timestamp_utc':'2026-09-15T11:00:00+00:00',
                    'git_sha':'a'*40,'command':'uv run mix --config '+str(cfgpath)})
    prep.write_json(tmp_path/'mixture_stats.json', {'by_source':{k:{'examples':n} for k,n in {**prep.REPLAY_COUNTS,sample.source['style']:716}.items()},
        'reasoning_traces':{'model':'qwen/qwen3.6-27b','family':'qwen36','inherited_from':{'repo':prep.BASE_REPO,'revision':prep.BASE_REVISION}}})
    sourcecfg=tmp_path/'synth_config.json'
    prep.write_json(sourcecfg,{'pipeline':sample.source['style'],'constitution':prep.CONSTITUTION,
        'constitution_sha256':'ab'*32,'models':{'respond':{'model':'anthropic/claude-sonnet-5'}}})
    monkeypatch.setattr(prep,'origin_url',lambda:'test-repository')
    return cfgpath,auditpath,sourcecfg,manifest


def test_single_card_requires_reference_and_discloses_no_observed_pair(sample,tmp_path,monkeypatch):
    cfg,report,sourcecfg,manifest=single_card_fixture(sample,tmp_path,monkeypatch)
    name=prep.prepare_card(cfg,tmp_path,report,sourcecfg)
    assert name=='2026-09-15-da-lowstakes-refresh-7-mix'
    card=(tmp_path/'README.md').read_text()
    assert 'no observed two-arm equality is claimed yet' in card
    assert (tmp_path/'replay_reference.json').read_bytes()==sample.refpath.read_bytes()


@pytest.mark.parametrize('change',['reference_bytes','false_pair_claim','reference_positions','audit_reference_sha'])
def test_single_card_rejects_changed_or_misrepresented_reference(sample,tmp_path,monkeypatch,change):
    cfg,report,sourcecfg,manifest=single_card_fixture(sample,tmp_path,monkeypatch)
    value=json.loads(report.read_text())
    if change=='reference_bytes':
        Path(manifest['replay_reference']['path']).write_text('{}')
    elif change=='false_pair_claim':value['replay_payloads_and_positions_equal']=True
    elif change=='reference_positions':value['mixtures'][0]['replay_positions_sha256']='bad'
    else:value['replay_reference']['sha256']='bad'
    prep.write_json(report,value)
    with pytest.raises(ValueError,match='replay'):
        prep.prepare_card(cfg,tmp_path,report,sourcecfg)
    assert not (tmp_path/'README.md').exists()


def test_paired_default_still_checks_actual_cross_arm_positions(sample, tmp_path, monkeypatch):
    tokenizer,_=fake_tokens(monkeypatch)
    other=tmp_path/'other_mixture.jsonl'
    write_rows(other,sample.mixed)
    result=audit.audit([sample.mixture,other],sample.basepath,tokenizer,synthetic_sources=[sample.source,sample.source])
    assert result['audit_mode']=='paired' and result['replay_payloads_and_positions_equal'] is True
    assert result['replay_matches_frozen_reference'] is False
    changed=deepcopy(sample.mixed)
    indices=[i for i,r in enumerate(changed) if r['source']=='no_robots']
    a,b=indices[:2];changed[a],changed[b]=changed[b],changed[a]
    write_rows(other,changed)
    with pytest.raises(ValueError,match='Arms differ'):
        audit.audit([sample.mixture,other],sample.basepath,tokenizer,synthetic_sources=[sample.source,sample.source])
