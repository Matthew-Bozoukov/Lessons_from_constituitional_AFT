# ABOUTME: Offline tests of exact mixture preparation and fail-closed publication-card provenance.
import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from scratch.dataset_refresh import prepare_mixtures as mod
from src.data.mixture.build_mixture import blend


def pins():
    return {style: (f'test-org/2026-09-15-{style}-synth', '0123456789abcdef' * 2 + '01234567') for style in mod.STYLES}


def test_two_configs_exact_dose_same_replay_and_no_generation_or_publish(tmp_path):
    manifest = mod.prepare(tmp_path / 'prepared', pins())
    replay = []
    for style, entry in manifest['configs'].items():
        cfg = OmegaConf.to_container(OmegaConf.load(entry['path']), resolve=True)
        assert Path(entry['path']).stem == style
        assert cfg['seed'] == 0 and cfg['max_seq_len'] == 8192
        assert cfg['synthetic_examples'] == 716 and cfg['synthetic_pct'] == 7
        assert cfg['base_mixture']['revision'] == mod.BASE_REVISION
        assert not any(k in cfg for k in ('hf', 'filter', 'reasoning_backfill'))
        quotas = blend(OmegaConf.to_container(OmegaConf.load(cfg['base']))['sources'], cfg['sources'], 7, 10000, synthetic_examples=716)
        assert quotas[style]['examples'] == 716
        replay.append({k: v['examples'] for k, v in quotas.items() if k != style})
    assert replay == [mod.REPLAY_COUNTS, mod.REPLAY_COUNTS]
    assert sum(replay[0].values()) == 9284


@pytest.mark.parametrize('bad', ['main', 'TO_BE_SUPPLIED', '7e991f58', 'a' * 40])
def test_missing_real_commit_fails_before_any_prepared_file(tmp_path, bad):
    specs = pins()
    specs[mod.STYLES[0]] = (specs[mod.STYLES[0]][0], bad)
    with pytest.raises(ValueError):
        mod.prepare(tmp_path / 'prepared', specs)
    assert not (tmp_path / 'prepared').exists()


def test_wrong_corpus_style_rejected(tmp_path):
    specs = pins()
    specs[mod.STYLES[0]] = specs[mod.STYLES[1]]
    with pytest.raises(ValueError, match='expected style'):
        mod.prepare(tmp_path / 'prepared', specs)


def card_fixture(tmp_path, style=mod.STYLES[0]):
    prepared = tmp_path / 'prepared'
    manifest = mod.prepare(prepared, pins())
    cfgpath = Path(manifest['configs'][style]['path'])
    cfg = OmegaConf.to_container(OmegaConf.load(cfgpath), resolve=True)
    dirs = [tmp_path / 'low_build', tmp_path / 'nonmoral_build']
    records = []
    for d in dirs:
        d.mkdir()
        # Stand-in bytes: the card gate consumes a trusted audit, not a second tokenizer run.
        (d / 'mixture.jsonl').write_text('fixture bytes', encoding='utf-8')
        records.append({'path': str(d / 'mixture.jsonl'), 'sha256': mod.sha(d / 'mixture.jsonl'), 'rows': 10000, 'synthetic_rows': 716, 'replay_rows': 9284})
    mod.write_json(dirs[0] / 'run_meta.json', {'smoke': False, 'config': cfg, 'timestamp_utc': '2026-09-15T11:00:00+00:00', 'git_sha': 'f' * 40, 'command': f'uv run mix --config {cfgpath}'})
    mod.write_json(dirs[0] / 'mixture_stats.json', {'by_source': {k: {'examples': n} for k,n in {**mod.REPLAY_COUNTS,style:716}.items()}, 'reasoning_traces': {'model': 'qwen/qwen3.6-27b', 'family': 'qwen36', 'inherited_from': {'repo': mod.BASE_REPO, 'revision': mod.BASE_REVISION}}})
    audit = tmp_path / 'audit.json'
    mod.write_json(audit, {'status': 'passed', 'replay_payloads_and_positions_equal': True, 'max_train_tokens': 8192, 'base': {'repo': mod.BASE_REPO, 'revision': mod.BASE_REVISION}, 'mixtures': records})
    sourcecfg = tmp_path / 'synth_config.json'
    mod.write_json(sourcecfg, {'pipeline': style, 'constitution': mod.CONSTITUTION, 'constitution_sha256': '1a' * 32, 'models': {'respond': {'model': 'anthropic/claude-sonnet-5'}}, 'craft_spec': 'preferences/craft_tensions_09/preferences.md', 'craft_spec_sha256': '2b' * 32})
    return cfgpath, dirs, audit, sourcecfg


@pytest.mark.parametrize('style', mod.STYLES)
def test_card_records_exact_share_native_reasoning_and_pins(tmp_path, monkeypatch, style):
    monkeypatch.setattr(mod, 'origin_url', lambda: 'test-repository')
    cfg, dirs, audit, sourcecfg = card_fixture(tmp_path, style)
    assert mod.prepare_card(cfg, dirs[0], audit, sourcecfg) == f'2026-09-15-{style}-7-mix'
    card = (dirs[0] / 'README.md').read_text(encoding='utf-8')
    for evidence in ('7.16%', '9,284', 'native reasoning_content', mod.BASE_REVISION, 'anthropic/claude-sonnet-5', 'qwen/qwen3.6-27b'):
        assert evidence in card
    assert (dirs[0] / 'frozen_synthetic_config.json').read_bytes() == sourcecfg.read_bytes()
    if style == 'nonmoral-advice':
        assert 'compatibility review target' in card and 'craft_spec_sha256' in card


def test_changed_other_arm_invalidates_joint_audit(tmp_path):
    cfg, dirs, audit, sourcecfg = card_fixture(tmp_path)
    (dirs[1] / 'mixture.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='Audit inputs changed'):
        mod.prepare_card(cfg, dirs[0], audit, sourcecfg)
    assert not (dirs[0] / 'README.md').exists()
