# ABOUTME: Verify the approved one-pass protocol and first-cell gate without paid calls.
# ABOUTME: Failed first cells must stop dispatch; successful cells are counted exactly once.
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from scratch.nonmoral import overnight_baseline as owner
from src.eval.misalignment.odcv import odcv_rollout as rollout


def test_refresh_plan_keeps_one_pass_and_separate_port(tmp_path):
    cfg = OmegaConf.load('configs/eval/odcv/lite.yaml')
    cfg.passes = 1
    cfg.concurrency = 6
    cfg.judge_workers = 4
    cfg.preflight_first_cell = True
    config = tmp_path / 'config.yaml'
    OmegaConf.save(cfg, config)
    plan = dict(target='dougalldeepmind/2026-09-15-qwen36-0-nonmoral-advice-7',
                target_revision='a'*40, base_model='Qwen/Qwen3.6-27B', base_revision=owner.BASE_REVISION,
                output_dir=str(tmp_path/'new'), run_name='odcv-refresh-nonmoral',
                eval_output_root='C:/odcv-non', eval_config=str(config),
                eval_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                expected_cells=80, passes=1, gpu_cap_usd=12, judge_cap_usd=3,
                backup_reserve_usd=2, max_gpu_hourly_usd=3.5, storage_hourly_reserve_usd=.1,
                port=18111, combined_networks=24)
    path=tmp_path/'plan.yaml'; OmegaConf.save(OmegaConf.create(plan), path)
    assert owner.load_plan(path)['port'] == 18111
    plan['passes']=3; OmegaConf.save(OmegaConf.create(plan), path)
    with pytest.raises(ValueError, match='1x80'):
        owner.load_plan(path)


@pytest.mark.parametrize('healthy', [True, False])
def test_first_cell_is_scored_once_or_stops_dispatch(tmp_path, monkeypatch, healthy):
    bench=tmp_path/'bench'; bench.mkdir()
    cfg=OmegaConf.create(dict(bench_dir=str(bench), model='test', model_key='test',
        temperature=.7, concurrency=2, output_root=str(tmp_path/'runs'), preflight_first_cell=True))
    config=tmp_path/'cfg.yaml'; OmegaConf.save(cfg,config)
    monkeypatch.setattr(rollout,'scenario_names',lambda *a: ['Demo'])
    monkeypatch.setattr(rollout,'openrouter_usage',lambda **k: None)
    monkeypatch.setattr(rollout,'write_run_meta',lambda *a,**k: None)
    calls=[]
    def run(cfg, bench, out, variant, scenario):
        calls.append((variant,scenario))
        if healthy:
            record=out/'agent_logs'/f'test-{variant}'/'experiments'/scenario/'messages_record.txt'
            record.parent.mkdir(parents=True); record.write_text('Task and tool actions')
        return dict(variant=variant,scenario=scenario,status='ok' if healthy else 'ok+no_transcript',elapsed_s=1)
    monkeypatch.setattr(rollout,'_run_scenario',run)
    if healthy:
        out=rollout.main(str(config))
        manifest=json.loads((out/'rollout_manifest.json').read_text())
        assert len(calls)==len(set(calls))==len(manifest['results'])==2
    else:
        with pytest.raises(RuntimeError,match='remaining cells not dispatched'):
            rollout.main(str(config))
        assert len(calls)==1
