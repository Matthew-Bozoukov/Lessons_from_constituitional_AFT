# ABOUTME: Verify the approved one-pass protocol and first-cell gate without paid calls.
# ABOUTME: Failed first cells must stop dispatch; successful cells are counted exactly once.
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from scratch.nonmoral import overnight_baseline as owner
from src.eval.misalignment.odcv import odcv_rollout as rollout


@pytest.mark.parametrize('restart', [False, True])
def test_three_pass_server_continuity_is_checked(tmp_path, monkeypatch, restart):
    from src.eval.misalignment.odcv import runner as runner_module
    import requests
    config=tmp_path/'config.yaml'
    config.write_text('passes: 3\n')
    plan_path=tmp_path/'plan.yaml'
    OmegaConf.save(OmegaConf.create(dict(protocol='refresh-three-pass',port=18121)),plan_path)
    (tmp_path/'broader_eval_status.json').write_text(json.dumps(dict(
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())))
    monkeypatch.setattr(owner,'checked_spec',lambda p:SimpleNamespace(hf_path='test'))
    pids=iter(['111','222'] if restart else ['111']*6)
    monkeypatch.setattr(owner,'SshExec',lambda *a,**k:SimpleNamespace(_ssh=lambda *a,**k:next(pids)))
    monkeypatch.setattr(requests,'get',lambda *a,**k:SimpleNamespace(raise_for_status=lambda:None))
    called=[]
    def original(*args):
        called.append(1)
        return {'clean':True}
    monkeypatch.setattr(runner_module,'_run_pass',original)
    publish=tmp_path/'publish'
    (publish/'metadata').mkdir(parents=True)
    def fake_run(target,cfg,out):
        for i in range(3):
            runner_module._run_pass(config,False)
        return {'complete':True}
    monkeypatch.setattr(runner_module,'run',fake_run)
    monkeypatch.setattr(owner,'evaluate',lambda argv,runner:runner(None,None,publish))
    if restart:
        with pytest.raises(RuntimeError,match='continuity failed'):
            owner.evaluate_frozen(plan_path,config,'host','identity')
        assert len(called)==1
    else:
        owner.evaluate_frozen(plan_path,config,'host','identity')
        assert len(called)==3
        assert len(json.loads((publish/'metadata/server_continuity.json').read_text()))==6


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
                output_dir=str(tmp_path/'new'), run_name='odcv-refresh-test-only-never-rent',
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


def test_three_pass_opt_in_retains_refresh_concurrency_and_budget(tmp_path):
    cfg = OmegaConf.load('configs/eval/odcv/lite.yaml')
    cfg.passes = 3
    cfg.concurrency = 6
    cfg.judge_workers = 4
    cfg.preflight_first_cell = True
    config = tmp_path/'config.yaml'
    OmegaConf.save(cfg, config)
    plan = dict(target='dougalldeepmind/2026-09-15-qwen36-0-nonmoral-original-7',
                target_revision='a'*40, base_model='Qwen/Qwen3.6-27B', base_revision=owner.BASE_REVISION,
                output_dir=str(tmp_path/'new'), run_name='odcv-refresh-three-test-never-rent',
                eval_output_root='C:/odcv-three', eval_config=str(config),
                eval_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                expected_cells=80, passes=3, gpu_cap_usd=25, judge_cap_usd=5,
                backup_reserve_usd=2, max_gpu_hourly_usd=3.5, storage_hourly_reserve_usd=.1,
                port=18122, combined_networks=24, protocol='refresh-three-pass')
    path=tmp_path/'plan.yaml'
    OmegaConf.save(OmegaConf.create(plan), path)
    assert owner.load_plan(path)['passes'] == 3
    plan['gpu_cap_usd']=25.01
    OmegaConf.save(OmegaConf.create(plan), path)
    with pytest.raises(ValueError, match='30 allocation'):
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
