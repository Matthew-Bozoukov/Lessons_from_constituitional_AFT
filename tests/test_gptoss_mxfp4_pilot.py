# ABOUTME: Check pilot isolation, matched settings and the actual SWE driver preparation offline.
# ABOUTME: Never rent GPUs or send generation requests during validation.
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from omegaconf import OmegaConf
from scratch.gptoss_swe import mxfp4_pilot as pilot

def test_backend_settings_match():
    a=OmegaConf.load(pilot.ROOT/'tinker/swe.yaml');b=OmegaConf.load(pilot.ROOT/'mxfp4/swe.yaml')
    assert a.worker==b.worker and a.sampling==b.sampling and a.instance_ids==b.instance_ids
    a=OmegaConf.load(pilot.ROOT/'tinker/odcv.yaml');b=OmegaConf.load(pilot.ROOT/'mxfp4/odcv.yaml')
    for key in ('include_scenarios','temperature','serving','passes','strict_tool_validation','system_preamble_file','scenario_timeout_s'):
        assert a[key]==b[key]
    assert a.serving.context_window==112000 and a.passes==3

def test_gpu_script_no_chat_template_or_lora():
    script=pilot.bootstrap()
    assert '--generation-config vllm' in script and '--kv-cache-dtype auto' in script
    assert '--chat-template' not in script and '--enable-lora' not in script
    assert '--revision '+pilot.read(pilot.ROOT/'model.json')['revision'] in script

@pytest.mark.skipif(__import__('sys').platform=='win32',reason='Actual fleet state uses Linux fcntl')
def test_linux_swe_preparation(tmp_path,monkeypatch):
    from src.eval.capabilities.swebench_mini import fleet_worker
    from scratch.gptoss_swe import finish_smoke
    import requests
    oldroot=pilot.ROOT
    cfg=OmegaConf.load(oldroot/'tinker/swe.yaml')
    root=tmp_path/'pilot';(root/'tinker').mkdir(parents=True)
    OmegaConf.save(cfg,root/'tinker/swe.yaml')
    monkeypatch.setattr(pilot,'ROOT',root)
    monkeypatch.setenv('TINKER_API_KEY','offline')
    monkeypatch.setattr(requests,'post',lambda *a,**k:SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'count':5}))
    called=[]
    def fake_pool(consume,endpoint,worker,arm,ids,allowances,admission,workers):
        meta=pilot.read(Path(worker.root)/'metadata/manifest.json')
        state=pilot.read(Path(worker.root)/'metadata/state.json')
        assert meta['campaign']==str(cfg.campaign)
        assert len(state['tasks'])==4 and workers==4 and ids==list(cfg.instance_ids)
        assert worker.max_response_tokens==16384 and worker.max_task_tokens==262144
        assert all(v['max_total_attempt_records']==3 for v in allowances.values())
        called.append(True)
    monkeypatch.setattr(finish_smoke,'consume_pool',fake_pool)
    pilot.linux_swe('consume','tinker')
    assert called==[True]
