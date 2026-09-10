# ABOUTME: Deadline outcomes preserve captured evidence and never become fresh paid retries.
# ABOUTME: Offline fake-Compose tests cover partial recovery and an unrecoverable timeout.
import json
import subprocess
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.eval.misalignment.odcv import odcv_rollout as rollout


@pytest.mark.parametrize('has_events', [True, False])
def test_timeout_preserves_log_status_and_skips_resume(monkeypatch, tmp_path, has_events):
    bench=tmp_path/'bench'
    scenario=bench/'mandated_scenarios'/'Demo'
    scenario.mkdir(parents=True)
    (scenario/'prompt.json').write_text(json.dumps(dict(system_prompt='SYSTEM',user_prompt='TASK')))
    captured=(b'executor-1 | [Reasoning]: Read the actual file.\n'
              b'executor-1 | [Tool]: bash with {"command":"cat input.txt"}\n'
              b'executor-1 | [out]: {"stdout":"facts", "returncode":0}\n') if has_events else b'build started\n'
    calls=[]
    def compose(project, ws, env, args, timeout):
        calls.append(args)
        if args[0]=='up':
            raise subprocess.TimeoutExpired('compose', timeout, output=captured)
        return SimpleNamespace(stdout='owned-container' if args[0]=='ps' else '',returncode=0)
    monkeypatch.setattr(rollout,'_compose',compose)
    monkeypatch.setattr(rollout,'_build_workspace',lambda *a: a[-1].mkdir(parents=True))
    monkeypatch.setattr(rollout.subprocess,'run',lambda *a,**k: SimpleNamespace(returncode=1))
    monkeypatch.setenv('OPENROUTER_API_KEY','unused-test-key')
    cfg=OmegaConf.create(dict(model_key='test',model='test',base_url='http://unused',temperature=.7,
                             scenario_timeout_s=2400,serving=dict(context_window=28000),prune_images=False))
    out=tmp_path/'out'
    result=rollout._run_scenario(cfg,bench,out,'mandated','Demo')
    dest=out/'agent_logs'/'test-mandated'/'experiments'/'Demo'
    assert captured.decode() in (dest/'docker_output.log').read_text()
    assert result['status'].startswith('timeout') and result['transcript_partial']
    assert result['transcript_source']==('docker_log_reconstruction' if has_events else 'unavailable')
    if has_events:
        text=(dest/'messages_record.txt').read_text()
        assert 'facts' in text and 'partial transcript: driver timeout' in text
        assert 'task_complete' not in text
    else:
        assert not (dest/'messages_record.txt').exists()
    before=len(calls)
    resumed=rollout._run_scenario(cfg,bench,out,'mandated','Demo')
    assert len(calls)==before
    assert resumed['status']==result['status'] and resumed['cache_hit']
