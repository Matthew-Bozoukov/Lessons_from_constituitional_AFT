# ABOUTME: Verify the Windows short-path alias and immutable ODCV cache before recovery.
# ABOUTME: Exercise cached scenario dispatch without model requests or Docker execution.
from pathlib import Path
import pytest
from omegaconf import OmegaConf
from scratch.gptoss_swe import mxfp4_path_recovery as recovery
from src.eval.misalignment.odcv import odcv_rollout


@pytest.mark.skipif(not recovery.ALIAS.exists(),reason='Local Windows pilot alias required')
def test_alias_preserves_exact_workspace():
    ws=next((recovery.pilot.ROOT/'mxfp4/odcv').glob('*/20*/workspaces/incentivized/Ai-Moderator-Content-Censorship-Backdoor'))
    short=recovery.short_path(ws)
    assert short.resolve()==ws.resolve() and len(str(short))<240
    assert (short/'docker-compose.yml').read_bytes()==(ws/'docker-compose.yml').read_bytes()


def test_short_path_rejects_outside_repo(tmp_path):
    with pytest.raises(ValueError):recovery.short_path(tmp_path)


@pytest.mark.skipif(not (recovery.pilot.ROOT/'mxfp4/odcv.yaml').exists(),reason='Local saved pilot required')
def test_all_nine_paid_transcripts_are_cached(monkeypatch):
    r=recovery.pilot.ROOT/'mxfp4';cfg=OmegaConf.load(r/'odcv.yaml')
    old=next((r/'odcv'/cfg.model_key).glob('20*'))
    records=list(old.glob('agent_logs/*/experiments/*/messages_record.txt'));assert len(records)==9
    def forbidden(*a,**k):raise AssertionError('Cached outcomes must never execute Docker or rebuild workspace')
    monkeypatch.setattr(odcv_rollout,'_compose',forbidden)
    monkeypatch.setattr(odcv_rollout,'_build_workspace',forbidden)
    for rec in records:
        variant='incentivized' if '-incentivized' in str(rec) else 'mandated'
        assert odcv_rollout._run_scenario(cfg,Path(cfg.bench_dir),old,variant,rec.parent.name)['status']=='cached'
