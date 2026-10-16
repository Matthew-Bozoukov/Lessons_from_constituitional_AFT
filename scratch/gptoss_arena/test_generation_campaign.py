# ABOUTME: Offline checks for Arena publication arm registration and guard contention recovery.
# ABOUTME: Neither test creates a sampler or submits a paid request.
from scratch.gptoss_arena import campaign, host
from src.eval.capabilities.arena_hard.runner import _generation_settings


def test_publication_registers_all_arms_without_changing_sampling():
    cfg = campaign.comparison_config()
    assert {a.name for a in cfg.arms} == {campaign.key(a) for a in campaign.TARGETS}
    assert _generation_settings(cfg,'think') == _generation_settings(campaign.config(),'think')


def test_guard_retries_transient_replace_contention(monkeypatch):
    attempts = []
    def guarded():
        attempts.append(1)
        if len(attempts) == 1:
            raise PermissionError('Windows sharing contention')
    monkeypatch.setattr(host.host,'guard',guarded)
    monkeypatch.setattr(host.time,'sleep',lambda seconds: None)
    host.guard()
    assert len(attempts) == 2
