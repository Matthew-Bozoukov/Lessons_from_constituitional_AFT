# ABOUTME: Docker missing-container checks recognize case variants without hiding other errors.
# ABOUTME: Offline coverage for the completion owner's preflight ownership fence.
from types import SimpleNamespace
import pytest
from scratch.gptoss_swe import finish_smoke


@pytest.mark.parametrize('error',['error: no such object: old','Error: No such container: old'])
def test_missing_container_is_stopped(monkeypatch,error):
    monkeypatch.setattr(finish_smoke.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stderr=error))
    assert finish_smoke.running('old') is False


def test_daemon_error_is_not_stopped(monkeypatch):
    monkeypatch.setattr(finish_smoke.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stderr='cannot connect to daemon'))
    with pytest.raises(AssertionError):finish_smoke.running('old')
