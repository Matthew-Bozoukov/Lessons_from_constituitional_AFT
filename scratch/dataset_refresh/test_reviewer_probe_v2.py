# ABOUTME: Verify v2 preserves frozen examples, omits reasoning override, and shares its $2 cap.
import json

import pytest

from scratch.dataset_refresh import reviewer_probe_v2 as v2
from scratch.dataset_refresh.test_reviewer_probe import frozen
from src.infra.endpoints.openrouter import ChatResult


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(v2.runtime, 'OpenRouterClient', lambda: pytest.fail('Paid transport forbidden'))
    monkeypatch.setattr(v2.runtime, 'provider_price', lambda _: {'in': 2., 'out': 10.})


def test_v2_exact_cases_default_reasoning_no_retry(tmp_path):
    baseline, original = frozen(tmp_path)
    output = tmp_path / 'v2'
    manifest = v2.prepare(output, baseline)
    assert manifest['cases'] == original['cases']
    assert manifest['v2_allocation']['absolute_ceiling_usd'] == 7
    calls = []
    def send(**kw):
        calls.append(kw)
        return ChatResult(content=json.dumps({'accepted': True, 'issues': [], 'assessment': 'Pass.'}),
                          prompt_tokens=100, completion_tokens=10, cost=.001, finish_reason='stop', provider='test')
    assert v2.run_probe(output, send=send)['correct'] == 1
    assert 'extra_body' not in calls[0]
    assert v2.run_probe(output, send=lambda **kw: pytest.fail('No retry'))['correct'] == 1
    assert len(calls) == 1
    manifest2 = v2.prepare(tmp_path / 'v2_second_path', baseline)
    assert manifest2['v2_allocation'] == manifest['v2_allocation']
