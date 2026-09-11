# ABOUTME: Regression checks for the context rejection and preservation of author meaning.
# ABOUTME: Offline tests never rent GPUs or send requests to model providers.
from types import SimpleNamespace

from src.eval.misalignment.delegated_harm.recovery import token_allowance, TokenBudget


def test_budget_reduces_output_at_old_failure_boundary():
    assert token_allowance(49153, 65536, 16384) == 16319
    assert token_allowance(50000, 131072, 16384) == 16384
    assert token_allowance(131072, 131072, 16384) == 0


def test_tokenization_counts_tools_and_uses_same_model(monkeypatch):
    calls = []
    def post(url, json, timeout):
        calls.append((url, json))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"count": 60000})
    monkeypatch.setattr("src.eval.misalignment.delegated_harm.recovery.requests.post", post)
    budget = TokenBudget("http://localhost:9000/v1", 65536)
    result = budget("adapter", [{"role": "user", "content": "a"}], [{"type": "function"}], 16384)
    assert result["max_tokens"] == 5472
    assert calls[0][0] == "http://localhost:9000/tokenize"
    assert calls[0][1]["model"] == "adapter"
    assert calls[0][1]["tools"] == [{"type": "function"}]
