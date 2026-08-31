# ABOUTME: Suite-wide fixtures. The offline tests must never depend on a developer's
# ABOUTME: .env, so the one environment value the push path reads is pinned here.

import pytest


@pytest.fixture(autouse=True)
def _pinned_hf_org(monkeypatch):
    """Pin HF_ORG for every test (src.huggingface.hf_org).

    `hf_org()` resolves the push namespace from the environment, loading `.env` when the
    variable is unset — which would make these tests pass or fail depending on whose
    machine they run on, and fail outright in CI where there is no `.env`. An already-set
    variable wins over `.env`, so setting it here makes the resolution deterministic
    without touching the mechanism under test.
    """
    monkeypatch.setenv("HF_ORG", "test-org")
