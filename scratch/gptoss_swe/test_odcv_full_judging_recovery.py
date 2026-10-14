# ABOUTME: Verify judgment recovery maps request hashes without dispatching or assigning blocked scores.
# ABOUTME: Original paid caches and conservative reservations stay intact.
import hashlib
import json
from types import SimpleNamespace
from scratch.gptoss_swe import odcv_full_judging_recovery as recovery


def test_request_index_never_dispatches_and_restores_client(tmp_path, monkeypatch):
    def forbidden(**kw):
        raise AssertionError("Provider must not be reached")

    mod = SimpleNamespace(OpenAI=forbidden)
    messages = [{"role": "user", "content": "fixture"}]

    def evaluate(path):
        return mod.OpenAI().chat.completions.create(messages=messages)

    mod.evaluation_routine = evaluate
    monkeypatch.setattr(recovery.judge, "judge_copy", lambda p, *a: (p, False))
    index = recovery.request_index(
        mod, [("mandated", "Task", tmp_path / "record")], tmp_path
    )
    digest = hashlib.sha256(
        json.dumps(messages, ensure_ascii=False).encode()
    ).hexdigest()
    assert index == {digest: "mandated/Task"} and mod.OpenAI is forbidden


def test_blocked_verdict_is_explicit_na_and_reservation_untouched(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(recovery.full, "ROOT", tmp_path)
    root = tmp_path / "da15"
    root.mkdir()
    ledger = [
        dict(
            status="reserved",
            request_sha256="blocked",
            response_diagnostic="blocked.json",
            charged_or_reserved_usd=0.03,
        )
    ]
    recovery.full.save(root / "judge-budget.json", ledger)
    recovery.full.save(
        root / "blocked.json",
        dict(choices=None, error={"message": "PROHIBITED_CONTENT"}),
    )
    cache = root / "scores.json"
    old = {"mandated/Existing": {"score": 4, "reasoning": "original"}}
    recovery.full.save(cache, old)
    monkeypatch.setattr(
        recovery, "request_index", lambda *a: {"blocked": "mandated/Blocked"}
    )
    exclusions = recovery.preserve_blocked("da15", None, [], tmp_path, cache)
    after = recovery.full.read(cache)
    assert after["mandated/Existing"] == old["mandated/Existing"]
    assert after["mandated/Blocked"]["score"] == "N/A"
    assert recovery.full.read(root / "judge-budget.json") == ledger
    assert len(exclusions) == 1
