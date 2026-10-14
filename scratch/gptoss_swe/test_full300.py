# ABOUTME: Offline full-run admission, fresh selection and simultaneous arm-capacity tests.
# ABOUTME: Never contacts providers, creates containers or requests model generations.
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import pytest
from omegaconf import OmegaConf
from scratch.gptoss_swe import full300 as full
from scratch.gptoss_swe.finish_smoke import consume_pool


def test_all_900_outcomes_are_fresh_and_separate():
    ids = [f"task-{i}" for i in range(300)]
    states = [full.fresh_state(ids) for _ in range(3)]
    states[0]["tasks"][ids[0]]["attempts"].append({"id": "test"})
    assert states[1]["tasks"][ids[0]]["attempts"] == []
    assert states[2]["tasks"][ids[0]]["attempts"] == []
    assert all(t["status"] == "pending" for s in states for t in s["tasks"].values())
    with pytest.raises(AssertionError):
        full.fresh_state(ids[:-1] + [ids[0]])


def test_frozen_recipe_and_unique_ledgers():
    ids = [f"task-{i}" for i in range(300)]
    baseline = OmegaConf.load(full.C.baseline)
    configs = [full.arm_config(baseline, a, ids) for a in full.C.targets]
    assert sum(c.workers for c in configs) == 80
    assert len({c.tinker.budget_ledger for c in configs}) == 3
    assert all(
        c.tinker.context_window == c.serving.context_window == 65000 for c in configs
    )
    baseline.worker.max_response_tokens = 32768
    with pytest.raises(AssertionError):
        full.arm_config(baseline, "base", ids)


def test_preparation_does_not_authorize_sampling():
    with pytest.raises(AssertionError, match="does not authorize"):
        full.run()


def test_all_three_arms_reach_80_combined_and_retry_configs_are_isolated():
    barrier = threading.Barrier(80)
    guard = threading.Lock()
    active = peak = 0
    seen = []

    def consume(endpoint, model, cfg, name, ids, expires, admission):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
            seen.append((endpoint, ids[0], cfg.max_infrastructure_attempts))
        barrier.wait(timeout=20)
        cfg.max_infrastructure_attempts = -1
        time.sleep(0.02)
        with guard:
            active -= 1

    worker = OmegaConf.create({"max_infrastructure_attempts": 3})

    def arm(a):
        ids = [str(i) for i in range(full.C.workers[a])]
        consume_pool(
            consume,
            a,
            worker,
            a,
            ids,
            {i: {"max_total_attempt_records": 3} for i in ids},
            {},
            full.C.workers[a],
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(arm, full.C.targets))
    assert peak == 80 and active == 0
    assert len(seen) == len(set((a, i) for a, i, n in seen)) == 80
    assert all(n == 3 for a, i, n in seen)
    assert worker.max_infrastructure_attempts == 3
