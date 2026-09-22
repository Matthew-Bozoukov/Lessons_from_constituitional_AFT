# ABOUTME: Test explicit all-supervision opt-in in the scratch training owner commands.
# ABOUTME: Pure command construction checks; no provisioning, training or external calls.
import pytest


@pytest.mark.parametrize("opt_in", [None, False, "true", 1, True])
def test_training_owner_passes_default_supervision_only_on_explicit_opt_in(opt_in):
    import shlex
    from scratch.nonmoral.train_pair import commands

    plan = {"base_model_revision": "a" * 40,
            "arms": [{"data_repo": "org/2026-09-15-example-7-mix",
                      "data_revision": "b" * 40}]}
    if opt_in is not None:
        plan["allow_default_supervise"] = opt_in
    argv = shlex.split(commands(plan)[0])
    assert ("allow_default_supervise=true" in argv) is (opt_in is True)
    assert "seed=0" in argv and "--nproc_per_node=2" in argv


def test_single_h200_preserves_recipe_and_scales_budget():
    import shlex
    from scratch.nonmoral.train_pair import commands, budget_limits
    plan = {'gpu_count': 1, 'gpu_budget_usd': 45, 'base_model_revision': 'a' * 40,
            'arms': [{'data_repo': 'org/2026-09-15-nonmoral-original-7-mix', 'data_revision': 'b' * 40}]}
    argv = shlex.split(commands(plan)[0])
    assert argv[:4] == ['uv', 'run', '--no-sync', 'python']
    assert not any('nproc' in a or 'torchrun' in a for a in argv)
    assert 'configs/train/sft.yaml' in argv and 'seed=0' in argv
    count, budget, rate, lifetime = budget_limits(plan)
    assert (count, budget, rate) == (1, 45, 5)
    assert lifetime * rate / 3600 + 2 <= budget
