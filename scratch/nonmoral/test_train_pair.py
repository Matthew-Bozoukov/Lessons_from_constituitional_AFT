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
