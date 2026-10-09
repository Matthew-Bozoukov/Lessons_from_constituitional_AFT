# ABOUTME: Verify the response-limit intervention preserves all other task and sampling settings.
# ABOUTME: Run: uv run python -m unittest scratch.gptoss_swe.test_response_quad
import unittest
from omegaconf import OmegaConf
from scratch.gptoss_swe.response_quad import C, revised_config, verify_delta
from src.eval.capabilities.swebench_mini.fleet_admission import output_allowance


class Tests(unittest.TestCase):
    def prior(self):
        return OmegaConf.create(dict(campaign='prior',run_name='prior',output_root='/prior',
            instance_ids=['prior'],subset=dict(n=20,seed=0),workers=20,
            tinker=dict(max_tokens=16384,context_window=131072,budget_usd=None,budget_ledger='/prior/budget',reasoning='medium'),
            worker=dict(max_response_tokens=16384,max_task_tokens=262144,step_limit=500,
                        model_request_timeout_seconds=7200,model_request_attempts=2,max_infrastructure_attempts=3,
                        max_infrastructure_failures=6,tool_concurrency=32,min_available_memory_gib=32),
            sampling=dict(temperature=1.,top_p=1.,top_k=-1)))

    def test_exact_intervention_and_immutable_input(self):
        for arm in C.targets:
            old=self.prior(); snapshot=OmegaConf.to_container(old,resolve=True)
            new=revised_config(old,arm); verify_delta(old,new)
            self.assertEqual(OmegaConf.to_container(old,resolve=True),snapshot)
            self.assertEqual(new.sampling,old.sampling)
            self.assertEqual(new.worker.max_task_tokens,262144)
            self.assertEqual(new.worker.step_limit,500)
            self.assertEqual(new.tinker.context_window,131072)

    def test_sampling_drift_rejected(self):
        old=self.prior();new=revised_config(old,'control');new.sampling.temperature=.7
        with self.assertRaises(AssertionError):verify_delta(old,new)

    def test_task_limit_drift_rejected(self):
        old=self.prior();new=revised_config(old,'da15');new.worker.step_limit=1000
        with self.assertRaises(AssertionError):verify_delta(old,new)

    def test_selected_counts_and_parallelism(self):
        self.assertEqual(sum(C.workers.values()),6)
        for arm in C.targets:
            self.assertEqual(len(C.selected[arm]),C.workers[arm])
            self.assertEqual(len(set(C.selected[arm])),C.workers[arm])

    def test_remaining_budgets_still_bind(self):
        self.assertEqual(output_allowance(1000,65536,262144,131072),65536)
        self.assertEqual(output_allowance(100000,65536,262144,131072),31072)
        self.assertEqual(output_allowance(1000,65536,4000,131072),4000)
        self.assertEqual(output_allowance(131072,65536,262144,131072),0)


if __name__=='__main__': unittest.main()
