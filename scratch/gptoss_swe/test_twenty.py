# ABOUTME: Offline selection, retained-outcome, fresh-replicate and exact failure-reason checks.
# ABOUTME: Run with uv run python -m unittest scratch.gptoss_swe.test_twenty; no network or sampling.
import json
import unittest
from scratch.gptoss_swe.twenty import expanded_state,limit_reason,C

class Tests(unittest.TestCase):
    def setUp(self):
        self.s=dict(old_ids=[str(i) for i in range(10)],new_ids=[str(i) for i in range(10,20)],ids=[str(i) for i in range(20)])
        self.p=dict(tasks={str(i):dict(status='valid',attempts=[dict(id=str(i),valid=True,exit_status='LimitsExceeded')]) for i in range(10)},halt=None)
    def test_retention(self):
        before=json.dumps(self.p);out=expanded_state(self.p,self.s,'base')
        self.assertEqual(sum(t['status']=='pending' for t in out['tasks'].values()),10)
        self.assertTrue(all(out['tasks'][i]==self.p['tasks'][i] for i in self.s['old_ids']))
        self.assertEqual(before,json.dumps(self.p))
    def test_fresh_replicates(self):
        for arm in ['control','da15']:
            out=expanded_state(self.p,self.s,arm)
            self.assertTrue(all(t==dict(status='pending',attempts=[]) for t in out['tasks'].values()))
    def test_duplicate_rejected(self):
        self.s['ids'][-1]='0'
        with self.assertRaises(AssertionError):expanded_state(self.p,self.s,'base')
    def test_live_prior_rejected(self):
        self.p['tasks']['0']['status']='running'
        with self.assertRaises(AssertionError):expanded_state(self.p,self.s,'base')
    def test_limit_evidence(self):
        self.assertEqual(limit_reason(dict(messages=[dict(extra=dict(limit_reason='context_limit'))],info=dict(model_stats=dict(api_calls=430)))),'context_limit')
        self.assertEqual(limit_reason(dict(info=dict(model_stats=dict(api_calls=500)))),'step_limit')
        self.assertEqual(limit_reason(dict(info=dict(model_stats=dict(api_calls=100)))),'other_limit')
    def test_parallel_total(self):
        self.assertEqual(sum(C.workers.values()),50)
        self.assertEqual(C.tool_concurrency,32)
if __name__=='__main__':unittest.main()
