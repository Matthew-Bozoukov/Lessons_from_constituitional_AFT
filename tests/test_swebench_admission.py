# ABOUTME: GPU-free admission tests cover concurrency, starvation, crash fencing and growing contexts.
# ABOUTME: Processes use real Linux file locks; no model or paid provider is called.
import os
import unittest
if os.name != 'posix':
    raise unittest.SkipTest('Linux fleet only')
import multiprocessing as mp
from pathlib import Path
import tempfile
import time
from src.eval.capabilities.swebench_mini.fleet_admission import (
    cache_capacity, output_allowance, token_slot, may_admit)
from src.eval.capabilities.swebench_mini.fleet_task import eligible_source


def reserve(path, tokens, capacity, ready, release):
    with token_slot(path, tokens, capacity, expires=time.time()+30):
        ready.set()
        release.wait(20)


class TokenAdmissionTests(unittest.TestCase):
    def test_four_short_requests_fit_but_next_waits(self):
        with tempfile.TemporaryDirectory() as path:
            release = mp.Event()
            events = [mp.Event() for _ in range(5)]
            ps = [mp.Process(target=reserve, args=(path, 105536, 450000, e, release)) for e in events]
            try:
                for p, e in zip(ps[:4], events[:4]):
                    p.start()
                    self.assertTrue(e.wait(5))
                ps[4].start()
                self.assertFalse(events[4].wait(.3))
                release.set()
                self.assertTrue(events[4].wait(5))
            finally:
                release.set()
                for p in ps:
                    if p.pid:
                        p.join(5)
                        if p.is_alive(): p.kill(); p.join()

    def test_dead_active_owner_fences_replica(self):
        with tempfile.TemporaryDirectory() as path:
            ready, release = mp.Event(), mp.Event()
            p = mp.Process(target=reserve, args=(path, 70, 100, ready, release))
            p.start()
            self.assertTrue(ready.wait(5))
            p.kill(); p.join(5)
            with self.assertRaises(ConnectionError):
                with token_slot(path, 30, 100, expires=time.time()+1): pass
            self.assertTrue((Path(path)/'poison.json').exists())

    def test_dead_queued_owner_is_reaped_without_fencing(self):
        with tempfile.TemporaryDirectory() as path:
            ready, release = mp.Event(), mp.Event()
            with token_slot(path, 80, 100, expires=time.time()+10):
                p = mp.Process(target=reserve, args=(path, 50, 100, ready, release))
                p.start()
                until = time.time()+5
                while len(list(Path(path).glob('*.json'))) < 2 and time.time()<until: time.sleep(.05)
                p.kill(); p.join(5)
                with token_slot(path, 20, 100, expires=time.time()+1): pass
            self.assertFalse((Path(path)/'poison.json').exists())

    def test_large_waiter_eventually_blocks_bypass(self):
        rows = [{'id':'a','state':'active','tokens':60,'created':0},
                {'id':'b','state':'waiting','tokens':80,'created':1},
                {'id':'c','state':'waiting','tokens':20,'created':2}]
        self.assertTrue(may_admit(rows,'c',100,10,30))
        self.assertFalse(may_admit(rows,'c',100,40,30))
        self.assertTrue(may_admit(rows[1:],'b',100,40,30))

    def test_context_and_task_budget_are_independent_of_gpu_budget(self):
        self.assertEqual(output_allowance(40000,65536,262144,262144),65536)
        self.assertEqual(output_allowance(250000,65536,262144,262144),12144)
        self.assertEqual(output_allowance(40000,65536,123,262144),123)
        self.assertEqual(output_allowance(262144,65536,123,262144),0)
        self.assertEqual(cache_capacity('GPU KV cache size: 503,949 tokens',.9,262144)['budget_tokens'],453554)
        with self.assertRaises(AssertionError): cache_capacity('missing',.9,262144)
        with self.assertRaises(AssertionError): cache_capacity('GPU KV cache size: 100 tokens',.9,262144)

    def test_forced_patch_excludes_tests_and_helpers(self):
        self.assertTrue(eligible_source('django/db/models/query.py'))
        for path in ['tests/a.py','a/test_x.py','setup.py','docs/conf.py','scripts/repro.py','a/conftest.py']:
            self.assertFalse(eligible_source(path))


if __name__ == '__main__': unittest.main()
