# ABOUTME: Verify shared-account admission cannot consume the running SWE-bench reservation.
# ABOUTME: CPU-only tests mock providers and remote state; no rentals or network calls.
import json
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from scratch.lowstakes_plain.campaign import reservation
from scratch.nonmoral_plain.campaign import funds


class AdmissionTests(unittest.TestCase):
    def account(self, balance):
        return {'runpod_http_status': 200, 'openrouter_http_status': 200,
                'runpod': {'clientBalance': balance},
                'openrouter': {'total_credits': 20, 'total_usage': 0}}

    def test_running_budget_is_additional_to_new_job_and_reserve(self):
        with patch('scratch.nonmoral_plain.campaign.snapshot', return_value=self.account(183.99)):
            with self.assertRaises(AssertionError):
                funds(34, 100)
        with patch('scratch.nonmoral_plain.campaign.snapshot', return_value=self.account(184)):
            self.assertEqual(funds(34, 100)['runpod']['clientBalance'], 184)

    def test_missing_judge_credit_prevents_rental(self):
        account = self.account(300)
        account['openrouter']['total_usage'] = 11
        with patch('scratch.nonmoral_plain.campaign.snapshot', return_value=account):
            with self.assertRaises(AssertionError):
                funds(34, 100)

    def remote(self, status='complete', verified=True):
        data = {'state': {'pods': [{'id': 'owned'}]},
                'supervisor': {'status': status, 'verified_hf_revision': 'abc' if verified else None}}
        return SimpleNamespace(returncode=0, stdout=json.dumps(data))

    def test_release_requires_verified_completion_and_provider_absence(self):
        for remote, pods, expected in [
            (self.remote('running'), [], 100),
            (self.remote(verified=False), [], 100),
            (self.remote(), [{'id': 'owned'}], 100),
            (self.remote(), [{'id': 'late', 'name': 'nika-swe-lite-d742aeb6-99'}], 100),
            (self.remote(), [{'id': 'unrelated', 'name': 'jamie'}], 0),
            (SimpleNamespace(returncode=255), [], 100),
        ]:
            with self.subTest(expected=expected), patch('pathlib.Path.exists', return_value=False), patch('scratch.lowstakes_plain.campaign.subprocess.run', return_value=remote), patch('scratch.lowstakes_plain.campaign.runpod.active_pods', return_value=pods):
                self.assertEqual(reservation(), expected)

    def test_explicit_user_override_releases_only_extra_reservation(self):
        override = json.dumps({'approved': True, 'policy': 'user_requested_concurrent_start'})
        with patch('pathlib.Path.exists', return_value=True), patch('pathlib.Path.read_text', return_value=override):
            self.assertEqual(reservation(), 0)
        with patch('scratch.nonmoral_plain.campaign.snapshot', return_value=self.account(83.99)):
            with self.assertRaises(AssertionError):
                funds(34, 0)


if __name__ == '__main__':
    unittest.main()
