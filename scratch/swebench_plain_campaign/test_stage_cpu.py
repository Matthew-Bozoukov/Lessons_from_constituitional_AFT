# ABOUTME: Offline ownership, credential minimization and sanitized-error tests for cold CPU staging.
# ABOUTME: No provider, SSH, Docker, provisioning or bootstrap calls occur in these tests.
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scratch.swebench_plain_campaign.stage_cpu import (
    INSTALL, PROBE, REQUIRED, StageError, credentials, endpoint, ssh_call,
)


class StageCpuTests(unittest.TestCase):
    def setUp(self):
        self.receipt = {'instance_id': 123, 'label': 'owned-cpu'}
        self.instance = {'id': 123, 'label': 'owned-cpu', 'actual_status': 'running',
                         'public_ipaddr': '192.0.2.5', 'ports': {'22/tcp': [{'HostPort': '1234'}]}}

    def test_duplicate_ipv4_ipv6_mapping_same_port_is_valid(self):
        self.instance['ports']['22/tcp'].append({'HostPort': '1234', 'HostIp': '::'})
        self.assertEqual(endpoint(self.receipt, self.instance), ('192.0.2.5', 1234))

    def test_ambiguous_ports_fail(self):
        self.instance['ports']['22/tcp'].append({'HostPort': '5678'})
        with self.assertRaises(StageError):
            endpoint(self.receipt, self.instance)

    def test_proxy_only_not_accepted(self):
        self.instance.pop('ports')
        self.instance.update(ssh_host='proxy', ssh_port=22)
        with self.assertRaises(StageError):
            endpoint(self.receipt, self.instance)

    def test_mismatched_owner_and_stopped_instances_fail(self):
        for field, value in [('id', 124), ('label', 'another-cpu'), ('actual_status', 'stopped')]:
            with self.subTest(field=field), self.assertRaises(StageError):
                endpoint(self.receipt, dict(self.instance, **{field: value}))

    def test_only_allowlisted_credentials_selected_without_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / 'values.env'
            env.write_text('\n'.join(f'{name}=value_{name}' for name in REQUIRED) + '\nUNRELATED_API_KEY=do-not-send\n')
            values = credentials(env)
            self.assertEqual(set(values), set(REQUIRED))
            self.assertNotIn('UNRELATED_API_KEY', values)

    def test_secret_values_never_in_argv_or_reported_errors(self):
        secret = 'PRIVATE_TOKEN_VALUE'
        result = subprocess.CompletedProcess([], 1, b'', ('failed ' + secret).encode())
        with patch('subprocess.run', return_value=result) as run:
            with self.assertRaises(StageError) as error:
                ssh_call(['ssh', 'root@192.0.2.5'], 'pass', {'token': secret})
            self.assertNotIn(secret, str(error.exception))
            self.assertNotIn(secret, str(run.call_args.args))
            self.assertIn(secret.encode(), run.call_args.kwargs['input'])

    def test_docker_absence_is_reported_clearly_without_raw_output(self):
        result = subprocess.CompletedProcess([], 1, b'', b'PRIVATE_TOKEN\nRuntimeError: Native Docker is absent; install/prepare it explicitly before staging')
        with patch('subprocess.run', return_value=result):
            with self.assertRaisesRegex(StageError, '^Native Docker is absent;') as error:
                ssh_call(['ssh'], 'pass', {})
            self.assertNotIn('PRIVATE_TOKEN', str(error.exception))

    def test_remote_scripts_compile(self):
        compile(PROBE, 'probe', 'exec')
        compile(INSTALL, 'install', 'exec')


if __name__ == '__main__':
    unittest.main()
