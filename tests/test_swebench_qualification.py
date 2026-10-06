# ABOUTME: Verify exact-config qualification refuses mismatched or failed executed evidence.
# ABOUTME: Linux-only; synthetic local fixtures and mocked publication never qualify a paid campaign.
import os
import unittest

if os.name != 'posix':
    raise unittest.SkipTest('Fleet qualification runs on Linux')

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from omegaconf import OmegaConf
from scratch import swebench_qualify_recipe as qualify


class PublicationReached(Exception):
    pass


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = OmegaConf.load('configs/eval/swebench_mini/lite.yaml')
        self.cfg.gpu = 'NVIDIA H200'
        self.cfg.replicas = 12
        self.cfg.fallback_gpus = ['NVIDIA H100 NVL']
        self.cfg.recipe_path = str(self.root/'qualified-recipe.json')
        self.cfg.cpu_qualification_path = str(self.root/'capacity.json')
        self.cfg.readiness = str(self.root/'ready')
        self.config = self.root/'custom.yaml'
        OmegaConf.save(self.cfg, self.config)
        self.config_hash = qualify.digest(self.config)
        self.smoke = self.root/'smoke'
        self.write(self.root/'ready/metadata/images.json', {})
        phases = []
        for agents, limit in [(6, 1), (40, 40), (60, 60), (80, 80), (80, 32)]:
            phases.append({'agents': agents, 'tool_limit': limit, 'rounds': 1, 'seconds': 1,
                           'samples': [{'available_gib': 40}],
                           'tasks': [{'resources': {'memory.events': {'oom_kill': 0}},
                                      'commands': [{'returncode': 0, 'execution_seconds': 1}]}
                                     for _ in range(agents)]})
        self.load_path = self.root/'load/results.json'
        self.write(self.load_path, {'status': 'finished', 'model_inference': False,
                                   'memory_gib': 200, 'cpu_count': 64, 'phases': phases})
        self.smoke_path = self.smoke/'results/infrastructure.json'
        self.write(self.smoke_path, {'status': 'passed', 'synthetic': True,
                                    'model_evaluation': False, 'config_sha256': self.config_hash})
        (self.smoke/'metadata').mkdir()
        (self.root/'template-evidence').mkdir()
        self.template_path = self.root/'template.json'
        self.write(self.template_path, {
            'status': 'passed', 'model_inference': False, 'config_sha256': self.config_hash,
            'base_revision': self.cfg.base_revision, 'checks': [True] * 5,
            'source_sha256': {'src/infra/endpoints/vllm.py': qualify.digest('src/infra/endpoints/vllm.py')},
            'evidence_dir': str(self.root/'template-evidence')})
        for name, text in [('tests.log', '10 passed\n'), ('transport.log', '\nOK\n'),
                           ('protocol.log', 'Ran 5 tests\nOK\n')]:
            (self.root/name).write_text(text)
        self.argv = ['qualify', '--config', str(self.config), '--load-dir', str(self.root/'load'),
                     '--smoke-root', str(self.smoke), '--test-log', str(self.root/'tests.log'),
                     '--transport-log', str(self.root/'transport.log'),
                     '--protocol-log', str(self.root/'protocol.log'),
                     '--template-proof', str(self.template_path)]

    @staticmethod
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def invoke(self):
        with patch('sys.argv', self.argv), patch.object(qualify, 'load_dotenv'), \
             patch.object(qualify.fleet, 'sources', return_value={}), \
             patch.object(qualify, 'hf_repo_id', return_value='test/no-upload'), \
             patch.object(qualify, 'push_run_dir', side_effect=PublicationReached):
            qualify.main()

    def test_custom_config_recipe_and_archive_match_executed_evidence(self):
        with self.assertRaises(PublicationReached):
            self.invoke()
        recipe = json.loads(Path(self.cfg.recipe_path).read_text())
        self.assertEqual(recipe['settings']['gpu'], 'NVIDIA H200')
        self.assertEqual(recipe['settings']['replicas'], 12)
        self.assertEqual(recipe['settings']['fallback_gpus'], ['NVIDIA H100 NVL'])
        self.assertEqual(recipe['configuration']['sha256'], self.config_hash)
        self.assertEqual((self.smoke/'metadata/qualified-config.yaml').read_bytes(), self.config.read_bytes())
        self.assertFalse(recipe['validated_full_run'])

    def test_mismatched_smoke_or_template_cannot_stamp_recipe(self):
        for evidence in (self.smoke_path, self.template_path):
            with self.subTest(evidence=evidence):
                original = json.loads(evidence.read_text())
                self.write(evidence, original | {'config_sha256': 'default-config-hash'})
                with self.assertRaisesRegex(AssertionError, 'exact --config'):
                    self.invoke()
                self.assertFalse(Path(self.cfg.recipe_path).exists())
                self.write(evidence, original)

    def test_failed_executed_protocol_cannot_stamp_recipe(self):
        (self.root/'protocol.log').write_text('Ran 5 tests\nFAILED (failures=1)\n')
        with self.assertRaises(AssertionError):
            self.invoke()
        self.assertFalse(Path(self.cfg.recipe_path).exists())

    def test_default_config_path_is_preserved(self):
        self.argv = self.argv[:1] + self.argv[3:]
        with patch('sys.argv', self.argv), \
             patch.object(qualify.OmegaConf, 'load', wraps=OmegaConf.load) as load, \
             patch.object(qualify, 'load_dotenv', side_effect=PublicationReached):
            with self.assertRaises(PublicationReached):
                qualify.main()
        self.assertEqual(load.call_args.args[0], Path('configs/eval/swebench_mini/lite.yaml'))


if __name__ == '__main__':
    unittest.main()
