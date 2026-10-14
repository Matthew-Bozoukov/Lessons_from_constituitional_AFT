# ABOUTME: Verify atomic grading-cache promotion and stock official report reuse without Docker.
# ABOUTME: Exercise both race winners and preserve original grading evidence exactly.
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('early_grade', Path(__file__).with_name('early_grade.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Tests(unittest.TestCase):
    def make(self, root):
        directory = root/'stage'
        log = directory/'logs/run_evaluation/early/model/foo__bar-1'
        log.mkdir(parents=True)
        (log/'report.json').write_text(json.dumps({'foo__bar-1': {'resolved': True}}))
        (log/'test_output.txt').write_text('original official test output')
        return directory

    def test_cache_used_by_stock_harness(self):
        from swebench.harness import run_evaluation as re
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); src = self.make(root); dst = root/'canonical'
            proof = m.offer_cache(src, dst, 'early', 'canonical')
            self.assertTrue(proof['installed'])
            self.assertTrue((src/'logs/run_evaluation/early/model/foo__bar-1/report.json').exists())
            preds = {'foo__bar-1': dict(model_name_or_path='model',model_patch='patch',instance_id='foo__bar-1')}
            with patch.object(re, 'RUN_EVALUATION_LOG_DIR', dst/'logs/run_evaluation'), patch.object(re, 'load_swebench_dataset', return_value=[{'instance_id':'foo__bar-1'}]):
                self.assertEqual(re.get_dataset_from_preds('unused','test',['foo__bar-1'],preds,'canonical',False), [])

    def test_existing_finalizer_directory_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); src = self.make(root); dst = root/'canonical'
            dst.mkdir(); (dst/'request.json').write_text('finalizer owns this')
            proof = m.offer_cache(src,dst,'early','canonical')
            self.assertFalse(proof['installed'])
            self.assertEqual((dst/'request.json').read_text(),'finalizer owns this')
            self.assertFalse((dst/'logs').exists())

    def test_atomic_mkdir_race(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(100):
                src=root/f'src{i}'; dst=root/f'dst{i}'; src.mkdir(); (src/'complete').write_text('ready')
                barrier=threading.Barrier(2); outcomes=[]
                def finalizer():
                    barrier.wait()
                    try: dst.mkdir(); outcomes.append('finalizer')
                    except FileExistsError: outcomes.append('cache')
                thread=threading.Thread(target=finalizer);thread.start();barrier.wait()
                installed=m.install_directory(src,dst);thread.join()
                self.assertEqual(installed,outcomes==['cache'])
                self.assertEqual((dst/'complete').exists(),installed)


if __name__ == '__main__':
    unittest.main()
