# ABOUTME: Verify MMLU preserves completed evidence on interruption and pins sampling.
# ABOUTME: Offline endpoint doubles; no GPU or network calls.
import json
from types import SimpleNamespace as NS

import pytest
from omegaconf import OmegaConf

from src.eval.capabilities.mmlu import runner


def test_completed_response_survives_interrupted_batch(tmp_path, monkeypatch):
    cfg = OmegaConf.load('configs/eval/mmlu.yaml')
    cfg.generation.extra_body = {'top_k': 20, 'min_p': 0.0, 'repetition_penalty': 1.0}
    questions = [{'uid': f'anatomy/{i}', 'subject': 'anatomy', 'category': 'STEM',
                  'question': 'Choose A', 'choices': ['yes', 'no', 'no', 'no'],
                  'n_choices': 4, 'answer_letter': 'A'} for i in range(2)]
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return NS(choices=[NS(message=NS(content='A', reasoning='reason'), finish_reason='stop')])
    monkeypatch.setattr(runner, 'OpenAI', lambda **kw: NS(chat=NS(completions=NS(create=create))))
    def interrupted(fn, n, **kwargs):
        fn(0)
        assert len((tmp_path / 'think/test/records.jsonl').read_text().splitlines()) == 1
        raise KeyboardInterrupt()
    monkeypatch.setattr(runner, 'map_threaded', interrupted)
    arm = {'name': 'test', 'served': 'test', 'adapter': None, 'revision': 'pinned',
           'synthetic_fraction': None, 'role': 'target'}
    with pytest.raises(KeyboardInterrupt):
        runner.run_arm(arm, questions, {}, cfg, 'http://unused', tmp_path)
    rec = json.loads((tmp_path / 'think/test/attempts.jsonl').read_text())
    assert rec['correct'] and rec['think'] == 'reason'
    assert rec['sampling']['seed'] == calls[0]['seed']
    assert calls[0]['extra_body'] == dict(cfg.generation.extra_body)
    assert calls[0]['temperature'] == 1.0 and calls[0]['max_tokens'] == 32768
    seeds = [calls[0]['seed']]
    def resumed(fn, n, **kwargs):
        assert n == 1
        return [fn(0)]
    monkeypatch.setattr(runner, 'map_threaded', resumed)
    runner.run_arm(arm, questions, {}, cfg, 'http://unused', tmp_path)
    assert seeds[0] != calls[1]['seed']
    records = [json.loads(s) for s in (tmp_path / 'think/test/records.jsonl').read_text().splitlines()]
    assert len(records) == 2 and len({r['uid'] for r in records}) == 2
