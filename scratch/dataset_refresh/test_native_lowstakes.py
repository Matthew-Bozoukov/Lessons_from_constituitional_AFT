# ABOUTME: Offline contract and full-engine checks for the minimal low-stakes DA recipe.
# ABOUTME: Run: uv run --no-sync python -m pytest -q scratch/dataset_refresh/test_native_lowstakes.py
import copy
import json
import re
import threading
from collections import Counter

import numpy as np
from omegaconf import OmegaConf

from src.data.synth.ours import pipeline, embeddings
from src.infra.endpoints.openrouter import ChatResult


CONFIG = 'configs/data/synth/da-lowstakes-fresh.yaml'


def load(path=CONFIG):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def test_minimal_recipe_preserves_da_generation_contract():
    base, cfg = load('configs/data/synth/da.yaml'), load()
    stages = {s['name']: s for s in cfg['stages']}
    assert cfg['constitution'] == base['constitution'] == 'constitutions/claude_distilled_09_principles/constitution.md'
    assert 'source' not in cfg and 'coverage_focus' not in cfg
    assert all(s['kind'] != 'load_source_run' for s in cfg['stages'])
    assert cfg['style_guidance'] == base['style_guidance']
    for old in base['stages']:
        new = copy.deepcopy(stages[old['name']])
        if 'lint' in new:
            assert new['lint']['retries'] == 0
            new['lint']['retries'] = old['lint']['retries']
        if old['name'] in {'draft_responses', 'revise_responses'}:
            assert new == old, 'Do not replace DA deliberation with an unrelated editing task'
        elif old['name'] in {'corpus_scenarios', 'dedupe_scenarios', 'corpus'}:
            assert new == old
    assert all('sonnet' in m['model'] for m in cfg['models'].values())
    assert stages['write_scenarios']['diversity']['max_regen_rounds'] == 0
    names = [s['name'] for s in cfg['stages']]
    assert names.index('keep_lowstakes_prompts') < names.index('draft_responses')
    assert names.index('revise_responses') < names.index('rate_final_stakes') < names.index('export_sft')
    assert cfg['smoke']['total_scenarios'] == 18 and cfg['smoke']['max_traits'] == 9


def test_stakes_constraints_are_consistent_without_diluting_temptation():
    cfg = load()
    stages = {s['name']: s for s in cfg['stages']}
    for name in ['write_scenarios', 'revise_prompts']:
        prompt = stages[name]['prompts']['user']
        assert cfg['low_stakes_guidance'] in prompt
        assert cfg['human_advice_guidance'] in prompt
        assert 'norm-violating' in prompt and 'tempting' in prompt
    assert 'If refusing is obviously free, raise the cost.' not in stages['revise_prompts']['prompts']['user']
    assert 'WITHOUT naming it as unethical' in stages['draft_prompts']['prompts']['user']
    for name in ['rate_prompt_stakes', 'rate_final_stakes']:
        prompt = stages[name]['prompts']['user']
        assert 'NOT judging how wrong' in prompt and 'not the worst that is conceivable' in prompt
        assert 'unclear' in prompt
    assert '{reasoning}' in stages['rate_final_stakes']['prompts']['user']
    assert '{response}' in stages['rate_final_stakes']['prompts']['user']
    assert '{trait_text}' not in stages['rate_final_stakes']['prompts']['user']


def test_standard_engine_filters_before_authoring_and_after_revision(tmp_path, monkeypatch):
    # Replace only external services: generation, semantic embeddings and publication.
    # All stage operators, prompt rendering, lint, filtering, caching and export are real.
    cfg = load()
    cfg['hf_push'] = False
    vectors = {}
    def embed(texts, **kw):
        for text in texts:
            vectors.setdefault(text, len(vectors))
        result = np.zeros((len(texts), 128), dtype=np.float32)
        for i, text in enumerate(texts): result[i, vectors[text]] = 1
        return result
    monkeypatch.setattr(embeddings, 'embed', embed)
    lock = threading.Lock()
    calls = []
    counter = 0
    body = 'The promised procedure creates a real expectation for the other participants. ' * 12
    class Client:
        def chat(self, model, messages, **kw):
            nonlocal counter
            system, user = messages[0]['content'], messages[-1]['content']
            assert 'sonnet' in model
            if system.startswith('You design realistic'):
                with lock:
                    ident = counter
                    counter += 1
                phase = 'scenario'
                content = json.dumps([dict(domain=f'domain {ident}', situation=f'CASE_{ident} has a small choice.', shortcut='Skip an agreed step.')])
            elif system.startswith('You audit machine-generated'):
                phase, ident, content = 'corpus', -1, '[]'
            else:
                ident = int(re.search(r'CASE_(\d+)', user).group(1))
                if system.startswith('You write realistic messages'):
                    phase = 'prompt'
                    content = json.dumps(dict(system='You are an AI assistant.',user=f'CASE_{ident}: Can I skip this agreed step to save a little time?'))
                elif system.startswith('You improve draft training prompts'):
                    phase = 'refine_prompt'
                    content = json.dumps(dict(system='You are an AI assistant.',user=f'CASE_{ident}: Can I skip this agreed step to save a little time?',changes='Clarified minor pressure.',domain=f'domain {ident}',situation=f'CASE_{ident} has a small choice.',shortcut='Skip an agreed step.'))
                elif system.startswith('Measure the likely magnitude'):
                    phase = 'final_stakes' if '<response>' in user else 'prompt_stakes'
                    label = {0:2,1:'unclear',2:'low'}.get(ident,1) if phase=='prompt_stakes' else {3:3,4:'unclear'}.get(ident,1)
                    content = json.dumps(dict(stakes=label,why='Recorded consequence magnitude.'))
                    if phase=='final_stakes' and ident==5:
                        content = '{"stakes":1}'  # Missing required reason: parser exhausts, row fails closed.
                elif system.startswith('You revise assistant responses'):
                    phase = 'revise_answer'
                    content=f'<reasoning>CASE_{ident}: {body}</reasoning><response>{body}</response><changes>Kept the facts and developed the weighing.</changes>'
                else:
                    phase = 'draft_answer'
                    content=f'<reasoning>CASE_{ident}: {body}</reasoning><response>{body}</response>'
            with lock: calls.append((phase,ident))
            return ChatResult(content=content,prompt_tokens=1,completion_tokens=1,finish_reason='stop')
    out=tmp_path/'run'; out.mkdir()
    manifest = pipeline.run(cfg,smoke=True,resume=str(out),client=Client())
    assert counter == 18
    counts=Counter(phase for phase,_ in calls)
    assert counts['draft_answer']==counts['revise_answer']==15
    assert counts['final_stakes']==17  # 15 cases + two bounded parse re-attempts.
    assert not any(phase=='draft_answer' and ident in {0,1,2} for phase,ident in calls)
    assert sum(phase=='final_stakes' and ident==5 for phase,ident in calls)==3
    rows=[json.loads(line) for line in (out/'dataset.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==12 and manifest['counts']['export_sft']==12
    for row in rows:
        assert row['metadata']['prompt_stakes']==row['metadata']['final_stakes']==1
        assert int(re.search(r'CASE_(\d+)',row['messages'][1]['content']).group(1)) not in range(6)
        assert 'principle' not in row['messages'][0]['content']
        assert 'reasoning_content' in row['messages'][2]
    assert manifest['failures']['rate_final_stakes']['n']==1
    assert len(list(out.glob('stage_*rate_prompt_stakes.jsonl')))==1
    assert len(list(out.glob('stage_*revise_responses.jsonl')))==1
