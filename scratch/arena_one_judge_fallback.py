# ABOUTME: Request the explicitly approved Gemini fallback for one unresolved Arena ordering.
# ABOUTME: Preserve exact prompt text and every attempt separately from GPT-4.1 primary records.
import hashlib
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import shutil

from dotenv import load_dotenv
from omegaconf import OmegaConf
import yaml

from src.eval.capabilities.arena_hard.arena_hard_judge import _judge_settings
from src.utils import read_jsonl

ROOT = Path('output/arena_qwen_base_20261013')
VENDOR = Path('src/eval/capabilities/arena_hard/third_party/arena-hard-auto').resolve()
UID = 'bb45a7aed6cf45cb'
MODEL = 'google/gemini-3-flash-preview'


def finalize():
    from src.eval.capabilities.arena_hard.pool import _score_slices, _generation_health
    from src.eval.capabilities.arena_hard.arena_hard_stats import battles_from_judgments, style_controlled_win_rate
    from src.utils import write_run_meta
    source = ROOT / 'da15/comparison'
    exit_record = json.loads((ROOT / 'da15_exit.json').read_text())
    launch = json.loads((ROOT / 'da15_launch.json').read_text())
    assert exit_record['exit_code'] != 0 and exit_record['finished_utc'] >= launch['started_utc']
    cfg = OmegaConf.load(source / 'metadata/arena_hard_config.yaml')
    primary_path = Path(cfg.vendor_dir) / 'data/arena-hard-v2.0/model_judgment/openai/gpt-4.1/qwen36_0_da_15.jsonl'
    original = read_jsonl(primary_path)
    missing = [(r['uid'], i) for r in original for i, g in enumerate(r['games']) if g['status'] != 'complete']
    assert missing == [(UID, 0)], missing
    fallback = json.loads((ROOT / 'fallback/gemini_one_ordering.json').read_text(encoding='utf-8'))
    game = copy.deepcopy(fallback['attempts'][-1])
    assert game['status'] == 'complete' and game['judgment']['finish_reason'] == 'stop'
    record = next(r for r in original if r['uid'] == UID)
    assert game['prompt'] == record['games'][0]['prompt']
    assert fallback['primary_request_hash'] == record['games'][0]['request_hash']
    game['attempts'] = fallback['attempts']
    game['judge_model'] = MODEL
    game['fallback_for_primary_request_hash'] = fallback['primary_request_hash']
    effective = copy.deepcopy(original)
    replaced = next(r for r in effective if r['uid'] == UID)
    replaced['judge'] = 'mixed: openai/gpt-4.1 with one approved Gemini ordering'
    replaced['games'][0] = game
    for a, b in zip(original, effective):
        for i, g in enumerate(a['games']):
            if g['status'] == 'complete':
                assert g == b['games'][i]
    raw = source / 'rollouts/judgments_qwen36_0_da_15.jsonl'
    raw.parent.mkdir(exist_ok=True)
    raw.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in effective), encoding='utf-8')
    shutil.copy2(primary_path, source / 'rollouts/primary_gpt41_judgments_qwen36_0_da_15.jsonl')
    shutil.copy2(ROOT / 'fallback/gemini_one_ordering.json', source / 'rollouts/gemini_fallback_one_ordering.json')
    limits = {'hard_prompt': 500, 'creative_writing': 250}
    slices = _score_slices(cfg, 'qwen36_0_da_15', {'limits': limits}, raw)
    details = {'uid': UID, 'ordering': 0, 'primary_judge': 'openai/gpt-4.1',
               'primary_attempts': len(record['games'][0]['attempts']),
               'fallback_judge': MODEL, 'fallback_attempts': len(fallback['attempts']),
               'fallback_score': game['score'], 'messages_identical': True,
               'valid_primary_verdicts_preserved': 1499,
               'primary_complete': False, 'effective_complete': True,
               'authorization': fallback['authorization']}
    control = json.loads((ROOT / 'control/comparison/results/results.json').read_text())
    summary = copy.deepcopy(control)
    primary = slices['hard_prompt']['controlled']
    repo = 'dougalldeepmind/2026-10-12-ah-qwen36-0-da-15'
    summary['report_version'] = 4
    summary['leaderboard'] = [{'model_key': 'qwen36_0_da_15', 'target': repo,
        'repo': 'https://huggingface.co/datasets/' + repo, 'primary_slice': 'hard_prompt',
        'win_rate': primary['mean'], 'ci_lower': primary['ci_lower'], 'ci_upper': primary['ci_upper'],
        'n_prompts': primary['n'], 'passes': slices['hard_prompt']['passes'], 'by_slice': slices,
        'judge_fallbacks': [details]}]
    summary['judge_fallbacks'] = [details]
    summary['judge_policy']['primary_completion_required'] = False
    summary['judge_policy']['approved_exception'] = 'One named Gemini ordering after 12 invalid GPT-4.1 attempts'
    summary['judge_policy']['fallback_judge'] = MODEL
    summary['judge_validation'] = json.loads((source / 'results/judge_validation.json').read_text())
    summary['generation_protocol_validation'] = json.loads((source / 'metadata/generation_protocol_validation.json').read_text())
    summary['generation_health'] = {key: _generation_health({'out_dir': ROOT / 'da15/arms' / key})
                                    for key in ('qwen36', 'qwen36_0_da_15')}
    summary['pooled_from'] = []
    for key in ('qwen36', 'qwen36_0_da_15'):
        receipt = json.loads((ROOT / 'da15' / f'{key}.json').read_text())
        run = receipt['published']
        meta = json.loads((Path(run['out_dir']) / 'metadata/sources.json').read_text())
        summary['pooled_from'].append({'target': run['target'], 'model_key': key, 'repo': run['repo'],
                                      'reference_arm': key == 'qwen36', 'answers': meta['answers']})
    # Bound the effect of this one verdict without inventing an alternative outcome.
    arm_meta = {r['uid']: r['metadata'] for r in read_jsonl(ROOT / 'da15/arms/qwen36_0_da_15/rollouts/answers.jsonl')}
    base_meta = {r['uid']: r['metadata'] for r in read_jsonl(ROOT / 'base/arms/qwen36/rollouts/answers.jsonl')}
    sensitivity = []
    for verdict in ('A>B', 'A=B', 'B>A'):
        candidate = copy.deepcopy(effective)
        next(r for r in candidate if r['uid'] == UID)['games'][0]['score'] = verdict
        battles = [b for b in battles_from_judgments(candidate) if b['category'] == 'hard_prompt']
        result = style_controlled_win_rate(battles, arm_meta, base_meta, rounds=1, seed=int(cfg.seed))
        sensitivity.append({'hypothetical_verdict': verdict, 'controlled_mean': result['mean'],
                            'raw_mean': sum(b['score'] for b in battles)/len(battles)})
    summary['single_fallback_sensitivity'] = sensitivity
    for filename, value in [('results.json', summary), ('leaderboard.json', summary),
                            ('judge_fallback.json', details), ('fallback_sensitivity.json', sensitivity),
                            ('judgment_qwen36_0_da_15.json', {'statistics_by_slice': slices, 'fallback': details})]:
        (source / 'results' / filename).write_text(json.dumps(value, indent=2), encoding='utf-8')
    (source / 'metadata/sources.json').write_text(json.dumps(summary['pooled_from'], indent=2))
    write_run_meta(source / 'metadata', OmegaConf.to_container(cfg, resolve=True),
                   extra={'command': 'uv run python -X utf8 scratch/arena_one_judge_fallback.py finalize',
                          'judge_fallback': details, 'pooled_from': summary['pooled_from']})
    (ROOT / 'fallback_application.json').write_text(json.dumps(details | {'sensitivity': sensitivity}, indent=2))
    print(json.dumps({'scores': {k: v['controlled'] for k, v in slices.items()}, 'fallback': details,
                      'sensitivity': sensitivity}, indent=2))


def main():
    load_dotenv('.env')
    sys.path.insert(0, str(VENDOR))
    package = types.ModuleType('utils')
    package.__path__ = [str(VENDOR / 'utils')]
    sys.modules['utils'] = package
    spec = importlib.util.spec_from_file_location('arena_one_fallback_vendor', VENDOR / 'gen_judgment.py')
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)
    cfg = OmegaConf.load(ROOT / 'da15/comparison/metadata/arena_hard_config.yaml')
    judge = _judge_settings(cfg, MODEL)
    upstream = yaml.safe_load((VENDOR / 'config/arena-hard-v2.0.yaml').read_text(encoding='utf-8'))
    configs = {'judge_model': MODEL, 'temperature': float(judge.temperature),
               'max_tokens': int(judge.max_tokens), 'regex_patterns': upstream['regex_patterns'],
               'prompt_template': upstream['prompt_template']}
    settings = {'model': MODEL, 'api_type': 'openai', 'parallel': 1,
                'max_tokens': int(judge.max_tokens), 'temperature': float(judge.temperature),
                'extra_body': OmegaConf.to_container(judge.extra_body, resolve=True),
                'endpoints': [{'api_base': str(judge.api_base), 'api_key': os.environ[str(judge.api_key_env)]}]}
    def selected(path):
        return next(row for row in read_jsonl(path) if row['uid'] == UID)
    question = selected(VENDOR / 'data/arena-hard-v2.0/question.jsonl')
    base = selected(ROOT / 'base/arms/qwen36/rollouts/answers.jsonl')
    answer = selected(ROOT / 'da15/arms/qwen36_0_da_15/rollouts/answers.jsonl')
    prior = selected(ROOT / 'recovery2_primary_checkpoint.jsonl')['games'][0]
    messages, request_hash = vendor.judgment_request(question, base, answer, None, configs, settings, 0)
    assert messages == prior['prompt'], 'Fallback must use the exact original messages'
    directory = ROOT / 'fallback'
    directory.mkdir(exist_ok=True)
    path = directory / 'gemini_one_ordering.json'
    record = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'uid': UID, 'ordering': 0, 'primary_judge': 'openai/gpt-4.1', 'fallback_judge': MODEL,
        'authorization': 'User explicitly approved one labelled Gemini fallback on 2026-10-14.',
        'primary_request_hash': prior['request_hash'], 'fallback_request_hash': request_hash,
        'messages_identical_to_primary': True, 'configs': configs,
        'settings': vendor.public_settings(settings), 'attempts': []}
    for _ in range(3):
        if record['attempts'] and vendor.complete_game(record['attempts'][-1], request_hash):
            break
        game = vendor.pairwise_judgment(question, base, answer, None, configs, settings, 0)
        record['attempts'].append(game)
        path.write_text(json.dumps(record, indent=2), encoding='utf-8')
        print('fallback attempt', len(record['attempts']), 'status', game['status'], flush=True)
    if not vendor.complete_game(record['attempts'][-1], request_hash):
        raise RuntimeError('Gemini fallback did not produce a valid complete verdict')
    print('Valid fallback saved separately; primary records are untouched.', flush=True)


if __name__ == '__main__':
    finalize() if len(sys.argv) > 1 and sys.argv[1] == 'finalize' else main()
