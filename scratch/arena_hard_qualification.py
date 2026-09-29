# ABOUTME: Bounded Arena-Hard instrument qualification using pinned historical or fresh smoke answers.
# ABOUTME: Preserves live judge transcripts separately from model capability results and publishes the evidence.

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.capabilities.arena_hard import arena_hard_judge as judge
from src.eval.capabilities.arena_hard.arena_hard_stats import battles_from_judgments, evaluate_arm
from src.eval.capabilities.arena_hard.runner import isolate_harness
from src.eval.layout import assert_layout, publish_layout, run_tags
from src.eval.run_eval import _card_fields
from src.infra.huggingface import push_run_dir
from src.naming import artifact_name
from src.utils import read_jsonl, write_run_meta

HISTORY_REPO = 'dougalldeepmind/2026-07-31-qwen36-27b-capability-eval-arena-hard'
HISTORY_SHA = '9b1c2802a5b9aded7c086cdb0b7cb06ebe714229'


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def prepare(out: Path, historical: Path, smoke_source: Path | None):
    rollouts, results, metadata = publish_layout(out)
    cfg = OmegaConf.load('configs/eval/arena_hard.yaml')
    cfg.output_dir = str(out.resolve())
    isolate_harness(cfg, metadata)
    cfg.baseline_arm = 'qualification_reference'
    cfg.judge_validation.comparison_arm = 'qualification_candidate'
    cfg.arms = [dict(name=name, adapter='saved-answer-fixture', role=role,
                    n_hard_prompt=100 if smoke_source is None else 4,
                    n_creative_writing=0 if smoke_source is None else 4)
                for name, role in [('qualification_reference', 'baseline'),
                                   ('qualification_candidate', 'target')]]
    answer_dir = Path(cfg.vendor_dir) / 'data' / cfg.bench_name / 'model_answer'
    answer_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, old in [('qualification_reference', 'arm_b_synth10'),
                      ('qualification_candidate', 'arm_c_synth20')]:
        source = (smoke_source / 'rollouts' / 'answers.jsonl' if smoke_source else
                  historical / 'model_answer' / f'{old}.jsonl')
        rows = read_jsonl(source)
        # Deliberate aliases distinguish two positions in an identical-answer control;
        # they do not claim that two different checkpoints were generated.
        for row in rows:
            row['model'] = name
        dest = answer_dir / f'{name}.jsonl'
        dest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        shutil.copy2(dest, rollouts / f'answers_{name}.jsonl')
        sources[name] = {'source': str(source), 'historical_arm': old if not smoke_source else None}
    protocol = {'kind': 'identical_answer_control' if smoke_source else 'historical_answer_judge_validation',
                'historical_repo': HISTORY_REPO if not smoke_source else None,
                'historical_revision': HISTORY_SHA if not smoke_source else None,
                'sources': sources, 'capability_comparison': False,
                'limitations': 'Judge qualification evidence; not a new candidate-versus-nosynth capability result.'}
    if smoke_source:
        protocol['qwen_source_run_meta'] = json.loads((smoke_source / 'metadata' / 'run_meta.json').read_text(encoding='utf-8'))
        shutil.copytree(smoke_source / 'metadata', metadata / 'qwen_generation', dirs_exist_ok=True)
    else:
        shutil.copy2(historical / 'report' / 'manifest.json', metadata / 'historical_manifest.json')
    dump(metadata / 'sources.json', protocol)
    OmegaConf.save(cfg, metadata / 'qualification_config.yaml')
    path = write_run_meta(out, OmegaConf.to_container(cfg, resolve=True),
                          extra={'command': ' '.join(sys.argv), **protocol})
    shutil.move(str(path), metadata / 'run_meta.json')
    print(json.dumps(protocol, indent=2))


def reproduce(out: Path, historical: Path):
    records = read_jsonl(historical / 'model_judgment' / 'arm_c_synth20.jsonl')
    meta = {}
    for name in ('arm_c_synth20', 'arm_b_synth10'):
        meta[name] = {r['uid']: r['metadata'] for r in read_jsonl(historical / 'model_answer' / f'{name}.jsonl')}
    battles = [r for r in battles_from_judgments(records) if r['category'] == 'hard_prompt']
    result = evaluate_arm(battles, meta['arm_c_synth20'], meta['arm_b_synth10'], rounds=2000, seed=0)
    assert abs(result['controlled']['mean'] - .4921214444102081) < 1e-12
    assert result['controlled']['n'] == 148
    result['interpretation'] = 'Exact historical reproduction only: 148 scored prompts, not proof of 150 complete paired judgments.'
    dump(out / 'results' / 'historical_reproduction.json', result)
    print(json.dumps(result, indent=2))


def live(out: Path, n: int, self_control: bool, judge_only: str | None = None):
    cfg = OmegaConf.load(out / 'metadata' / 'qualification_config.yaml')
    cfg.judge_validation.n_questions = n
    if n < 1 or n > (4 if self_control else 100):
        raise ValueError('Outside bounded qualification sample size')
    OmegaConf.save(cfg, out / 'metadata' / 'qualification_config.yaml')
    try:
        if judge_only:
            if judge_only not in (str(cfg.judge.model), str(cfg.judge_validation.reference_judge)):
                raise ValueError('Only the two declared qualification judges may run')
            result = judge.judge_arm(cfg, str(cfg.judge_validation.comparison_arm), n, judge_only)
            result['capability_comparison'] = False
            result['calibration_status'] = 'not_assessed_single_judge'
            dump(out / 'results' / f'single_judge_{judge_only.replace("/", "_")}_{n}.json', result)
            print(json.dumps(result, indent=2))
            return
        if self_control:
            control = judge.judge_arm(cfg, str(cfg.judge_validation.comparison_arm), None, str(cfg.judge.model))
            control['capability_comparison'] = False
            dump(out / 'results' / 'identical_answer_control.json', control)
        result = judge.validate_judge(cfg)
        result['qualification_kind'] = 'identical_answer_control' if self_control else 'historical_answer_judge_validation'
        result['calibration_status'] = ('not_assessed' if self_control or n < 100 or cfg.judge_validation.get('policy') == 'diagnostic' else
                                        'passed' if result['passes'] else 'failed')
        if self_control or n < 100:
            result['diagnostic_threshold_passes'] = result.pop('passes')
            result['passes'] = None
        result['capability_comparison'] = False
        dump(out / 'results' / f'validation_{n}.json', result)
        (out / 'results' / f'validation_{n}.md').write_text(
            '# Arena-Hard instrument qualification\n\nThis is not a new model capability score.\n\n```json\n'
            + json.dumps(result, indent=2) + '\n```\n', encoding='utf-8')
        print(json.dumps(result, indent=2))
    except BaseException as exc:
        dump(out / 'results' / f'validation_{n}_failure.json', {'error_type': type(exc).__name__, 'error': str(exc)})
        raise
    finally:
        source = Path(cfg.vendor_dir) / 'data' / cfg.bench_name / 'model_judgment'
        if source.exists():
            shutil.copytree(source, out / 'rollouts' / 'judgments', dirs_exist_ok=True)


def publish(out: Path):
    cfg = OmegaConf.load(out / 'metadata' / 'qualification_config.yaml')
    provenance = json.loads((out / 'metadata' / 'sources.json').read_text(encoding='utf-8'))
    validations = sorted((out / 'results').glob('validation_[0-9]*.json'),
                         key=lambda path: int(path.stem.split('_')[1]))
    validations = [path for path in validations if not path.stem.endswith('_failure')]
    summary = {'instrument_qualification': True, 'capability_comparison': False,
               'qualification_kind': provenance['kind'],
               'validation': json.loads(validations[-1].read_text(encoding='utf-8')) if validations else None,
               'cost': {}}
    summary['failures'] = [json.loads(p.read_text(encoding='utf-8'))
                           for p in sorted((out / 'results').glob('*failure.json'))]
    summary['judge_coverage'] = {}
    for model in (str(cfg.judge.model), str(cfg.judge_validation.reference_judge)):
        path = out / 'rollouts' / 'judgments' / model / 'qualification_candidate.jsonl'
        if path.exists():
            rows = read_jsonl(path)
            summary['cost'][model] = judge._cost(rows, model)
            summary['judge_coverage'][model] = {
                'recorded_prompts': len(rows),
                'complete_pairs': sum(len(r.get('games') or []) == 2 and all(
                    g and g.get('status') == 'complete' for g in r['games']) for r in rows)}
    summary['calibration_status'] = ('blocked_incomplete' if summary['failures'] else
                                     (summary['validation'] or {}).get('calibration_status', 'not_assessed'))
    dump(out / 'results' / 'results.json', summary)
    (out / 'results' / 'results.md').write_text(
        '# Arena-Hard instrument qualification\n\nNot a candidate-versus-control capability result.\n\n```json\n'
        + json.dumps(summary, indent=2) + '\n```\n', encoding='utf-8')
    subject = 'arena-hard-' + provenance['kind'].replace('_', '-')
    fields = _card_fields('arena_hard', cfg, ' '.join(sys.argv),
                          experiment='Arena-Hard instrument qualification; no new capability score',
                          models=json.dumps({'judges': [cfg.judge.model, cfg.judge_validation.reference_judge],
                                             'answers': provenance}),
                          source_revision=json.loads((out / 'metadata' / 'run_meta.json').read_text(encoding='utf-8'))['git_sha'])
    # The run-local harness may contain credential files only during a subprocess;
    # publication refuses even an accidentally retained one.
    if list(out.rglob('generated_api_config.yaml')):
        raise ValueError('Refusing publication with generated endpoint credentials')
    assert_layout(out)
    url = push_run_dir(out, artifact_name(subject), fields,
                       front_matter={'tags': run_tags('arena_hard', 'instrument_qualification', 'think') + ['instrument-qualification']})
    print(url)


def reinterpret(out: Path):
    """Apply the approved policy to saved evidence, without changing its requests."""
    cfg = OmegaConf.load(out / 'metadata' / 'qualification_config.yaml')
    original = OmegaConf.to_container(cfg, resolve=True)
    active = OmegaConf.load('configs/eval/arena_hard.yaml')
    cfg.judge = active.judge
    cfg.judge_validation.policy = active.judge_validation.policy
    cfg.judge_validation.reference_judge = active.judge_validation.reference_judge
    cfg.judge_validation.n_questions = 100
    questions = judge._expected_questions(cfg, {'hard_prompt': 100, 'creative_writing': 0})
    allowed = read_jsonl(Path(cfg.vendor_dir) / 'data' / cfg.bench_name / 'question.jsonl')
    answers = {arm: judge._validate_answers(cfg, arm, questions)
               for arm in (str(cfg.baseline_arm), str(cfg.judge_validation.comparison_arm))}
    records = {model: read_jsonl(out / 'rollouts' / 'judgments' / model / 'qualification_candidate.jsonl')
               for model in (str(cfg.judge.model), str(cfg.judge_validation.reference_judge))}
    result = judge.summarise_judge_validation(cfg, questions, records,
                    expected_answers=answers, allowed_questions=allowed)
    result.update(capability_comparison=False, calibration_status='not_assessed',
                  interpretation='Offline application of the user-approved policy to prior saved requests; no new calls.',
                  original_request_config=original,
                  limitation='Historical GPT-4.1 requests inherited low reasoning extra_body; current primary uses empty extra_body.')
    dump(out / 'results' / 'approved_policy_reanalysis.json', result)
    print(json.dumps({k: result[k] for k in ('primary_complete', 'n_primary_complete',
         'n_auxiliary_complete', 'n_compared', 'verdict_agreement', 'win_rate_gap_pp', 'passes')}, indent=2))


def main():
    load_dotenv('.env', override=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'reproduce', 'live', 'publish', 'reinterpret'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--historical', type=Path, default=Path('output/arena_hard_qualification/historical'))
    parser.add_argument('--smoke-source', type=Path)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--self-control', action='store_true')
    parser.add_argument('--judge-only')
    a = parser.parse_args()
    if a.action == 'prepare':
        prepare(a.out, a.historical, a.smoke_source)
    elif a.action == 'reproduce':
        reproduce(a.out, a.historical)
    elif a.action == 'live':
        live(a.out, a.n, a.self_control, a.judge_only)
    elif a.action == 'reinterpret':
        reinterpret(a.out)
    else:
        publish(a.out)


if __name__ == '__main__':
    main()
