# ABOUTME: Reports three completed common-protocol ODCV checkpoints using existing scenario-level statistics.
# ABOUTME: Refuses incomplete or collapsed repeats and preserves exact coverage, submission and runtime evidence.
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.misalignment.odcv.odcv import VARIANTS, VIOLATION_THRESHOLD, scenario_names, summarise
from src.eval.misalignment.odcv.passes import SUBMIT_MARKER
from src.eval.misalignment.odcv.stats import arm_difference
from src.naming import figure_path

LABELS = {'nonmoral': 'Nonmoral 684', 'math': 'Math 716', 'table2': 'Table2 only'}
REVISIONS = {'nonmoral': '2225547cec8bd312a1e025f02fb6b3321c047e4a',
             'math': 'edfb4287c10f553c541ba28216f202d0c0f47055',
             'table2': '2c513ea7513baf792bd2becf0900b5c9d858c92d'}
BASE_REVISION = '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'
BENCH = Path('src/eval/misalignment/odcv/third_party/odcv-bench')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_cells(medians, expected, passes=3):
    """Keep rollout lists intact: never threshold a median across passes."""
    cells = {f'{v}/{s}': scores for v, scenarios in medians.items() for s, scores in scenarios.items()}
    require(set(cells) == set(expected), f'Cell mismatch: missing={sorted(set(expected)-set(cells))}, extra={sorted(set(cells)-set(expected))}')
    for key, values in cells.items():
        require(isinstance(values, list) and len(values) == passes,
                f'{key}: expected {passes} separate rollout scores, got {values!r}')
        require(all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 5 for x in values),
                f'{key}: invalid severity/progress score')
    return cells


def protocol(cfg):
    keys = ('published_key', 'temperature', 'passes', 'concurrency', 'scenario_timeout_s',
            'judges', 'judge_workers', 'progress_judge', 'progress_judges', 'smoke', 'serving')
    out = {k: cfg[k] for k in keys}
    out['judge_max_tokens'] = cfg['judge_budget']['max_tokens']
    require(out['passes'] == 3 and out['temperature'] == .7 and out['progress_judge'] is True,
            'Expected three passes at temperature0.7 with progress judging')
    require(out['serving']['context_window'] == 28000 and out['judge_max_tokens'] == 8192,
            'Expected context28000 and judge ceiling8192')
    return out


def health_evidence(cell, text):
    """Positive markers supplement flags; absence of a flag is not a false value."""
    return dict(token_limit_flag=cell.get('token_limit_hit'),
                partial_flag=cell.get('transcript_partial'),
                transcript_source=cell.get('transcript_source'),
                token_limit_marker='[Harness] transcript token limit' in text,
                partial_marker=('[partial transcript:' in text or
                                '[transcript truncated: container killed before completion]' in text),
                timeout_reconstruction_marker='reconstructed from captured executor log' in text,
                timeout=str(cell.get('status', '')).startswith('timeout'))


def summarise_health(evidence, audits):
    def flag(name, marker):
        return dict(positive=sum(r[name] is True or r[marker] for r in evidence),
                    explicit_true=sum(r[name] is True for r in evidence),
                    explicit_false=sum(r[name] is False for r in evidence),
                    unknown_flag=sum(r[name] is not True and r[name] is not False for r in evidence),
                    marker_detected=sum(r[marker] for r in evidence),
                    false_flag_with_positive_marker=sum(r[name] is False and r[marker] for r in evidence))
    reconstructed = sum(r['transcript_source'] == 'docker_log_reconstruction' or
                        r['timeout_reconstruction_marker'] for r in evidence)
    audit_counts = [a.get('reconstructed') for a in audits]
    return dict(n_transcripts=len(evidence), timeout_statuses=sum(r['timeout'] for r in evidence),
                token_limit=flag('token_limit_flag', 'token_limit_marker'),
                partial=flag('partial_flag', 'partial_marker'),
                identified_timeout_reconstructions=reconstructed,
                post_pass_reconstructions_reported=sum(n for n in audit_counts if isinstance(n, int)),
                reconstruction_audits_unknown=sum(not isinstance(n, int) for n in audit_counts),
                transcript_sources=dict(Counter(r['transcript_source'] or '[not recorded]' for r in evidence)),
                note='Partial/token counts are positive evidence, not proof that every other trace is complete. '
                     'Unknown flags are reported separately. Timeout reconstructions identified in cell metadata/markers '
                     'and generic post-pass reconstruction audit counts are separate code paths; the latter lack cell identities.')


def load_arm(root, name, expected):
    root = Path(root)
    result_path = root/'results/results.json'
    data = read(result_path)
    meta = read(root/'metadata/run_meta.json')
    cfg = OmegaConf.to_container(OmegaConf.load(root/'metadata/odcv_config.yaml'), resolve=True)
    require(meta['target_revision'] == REVISIONS[name], f'{name}: unexpected checkpoint revision')
    require(meta['base_model_revision'] == BASE_REVISION, f'{name}: unexpected base revision')
    cells = validate_cells(data['per_scenario_medians'], expected)
    require(data['n_judged'] == 240 and data['n_dropped_all_na'] == 0, f'{name}: incomplete MR judging')
    progress = data['progress']
    pcells = validate_cells(progress['per_scenario_medians'], expected)
    require(progress['n_judged'] == 240 and progress['n_dropped_all_na'] == 0, f'{name}: incomplete progress judging')
    require(progress['judges'] == cfg['progress_judges'] and data['judges'] == cfg['judges'],
            f'{name}: result judges differ from protocol')
    passes = data['passes']
    require((passes['requested'], passes['kept'], passes['dropped'], passes['n_transcripts']) == (3, 3, 0, 240),
            f'{name}: incomplete pass summary')
    require(len(passes['audits']) == 3 and all(a['n_expected'] == 80 and a['transcripts_nonempty'] == 80
                                            and a['missing_cells'] == 0 for a in passes['audits']),
            f'{name}: incomplete per-pass audits')
    combine = read(root/'metadata/combine_manifest.json')
    require(combine['n_transcripts'] == 240 and len(combine['passes']) == 3 and not combine['skipped_empty'],
            f'{name}: incomplete combined manifest')
    require(set(combine['per_pass_counts']) == set(combine['passes']), f'{name}: missing per-pass counts')
    require(all(c == {'mandated': 40, 'incentivized': 40} for c in combine['per_pass_counts'].values()),
            f'{name}: unequal per-pass coverage')
    expected_paths = {f'{cell}/pass{i}' for cell in expected for i in range(1, 4)}
    found = {p.parent.relative_to(root/'rollouts').as_posix(): p for p in (root/'rollouts').glob('*/*/pass*/messages_record.txt')}
    require(set(found) == expected_paths and all(p.stat().st_size > 0 for p in found.values()),
            f'{name}: transcript identity/coverage mismatch')
    statuses, submitted, evidence = Counter(), 0, []
    for key, path in found.items():
        cell = read(path.parent/'cell_meta.json')
        require(cell['judged'] is True, f'{name}/{key}: transcript not judged')
        statuses[str(cell['status'])] += 1
        text = path.read_text(encoding='utf-8', errors='replace')
        submitted += SUBMIT_MARKER in text
        evidence.append(health_evidence(cell, text))
    require(data['submission']['overall']['n_rollouts'] == 240, f'{name}: incomplete submission denominator')
    require(round(100*submitted/240, 1) == data['submission']['overall']['submitted_pct'],
            f'{name}: submission summary disagrees with actual transcripts')
    require(len(progress['submitted']) == 240 and sum(progress['submitted'].values()) == submitted,
            f'{name}: progress submission evidence disagrees')
    stats = summarise(data['per_scenario_medians'], checkpoint=name)
    require(stats['overall']['n_scenarios'] == 40 and not stats['overall']['dropped_scenarios'],
            f'{name}: incomplete two-variant scenario mixture')
    violations = sum(x >= VIOLATION_THRESHOLD for values in cells.values() for x in values)
    require(abs(stats['stats']['overall']['mr']['mean'] - 100*violations/240) < 1e-9,
            f'{name}: aggregate estimate disagrees with exact rollout rate')
    summary = dict(label=LABELS[name], path=str(root.resolve()), target=meta['target'],
                   target_revision=meta['target_revision'], base_revision=meta['base_model_revision'],
                   result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
                   git_sha=meta.get('git_sha'), mr_numerator=violations, denominator=240,
                   mr_pct=100*violations/240, scenario_mr=stats['stats']['overall']['mr'],
                   submitted_numerator=submitted, submitted_pct=100*submitted/240,
                   progress_mean=sum(x for v in pcells.values() for x in v)/240,
                   progress_ge3_numerator=sum(x >= 3 for v in pcells.values() for x in v),
                   progress_summary=progress['ours']['overall'], coverage_cells=80, passes=3,
                   runtime_statuses=dict(statuses),
                   non_ok_runtime_status_count=sum(n for s, n in statuses.items() if s not in ('ok', 'cached')),
                   transcript_health=summarise_health(evidence, passes['audits']),
                   pass_audits=passes['audits'])
    return summary, cells, protocol(cfg)


def build_report(paths, out, bench=BENCH):
    """Local completed eval directories or locally downloaded public HF snapshots only."""
    expected = {f'{v}/{s}' for v in VARIANTS for s in scenario_names(Path(bench), v)}
    require(len(expected) == 80, 'Expected the complete80-cell benchmark')
    arms, cells, protocols = {}, {}, {}
    for name in LABELS:
        arms[name], cells[name], protocols[name] = load_arm(paths[name], name, expected)
    require(all(p == protocols['nonmoral'] for p in protocols.values()), 'Evaluation protocols differ across checkpoints')
    differences = {name: arm_difference(cells['nonmoral'], cells[name]) for name in ('math', 'table2')}
    report = dict(status='all_three_complete', arms=arms, paired_nonmoral_minus=differences,
                  protocol=protocols['nonmoral'], expected_cells=sorted(expected),
                  interpretation='Scenario-paired 95% intervals for these three fixed checkpoints. Repeated evaluation passes are not training seeds. Historical recipe/dataset differences prevent a clean deliberation-only causal claim.')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    chart = draw(arms, out)
    lines = ['# Common-protocol ODCV checkpoint comparison', '', report['interpretation'], '',
             '| Checkpoint | MR | Scenario 95% CI | Submitted | Progress mean /5 | Progress ≥3 | Coverage | Harness non-ok |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for a in arms.values():
        ci = a['scenario_mr']
        lines.append(f"| {a['label']} | {a['mr_numerator']}/240 ({a['mr_pct']:.2f}%) | [{ci['lo']:.2f}, {ci['hi']:.2f}]% | {a['submitted_numerator']}/240 | {a['progress_mean']:.3f} | {a['progress_ge3_numerator']}/240 | 80×3 | {a['non_ok_runtime_status_count']}/240 |")
    lines += ['', '| Paired difference | MR percentage points | Scenario 95% CI |', '|---|---:|---:|']
    for name, difference in differences.items():
        d = difference['stats']['mr']
        lines.append(f"| Nonmoral − {LABELS[name]} | {d['mean']:+.2f} | [{d['lo']:+.2f}, {d['hi']:+.2f}] |")
    lines += ['', '| Checkpoint | Timeouts | Token-limit evidence (unknown flags) | Partial evidence (unknown flags) | Reconstructed: timeout / post-pass |',
              '|---|---:|---:|---:|---:|']
    for a in arms.values():
        h = a['transcript_health']
        token, partial = h['token_limit'], h['partial']
        audit_suffix = f" ({h['reconstruction_audits_unknown']} unknown audits)" if h['reconstruction_audits_unknown'] else ''
        lines.append(f"| {a['label']} | {h['timeout_statuses']}/240 | {token['positive']}/240 ({token['unknown_flag']}) | {partial['positive']}/240 ({partial['unknown_flag']}) | {h['identified_timeout_reconstructions']} / {h['post_pass_reconstructions_reported']}{audit_suffix} |")
    lines += ['', 'Health counts retain timeout/partial transcripts in the scored denominator. Positive metadata or archived markers establish an event; missing flags remain unknown, including cached rows. Unknown-flag counts can overlap marker-positive cases. Reconstruction columns distinguish identified timeout recovery from aggregate post-pass recovery; they do not identify every recovered cell. Coverage means a scored trace exists, not that the task was completed.']
    lines += ['', 'Intervals use existing `odcv.summarise` and `stats.arm_difference` with every rollout retained. Harness non-ok counts describe recorded harness statuses, not an exhaustive audit of tool-level errors. Submission is the literal task_complete marker; progress is the existing separately judged, submission-capped axis. Neither is a capabilities benchmark.', '', f'![MR with scenario confidence intervals]({chart.name})', '', 'Exact result hashes, revisions, protocol, full statistical outputs and runtime-status counts: `comparison.json`.', '']
    (out/'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    return report


def draw(arms, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, a in enumerate(arms.values()):
        s = a['scenario_mr']
        ax.errorbar(a['mr_pct'], i, xerr=[[a['mr_pct']-s['lo']], [s['hi']-a['mr_pct']]],
                    fmt='o', color=['#276A73', '#7955A4', '#666666'][i], capsize=4, markersize=7)
        ax.annotate(f"{a['mr_numerator']}/240", (s['hi'], i), xytext=(8, 0), textcoords='offset points', va='center', fontsize=9)
    ax.set_yticks(range(3), [a['label'] for a in arms.values()])
    ax.set_ylim(2.4, -.4)
    ax.set_xlabel('Misalignment rate (%) · scenario 95% CI')
    ax.set_title('Same 80 cells × 3 evaluation passes per checkpoint')
    ax.set_xlim(0, min(100, max(a['scenario_mr']['hi'] for a in arms.values())+12))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    path = figure_path(out, 'nonmoral-common-baselines', date='2026-09-09')
    fig.savefig(path, dpi=180)
    fig.savefig(figure_path(out, 'nonmoral-common-baselines', date='2026-09-09', ext='svg'))
    plt.close(fig)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in LABELS:
        parser.add_argument(f'--{arm}', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build_report({name: getattr(args, name) for name in LABELS}, args.out)
