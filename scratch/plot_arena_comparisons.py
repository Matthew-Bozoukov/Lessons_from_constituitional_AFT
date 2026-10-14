# ABOUTME: Draw vertical bars for the three October Qwen Arena pairwise comparisons.
# ABOUTME: Read immutable published results, preserve confidence intervals, and label the judge exception.
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from src.infra.huggingface import hf_download
from src.naming import artifact_name
from src.utils import write_run_meta

ROOT = Path('output/arena_qwen_base_20261013')
OUT = ROOT / 'figures'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    prior_repo = 'dougalldeepmind/2026-10-12-ah-vs-qwen36-0-plain'
    prior_sha = '4f19416df56ca922b94fc3403dfe7dbea79ce698'
    prior = read(hf_download(prior_repo, 'results/results.json', repo_type='dataset', revision=prior_sha))
    sources = [{'repo': prior_repo, 'revision': prior_sha, 'file': 'results/results.json'}]
    current_repo = 'dougalldeepmind/2026-10-13-ah-vs-qwen36'
    current_sha = '99f48c89e31c41626d9311a4f4a95fc965888302'
    current = read(hf_download(current_repo, 'results/results.json', repo_type='dataset', revision=current_sha))
    rows = {r['model_key']: r for r in current['leaderboard']}
    sources.append({'repo': current_repo, 'revision': current_sha, 'file': 'results/results.json'})
    comparisons = [
        ('Control\nvs base', rows['qwen36_0_plain'], '#6D7786'),
        ('DA15\nvs base', rows['qwen36_0_da_15'], '#8651B2'),
        ('DA15\nvs control', prior['leaderboard'][0], '#8651B2'),
    ]
    has_fallback = bool(rows.get('qwen36_0_da_15', {}).get('judge_fallbacks'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.spines.bottom': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.7), sharey=True)
    fig.patch.set_facecolor('#FAFAFC')
    fig.subplots_adjust(left=.08, right=.98, top=.74, bottom=.24, wspace=.14)
    fig.text(.08, .935, 'Arena-Hard: Qwen3.6 comparisons', fontsize=23, weight='bold', color='#192338')
    fig.text(.08, .885, 'Style-controlled preference score  ·  50% = tie  ·  Error bars = 95% confidence intervals',
             fontsize=12, color='#586174')
    values = []
    for ax, (category, title) in zip(axes, [('hard_prompt', 'Hard prompts · 500 questions'),
                                          ('creative_writing', 'Creative writing · 250 questions')]):
        ax.set_facecolor('#FAFAFC')
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E2E5EC', linewidth=.7)
        ax.set_ylim(0, .70)
        ax.set_xlim(-.55, 2.55)
        ax.set_yticks([0, .1, .2, .3, .4, .5, .6, .7])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.tick_params(axis='both', length=0, pad=8, labelcolor='#414C60')
        ax.axhline(.5, color='#4D596E', linestyle=(0, (4, 4)), linewidth=1.2, zorder=2)
        ax.set_title(title, fontsize=15, loc='left', pad=18, weight='bold', color='#263248')
        ax.set_xticks(range(3), [c[0] for c in comparisons])
        for i, (label, row, color) in enumerate(comparisons):
            data = row['by_slice'][category]['controlled']
            mean, lo, hi = data['mean'], data['ci_lower'], data['ci_upper']
            ax.bar(i, mean, width=.55, color=color, zorder=3)
            ax.errorbar(i, mean, yerr=[[mean-lo], [hi-mean]], fmt='none', color='#273247',
                        elinewidth=1.6, capsize=6, capthick=1.6, zorder=4)
            ax.text(i, hi+.025, f'{100*mean:.1f}%', ha='center', fontsize=14, weight='bold', color='#263248')
            values.append({'comparison': label.replace('\n', ' '), 'slice': category, **data})
    axes[0].set_ylabel('Target preference score', labelpad=12, color='#414C60')
    fig.text(.08, .125, 'Base = original Qwen/Qwen3.6-27B, before our SFT. Each bar uses the reference named below it.',
             fontsize=10.5, color='#586174')
    fig.text(.08, .082, 'Matched thinking / 12k-token protocol; GPT-4.1 judge; length + Markdown control; prompt-bootstrap intervals.',
             fontsize=10, color='#586174')
    if has_fallback:
        fig.text(.08, .035, 'DA15 vs base: one hard-prompt verdict used an approved Gemini fallback (1 of 1,500); all other verdicts use GPT-4.1.',
                 fontsize=9.7, color='#885F28')
    else:
        fig.text(.08, .035, 'Pairwise preference scores are not absolute accuracy. The comparisons share answers and are not independent.',
                 fontsize=10, color='#586174')
    OUT.mkdir(parents=True, exist_ok=True)
    stem = artifact_name('arena-qwen36-pairwise-comparisons')
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'{stem}.{extension}', dpi=200, facecolor=fig.get_facecolor())
    write_run_meta(OUT, {'sources': sources, 'values': values, 'pending': len(values) < 6},
                   extra={'command': 'uv run python -X utf8 scratch/plot_arena_comparisons.py'})
    lines = ['# Arena-Hard pairwise comparison figure', '', '| Comparison | Slice | Score | 95% CI |', '|---|---|---:|---|']
    lines += [f"| {v['comparison']} | {v['slice']} | {v['mean']:.6f} | [{v['ci_lower']:.6f}, {v['ci_upper']:.6f}] |" for v in values]
    lines += ['', 'Sources:', '```json', json.dumps(sources, indent=2), '```']
    (OUT / f'{stem}_results.md').write_text('\n'.join(lines), encoding='utf-8')
    print(OUT / f'{stem}.png')


if __name__ == '__main__':
    main()
