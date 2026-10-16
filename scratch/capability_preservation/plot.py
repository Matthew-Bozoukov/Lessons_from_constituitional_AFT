# ABOUTME: Render the user's pairwise Arena tables and grouped capability bars for two families.
# ABOUTME: Preserve pinned provenance, exact score direction, and published bootstrap intervals.
from __future__ import annotations

import hashlib
import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from omegaconf import OmegaConf

from src.infra.huggingface import hf_download
from src.naming import artifact_name

HERE = Path(__file__).parent
INK, MUTED, LINE = '#202C3E', '#596474', '#DDE2E8'
ARMS = ['Base', 'Control', 'DA15']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bars-only', action='store_true')
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(HERE / 'figures.yaml'), resolve=True)
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    source_data, receipts = {}, []
    def published(source, filename):
        p = Path(hf_download(source['repo'], filename,
                            repo_type='dataset', revision=source['revision']))
        receipts.append({**source, 'file': filename,
                         'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
        return json.loads(p.read_text(encoding='utf-8'))

    for key, source in cfg['arena_sources'].items():
        source_data[key] = published(source, 'results/results.json')
    # Fail before drawing if the handwritten display counts disagree with the pinned evidence.
    for family, model in cfg['models'].items():
        for i, (repo, revision) in enumerate(model['mmlu_repos']):
            data = published(dict(repo='dougalldeepmind/'+repo, revision=revision), 'results/results.json')
            assert data['n'] == 570 and data['n_correct'] == model['mmlu'][i]
        for i, (repo, revision) in enumerate(model['swe_repos']):
            source = dict(repo='dougalldeepmind/'+repo, revision=revision)
            if family == 'gptoss120b':
                data = published(source, 'metadata/summary.json')
                assert [data[a]['success'] for a in ['base','control','da15']] == model['swe']
                assert all(data[a]['graded'] == 300 for a in data)
            else:
                data = published(source, 'results/results.json')
                assert data['n_graded'] == 300 and data['n_resolved'] == model['swe'][i]

    qbase = {r['model_key']: r for r in source_data['qwen_base']['leaderboard']}
    qcontrol = source_data['qwen_control']['leaderboard'][0]
    arena = {
        'qwen36': [qbase['qwen36_0_plain']['by_slice']['hard_prompt']['controlled'],
                   qcontrol['by_slice']['hard_prompt']['controlled'],
                   qbase['qwen36_0_da_15']['by_slice']['hard_prompt']['controlled']],
        'gptoss120b': [source_data['gptoss']['results'][pair]['hard_prompt']['controlled']
                       for pair in ['control-vs-base', 'da15-vs-control', 'da15-vs-base']],
    }
    # Display the sketch's first-named arm. Complement both mean and interval endpoints.
    for key, values in arena.items():
        arena[key] = [
            {'left': left, 'right': right, 'mean': 1-v['mean'] if flip else v['mean'],
             'lo': 1-v['ci_upper'] if flip else v['ci_lower'],
             'hi': 1-v['ci_lower'] if flip else v['ci_upper']}
            for v, (left, right, flip) in zip(values, [
                ('Base', 'Control', True), ('Control', 'DA15', True), ('DA15', 'Base', False)])
        ]

    def save(fig, subject, mirror):
        stem = artifact_name(subject, date=cfg['date'])
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(out / f'{stem}.{ext}', dpi=220, facecolor='white')
        (out / f'{stem}_results.md').write_text(mirror, encoding='utf-8')
        plt.close(fig)
        print(out / f'{stem}.png')

    for key, model in cfg['models'].items():
        source_links = '\n'.join(
            f'- https://huggingface.co/datasets/dougalldeepmind/{repo}/tree/{rev}'
            for field in ['mmlu_repos', 'swe_repos'] for repo, rev in model[field])
        fig = plt.figure(figsize=(10.5, 6.0))
        fig.text(.09, .93, model['title'], size=24, weight='bold', color=INK)
        ax = fig.add_axes([.13, .12, .83, .65])
        ax.set_ylim(0, 100)
        ax.set_xlim(-.65, 2.05)
        ax.set_yticks(range(0, 101, 20))
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_ylabel('Accuracy / tasks resolved (%)', labelpad=12, color=MUTED)
        ax.set_xticks([0, 1.4], ['MMLU', 'SWE-bench Lite'])
        ax.tick_params(axis='both', length=0, labelcolor=INK, pad=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis='y', color=LINE, lw=.8)
        ax.set_axisbelow(True)
        for i, arm in enumerate(ARMS):
            xs = [(i-1)*.25, 1.4+(i-1)*.25]
            vals = [100*model['mmlu'][i]/570, 100*model['swe'][i]/300]
            ax.bar(xs, vals, width=.22, color=cfg['colours'][arm], zorder=3)
            for j, (x, v) in enumerate(zip(xs, vals)):
                ax.text(x, v+1.65, f'{v:.1f}%', ha='center', color=INK,
                        size=12, weight='bold')
        fig.legend(handles=[Patch(facecolor=cfg['colours'][a], label=a)
                            for a in ARMS], loc='upper left', bbox_to_anchor=(.083, .88),
                   ncol=3, frameon=False, fontsize=11, handlelength=1.3, columnspacing=1.6)
        notes = [model['checkpoints'],
                 'MMLU: final-answer scoring; not full MMLU. SWE: official grading, with two no-fix fixture passes.']
        if key == 'gptoss120b':
            notes += ['Control: 41/570 unparseable; below the 95% parsing-health threshold.',
                      'This MMLU point is descriptive, not a clean capability estimate. DA15: 17 unparseable.']
        else:
            notes += ['All MMLU responses parsed. Single training seed; bars show observed scores.']
        mirror = f"# {model['title']} capability preservation\n\n| Arm | MMLU /570 | SWE /300 |\n|---|---:|---:|\n"
        mirror += '\n'.join(f"| {a} | {model['mmlu'][i]} ({100*model['mmlu'][i]/570:.4f}%) | {model['swe'][i]} ({100*model['swe'][i]/300:.4f}%) |" for i,a in enumerate(ARMS))
        mirror += '\n\n' + '\n'.join(notes) + '\n\n## Immutable sources\n\n' + source_links + '\n'
        save(fig, f'{key}_capability_preservation', mirror)
        if args.bars_only:
            continue

        fig = plt.figure(figsize=(10.5, 4.8))
        fig.text(.08, .91, f"{model['title']} · Arena-Hard", size=23, weight='bold', color=INK)
        fig.text(.08, .845, 'Style-adjusted preference score · hard prompts', size=14, color=MUTED)
        ax = fig.add_axes([.07, .07, .87, .67])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.text(.025, .94, 'COMPARISON', size=10, weight='bold', color=MUTED)
        ax.text(.66, .94, 'FIRST MODEL’S SCORE', size=10, weight='bold', color=MUTED, ha='center')
        ax.text(.935, .94, '95% CI', size=10, weight='bold', color=MUTED, ha='center')
        ax.axhline(.875, color=INK, lw=1.1)
        for i, row in enumerate(arena[key]):
            y = .735 - i*.275
            ax.text(.025, y, row['left'], size=20, weight='bold', color=cfg['colours'][row['left']], va='center')
            ax.text(.225, y, 'vs', size=16, color=MUTED, va='center')
            ax.text(.295, y, row['right'], size=20, weight='bold', color=cfg['colours'][row['right']], va='center')
            ax.text(.66, y, f"{100*row['mean']:.1f}%", size=25, weight='bold', color=INK, ha='center', va='center')
            ax.text(.935, y, f"{100*row['lo']:.1f}–{100*row['hi']:.1f}%", size=12, color=MUTED, ha='center', va='center')
            ax.axhline(y-.1375, color=LINE, lw=.8)
        notes = ['Above 50% favors the first model; below 50% favors the second. Ties count half.',
                 *model['arena_notes'], 'Length/Markdown-adjusted estimates; 95% prompt-bootstrap intervals.']
        mirror = f"# {model['title']} Arena-Hard\n\n| First model | Second model | First-model score | 95% CI |\n|---|---|---:|---|\n"
        mirror += '\n'.join(f"| {r['left']} | {r['right']} | {r['mean']*100:.6f}% | [{r['lo']*100:.6f}, {r['hi']*100:.6f}] |" for r in arena[key])
        mirror += '\n\n'+'\n'.join(notes)+'\n\nReverse directions use 1-p and [1-upper, 1-lower], without refitting.\n'
        mirror += 'These are repository style-controlled estimates; not a claim of exact upstream estimator equivalence.\n'
        mirror += '\n'.join(f"\n- https://huggingface.co/datasets/{s['repo']}/blob/{s['revision']}/results/results.json" for k,s in cfg['arena_sources'].items() if (k.startswith('qwen') if key=='qwen36' else k=='gptoss'))
        save(fig, f'{key}_arena_hard_pairwise', mirror)
    meta = {'git_sha': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
            'config': cfg, 'sources': receipts, 'displayed_arena': arena,
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'note': 'Figures only; no new inference, judging or rescoring. Cross-family protocols differ.'}
    (out/'run_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
