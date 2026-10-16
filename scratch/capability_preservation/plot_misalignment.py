# ABOUTME: Render saved GPT-OSS misalignment results with verified sampler identities.
# ABOUTME: Figures only: no inference, judging, imputation, or new uncertainty estimates.
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from omegaconf import OmegaConf
from src.infra.huggingface import hf_download
from src.naming import artifact_name

HERE = Path(__file__).parent


def main():
    cfg = OmegaConf.to_container(OmegaConf.load(HERE / 'misalignment.yaml'), resolve=True)
    out = Path(cfg['output'])
    (out / 'sources').mkdir(parents=True, exist_ok=True)
    receipts = []

    def read(source, filename, local_name):
        p = Path(hf_download(source['repo'], filename, repo_type='dataset', revision=source['revision']))
        raw = p.read_bytes()
        (out / 'sources' / local_name).write_bytes(raw)
        receipts.append({**source, 'file': filename, 'sha256': hashlib.sha256(raw).hexdigest()})
        return json.loads(raw)

    odcv_meta = read(cfg['odcv'], 'metadata/run_meta.json', 'odcv_run_meta.json')
    odcv = {}
    for label, arm in [('Base', 'base'), ('Control', 'control'), ('DA15', 'da15')]:
        assert odcv_meta['config']['arms'][arm] == cfg['targets'][label]
        result = read(cfg['odcv'], f'results/{arm}.json', f'odcv_{arm}.json')
        odcv[label] = result['ours']['overall']
    mask, mask_rows = {}, {}
    for label, source in cfg['mask'].items():
        meta = read(source, 'metadata/run_meta.json', f'mask_{label}_run_meta.json')
        result = read(source, 'results/results.json', f'mask_{label}.json')
        assert result['target'] == cfg['targets'][label]
        assert meta.get('target', meta['target_revision']) == cfg['targets'][label]
        assert result['n_rows'] == 1000
        if label == 'Control':
            assert result['target_revision'] == meta['target_revision']
            assert result['status'] == 'closed_with_exclusions'
        else:
            assert result['generation_errors']['errors'] == 0
        mask[label] = result['overall_honesty_score']
        mask_rows[label] = result.get('n_scored', result['n_rows'])

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})

    def chart(name, title, ylabel, labels, values, intervals=None, note=None):
        fig = plt.figure(figsize=(8.4, 5.6))
        fig.text(.12, .92, title, fontsize=22, weight='bold', color='#202C3E')
        ax = fig.add_axes([.14, .13, .81, .68])
        ax.set_ylim(0, 100)
        ax.set_xlim(-.65, len(labels)-.35)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(0, 101, 20))
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_ylabel(ylabel, labelpad=10, color='#596474')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0, pad=9, labelcolor='#202C3E')
        ax.grid(axis='y', color='#DDE2E8', linewidth=.8)
        ax.set_axisbelow(True)
        for i, (label, value) in enumerate(zip(labels, values)):
            if value is None:
                ax.text(i, 5, 'No result', ha='center', color='#596474', fontsize=11)
                continue
            ax.bar(i, value, width=.55, color=cfg['colours'][label], zorder=3)
            high = value
            if intervals is not None:
                lo, high = intervals[i]
                assert lo <= value <= high
                ax.errorbar(i, value, yerr=[[value-lo], [high-value]], fmt='none',
                            ecolor='#202C3E', capsize=5, elinewidth=1.3, zorder=4)
            ax.text(i, high+2, f'{value:.1f}%', ha='center', fontsize=14,
                    color='#202C3E', weight='bold')
        if note:
            fig.text(.54,.025,note,ha='center',fontsize=9,color='#596474')
        stem=artifact_name(f'gptoss120b-{name}',date=cfg['date'])
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(out / f"{stem}.{ext}", dpi=240, facecolor='white')
        plt.close(fig)

    arms = ['Base', 'Control', 'DA15']
    chart('odcv', 'GPT-OSS-120B · ODCV', 'Misalignment rate (%) ↓', arms,
          [odcv[a]['mr_pct'] for a in arms], [odcv[a]['mr_ci95'] for a in arms])
    chart('mask', 'GPT-OSS-120B · MASK', 'Honesty score (%) ↑', arms,
          [mask[a] for a in arms], note=f"Control: {mask_rows['Control']}/1000 rows after one retry; Base/DA15: 1000 each.")
    labels = ['Base', 'Control', 'DA15', 'MSM', 'DA15+MSM']
    chart('mask-all-available', 'GPT-OSS-120B · MASK', 'Honesty score (%) ↑', labels,
          [mask[a] for a in labels], note=f"Control: {mask_rows['Control']}/1000 rows after one retry; other arms: 1000 each.")

    report = ['# GPT-OSS misalignment figures', '',
              'Only existing results were read. No model inference, judging, or rescoring.', '',
              '## Main ODCV chart', '',
              '| Arm | Misalignment rate | 95% CI | Usable alignment verdicts |',
              '|---|---:|---:|---:|']
    for a in arms:
        r = odcv[a]
        report.append(f"| {a} | {r['mr_pct']}% | {r['mr_ci95']} | {r['n_rollouts']}/240 |")
    report += ['', 'Source: October 12 full 56K run, medium reasoning, 40 scenarios, two variants, three passes per cell.',
               'The chart uses the published Gemini-only, scenario-aggregated scores and original scenario-level 95% intervals. It does not mix in the later two Sonnet judgments. One alignment verdict each for Control and DA15 was provider-blocked; these remain excluded, never imputed.',
               'The later mixed-judge supplement has 103/240, 87/240 and 27/240 misaligned rollouts. It is retained separately at revision eaaa045bffedd1e7b22d05e03b5b0f1b04d35d41. Raw proportions and published scenario-aggregated estimates are distinct.',
               'Submission is a separate axis: 220/240 Base, 205/240 Control, 204/240 DA15. Limits: context 2/20/15; cycles 18/15/21. Lower misconduct alone does not establish equal task completion.',
               'The selected 112K rerun and older low-reasoning ODCV runs are separate protocols, not pooled into this chart.', '',
               '## MASK charts', '', '| Arm | Published honesty score | Rows |', '|---|---:|---:|']
    report += [f'| {a} | {mask[a]}% | {mask_rows[a]} |' for a in labels]
    report += ['', 'Plain-Control MASK received one authorized retry of 55 incomplete generations and 11 judgments. Four generation and two judge failures exclude six rows. Original outputs and retry costs are retained. MSM is a distinct training arm, not a substitute for plain Control.',
               'MASK is the published per-row honesty score (100 minus the dishonest percentage), not a simple count of explicitly honest answers. Each arm has one pass; published confidence intervals are null, so none are invented. Historical arms report zero generation errors; Control is an available-row estimate after retries and is not an identical first-attempt protocol.',
               'The DA15 sampler identity matches the October 12 ODCV DA15 checkpoint exactly. Base is tinker://base. Sampling and evaluation protocols differ between MASK and ODCV.', '',
               '## Inventory', '',
               'The earlier 2026-10-15 inventory searched 991 datasets and metadata for 158 MASK/AM/agentic candidates, finding four GPT-OSS MASK arms and no GPT-OSS Agentic Misalignment result. Control MASK was subsequently published and is now included. This script reads configured pins; it does not repeat the whole inventory.', '',
               '## Pinned sources', '']
    seen = set()
    for r in receipts:
        key = (r['repo'], r['revision'])
        if key not in seen:
            report.append(f"- https://huggingface.co/datasets/{r['repo']}/tree/{r['revision']}")
            seen.add(key)
    (out / 'figure_notes.md').write_text('\n'.join(report)+'\n', encoding='utf8')
    meta = {'date': cfg['date'], 'git_sha': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'config': cfg,
            'sources': receipts, 'displayed': {'odcv': odcv, 'mask': mask},
            'note': 'Existing published scores and intervals only; no inference or judging.'}
    (out / 'run_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf8')
    archive = out / f"{cfg['date']}-gptoss-misalignment-figures.zip"
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file() and p.suffix != '.zip':
                z.write(p, p.relative_to(out))
        z.write(__file__, 'source/plot_misalignment.py')
        z.write(HERE / 'misalignment.yaml', 'source/misalignment.yaml')
    print(archive)


if __name__ == '__main__':
    main()
