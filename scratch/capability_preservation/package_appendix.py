# ABOUTME: Assemble reviewed appendix prose, figures and immutable HF links into a local paper handoff.
# ABOUTME: Verify saved metadata hashes and package evidence without inference, judging or publication.
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output/appendix_20261015'
FIGURES = [
    ('A1', 'capability_preservation', '2026-10-14-qwen36-capability-preservation'),
    ('A2', 'capability_preservation', '2026-10-14-qwen36-arena-hard-pairwise'),
    ('A3', 'gptoss_misalignment', '2026-10-15-gptoss120b-odcv'),
    ('A4', 'gptoss_misalignment', '2026-10-15-gptoss120b-mask'),
    ('A5', 'capability_preservation', '2026-10-14-gptoss120b-capability-preservation'),
    ('A6', 'capability_preservation', '2026-10-14-gptoss120b-arena-hard-pairwise'),
    ('A7', 'gptoss_misalignment', '2026-10-15-gptoss120b-mask-all-available'),
]


def main():
    records = {}
    for section, filename in [('sources', 'sources.json'), ('lineage', 'lineage.json')]:
        data = json.loads((OUT/filename).read_text(encoding='utf-8'))
        for key, rec in data.items():
            assert re.fullmatch('[0-9a-f]{40}', rec['revision']), (key, rec['revision'])
            for name, evidence in rec['files'].items():
                raw = (OUT/section/key/name).read_bytes()
                assert hashlib.sha256(raw).hexdigest() == evidence['sha256'], (key, name)
            records[key] = rec
    groups = [
        ('Qwen models and training mixtures', lambda k: k.startswith('qwen') and '_mmlu_' not in k and '_swe_' not in k),
        ('Qwen evaluation runs and Arena answers', lambda k: k.startswith('arena_qwen') or k.startswith('qwen36_')),
        ('GPT-OSS models and training mixtures', lambda k: k in ['gptoss_control','gptoss_da15','gptoss_control_data','gptoss_da15_configured_data']),
        ('GPT-OSS capability runs and Arena answers', lambda k: k.startswith('gptoss120b_') or k.startswith('arena_gptoss')),
        ('GPT-OSS misalignment results', lambda k: k.startswith('mask_') or k == 'gptoss_odcv'),
        ('Synthetic source', lambda k: k == 'da_synthetic_source'),
    ]
    lines = ['<!-- ABOUTME: Immutable HF source catalogue for the October appendix figures. -->',
             '<!-- ABOUTME: Generated from downloaded and SHA256-verified run metadata by package_appendix.py. -->',
             '# HF links for the appendix', '',
             'Every experiment link below is pinned. Dataset repositories contain raw/evaluation/metadata evidence; model repositories contain exported adapters. Index 0/1/2 means Base/Control/DA15. GPT-OSS SWE is a single three-arm repository.', '',
             '**Provenance qualification:** GPT-OSS DA15 has no completed training metadata in its model export. Its configured-data link is from the historical launch configuration, not a recovered training receipt. The model card does bind the evaluated sampler identity. Optional MSM results have sampler provenance in their eval metadata; no separately verified adapter export is claimed.', '']
    used = set()
    for title, matches in groups:
        lines += ['## '+title, '', '| Artifact | Immutable repository |', '|---|---|']
        for key, rec in sorted(records.items()):
            if not matches(key): continue
            used.add(key)
            prefix = 'datasets/' if rec.get('repo_type','dataset') == 'dataset' else ''
            url = f"https://huggingface.co/{prefix}{rec['repo']}/tree/{rec['revision']}"
            lines.append(f"| {key} | [{rec['repo']} @ {rec['revision'][:8]}]({url}) |")
        lines.append('')
    assert used == records.keys(), records.keys()-used
    lines += ['## Base models and original benchmark inputs', '',
        '- [Qwen3.6-27B base weights and tokenizer](https://huggingface.co/Qwen/Qwen3.6-27B/tree/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9).',
        '- [GPT-OSS-120B public tokenizer/configuration revision](https://huggingface.co/openai/gpt-oss-120b/tree/b5c939de8f754692c1647ca79fbf85e8c1e70f8a). Provider-served Base identity is `tinker://base`; this public revision is not proof of provider weight bytes or precision.',
        '- [MMLU input](https://huggingface.co/datasets/cais/mmlu/tree/c30699e8356da336a370243923dbaf21066bb9fe): 10 test questions/subject, 57 subjects, five dev examples, seed 0.',
        '- [SWE-bench Lite input](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/tree/6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2): all 300 test problems.',
        '- [Arena-Hard v2.0 vendored input](https://github.com/lmarena/arena-hard-auto/tree/196f6b826783b3da7310e361a805fa36f0be83f3): 500 hard and 250 creative prompts before the GPT-OSS judge exclusion.',
        '- [ODCV vendored scenarios](https://github.com/McGill-DMaS/ODCV-Bench/tree/7353f1cf4b2579a3a8a5b8a5061d7c7d41f60668). Project harness patches and exact configs are retained with each run.',
        '- [MASK vendored input](https://github.com/centerforaisafety/mask/tree/25e0b1201e6c928ebe69f7c5aad6fa9063a377ea). The Control publication contains frozen CSV hashes, historical scoring sources and raw request/response archives.',
        '', '## Upstream sources of the plain mixture', '',
        'These are the revisions documented by the plain mixture card. The released mixture/provenance files, rather than re-sampling upstream HEAD, define the training rows.', '']
    for repo,rev in [
        ('nvidia/Nemotron-SFT-Instruction-Following-Chat-v3','be3b3e04ef605ac9d3f8f35b9d5a632f4a3a3402'),
        ('allenai/WildChat-1M','7d6490e462285cf85d91eabea0f9a954fbddcd1f'),
        ('nvidia/Nemotron-SFT-Math-v4','84d42ad0cb960f07f951b9baa9ed2b46a5a18c66'),
        ('nvidia/OpenCodeReasoning','20a1ca19c0d050fe9057fc08339d6b370ec1c67a'),
        ('nvidia/Nemotron-SFT-Science-v2','6536a5021222a94126968e8c92f29ee47fc8a7df')]:
        lines.append(f'- [{repo}](https://huggingface.co/datasets/{repo}/tree/{rev})')
    links = '\n'.join(lines)+'\n'
    (OUT/'HF_LINKS.md').write_text(links,encoding='utf-8')
    (ROOT/'docs/appendix_hf_links_2026-10-15.md').write_text(links,encoding='utf-8')
    text = (ROOT/'docs/appendix_capability_replication_2026-10-15.md').read_text(encoding='utf-8')
    for number, folder, stem in FIGURES:
        for ext in ['png','pdf','svg']:
            dest=OUT/'figures'/f'{stem}.{ext}';dest.parent.mkdir(exist_ok=True)
            shutil.copy2(ROOT/'output/figures'/folder/f'{stem}.{ext}',dest)
        text=text.replace(f'File: `{stem}.pdf`.',f'![Figure {number}](figures/{stem}.png)\n\n[PDF](figures/{stem}.pdf) · [SVG](figures/{stem}.svg)')
    text += '\n\n[Complete immutable HF link catalogue](HF_LINKS.md)\n'
    (OUT/'APPENDIX.md').write_text(text,encoding='utf-8')
    for folder in ['capability_preservation','gptoss_misalignment']:
        dest=OUT/'figure_provenance'/folder;dest.mkdir(parents=True,exist_ok=True)
        for p in (ROOT/'output/figures'/folder).glob('*'):
            if p.suffix in ['.md','.json']: shutil.copy2(p,dest/p.name)
    meta=dict(source_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              figure_count=len(FIGURES),note='Local appendix handoff; no new inference, judging or HF publication.')
    (OUT/'package_meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    selected=[p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ['.zip','.log','.txt'] and p.name!='SHA256SUMS.json']
    hashes={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(selected)}
    sums=OUT/'SHA256SUMS.json';sums.write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    archive=OUT/'2026-10-15-paper-appendix.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in selected+[sums]:z.write(p,p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,sha in hashes.items(): assert hashlib.sha256(z.read(name)).hexdigest()==sha
    print(f'{archive}: {len(hashes)} files verified, {len(FIGURES)} figures')


if __name__=='__main__': main()
