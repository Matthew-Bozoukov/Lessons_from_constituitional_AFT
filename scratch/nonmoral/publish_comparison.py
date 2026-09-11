# ABOUTME: Publish the completed three-checkpoint comparison with source, cost and audit evidence.
# ABOUTME: Requires verified teardown, matching shell bytes and public result hashes; excludes account snapshots.
import hashlib
import json
from pathlib import Path
import subprocess
import argparse
import re
import yaml
from datetime import date
from huggingface_hub import CommitOperationAdd, CommitOperationDelete

from src.infra.huggingface import card_markdown, gate_push, hf_api, hf_download, hf_org, push_run_dir
from src.naming import artifact_name
from scratch.nonmoral.publish_invalid_baseline import secret_values, scan

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'output/nonmoral_overnight/20260909'
PREFIXES = {'nonmoral': 'baseline_lf', 'math': 'math_baseline', 'table2': 'table2_baseline'}


def correct_publication_metadata(audit_dir):
    """Correct audited presentation metadata, retaining immutable scientific evidence."""
    audit_dir = Path(audit_dir)
    inventory = json.loads((audit_dir/'inventory.json').read_text())
    api = hf_api(); receipts = []; secrets = secret_values()
    for row in inventory:
        if not row['issues']:
            continue
        repo = row['repo']; fields = row['card_fields']
        assert row['kind'] == 'dataset' and repo.startswith('dougalldeepmind/')
        gate_push(repo, fields)
        current = api.dataset_info(repo)
        assert current.sha == row['revision'], 'Repo changed after audit; inspect before updating'
        card = (audit_dir/repo.split('/')[-1]/'README.md').read_text(encoding='utf-8')
        fm_text, body = card.split('---\n', 2)[1:]
        fm = yaml.safe_load(fm_text)
        removed = []
        operations = []
        presentation_files = ['README.md']
        for filename in row['figures']:
            path = Path(hf_download(repo, filename, repo_type='dataset', revision=row['revision']))
            raw = path.read_bytes()
            local = audit_dir/'retained_figures'/repo.split('/')[-1]/Path(filename).name
            local.parent.mkdir(parents=True, exist_ok=True); local.write_bytes(raw)
            removed.append(dict(file=filename, sha256=hashlib.sha256(raw).hexdigest()))
            operations.append(CommitOperationDelete(path_in_repo=filename))
        if removed:
            body = '\n'.join(line for line in body.splitlines() if not line.startswith('!['))+'\n'
            body = body.replace('and figures;', '(figures remain local);').replace('and figures |', '(figures remain local) |')
            for filename in row['files']:
                if not (filename.startswith('results/') and filename.endswith('.md')):
                    continue
                path = Path(hf_download(repo, filename, repo_type='dataset', revision=row['revision']))
                text = path.read_text(encoding='utf-8')
                clean = '\n'.join(line for line in text.splitlines() if not line.startswith('!['))+'\n'
                if clean != text:
                    scan(clean.encode(), filename, secrets)
                    operations.append(CommitOperationAdd(path_in_repo=filename, path_or_fileobj=clean.encode()))
                    presentation_files.append(filename)
        invalid = 'invalid' in row['tags'] and 'unjudged' in row['tags']
        if invalid:
            fm['tags'] = [t for t in fm['tags'] if t not in ('eval-run', 'eval:odcv')]
            fm['tags'] += ['research-incident', 'benchmark:odcv']
        assert removed or invalid, row['issues']
        note = ('Presentation correction: figures remain local under repository policy; old figure files and manifests remain available at the prior revision.' if removed else
                'Discovery correction: this INVALID/UNJUDGED archive is tagged as a research incident, not a completed eval-run. No verdict or missing result was invented.')
        body += '\n'+note+'\n'
        card = '---\n'+yaml.safe_dump(fm, sort_keys=False)+'---\n'+body
        proof = dict(previous_revision=row['revision'], date=date.today().isoformat(), reason=note,
            removed_figures=removed, presentation_files=presentation_files, scientific_data_unchanged=True,
            code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
        raw = (json.dumps(proof, indent=2)+'\n').encode()
        scan(card.encode(), 'README.md', secrets); scan(raw, 'correction.json', secrets)
        operations += [CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=card.encode()),
            CommitOperationAdd(path_in_repo='metadata/'+artifact_name('publication-correction')+'.json', path_or_fileobj=raw)]
        result = api.create_commit(repo_id=repo, repo_type='dataset', operations=operations,
            parent_commit=row['revision'], commit_message='Apply documented figure storage and incident discovery standards')
        before = api.dataset_info(repo, revision=row['revision'], files_metadata=True)
        after = api.dataset_info(repo, revision=result.oid, files_metadata=True)
        remaining = {f.rfilename: (f.blob_id, f.lfs.sha256 if f.lfs else None) for f in after.siblings}
        changed = {*presentation_files, *(r['file'] for r in removed)}
        for f in before.siblings:
            if f.rfilename not in changed:
                assert remaining[f.rfilename] == (f.blob_id, f.lfs.sha256 if f.lfs else None), f.rfilename
        receipts.append(dict(repo=repo, revision=after.sha, public=not after.private, **proof))
        (audit_dir/'corrections.json').write_text(json.dumps(receipts, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipts, indent=2))
REPOS = {'nonmoral': '2026-09-09-odcv-nonmoral-lf-common-3x',
         'math': '2026-09-09-odcv-math-common-3x',
         'table2': '2026-09-09-odcv-table2-common-3x'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    assert hf_org() == 'dougalldeepmind'
    report = read(BASE / 'baseline_comparison/comparison.json')
    assert report['status'] == 'all_three_complete'
    states, costs, public = {}, {}, {}
    api = hf_api()
    for arm, prefix in PREFIXES.items():
        state = states[arm] = read(BASE / f'{prefix}_status.json')
        assert state['phase'] == 'evaluation_completed' and state['termination_verified']
        assert state['target_revision'] == report['arms'][arm]['target_revision']
        costs[arm] = read(BASE / f'{prefix}_cost_accounting.json')
        assert costs[arm]['judge_requests'] >= 480
        repo = hf_org() + '/' + REPOS[arm]
        info = api.dataset_info(repo)
        assert not info.private
        path = hf_download(repo, 'results/results.json', repo_type='dataset', revision=info.sha)
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == report['arms'][arm]['result_sha256']
        public[arm] = dict(url='https://huggingface.co/datasets/' + repo, revision=info.sha,
                           result_sha256=report['arms'][arm]['result_sha256'])
    assert all(s['shell_sha256'] == states['nonmoral']['shell_sha256'] for s in states.values())
    auth = read(BASE / 'authorization.json')
    data = read(BASE / 'data/terminal_decision.json')
    invalid = read(BASE / 'baseline_status.json')
    assert invalid['termination_verified']
    invalid_cost = ((invalid['terminated_at_unix'] - invalid['rented_at_unix']) / 3600
                    * (invalid['actual_gpu_hourly_usd'] + invalid['storage_hourly_reserve_usd']))
    budget = dict(ceiling_usd=auth['total_ceiling_usd'],
                  prior_settled_usd=auth['prior_settled_usd'],
                  prior_uncertain_reserved_usd=auth['prior_uncertain_reserved_usd'],
                  fresh_data_usd=data['settled_usd'] + data['reserved_usd'],
                  invalid_attempt_gpu_storage_estimate_usd=invalid_cost,
                  completed_checkpoint_estimates_usd={k: v['total_attributed_estimate_including_reserved_usd']
                                                     for k, v in costs.items()},
                  judge_unsettled_requests={k: v['judge_unsettled_requests'] for k, v in costs.items()})
    budget['total_exposure_estimate_usd'] = (budget['prior_settled_usd'] + budget['prior_uncertain_reserved_usd']
        + budget['fresh_data_usd'] + invalid_cost + sum(budget['completed_checkpoint_estimates_usd'].values()))
    assert budget['total_exposure_estimate_usd'] < budget['ceiling_usd']
    budget['basis'] = 'Token-rate judge estimates and full owned-pod lifetimes, including storage/uncertain reservations; not provider invoices. Shared-account deltas excluded.'
    dest = BASE / 'publication/comparison'
    dest.mkdir(parents=True, exist_ok=True)
    secrets = secret_values()
    files = []

    def copy(path, relative):
        raw = path.read_bytes()
        if relative == 'results/comparison.md':
            raw = '\n'.join(line for line in raw.decode('utf-8').splitlines() if not line.startswith('![')).encode()
        scan(raw, str(path), secrets)
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        files.append(dict(path=target.relative_to(dest).as_posix(), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))

    for path in sorted((BASE / 'baseline_comparison').iterdir()):
        if path.is_file() and path.suffix in {'.json', '.md'}:
            copy(path, 'results/' + path.name)
    for path in (dest/'results').iterdir():
        if path.suffix in {'.png', '.svg', '.pdf'}:
            path.unlink()  # Remove stale staged figures; originals remain in baseline_comparison.
    for pattern in ('*_status.json', '*_cost_accounting.json', '*_noncompletion_audit.json', '*_noncompletion_audit.md',
                    'authorization.json', 'shell_line_ending_repair.json', 'timeout_runtime_provenance.json',
                    '*_api_timeout_provenance.json', '*_terminal*.json',
                    '*_lane_closed.json', 'comparison_public_verification.json', 'keep_awake.json',
                    'validation_publication.json', 'invalid_baseline_publication.json'):
        for path in sorted(BASE.glob(pattern)):
            copy(path, 'metadata/' + path.name)
    for folder in ('baseline_harness_audit_lf', 'math_harness_audit', 'table2_harness_audit'):
        assert (BASE / folder).is_dir(), f'Missing completed health audit: {folder}'
        for path in sorted((BASE / folder).rglob('*')):
            if path.is_file() and path.suffix in {'.json', '.md', '.txt', '.log'}:
                copy(path, 'metadata/' + path.relative_to(BASE).as_posix())
    copy(BASE / 'data/terminal_decision.json', 'metadata/data_terminal_decision.json')
    copy(BASE / 'loaded_timeout_guard/status.json', 'metadata/loaded_timeout_guard_final.json')
    for name, value in [('public_sources.json', public), ('budget.json', budget), ('files_manifest.json', files)]:
        (dest / 'metadata' / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    fields = dict(experiment='Nonmoral versus math and Table-2-only: three fixed checkpoints under one ODCV protocol',
        date_generated='2026-09-09', constitution='none supplied at evaluation; existing checkpoint recipes differ',
        source_repo='https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ ' + revision,
        models={k: dict(target=v['target'], revision=v['target_revision'], base_revision=v['base_revision'])
                for k,v in report['arms'].items()}, generation_config=report['protocol'],
        schema='results/: exact counts, scenario intervals and paired contrasts; metadata/: pinned public sources, costs, shell hashes and qualitative audits; figures remain local',
        provenance='uv run python scratch/nonmoral/baseline_report.py --nonmoral <completed> --math <completed> --table2 <completed> --out output/nonmoral_overnight/20260909/baseline_comparison; uv run python scratch/nonmoral/publish_comparison.py',
        limitations=report['interpretation'] + ' Formal capability tests remain deferred. Fresh paired development accepted 13/32 and failed the scaling gate; no new SFT ran. The invalid CRLF attempt is excluded.')
    name = artifact_name('nonmoral-baseline-comparison', date='2026-09-09')
    front = {'tags': ['nonmoral-deliberation', 'research-comparison']}
    url = push_run_dir(dest, name, fields, private=False, front_matter=front)
    body = (dest / 'results/comparison.md').read_text(encoding='utf-8')
    body = '\n'.join(line for line in body.splitlines() if not line.startswith('!['))
    (dest/'results/comparison.md').write_text(body, encoding='utf-8')
    heading, rest = body.split('\n\n', 1)
    summary = (f"**720 scored rollouts; ${budget['total_exposure_estimate_usd']:.2f} estimated total exposure / $300. "
               "All owned pods terminated.** No new SFT: fresh paired development accepted 13/32 and failed its frozen gate.")
    metadata = card_markdown(fields, front)
    assert metadata.startswith('---\n')
    yaml_end = metadata.index('---\n', 4) + 4
    card = (metadata[:yaml_end] + heading + '\n\n' + summary + '\n\n' + rest
            + '\n<details>\n<summary>Reproduction metadata and limitations</summary>\n\n'
            + metadata[yaml_end:] + '\n</details>\n')
    (dest / 'README.md').write_text(card, encoding='utf-8')
    api.upload_file(path_or_fileobj=str(dest / 'README.md'), path_in_repo='README.md',
                    repo_id=hf_org() + '/' + name, repo_type='dataset',
                    commit_message='Display exact completed comparison on dataset card')
    info = api.dataset_info(hf_org() + '/' + name)
    assert not info.private
    receipt = dict(url=url, revision=info.sha, private=False, budget=budget, sources=public)
    (BASE / 'comparison_publication.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--correct-audit', type=Path)
    args = parser.parse_args()
    if args.correct_audit:
        correct_publication_metadata(args.correct_audit)
    else:
        main()
