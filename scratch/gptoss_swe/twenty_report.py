# ABOUTME: Add canonical combined scores, Markdown and closeout metadata after the smoke owner exits.
# ABOUTME: Publication-only operation; preserves every rollout, grade and original publication revision.
import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
from scratch.gptoss_swe.twenty import ROOT, C, read, save, sha, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    load_dotenv('/srv/lasr/credentials.env')
    assert read(ROOT/'status.json')['phase'] == 'complete'
    owner = read(ROOT/'owner-started.json')
    process = Path(f"/proc/{owner['pid']}/cmdline")
    assert not process.exists() or b'scratch.gptoss_swe.twenty run' not in process.read_bytes()
    prior = read(ROOT/'publication.json')
    assert prior['complete'] and read(ROOT/'cleanup.json')['owned_remaining'] == []
    with (ROOT/'reporting-claim.json').open('x') as f:
        json.dump({'source': args.source, 'prior_revision': prior['revision']}, f)
    summary = status()
    selection = read(ROOT/'selection.json')
    for arm, row in summary.items():
        assert row['graded'] == 20 and row['running'] == row['waiting'] == row['reservations'] == 0
        state = read(ROOT/arm/'swe/metadata/state.json')
        grade = read(ROOT/arm/'summary.json')
        assert all(t['status'] == 'valid' for t in state['tasks'].values())
        row['resolved_ids'] = grade['resolved_ids']
        row['resolved_limit_endings'] = [i for i in grade['resolved_ids'] if state['tasks'][i]['attempts'][-1]['exit_status'] == 'LimitsExceeded']
        row['old_ten_resolved'] = len(set(grade['resolved_ids']) & set(selection['old_ids']))
        row['new_ten_resolved'] = len(set(grade['resolved_ids']) & set(selection['new_ids']))
    out = ROOT/'reporting-package'
    save(out/'results/results.json', dict(arms=summary, selection=selection, scope='Matched twenty-task diagnostic subset; not full SWE-bench Lite'))
    lines = ['# Twenty-task SWE smoke', '', '| Metric | Base | Control | DA15 |', '|---|---:|---:|---:|']
    for field in ('success','graded','failed','submitted','awaiting_grading','running','waiting','infrastructure_excluded','cost_usd'):
        lines.append('| '+field+' | '+' | '.join(str(summary[a][field]) for a in C.targets)+' |')
    for reason in ('context_limit','response_token_limit','step_limit','task_token_limit','other_limit'):
        lines.append('| '+reason+' | '+' | '.join(str(summary[a]['limits'].get(reason,0)) for a in C.targets)+' |')
    lines += ['', 'Limit endings can still resolve: base1, control2, DA15 1. Ending categories are separate from official grades.',
              'Base includes ten retained outcomes and its historical cumulative ledger. Control and DA15 are fresh replicates. Costs are estimates, not invoices.']
    (out/'results/results.md').write_text('\n'.join(lines)+'\n')
    save(out/'metadata/run_meta.json', dict(source=read('/srv/lasr/deployment.json'), reporting_source=args.source,
         targets=dict(C.targets), selection=selection, configs={a:read(ROOT/a/'swe/metadata/manifest.json')['config'] for a in C.targets},
         status='complete', original_publication=prior['revision'], inherited_base_publication=read(ROOT/'prior-publication.json')))
    save(out/'metadata/cleanup.json', read(ROOT/'cleanup.json'))
    save(out/'metadata/status.json', read(ROOT/'status.json'))
    save(out/'metadata/initial-smoke20-publication.json', prior)
    hashes = dict(prior['sha256']); hashes.pop('metadata/file-hashes.json')
    paths = sorted(p for p in out.rglob('*') if p.is_file())
    hashes.update({p.relative_to(out).as_posix():sha(p) for p in paths})
    save(out/'metadata/file-hashes.json', hashes)
    paths.append(out/'metadata/file-hashes.json')
    additions = {p.relative_to(out).as_posix():sha(p) for p in paths}
    api = HfApi(token=os.environ['HF_TOKEN'])
    revision = api.create_commit(repo_id=prior['repo'], repo_type='dataset', parent_commit=prior['revision'],
        operations=[CommitOperationAdd(path_in_repo=p.relative_to(out).as_posix(), path_or_fileobj=p) for p in paths],
        commit_message='Add combined score table and verified closeout metadata; preserve all raw outcomes').oid
    receipt = dict(repo=prior['repo'], revision=revision, complete=False, parent_revision=prior['revision'],
                   changed_file_hashes=additions, inherited_verified_hashes={k:v for k,v in prior['sha256'].items() if k not in additions})
    save(ROOT/'reporting-publication.json', receipt)
    for name, digest in additions.items():
        assert sha(hf_hub_download(prior['repo'], name, repo_type='dataset', revision=revision, token=os.environ['HF_TOKEN'])) == digest
    receipt['complete'] = True
    save(ROOT/'reporting-publication.json', receipt)
    print(json.dumps({'revision':revision, 'changed_files_verified':len(additions)}))


if __name__ == '__main__':
    main()
