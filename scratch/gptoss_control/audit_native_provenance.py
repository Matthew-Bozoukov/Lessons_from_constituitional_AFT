# ABOUTME: Verify every existing CoT against pinned GPT-OSS generation receipts without inference.
# ABOUTME: Publish per-turn provenance and resolve misleading failure labels while preserving historical judgments.
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.infra.huggingface import hf_api, hf_download, gate_push
from src.infra.endpoints.harmony import make_renderer, render_prompt


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_rows(path, data):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in data), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='scratch/gptoss_control/audit_native_provenance.yaml')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    common = Path(subprocess.check_output(['git', 'rev-parse', '--git-common-dir'], text=True).strip()).resolve()
    load_dotenv(common.parent/'.env')
    api = hf_api()
    out = ROOT/cfg.output
    final = out/'dataset'
    final.mkdir(parents=True, exist_ok=True)
    current_dir, original_dir, strict_dir = [ROOT/cfg[k] for k in ('current_output', 'original_output', 'strict_output')]
    current_path = current_dir/'mixture.jsonl'
    assert sha(current_path.read_bytes()) == cfg.source_file_sha256
    published = Path(hf_download(cfg.source_repo, 'mixture.jsonl', repo_type='dataset', revision=cfg.source_revision))
    assert current_path.read_bytes() == published.read_bytes()
    original_published = Path(hf_download(cfg.original_repo, 'mixture.jsonl', repo_type='dataset', revision=cfg.original_revision))
    original_rows, current = rows(original_published), rows(current_path)
    assert rows(strict_dir/'converted_unenriched.jsonl') == original_rows
    assert len(current) == len(original_rows) == 10000

    trees = {}
    for repo, revision in [(cfg.source_repo, cfg.source_revision), (cfg.original_repo, cfg.original_revision)]:
        trees[repo] = {f.path: f for f in api.list_repo_tree(repo, repo_type='dataset', revision=revision, recursive=True)
                       if hasattr(f, 'blob_id')}

    def verify_file(local, repo, remote_name):
        data = local.read_bytes()
        entry = trees[repo][remote_name]
        if entry.lfs:
            assert sha(data) == entry.lfs.sha256, remote_name
        else:
            assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == entry.blob_id, remote_name
        return sha(data)

    verify_file(original_dir/'dataset/run_meta.json', cfg.original_repo, 'run_meta.json')
    original_meta = read(original_dir/'dataset/run_meta.json')
    assert original_meta['config']['base_model'] == cfg.base_model
    original_code = subprocess.check_output(['git', 'show', original_meta['git_sha']+':scratch/gptoss_control/run.py'])
    # The archived code selects the base sampler and excludes the target answer from its prompt.
    assert b'create_sampling_client(base_model=cfg.base_model)' in original_code
    assert b'prompt_messages = row["messages"][:turn]' in original_code
    verify_file(strict_dir/'backfill_identity.json', cfg.source_repo, 'prior_strict_replacement/backfill_identity.json')
    assert read(strict_dir/'backfill_identity.json')['base_model'] == cfg.base_model

    failed = rows(current_dir/'failed_replacements.jsonl')
    failed_keys = {(r['row'], r['turn']) for r in failed}
    adopted = {(r['row'], r['turn']): r for r in rows(current_dir/'cached_answer_replacements.jsonl')}
    originals = {(r['row'], r['turn']): (p, r) for p in (original_dir/'backfill_receipts').glob('*.json')
                 if (r := read(p)).get('accepted')}
    strict = {(r['row'], r['turn']): (p, r) for p in (strict_dir/'backfill_receipts').glob('*.json')
              if (r := read(p)).get('accepted')}
    assert len(failed_keys) == 508 and len(adopted) == 119 and len(strict) == 446
    assert not (failed_keys & adopted.keys() or failed_keys & strict.keys() or adopted.keys() & strict.keys())
    renderer = make_renderer(local_files_only=True)
    manifest, untraced = [], 0
    for i, (old_row, row) in enumerate(zip(original_rows, current)):
        assert {k:v for k,v in row.items() if k != 'messages'} == {k:v for k,v in old_row.items() if k != 'messages'}
        assert len(old_row['messages']) == len(row['messages'])
        for j, (old, msg) in enumerate(zip(old_row['messages'], row['messages'])):
            if not msg.get('reasoning_content'):
                assert msg == old, (i,j)
                untraced += msg['role'] == 'assistant'
                continue
            assert old.get('reasoning_content') and msg['role'] == 'assistant'
            key = (i,j)
            if key in failed_keys:
                stage = 'original_native_gptoss_retained'
                path, rec = originals[key]
                repo, revision = cfg.original_repo, cfg.original_revision
                assert msg == old and rec['accepted'] and rec['judge']['verdict'] == 'yes'
            elif key in adopted:
                stage = 'cached_answer_and_cot_pair'
                path = strict_dir/adopted[key]['receipt']
                rec = read(path)
                repo, revision = cfg.source_repo, cfg.source_revision
                assert msg['content'] == rec['response']['content']
                assert {k:v for k,v in msg.items() if k not in ('reasoning_content','content')} == {k:v for k,v in old.items() if k not in ('reasoning_content','content')}
            else:
                stage = 'strict_cot_replacement'
                path, rec = strict[key]
                repo, revision = cfg.source_repo, cfg.source_revision
                assert rec['accepted'] and rec['judge']['verdict'] == 'yes'
                assert {k:v for k,v in msg.items() if k != 'reasoning_content'} == {k:v for k,v in old.items() if k != 'reasoning_content'}
            assert (rec['row'], rec['turn']) == key
            receipt_path = 'backfill_receipts/'+path.name
            receipt_sha = verify_file(path, repo, receipt_path)
            assert rec['status'] == 'completed' and rec['termination'] == 'stop_sequence'
            response = rec['response']
            assert not response.get('tool_calls')
            trace = response['reasoning_content']
            assert rec['trace'] == trace.strip()
            assert msg['reasoning_content'] == (trace if key in adopted else rec['trace'])
            raw_verified = bool(rec.get('raw_tokens'))
            if raw_verified:
                parsed, stop = renderer.parse_response(rec['raw_tokens'])
                decoded = renderer.to_openai_message(parsed)
                assert stop.is_stop_sequence and rec['raw_tokens'][-1] == 200002
                assert decoded['reasoning_content'] == trace and decoded['content'] == response['content']
                assert rec['prompt_tokens'] == render_prompt(renderer, row['messages'][:j], row.get('tools')).to_ints()
            manifest.append({'row':i, 'turn':j, 'source':row['source'], 'model':cfg.base_model,
                'stage':stage, 'attempt':rec['attempt'], 'receipt_repo':repo, 'receipt_revision':revision,
                'receipt_path':receipt_path, 'receipt_sha256':receipt_sha,
                'trace_sha256':sha(msg['reasoning_content'].encode()), 'raw_tokens_and_prompt_verified':raw_verified,
                'evidence': 'raw_tokens_plus_parsed_response' if raw_verified else 'parsed_response_only_original_receipt',
                'historical_judge':rec.get('judge'), 'action_this_audit':'retain_unchanged'})
            if len(manifest)%200 == 0:
                print('Verified', len(manifest), 'traces', flush=True)
    assert len(manifest) == 1073 and untraced == 9307
    assert {(r['row'],r['turn']) for r in manifest} == failed_keys | adopted.keys() | strict.keys()
    counts = dict(collections.Counter(r['stage'] for r in manifest))
    report = {'passed':True, 'dataset_repo':cfg.source_repo, 'audited_revision':cfg.source_revision,
        'dataset_file_sha256':cfg.source_file_sha256, 'rows':10000, 'assistant_turns':10380,
        'gptoss_provenance_verified_cot_turns':1073, 'unresolved_model_provenance':0,
        'stages':counts, 'untraced_assistant_turns_unchanged':untraced,
        'raw_tokens_and_prompt_verified':sum(r['raw_tokens_and_prompt_verified'] for r in manifest),
        'parsed_response_only_entries':[{k:r[k] for k in ('row','turn','receipt_path','receipt_repo','receipt_revision')}
            for r in manifest if not r['raw_tokens_and_prompt_verified']],
        'dataset_content_changed':False, 'new_generator_calls':0, 'new_judge_calls':0,
        'new_provider_inference_cost_usd':0,
        'limitations':'Generation provenance is not independent answer-correctness certification. Original acceptance judged trace compatibility, not the later stricter answer equivalence. Some early receipts lack raw token arrays; these are verified by exact parsed response and pinned published receipt hash.'}
    write_rows(final/'native_cot_provenance.jsonl', manifest)
    write(final/'native_cot_provenance_summary.json', report)
    dispositions = []
    index = {(r['row'],r['turn']):r for r in manifest}
    for failure in failed:
        row = {**failure, 'current_status':'resolved_native_gptoss_provenance',
            'current_reason':'Retained trace exactly matches an accepted original base GPT-OSS-120B generation; later replacement failures do not invalidate model provenance.',
            'native_provenance':index[(failure['row'],failure['turn'])]}
        dispositions.append(row)
    write_rows(final/'failed_replacements.jsonl', dispositions)
    write(final/'failed_replacement_indices.json', [{k:r[k] for k in ('row','turn','source','current_status')} for r in dispositions])
    md = '# Historical replacement failures: all 508 have verified GPT-OSS provenance\n\nThese are failures of the later replacement attempt, not missing native CoT. All retained traces match accepted original GPT-OSS generations. No answers or CoTs were changed by this audit. Historical verdicts remain intact.\n\n| Row | Turn | Original accepted receipt |\n|---|---|---|\n'
    for r in dispositions:
        proof = r['native_provenance']
        url = f"https://huggingface.co/datasets/{proof['receipt_repo']}/blob/{proof['receipt_revision']}/{proof['receipt_path']}"
        md += f"| {r['row']} | {r['turn']} | [{proof['attempt']}]({url}) |\n"
    (final/'failed_replacements.md').write_text(md, encoding='utf-8')
    stats = read(current_dir/'mixture_stats.json')
    stats['reasoning_traces']['provenance_audit'] = {'verified_native_turns':1073, 'unresolved_model_provenance':0,
        'manifest':'native_cot_provenance.jsonl', 'summary':'native_cot_provenance_summary.json'}
    write(final/'mixture_stats.json', stats)
    card = (current_dir/'README.md').read_text(encoding='utf-8').split('## Current revision')[0]
    card += '## Current provenance audit\n\nAll **1,073 nonempty CoTs** are verified against saved base `openai/gpt-oss-120b` generation receipts. The 508 previously labeled unresolved already came from the original native GPT-OSS conversion. They failed later replacement attempts, not native provenance.\n\nBreakdown: 446 strict CoT replacements, 119 reviewed cached CoT/answer pairs, and 508 retained original native CoTs. All 9,307 untraced assistant turns remain unchanged. This revision changes metadata only; `mixture.jsonl` is byte-identical to the previous revision. No new model/judge calls, training, or evaluation.\n\n[Per-turn provenance](native_cot_provenance.jsonl) | [Audit summary and evidence limits](native_cot_provenance_summary.json) | [All 508 dispositions](failed_replacements.md).\n\nThe old failure judgments remain historical evidence. Original trace compatibility is weaker than strict answer equivalence; this audit does not newly certify answer correctness. Earlier metadata is archived under `prior_cached_answer_review/`.\n'
    (final/'README.md').write_text(card, encoding='utf-8')
    write(final/'native_provenance_run_meta.json', {'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'working_tree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        'command':subprocess.list2cmdline(sys.argv), 'config':OmegaConf.to_container(cfg),
        'original_generator_code_sha256':sha(original_code)})
    shutil.copy2(__file__, final/'audit_native_provenance.py')
    shutil.copy2(args.config, final/'audit_native_provenance.yaml')
    print(json.dumps(report, indent=2), flush=True)
    if args.publish:
        assert api.dataset_info(cfg.source_repo).sha == cfg.source_revision, 'Remote changed; inspect before publishing'
        gate_push(cfg.source_repo, {'date_generated':'2026-09-30'}, what='native-CoT provenance metadata audit')
        operations = []
        for name in ('README.md','failed_replacements.jsonl','failed_replacement_indices.json','failed_replacements.md','mixture_stats.json'):
            p = Path(hf_download(cfg.source_repo, name, repo_type='dataset', revision=cfg.source_revision))
            operations.append(CommitOperationAdd(path_in_repo='prior_cached_answer_review/'+name,path_or_fileobj=p))
        for p in sorted(final.iterdir()):
            operations.append(CommitOperationAdd(path_in_repo=p.name,path_or_fileobj=p))
        commit = api.create_commit(repo_id=cfg.source_repo, repo_type='dataset', parent_commit=cfg.source_revision,
            operations=operations, commit_message='Verify all native GPT-OSS CoT provenance and resolve 508 historical failure labels')
        for p in final.iterdir():
            fetched = Path(hf_download(cfg.source_repo,p.name,repo_type='dataset',revision=commit.oid))
            assert fetched.read_bytes() == p.read_bytes(), p.name
        fetched = Path(hf_download(cfg.source_repo,'mixture.jsonl',repo_type='dataset',revision=commit.oid))
        assert sha(fetched.read_bytes()) == cfg.source_file_sha256
        write(out/'publication.json',{'repo':cfg.source_repo,'revision':commit.oid,'all_files_verified':True,
            'dataset_file_sha256':cfg.source_file_sha256})
        print('Published and byte-verified',commit.oid)


if __name__ == '__main__':
    main()
