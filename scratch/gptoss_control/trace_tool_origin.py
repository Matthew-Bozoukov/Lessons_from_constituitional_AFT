# ABOUTME: Match every nosynth APIgen conversation verbatim to pinned upstream SmolTalk rows.
# ABOUTME: Archive row offsets and hashes, with upstream examples for every schema-defective row.
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

from dotenv import load_dotenv
from omegaconf import OmegaConf
import pyarrow.parquet as pq

from src.infra.huggingface import hf_download


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    cfg = OmegaConf.load(Path(__file__).with_suffix('.yaml'))
    common = Path(subprocess.check_output(['git', 'rev-parse', '--git-common-dir'], text=True).strip()).resolve()
    load_dotenv(common.parent / '.env')
    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    upstream = Path(hf_download(cfg.upstream_repo, cfg.upstream_file, repo_type='dataset', revision=cfg.upstream_revision))
    mixture = Path(hf_download(cfg.mixture_repo, 'mixture.jsonl', repo_type='dataset', revision=cfg.mixture_revision))
    violations = Path(hf_download(cfg.audit_repo, cfg.audit_prefix + '/schema_violations.json',
                                  repo_type='dataset', revision=cfg.audit_revision))
    rows = [json.loads(line) for line in mixture.read_text(encoding='utf-8').splitlines()]
    defects = json.loads(violations.read_text(encoding='utf-8'))
    bad_rows = {e['row'] for e in defects}
    lookup = defaultdict(list)
    for i, row in enumerate(rows):
        if row['source'] == 'apigen_function_calling':
            lookup[digest(row['messages'])].append(i)
    matched, examples, offset = {}, {}, 0
    for batch in pq.ParquetFile(upstream).iter_batches(batch_size=4096):
        for j, row in enumerate(batch.to_pylist()):
            key = digest(row['messages'])
            for i in lookup.get(key, []):
                assert rows[i]['messages'] == row['messages']
                matched.setdefault(i, {'mixture_row': i, 'messages_sha256': key,
                                      'upstream_rows': [], 'schema_defective': i in bad_rows})['upstream_rows'].append(offset+j)
                if i in bad_rows and i not in examples:
                    examples[i] = {'mixture_row': i, 'upstream_row': offset+j, 'messages': row['messages']}
        offset += batch.num_rows
    assert len(matched) == 1054 and len(examples) == len(bad_rows) == 95
    results = {'config': OmegaConf.to_container(cfg), 'upstream_rows_scanned': offset,
        'mixture_tool_rows': 1054, 'exact_full_conversation_matches': len(matched),
        'schema_defective_rows_matched': len(examples), 'schema_defective_calls_in_those_rows': len(defects),
        'all_roles_and_content_identical': True, 'offset_convention': 'zero-based parquet train row and mixture row',
        'upstream_file_sha256': hashlib.sha256(upstream.read_bytes()).hexdigest(),
        'mixture_file_sha256': hashlib.sha256(mixture.read_bytes()).hexdigest(),
        'comparison': 'Exact parsed message-list equality, including all whitespace inside content strings; no normalization',
        'inference': 'Defects predate our ingestion; our import/selection and later Harmony conversion did not generate them'}
    (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    (out / 'matches.json').write_text(json.dumps(list(matched.values()), indent=2) + '\n', encoding='utf-8')
    (out / 'upstream_bad_rows.jsonl').write_text(''.join(json.dumps(examples[i], ensure_ascii=False) + '\n'
                                                     for i in sorted(examples)), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
