# ABOUTME: Verify published training data and inspect all tool arguments, schemas, and supervised targets.
# ABOUTME: Read-only audit with historical renderer; no data repair, sampling, training, or tool execution.
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import types

from dotenv import load_dotenv
import jsonschema
from omegaconf import OmegaConf

from src.infra.huggingface import hf_download

ROOT = Path(__file__).resolve().parents[2]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in strings(v)]
    return []


def contains_array(value):
    return isinstance(value, list) or isinstance(value, dict) and any(contains_array(v) for v in value.values())


def main():
    cfg = OmegaConf.load(Path(__file__).with_suffix('.yaml'))
    common = Path(subprocess.check_output(['git', 'rev-parse', '--git-common-dir'], cwd=ROOT, text=True).strip())
    if not common.is_absolute():
        common = ROOT / common
    load_dotenv(common.resolve().parent / '.env')
    out = ROOT / cfg.output
    out.mkdir(parents=True, exist_ok=True)
    downloads = {}

    def fetch(label, repo, revision, filename, kind='dataset'):
        path = Path(hf_download(repo, filename, repo_type=kind, revision=revision))
        downloads[label] = {'repo': repo, 'revision': revision, 'filename': filename,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        return path.read_bytes().decode('utf-8')

    meta = json.loads(fetch('model_metadata', cfg.model_repo, cfg.model_revision, 'training_meta.json', 'model'))
    assert meta['data_repo'] == cfg.data_repo and meta['data_revision'] == cfg.data_revision
    assert meta['training_code_revision'] == cfg.training_revision
    data = fetch('dataset', cfg.data_repo, cfg.data_revision, 'mixture.jsonl')
    parent_data = fetch('parent_dataset', cfg.parent_repo, cfg.parent_revision, 'mixture.jsonl')
    old_audit = json.loads(fetch('training_audit', cfg.data_repo, cfg.data_revision, 'final_audit.json'))
    rows, parents = [[json.loads(line) for line in text.splitlines()] for text in [data, parent_data]]
    assert len(rows) == len(parents) == 10000
    assert data.encode() == (ROOT / 'output/gptoss_control/dataset/mixture.jsonl').read_bytes()
    historical = types.ModuleType('historical_harmony')
    source = subprocess.check_output(['git', 'show', f'{cfg.training_revision}:src/infra/endpoints/harmony.py'], cwd=ROOT, text=True)
    exec(compile(source, 'historical_harmony.py', 'exec'), historical.__dict__)
    renderer = historical.make_renderer(local_files_only=True)
    calls, errors, examples, schema_errors = [], [], [], []
    counts = Counter()
    violation_types = Counter()
    schema_error_shapes = Counter()
    lengths, string_lengths, arg_tokens, supervised_lengths = [], [], [], []
    all_strings = []
    required_error_rows = []
    for i, (row, parent) in enumerate(zip(rows, parents)):
        if row['source'] != 'apigen_function_calling':
            continue
        counts['tool_rows'] += 1
        assert len(row['messages']) == 3
        counts['rows_with_reasoning'] += any(m.get('reasoning_content') for m in row['messages'])
        tools = {t['function']['name']: t['function'] for t in json.loads(row['tools'])}
        parent_specs = json.loads(re.search(r'<tools>(.*?)</tools>', parent['messages'][0]['content'], re.S)[1])
        parent_tools = {t.get('function', t)['name']: t for t in parent_specs}
        counts['defined_bash_tools'] += 'bash' in tools
        for tool in tools.values():
            try:
                jsonschema.Draft202012Validator.check_schema(tool['parameters'])
            except jsonschema.SchemaError as error:
                schema_errors.append({'row': i, 'function': tool['name'],
                    'schema': tool['parameters'], 'error': error.message,
                    'path': list(error.path), 'parent_definition': parent_tools[tool['name']]})
        generated = [c for m in row['messages'] for c in (m.get('tool_calls') or [])]
        counts['rows_with_calls'] += bool(generated)
        counts['rows_with_multiple_calls'] += len(generated) > 1
        if generated:
            match = re.fullmatch(r'\s*<tool_call>(.*?)</tool_call>\s*', parent['messages'][-1]['content'], re.S)
            assert match
            parent_calls = json.loads(match.group(1))
            assert len(parent_calls) == len(generated)
        else:
            parent_calls = []
        rendered = historical.supervised_examples(renderer, row)
        assert len(rendered) == 1
        e = rendered[0]
        supervised_ids = [t for t, w in zip(e['target_tokens'], e['weights']) if w]
        counts['supervised_tokens'] += len(supervised_ids)
        supervised_lengths.append(len(supervised_ids))
        supervised = renderer.tokenizer.decode(supervised_ids)
        assert '<tool_call' not in supervised and '</tool_call>' not in supervised
        if generated:
            assert supervised_ids[-1] == 200012
            assert supervised_ids.count(200012) == 1
            parsed = renderer._parse_harmony_messages(supervised)
            projected = [m for m in parsed if (m.get('recipient') or '').startswith('functions.')]
            assert len(projected) == len(generated)
        bad_row = False
        for j, call in enumerate(generated):
            fn = call['function']
            value = json.loads(fn['arguments'])
            assert isinstance(value, dict)
            assert parent_calls[j] == {'name': fn['name'], 'arguments': value}
            assert projected[j]['recipient'] == 'functions.' + fn['name']
            assert json.loads(projected[j]['content']) == value
            counts['calls'] += 1
            counts['bash_calls'] += fn['name'] == 'bash'
            counts['calls_containing_arrays'] += contains_array(value)
            counts['calls_ending_array_close'] += fn['arguments'].endswith(']}')
            leaves = strings(value)
            counts['calls_with_newline_strings'] += any('\n' in s for s in leaves)
            counts['calls_with_quoted_strings'] += any('"' in s for s in leaves)
            counts['calls_with_backslash_strings'] += any('\\' in s for s in leaves)
            counts['strings_containing_newline'] += sum('\n' in s for s in leaves)
            counts['strings_containing_double_quote'] += sum('"' in s for s in leaves)
            all_strings.extend(leaves)
            string_lengths.extend(map(len, leaves))
            lengths.append(len(fn['arguments']))
            arg_tokens.append(len(renderer.tokenizer.encode(fn['arguments'], add_special_tokens=False)))
            tool = tools[fn['name']]
            validation = list(jsonschema.Draft202012Validator(tool['parameters']).iter_errors(value))
            entry = {'row': i, 'call_index': j, 'function': fn['name'], 'arguments': fn['arguments'],
                     'schema': tool['parameters'], 'same_arguments_as_parent': True,
                     'parent_definition': parent_tools[fn['name']],
                     'supervised_target': supervised,
                     'errors': [{'validator': error.validator, 'path': list(error.path),
                                 'schema_path': list(error.schema_path), 'message': error.message}
                                for error in validation]}
            calls.append({k: v for k, v in entry.items() if k not in ['supervised_target', 'schema']})
            if validation:
                if parent_tools[fn['name']].get('type') == 'function':
                    assert parent_tools[fn['name']]['function'] == tool
                    counts['bad_calls_with_unchanged_parent_jsonschema'] += 1
                else:
                    counts['bad_calls_with_python_type_parent_schema'] += 1
                bad_row = True
                errors.append(entry)
                violation_types.update(e.validator for e in validation)
                for error in validation:
                    schema_error_shapes[f'{error.validator}:{error.validator_value}:actual_{type(error.instance).__name__}'] += 1
                    if error.validator == 'required':
                        required_error_rows.append(i)
            if (validation and len(examples) < 5) or any('\n' in s for s in leaves):
                examples.append(entry)
        if bad_row:
            counts['rows_with_schema_violations'] += 1
            counts['supervised_tokens_in_schema_violation_rows'] += len(supervised_ids)
        if not generated:
            counts['no_call_rows'] += 1
    assert counts['supervised_tokens'] == old_audit['by_source']['apigen_function_calling']['supervised_tokens']
    counts['tool_result_turns_entire_dataset'] = sum(m['role'] == 'tool' for r in rows for m in r['messages'])
    counts['assistant_text_with_xml_tool_tags_entire_dataset'] = sum(
        bool(re.search(r'</?tool_call', (m.get('content') or '') + (m.get('reasoning_content') or '')))
        for r in rows for m in r['messages'] if m['role'] == 'assistant')
    counts['schema_invalid_calls'] = len(errors)
    counts['invalid_json_calls'] = 0  # Every call parsed and round-tripped above.

    def percentiles(values):
        v = sorted(values)
        return {str(q): v[int((len(v)-1)*q)] for q in [.5, .9, .95, .99, 1]}

    results = {'config': OmegaConf.to_container(cfg), 'downloads': downloads, 'counts': dict(counts),
        'total_supervised_tokens': old_audit['total']['supervised_tokens'],
        'tool_supervised_token_fraction': counts['supervised_tokens'] / old_audit['total']['supervised_tokens'],
        'call_argument_char_percentiles': percentiles(lengths),
        'call_argument_token_percentiles': percentiles(arg_tokens),
        'string_argument_char_percentiles': percentiles(string_lengths),
        'supervised_tool_row_token_percentiles': percentiles(supervised_lengths),
        'schema_violation_counts': dict(violation_types), 'schema_violation_shapes': dict(schema_error_shapes),
        'schema_required_error_rows': sorted(set(required_error_rows)),
        'all_original_call_values_preserved': True, 'all_supervised_calls_roundtrip': True,
        'schema_definitions_failing_metaschema': len(schema_errors),
        'single_handoff_per_tool_target': True, 'tools_executed': False,
        'generated_or_modified_training_data': False,
        'git_sha': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()}
    save(out / 'results.json', results)
    save(out / 'schema_violations.json', errors)
    save(out / 'invalid_schema_definitions.json', schema_errors)
    save(out / 'calls.json', calls)
    save(out / 'examples.json', examples)
    print(json.dumps({k: v for k, v in results.items() if k not in ['downloads', 'config']}, indent=2))


if __name__ == '__main__':
    main()
