# ABOUTME: Compare old/new tool instructions for base and nosynth GPT-OSS using frozen prefixes.
# ABOUTME: Archives exact input/output tokens; validates arguments but never executes commands.
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import types

from dotenv import load_dotenv
from omegaconf import OmegaConf
import tinker

from src.infra.endpoints import harmony

ROOT = Path(__file__).resolve().parents[2]


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def inspect_response(renderer, ids, tools):
    text = renderer.tokenizer.decode(ids)
    declared = {t['function']['name']: t['function']['parameters'] for t in tools}
    calls, errors = [], []
    for message in renderer._parse_harmony_messages(text):
        recipient = message.get('recipient') or ''
        if not recipient:
            continue
        name = recipient.removeprefix('functions.')
        call = {'name': name, 'arguments': message.get('content') or '', 'errors': []}
        if name not in declared:
            call['errors'].append('undeclared_tool')
        try:
            arguments = json.loads(call['arguments'])
        except json.JSONDecodeError as e:
            call['errors'].append('invalid_json')
            call['json_error'] = str(e)
        else:
            if not isinstance(arguments, dict):
                call['errors'].append('non_object')
            elif name in declared:
                schema = declared[name]
                missing = set(schema.get('required', [])) - arguments.keys()
                extras = arguments.keys() - schema.get('properties', {}).keys()
                if missing:
                    call['errors'].append('missing_parameter')
                if extras:
                    call['errors'].append('undeclared_parameter')
                for key, value in arguments.items():
                    if schema.get('properties', {}).get(key, {}).get('type') == 'string' and not isinstance(value, str):
                        call['errors'].append('wrong_parameter_type')
        calls.append(call)
        errors.extend(call['errors'])
    if not calls:
        errors.append('no_tool_call')
    if ids[-1:] != [200012]:
        errors.append('no_handoff')
    return {'text': text, 'calls': calls, 'errors': sorted(set(errors))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    cfg = OmegaConf.load(Path(__file__).with_suffix('.yaml'))
    common = Path(subprocess.check_output(['git', 'rev-parse', '--git-common-dir'], text=True).strip()).resolve()
    load_dotenv(common.parent / '.env')
    original_file = Path(cfg.source_prompts)
    frozen = json.loads(original_file.read_text(encoding='utf-8'))
    previous = types.ModuleType('historical_harmony')
    source = subprocess.check_output(['git', 'show', f'{cfg.historical_renderer_revision}:src/infra/endpoints/harmony.py'], text=True)
    exec(compile(source, 'historical_harmony.py', 'exec'), previous.__dict__)
    renderer = harmony.make_renderer(cfg.reasoning, local_files_only=True)
    prepared = []
    for i, entry in enumerate(frozen):
        body = entry['body']
        for version, module in [('old', previous), ('new', harmony)]:
            ids = module.render_prompt(renderer, body['messages'], body['tools']).to_ints()
            if version == 'old':
                assert ids == entry['prompt_token_ids'], 'Historical frozen tokens must match exactly'
            assert len(ids) + cfg.max_tokens <= cfg.context_window
            prepared.append({'prompt': i, 'version': version, 'source': entry['source'],
                             'body': body, 'tokens': ids, 'text': renderer.tokenizer.decode(ids)})
    ceiling = sum((len(p['tokens']) * cfg.input_usd_per_million + cfg.max_tokens * cfg.output_usd_per_million)
                  / 1e6 for p in prepared) * len(cfg.seeds) * len(cfg.checkpoints)
    assert ceiling <= cfg.max_total_usd
    print('Prepared', len(prepared), 'prefix/version pairs; 120 responses; maximum rate-card cost', ceiling, flush=True)
    if not args.execute:
        return
    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'prompts.json', prepared)
    write(out / 'run_meta.json', {'config': OmegaConf.to_container(cfg),
        'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'renderer_sha256': hashlib.sha256(Path(harmony.__file__).read_bytes()).hexdigest(),
        'source_prompts_sha256': hashlib.sha256(original_file.read_bytes()).hexdigest(),
        'selection': 'Three historical LoRA failures, frozen before their first malformed call',
        'old_tokens_match_previous_probe': True, 'tools_executed': False, 'judges_called': False,
        'maximum_rate_card_usd': ceiling, 'sdk_transport_retries': 'SDK defaults; no application resampling',
        'interpretation': 'Conditional syntax probe; no end-to-end task success or population rate estimate'} )
    service = tinker.ServiceClient()
    rows = []
    for arm, checkpoint in cfg.checkpoints.items():
        sampler = service.create_sampling_client(base_model=harmony.MODEL,
            **({} if checkpoint == 'base' else {'model_path': checkpoint}))
        # Alternate version order across seeds. Same seeds pair prompts, not identical completions.
        jobs = [(p, seed) for seed in cfg.seeds
                for p in (prepared if seed % 2 == 0 else list(reversed(prepared)))]

        def sample(pair):
            prompt, seed = pair
            started = time.monotonic()
            result = sampler.sample(prompt=tinker.ModelInput.from_ints(prompt['tokens']), num_samples=1,
                sampling_params=tinker.SamplingParams(max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature, top_p=cfg.top_p, seed=seed,
                    stop=renderer.get_stop_sequences())).result()
            ids = result.sequences[0].tokens
            return {'arm': arm, 'checkpoint': checkpoint, 'version': prompt['version'],
                    'prompt': prompt['prompt'], 'seed': seed, 'seconds': time.monotonic() - started,
                    'input_tokens': len(prompt['tokens']), 'output_tokens': len(ids), 'raw_tokens': ids,
                    'rate_card_upper_usd': (len(prompt['tokens']) * cfg.input_usd_per_million
                                           + len(ids) * cfg.output_usd_per_million) / 1e6,
                    **inspect_response(renderer, ids, prompt['body']['tools'])}

        with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
            futures = {pool.submit(sample, job): job for job in jobs}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                with (out / 'responses.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(len(rows), arm, row['version'], row['prompt'], row['seed'],
                      row['errors'] or 'valid', flush=True)
    results = {'responses': len(rows), 'rate_card_upper_usd': sum(r['rate_card_upper_usd'] for r in rows), 'arms': {}}
    for arm in cfg.checkpoints:
        for version in ['old', 'new']:
            group = [r for r in rows if r['arm'] == arm and r['version'] == version]
            results['arms'][f'{arm}_{version}'] = {'n': len(group),
                'invalid_json': sum('invalid_json' in r['errors'] for r in group),
                'any_error': sum(bool(r['errors']) for r in group),
                'per_prompt': [{ 'prompt': i, 'n': sum(r['prompt'] == i for r in group),
                    'invalid_json': sum(r['prompt'] == i and 'invalid_json' in r['errors'] for r in group),
                    'any_error': sum(r['prompt'] == i and bool(r['errors']) for r in group)} for i in range(len(frozen))]}
    write(out / 'results.json', results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
