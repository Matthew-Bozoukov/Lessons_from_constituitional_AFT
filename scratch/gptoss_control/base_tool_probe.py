# ABOUTME: Bounded Tinker sampling at recorded GPT-OSS tool-failure prefixes; executes no tools.
# ABOUTME: Run with the adjacent YAML config; --execute samples nine times with no HTTP retries.
import argparse
import ast
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from omegaconf import OmegaConf
from src.infra.endpoints.harmony import make_renderer, render_prompt
from src.infra.endpoints.tinker import tinker_shim


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def restore_prefix(text, before_step):
    messages, pending = [], deque()
    chunks = re.split(r'^== Step \d+ ==\n', text, flags=re.M)[1:]
    for chunk in chunks[:before_step - 1]:
        assert chunk.endswith('\n\n')
        chunk = chunk[:-2]
        role, body = chunk.split('\ncontent: ', 1)
        message = {'role': role.removeprefix('role: ')}
        if message['role'] == 'assistant':
            if '\ncall: ' in body:
                body, calls = body.rsplit('\ncall: ', 1)
                message['tool_calls'] = ast.literal_eval(calls)
                pending.extend(message['tool_calls'])
            if '\nreason: ' in body:
                body, reason = body.split('\nreason: ', 1)
                if reason != 'No explanation':
                    message['reasoning'] = reason
        elif message['role'] == 'tool':
            call = pending.popleft()
            message['tool_call_id'] = call['id']
        message['content'] = body
        messages.append(message)
    assert not pending, 'Prefix ends within a tool-result batch'
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--checkpoint', default='base')
    args = parser.parse_args()
    os.chdir(ROOT)
    cfg = OmegaConf.load(Path(__file__).with_suffix('.yaml'))
    out = ROOT / cfg.output
    source = ROOT / cfg.source_run
    common = Path(subprocess.check_output(['git', 'rev-parse', '--git-common-dir'], text=True).strip()).resolve()
    load_dotenv(common.parent / '.env')
    trajectory = json.loads((ROOT / 'output/gptoss_control/malformed_arguments_trajectories.json').read_text())
    decoded = json.loads((ROOT / 'output/gptoss_control/malformed_arguments_decoded.json').read_text())
    verified = json.loads((ROOT / 'output/gptoss_control/closeout_receipt.json').read_text())
    events = [json.loads(l) for l in (source / 'metadata/tinker_shim/sampling.jsonl').read_text().splitlines()]
    reserved = {e['id']: e for e in events if e['event'] == 'reserved'}
    module = ast.parse((ROOT / 'src/eval/misalignment/odcv/third_party/odcv-bench/mission_executor/agent_main.py').read_text(encoding='utf-8'))
    tools = next(ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'BASE_TOOLS' for t in n.targets))
    renderer = make_renderer(cfg.reasoning, local_files_only=True)
    prepared = []
    for path in cfg.prompts:
        file = source / path
        assert hashlib.sha256(file.read_bytes()).hexdigest() == verified['verified_files_sha256'][path]
        old = next(r for r in trajectory if r['path'] == path)
        first = old['errors'][0]
        messages = restore_prefix(file.read_text(encoding='utf-8'), first['step'])
        ids = render_prompt(renderer, messages, tools).to_ints()
        original = next(e for e in decoded if e['arguments'] == first['arguments'])
        original_count = reserved[original['id']]['prompt_tokens']
        assert len(ids) == original_count, (path, len(ids), original_count)
        prepared.append({'source': path, 'before_step': first['step'], 'source_transcript_sha256': verified['verified_files_sha256'][path],
                         'source_bad_request_id': original['id'], 'original_bad_arguments': first['arguments'],
                         'original_prompt_token_count': original_count, 'prompt_token_ids': ids,
                         'body': {'model': 'openai/gpt-oss-120b', 'messages': messages, 'tools': tools,
                                  'temperature': cfg.temperature, 'top_p': cfg.top_p, 'max_tokens': cfg.max_tokens}})
    write(out / 'prompts.json', prepared)
    write(out / 'run_meta.json', {'config': OmegaConf.to_container(cfg),
                                'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                                'selection': 'Three prefixes immediately before first malformed LoRA arguments; selected on historical failures',
                                'history': 'Recorded LoRA/tool histories replayed; base does not generate its own prior trajectory',
                                'tools_executed': False, 'judge_calls': False,
                                'raw_historical_input_tokens_available': False,
                                'reconstruction_check': 'Source transcript SHA256 and historical rendered input-token counts match',
                                'render_date': renderer.lasr_date})
    print('Prepared prefixes:', [(p['before_step'], p['original_prompt_token_count']) for p in prepared], flush=True)
    if not args.execute:
        return
    arm = 'base' if args.checkpoint == 'base' else 'lora'
    arm_dir = out / arm
    if (arm_dir / 'responses.jsonl').exists() or (arm_dir / 'sampling.jsonl').exists():
        raise RuntimeError('Arm already sampled; no silent repeat or overwrite')
    with tinker_shim(args.checkpoint, port=cfg.port, reasoning=cfg.reasoning, max_tokens=cfg.max_tokens,
                     context_window=cfg.context_window, max_cost_usd=cfg.max_cost_per_arm_usd, log_dir=arm_dir) as url:
        for i, prompt in enumerate(prepared):
            for repeat in range(cfg.samples_per_prompt):
                req = urllib.request.Request(url + '/chat/completions', data=json.dumps(prompt['body']).encode(),
                    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + os.environ['TINKER_SHIM_API_KEY']})
                try:
                    with urllib.request.urlopen(req, timeout=300) as response:
                        code, body = response.status, json.load(response)
                except urllib.error.HTTPError as error:
                    code, body = error.code, json.loads(error.read())
                row = {'prompt': i, 'repeat': repeat, 'checkpoint': args.checkpoint, 'http_status': code, 'response': body}
                with (arm_dir / 'responses.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(row) + '\n')
                print(arm, i, repeat, 'HTTP', code, 'output_tokens', body.get('usage', {}).get('completion_tokens'), flush=True)
                if code in [401, 402, 403, 429]:
                    raise RuntimeError(f'Provider/budget access failure: {code}')


if __name__ == '__main__':
    main()
