# ABOUTME: Replay real client rejection recovery through pinned vLLM transforms and the Qwen template.
# ABOUTME: Downloads public source/templates only; no model, GPU, paid host or credentials used.
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import runpy
import typing
import urllib.request

from jinja2.sandbox import ImmutableSandboxedEnvironment
import yaml


def extract(source, names, scope):
    nodes = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    for node in nodes:
        node.decorator_list = []
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[])),
                 '<pinned-source-functions>', 'exec'), scope)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/eval/swebench_mini/lite.yaml'))
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    cfg = yaml.safe_load(config_bytes)
    root = Path('output') / ('swebench-protocol-template-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    root.mkdir(parents=True)
    (root/'source-config.yaml').write_bytes(config_bytes)
    version = re.search(r'vllm==([\d.]+)', Path('pyproject.toml').read_text()).group(1)
    hashes = {}

    def fetch(url, name):
        data = urllib.request.urlopen(url, timeout=30).read()
        (root/name).write_bytes(data)
        hashes[url] = hashlib.sha256(data).hexdigest()
        return data.decode()

    def text_parts(role, content, tracker, **kwargs):
        assert not kwargs['wrap_dicts']
        assert all(isinstance(x, dict) and x.get('type') == 'text' for x in content)
        return [{'role': role, 'content': '\n'.join(x['text'] for x in content)}]

    scope = {'json': json, 'cast': typing.cast, '_AssistantParser': lambda x: x,
             '_ToolParser': lambda x: x, 'ChatCompletionContentPartTextParam': dict,
             '_parse_chat_message_content_parts': text_parts, 'VLLMValidationError': ValueError}
    for path, names in [
        ('vllm/entrypoints/openai/chat_completion/protocol.py', ['_normalize_messages_before']),
        ('vllm/entrypoints/chat_utils.py', ['_parse_chat_message_content', '_postprocess_messages'])]:
        extract(fetch(f'https://raw.githubusercontent.com/vllm-project/vllm/v{version}/{path}',
                      path.replace('/', '_')), names, scope)
    url = f"https://huggingface.co/{cfg['base']}/resolve/{cfg['base_revision']}/tokenizer_config.json"
    template = json.loads(fetch(url, 'tokenizer_config.json'))['chat_template']
    pin_path = Path('src/infra/endpoints/vllm.py')
    hashes[str(pin_path)] = hashlib.sha256(pin_path.read_bytes()).hexdigest()
    pins = {}
    extract(pin_path.read_text(), ['pin_template', 'pin_prefix'], pins)
    jinja = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def fail(message):
        raise ValueError(message)
    jinja.globals['raise_exception'] = fail
    preserve = cfg.get('serving', {}).get('preserve_thinking', False)
    renderer = jinja.from_string(pins['pin_template'](template, 'think', preserve_thinking=preserve))
    tests = runpy.run_path('tests/test_swebench_protocol.py')
    checks = []
    for kind in ('no_tool', 'bad_tool', 'bad_json', 'null_args', 'bad_command'):
        test = tests['ProtocolTests']()
        try:
            test.setUp()
            test.kind = kind
            try:
                test.agent.query()
                raise AssertionError('Expected rejected response')
            except tests['FormatError'] as exc:
                test.agent.add_messages(*exc.messages)
            test.kind = 'normal'
            test.agent.query()
            wire = test.calls[-1]
            normalized = scope['_normalize_messages_before'](None, copy.deepcopy(wire))
            conversation = []
            for message in normalized['messages']:
                conversation.extend(scope['_parse_chat_message_content'](message, None, 'string', False))
            scope['_postprocess_messages'](conversation)
            rendered = renderer.render(messages=conversation, tools=wire['tools'],
                add_generation_prompt=True, enable_thinking=False, preserve_thinking=False)
            assert rendered.count('TRACE-1') == 1, kind
            assert rendered.endswith('<|im_start|>assistant\n<think>\n'), kind
            if kind == 'bad_json':
                assert '{oops' in rendered
            if kind == 'null_args':
                assert 'null' in rendered
            (root/(kind+'-wire.json')).write_text(json.dumps(wire, indent=2), encoding='utf-8')
            (root/(kind+'-prompt.txt')).write_text(rendered, encoding='utf-8')
            checks.append({'case': kind, 'retained_rejected_reasoning_once': True,
                           'rendered_under_actual_pinned_template': True})
        finally:
            test.doCleanups()
    result = {'status': 'passed', 'model_inference': False, 'vllm': version,
              'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
              'preserve_thinking': preserve,
              'base': cfg['base'], 'base_revision': cfg['base_revision'], 'checks': checks,
              'source_sha256': hashes, 'evidence_dir': str(root),
              'limitations': 'Pinned text-only vLLM function replay and Jinja rendering, not a running CUDA server.'}
    for path in (root/'results.json', Path('output/swebench-protocol-template-latest.json')):
        path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('PINNED_TEMPLATE_RECOVERY_PASSED', root)


if __name__ == '__main__': main()
