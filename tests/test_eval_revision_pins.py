# ABOUTME: Explicit target revisions must reach metadata, template and context reads.
# ABOUTME: These offline tests guard against serving a frozen adapter with moving-head metadata.
import json
from unittest.mock import Mock

from src.infra.endpoints import vllm


def test_adapter_metadata_reads_requested_revision(monkeypatch, tmp_path):
    metadata = tmp_path / 'metadata.json'
    metadata.write_text('{}')
    download = Mock(return_value=str(metadata))
    monkeypatch.setattr(vllm, 'hf_download', download)
    spec = vllm.TargetSpec(hf_path='org/2026-09-22-qwen36-0-nosynth', base_model='Qwen/Qwen3.6-27B',
                           adapter=True, mode='think', model_key='qwen36_0_nosynth', base_revision='base-pin', lora_rank=64)
    monkeypatch.setattr(vllm, '_spec_from_files', lambda *a: spec)
    monkeypatch.setattr(vllm, '_repo_sha', Mock(side_effect=AssertionError('Moving head read')))
    got = vllm.resolve_target(spec.hf_path, revision='a' * 40)
    assert got.revision == 'a' * 40
    assert [c.args[1] for c in download.call_args_list] == ['adapter_config.json', 'training_meta.json']
    assert all(c.kwargs == {'revision': 'a' * 40} for c in download.call_args_list)


def test_template_and_native_context_use_base_revision(monkeypatch, tmp_path):
    metadata = tmp_path / 'metadata.json'
    metadata.write_text(json.dumps({'chat_template': 'template', 'max_position_embeddings': 262144}))
    download = Mock(return_value=str(metadata))
    monkeypatch.setattr(vllm, 'hf_download', download)
    monkeypatch.setattr(vllm, 'pin_template', lambda template, mode: 'pinned-' + template)
    server = object.__new__(vllm.VllmServer)
    server.executor = Mock()
    server._pinned_template_path('Qwen/Qwen3.6-27B', 'think', 'b' * 40)
    assert vllm.native_context_window('Qwen/Qwen3.6-27B', 'b' * 40) == 262144
    assert all(c.kwargs == {'revision': 'b' * 40} for c in download.call_args_list)
