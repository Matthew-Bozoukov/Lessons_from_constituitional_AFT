# ABOUTME: ServedTarget.sibling: a full model that IS the server's base takes the server's pinned
# ABOUTME: mode (base model in every Hospital seat), while any other mismatch is still refused.
import pytest

import src.infra.endpoints.vllm as vllm
from src.infra.endpoints.vllm import ServedTarget, TargetSpec

BASE = "Qwen/Qwen3.6-27B"


class FakeServer:
    def ensure(self, spec):
        return spec


def spec(hf_path, adapter, mode, base=BASE):
    return TargetSpec(hf_path=hf_path, base_model=base, adapter=adapter, mode=mode,
                      model_key=hf_path.split("/")[-1], lora_rank=None,
                      revision="r", base_revision="br")


def test_full_model_peer_takes_the_servers_pinned_mode(monkeypatch):
    monkeypatch.setattr(vllm, "resolve_target", lambda p: spec(p, False, "default"))
    target = ServedTarget(spec(BASE, False, "think"), FakeServer())
    peer = target.sibling(BASE)
    assert peer.mode == "think" and peer.revision == "br"


def test_explicit_mode_still_wins_and_mismatch_is_refused(monkeypatch):
    monkeypatch.setattr(vllm, "resolve_target", lambda p: spec(p, False, "default"))
    target = ServedTarget(spec(BASE, False, "think"), FakeServer())
    with pytest.raises(AssertionError, match="cannot co-serve"):
        target.sibling(BASE, mode="nothink")


def test_adapter_peer_in_another_mode_is_still_refused(monkeypatch):
    monkeypatch.setattr(vllm, "resolve_target", lambda p: spec(p, True, "nothink"))
    target = ServedTarget(spec(BASE, False, "think"), FakeServer())
    with pytest.raises(AssertionError, match="cannot co-serve"):
        target.sibling("org/some-lora")
