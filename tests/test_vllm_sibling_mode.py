# ABOUTME: ServedTarget.sibling: a full model that IS the server's base takes its pinned mode, and an adapter
# ABOUTME: on another commit of the same base is served on the caller's only when the weights match.
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


def test_adapter_on_another_base_commit_is_served_on_the_callers_when_weights_match(monkeypatch):
    monkeypatch.setattr(vllm, "resolve_target",
                        lambda p: vllm.replace(spec(p, True, "think"), base_revision="old",
                                               base_revision_from="training_meta"))
    monkeypatch.setattr(vllm, "_base_files_that_differ", lambda base, a, b: [])
    target = ServedTarget(spec("org/arm", True, "think"), FakeServer())
    peer = target.sibling("org/peer")
    assert peer.base_revision == "br"
    assert "old" in peer.base_revision_from and "br" in peer.base_revision_from


def test_adapter_on_another_base_commit_is_refused_when_weights_differ(monkeypatch):
    monkeypatch.setattr(vllm, "resolve_target",
                        lambda p: vllm.replace(spec(p, True, "think"), base_revision="old"))
    monkeypatch.setattr(vllm, "_base_files_that_differ",
                        lambda base, a, b: ["model-00001-of-00011.safetensors"])
    target = ServedTarget(spec("org/arm", True, "think"), FakeServer())
    with pytest.raises(AssertionError, match="cannot co-serve.*safetensors"):
        target.sibling("org/peer")


def test_base_files_that_differ_ignores_only_the_template_and_card(monkeypatch):
    class Sib:
        def __init__(self, name, h, lfs=False):
            self.rfilename, self.blob_id = name, h
            self.lfs = type("L", (), {"sha256": h})() if lfs else None

    trees = {
        "a": [Sib("w.safetensors", "1", True), Sib("config.json", "c"),
              Sib("chat_template.jinja", "t1"), Sib("tokenizer_config.json", "k1"),
              Sib("README.md", "r1"), Sib("tokenizer.json", "z")],
        "b": [Sib("w.safetensors", "1", True), Sib("config.json", "c"),
              Sib("chat_template.jinja", "t2"), Sib("tokenizer_config.json", "k2"),
              Sib("README.md", "r2"), Sib("tokenizer.json", "z2")],
    }

    class Api:
        def model_info(self, repo, revision, files_metadata):
            return type("I", (), {"siblings": trees[revision]})()

    import src.infra.huggingface as hfmod
    monkeypatch.setattr(hfmod, "hf_api", lambda: Api())
    assert vllm._base_files_that_differ(BASE, "a", "b") == ["tokenizer.json"]
    assert vllm._base_files_that_differ(BASE, "a", "a") == []
    assert vllm._base_files_that_differ(BASE, None, "a") == ["<unrecorded revision>"]
