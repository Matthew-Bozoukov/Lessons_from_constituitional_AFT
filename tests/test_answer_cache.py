# ABOUTME: Offline tests for the HF answer cache (local backend + mirror) and for lazy
# ABOUTME: ServedTarget serving — the two pieces that make cached arms cost nothing.

from __future__ import annotations

import json

import pytest

from src.eval.answer_cache import ANSWERS, META, AnswerCache, CacheKey, gen_hash

KEY = CacheKey(model_key="ft_model", mode="think", subset_hash="abc123", gen_hash="d4e5f6")


def entry(tmp_path, key=KEY, **meta_overrides):
    """Write a valid cache entry source dir for `key`."""
    src = tmp_path / "src_entry"
    src.mkdir(exist_ok=True)
    (src / ANSWERS).write_text(json.dumps({"id": 0, "answer": "a"}) + "\n")
    meta = {"mode": key.mode, "subset_hash": key.subset_hash, "gen_hash": key.gen_hash,
            **meta_overrides}
    (src / META).write_text(json.dumps(meta))
    return src


def test_gen_hash_is_order_insensitive_and_value_sensitive():
    a = gen_hash({"temperature": 0.7, "top_p": 0.95})
    b = gen_hash({"top_p": 0.95, "temperature": 0.7})
    c = gen_hash({"temperature": 0.8, "top_p": 0.95})
    assert a == b != c


def test_key_path_is_hierarchical():
    assert KEY.path == "ft_model/think/abc123-d4e5f6"


def test_push_probe_fetch_roundtrip_local(tmp_path):
    cache = AnswerCache(str(tmp_path / "store"))
    assert not cache.probe(KEY)
    cache.push(KEY, entry(tmp_path), card_fields={})
    assert cache.probe(KEY)
    got = cache.fetch(KEY, tmp_path / "dest")
    assert json.loads((got / ANSWERS).read_text())["answer"] == "a"


def test_push_refuses_silent_overwrite_but_allows_refresh(tmp_path):
    cache = AnswerCache(str(tmp_path / "store"))
    cache.push(KEY, entry(tmp_path), card_fields={})
    with pytest.raises(RuntimeError, match="already exists"):
        cache.push(KEY, entry(tmp_path), card_fields={})
    cache.push(KEY, entry(tmp_path), card_fields={}, refresh=True)


def test_fetch_refuses_an_entry_stored_under_the_wrong_key(tmp_path):
    cache = AnswerCache(str(tmp_path / "store"))
    cache.push(KEY, entry(tmp_path), card_fields={})
    lying = CacheKey(model_key=KEY.model_key, mode=KEY.mode,
                     subset_hash="other000", gen_hash=KEY.gen_hash)
    # Hand-plant the entry under a path its sidecar disagrees with.
    planted = tmp_path / "store" / lying.path
    planted.mkdir(parents=True)
    for name in (ANSWERS, META):
        planted.joinpath(name).write_bytes(
            (tmp_path / "store" / KEY.path / name).read_bytes())
    with pytest.raises(AssertionError, match="refusing to use it"):
        cache.fetch(lying, tmp_path / "dest")


def test_push_never_stores_an_entry_that_would_refuse_to_load(tmp_path):
    cache = AnswerCache(str(tmp_path / "store"))
    bad = entry(tmp_path, mode="nothink")  # sidecar disagrees with KEY.mode
    with pytest.raises(AssertionError, match="refusing to use it"):
        cache.push(KEY, bad, card_fields={})


def test_mirror_gives_read_through_without_the_backing_store(tmp_path):
    mirror = tmp_path / "mirror"
    writer = AnswerCache(str(tmp_path / "store"), mirror=mirror)
    writer.push(KEY, entry(tmp_path), card_fields={})
    # A second handle pointed at a MISSING backing store still hits via the mirror —
    # the same-invocation handoff between the reference arm and the target arms.
    reader = AnswerCache(str(tmp_path / "gone"), mirror=mirror)
    assert reader.probe(KEY)
    got = reader.fetch(KEY, tmp_path / "dest")
    assert (got / ANSWERS).exists()


# --- Lazy ServedTarget ----------------------------------------------------------------


def test_served_target_boots_only_on_base_url_access():
    from src.endpoints.vllm_server import ServedTarget, TargetSpec

    spec = TargetSpec(hf_path="org/m", base_model="Qwen/Qwen3.6-27B", adapter=True,
                      mode="think", model_key="m", lora_rank=16)

    class _Server:
        calls = 0

        def serve(self, spec):
            self.calls += 1
            return "http://localhost:8000/v1"

    server = _Server()
    served = ServedTarget(spec, server)
    # Identity access must not serve.
    assert served.model_name == "m" and served.spec.mode == "think"
    assert server.calls == 0
    # Endpoint access serves (idempotently at the VllmServer layer).
    assert served.base_url == "http://localhost:8000/v1"
    assert server.calls == 1


def test_served_target_model_name_for_full_models_is_base():
    from src.endpoints.vllm_server import ServedTarget, TargetSpec

    spec = TargetSpec(hf_path="Qwen/Qwen3.6-27B", base_model="Qwen/Qwen3.6-27B",
                      adapter=False, mode="default", model_key="Qwen3_6-27B",
                      lora_rank=None)
    assert ServedTarget(spec, server=None).model_name == "base"