# ABOUTME: Offline durability and corruption checks for the SWE message blob cache.
# ABOUTME: Requires Linux file locks; no provider, model, or Docker calls.
import os
from pathlib import Path
from unittest.mock import patch
import pytest


@pytest.mark.skipif(os.name != "posix", reason="fleet protocol uses Linux locks")
def test_blob_cache_reuses_only_verified_durable_writes(tmp_path):
    from src.eval.capabilities.swebench_mini.fleet_protocol import MessageBlobStore

    store = MessageBlobStore(tmp_path, capacity=1)
    msg = {"role": "user", "content": "alpha"}
    sha = store.put(msg)
    with patch.object(
        Path, "read_bytes", side_effect=AssertionError("redundant filesystem lookup")
    ):
        assert store.put(msg) == sha
    # A changed message obtains a new blob; eviction and reopen validate existing data.
    assert store.put(dict(msg, content="beta")) != sha
    assert store.put(msg) == sha
    (tmp_path / (sha + ".json")).write_text("{}")
    with pytest.raises(ValueError, match="Corrupt"):
        MessageBlobStore(tmp_path).put(msg)


@pytest.mark.skipif(os.name != "posix", reason="fleet protocol uses Linux locks")
def test_failed_blob_write_is_not_cached(tmp_path):
    from src.eval.capabilities.swebench_mini.fleet_protocol import MessageBlobStore

    store = MessageBlobStore(tmp_path)
    with patch(
        "src.eval.capabilities.swebench_mini.fleet_protocol.atomic",
        side_effect=OSError("disk full"),
    ):
        with pytest.raises(OSError):
            store.put({"content": "x"})
    assert not store.verified
    sha = store.put({"content": "x"})
    assert (tmp_path / (sha + ".json")).exists()
