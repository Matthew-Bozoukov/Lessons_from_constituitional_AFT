# ABOUTME: Verify the SURF corpus cache shares one copy across loaders without changing
# ABOUTME: sampling behaviour — the patch that stops --num-runs 3 from OOM-killing the sweep.

"""Test the _CORPUS_CACHE patch in surf.em_loop.sampling.

Run (from the SURF dir): uv run python <repo>/scratch/test_surf_corpus_cache.py

Uses a temporary JSONL corpus so it is fast and offline; the cache is source-agnostic, so
the HF path shares the same code.
"""

import json
import random
import sys
import tempfile
from pathlib import Path

SURF = Path("/home/matthewb/git repos/teaching_claude_why_replication"
            "/src/eval/audits/surf/third_party/SURF")
sys.path.insert(0, str(SURF))

from surf.em_loop.sampling import _CORPUS_CACHE, AttributeFileLoader  # noqa: E402


def main() -> None:
    """Assert identity-sharing, correct isolation by column, and unchanged sampling."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "corpus.jsonl"
        with open(path, "w") as f:
            for i in range(500):
                f.write(json.dumps({
                    "prompt": f"p{i}", "response": f"r{i}",
                    "sae_attributes": [f"a{i}_0", f"a{i}_1", f"a{i}_2"],
                    "attributes": [f"b{i}_0", f"b{i}_1"],
                }) + "\n")

        a = AttributeFileLoader(str(path))
        b = AttributeFileLoader(str(path))
        assert a.data is b.data, "second loader did not reuse the first corpus"
        assert len(a.data) == 500, f"expected 500 records, got {len(a.data)}"

        # A different column must NOT reuse the cache: it filters to a different subset.
        c = AttributeFileLoader(str(path), attribute_column="attributes")
        assert c.data is not a.data, "different column wrongly shared a corpus"
        assert len(_CORPUS_CACHE) == 2, f"expected 2 cache entries, got {len(_CORPUS_CACHE)}"

        # Sampling must be unchanged: same seed through either loader gives the same draw.
        random.seed(0)
        first = [a.sample_random_attributes() for _ in range(20)]
        random.seed(0)
        second = [b.sample_random_attributes() for _ in range(20)]
        assert first == second, "shared corpus changed sampling behaviour"

        # Sampling must not mutate the shared records, or runs would contaminate each other.
        before = json.dumps(a.data[:50], sort_keys=True)
        random.seed(1)
        for _ in range(200):
            a.sample_random_attributes()
        assert json.dumps(a.data[:50], sort_keys=True) == before, "sampling mutated corpus"

        print(f"PASS  shared={a.data is b.data}  entries={len(_CORPUS_CACHE)}  "
              f"records={len(a.data)}  sampling unchanged, corpus unmutated")


if __name__ == "__main__":
    main()
