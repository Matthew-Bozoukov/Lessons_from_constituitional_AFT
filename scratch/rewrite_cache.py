# ABOUTME: Pick a stable scenario sample once and cache every rewrite made against it, so a
# ABOUTME: later experiment with a different move set reuses the halves already paid for.
"""A reusable cache of difficult-advice rewrites.

The sample is chosen once, stratified across the nine principles and pinned by seed, and
written to sids.json. Every rewrite is stored under a key naming the half and the exact move
set, so re-running the same combination costs nothing and a new combination only pays for
what it actually changes.

Run:
  uv run python scratch/rewrite_cache.py init --n 20
  uv run python scratch/rewrite_cache.py show
"""
import json
import random
from pathlib import Path

import fire

from src.utils import read_jsonl, timestamp, write_run_meta

CACHE = Path("output/rewrite_cache")
SRC = "output/invert_all716/20260821_191814/inversions.jsonl"


def _key(half: str, moves: list[str]) -> str:
    """Cache key naming the half and the move set, order-independent."""
    return f"{half}__" + "_".join(sorted(m[:12] for m in moves))


def init(n: int = 20, seed: int = 0, include: str = "") -> None:
    """Choose the sample once and write the manifest.

    Args:
        n: Sample size, spread evenly over the nine principles.
        seed: RNG seed, pinned so the sample is reproducible.
        include: Comma-separated scenario ids to force into the sample (kept first),
            so an earlier experiment's rows stay comparable.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(SRC)
    # Fire coerces a comma-separated flag into a tuple, so accept both shapes.
    forced = ([x.strip() for x in include.split(",")] if isinstance(include, str) and include
              else [str(x) for x in include] if include else [])
    by_trait: dict[str, list] = {}
    for r in rows:
        if r["scenario_id"] in forced:
            continue
        by_trait.setdefault(r["scenario_id"].split("_")[0], []).append(r)
    traits = sorted(by_trait, key=lambda x: int(x[1:]))
    rng = random.Random(seed)
    need = n - len(forced)
    if need >= sum(len(v) for v in by_trait.values()):
        # The whole corpus: stratification is moot, so take everything in scenario order.
        sids = forced + sorted(r["scenario_id"] for t in traits for r in by_trait[t])
        assert len(sids) == n, f"{len(sids)} != {n}"
        (CACHE / "sids.json").write_text(json.dumps(
            {"sids": sids, "n": n, "seed": seed, "forced": forced, "source": SRC,
             "created_utc": timestamp(),
             "selection": "entire corpus; earlier sample kept first"}, indent=1))
        print(f"wrote {CACHE/'sids.json'} — all {n} scenarios "
              f"({len(forced)} from the earlier sample, kept first)")
        return
    base, extra = divmod(need, len(traits))
    picked = [r["scenario_id"] for i, t in enumerate(traits)
              for r in rng.sample(by_trait[t], base + (1 if i < extra else 0))]
    sids = forced + picked
    assert len(sids) == n, f"{len(sids)} != {n}"
    manifest = {"sids": sids, "n": n, "seed": seed, "forced": forced,
                "source": SRC, "created_utc": timestamp(),
                "selection": "stratified over the nine principles; forced ids kept first"}
    (CACHE / "sids.json").write_text(json.dumps(manifest, indent=1))
    counts: dict[str, int] = {}
    for s in sids:
        counts[s.split("_")[0]] = counts.get(s.split("_")[0], 0) + 1
    print(f"wrote {CACHE/'sids.json'} — {n} scenarios, per trait {dict(sorted(counts.items()))}")
    print(f"forced in: {forced or 'none'}")


def sids() -> str:
    """Return the cached scenario ids as a comma-separated string."""
    return ",".join(json.loads((CACHE / "sids.json").read_text())["sids"])


def store(half: str, moves: str, path: str) -> None:
    """Record a completed rewrite in the cache under its half+moves key.

    Args:
        half: "reasoning" or "answer".
        moves: Comma-separated move names that were targeted.
        path: Directory holding the rewrite output.
    """
    assert half in ("reasoning", "answer"), half
    ms = ([m.strip() for m in moves.split(",")] if isinstance(moves, str)
          else [str(m) for m in moves])
    idx_path = CACHE / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    idx[_key(half, ms)] = {"half": half, "moves": ms, "path": str(path),
                           "stored_utc": timestamp()}
    idx_path.write_text(json.dumps(idx, indent=1))
    print(f"cached {_key(half, ms)} -> {path}")


def show() -> None:
    """Print the manifest and everything cached against it."""
    m = json.loads((CACHE / "sids.json").read_text())
    print(f"sample: {m['n']} scenarios, seed {m['seed']}, created {m['created_utc']}")
    print(f"  {', '.join(m['sids'])}\n")
    idx_path = CACHE / "index.json"
    if not idx_path.exists():
        print("nothing cached yet")
        return
    idx = json.loads(idx_path.read_text())
    print(f"{len(idx)} cached rewrite(s):")
    for k, v in sorted(idx.items()):
        print(f"  {v['half']:9s} {'+'.join(v['moves'])}")
        print(f"            {v['path']}")


if __name__ == "__main__":
    fire.Fire({"init": init, "sids": sids, "store": store, "show": show})
