# ABOUTME: Carry ODCV judge scores from an already-judged run into a newly combined run.
# ABOUTME: Matches transcripts by content hash, never by rollout number, which combining renumbers.
import hashlib
import json
import shutil
import sys
from pathlib import Path

VARIANTS = ("mandated", "incentivized")


def index_by_hash(run: Path) -> dict[str, str]:
    """Map each transcript's content hash to its `variant/Scenario/rollout_NNN` key.

    Args:
        run: A combined ODCV run directory containing agent_logs/.

    Returns:
        Dict of sha1 hex digest -> unit key.
    """
    out: dict[str, str] = {}
    for variant in VARIANTS:
        for t in run.glob(f"agent_logs/*-{variant}/experiments/*/rollout_*/messages_record.txt"):
            key = f"{variant}/{t.parent.parent.name}/{t.parent.name}"
            digest = hashlib.sha1(t.read_bytes()).hexdigest()
            assert digest not in out or out[digest] == key, (
                f"duplicate transcript content: {key} and {out[digest]}"
            )
            out[digest] = key
    return out


def main(old_run: str, new_run: str) -> None:
    """Rewrite old judge caches onto the new run's rollout numbering.

    Args:
        old_run: Judged combined run whose evaluations/scores_*.json will be carried.
        new_run: Newly combined run to receive the cache.
    """
    old, new = Path(old_run), Path(new_run)
    old_hashes = index_by_hash(old)
    new_hashes = index_by_hash(new)
    old_by_key = {v: k for k, v in old_hashes.items()}

    (new / "evaluations").mkdir(parents=True, exist_ok=True)
    for src in sorted((old / "evaluations").glob("scores_*.json")):
        cache = json.loads(src.read_text())
        carried, moved, dropped = {}, 0, []
        for key, score in cache.items():
            digest = old_by_key.get(key)
            if digest is None or digest not in new_hashes:
                dropped.append(key)
                continue
            new_key = new_hashes[digest]
            carried[new_key] = score
            moved += new_key != key
        dst = new / "evaluations" / src.name
        dst.write_text(json.dumps(carried, indent=2))
        print(
            f"{src.name}: {len(cache)} old -> {len(carried)} carried "
            f"({moved} renumbered, {len(dropped)} dropped)"
        )
        for k in dropped:
            print(f"    dropped (transcript not in new run): {k}")

    shutil.copy(old / "evaluations" / "run_meta.json", new / "evaluations" / "run_meta.json")
    print(f"\nnew run holds {len(new_hashes)} transcripts; cache primed.")


if __name__ == "__main__":
    main(*sys.argv[1:])
