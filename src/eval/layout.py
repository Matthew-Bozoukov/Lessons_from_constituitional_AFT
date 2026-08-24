# ABOUTME: The published-layout contract every eval's out_dir must satisfy before the
# ABOUTME: epilogue pushes it verbatim to HF: rollouts/ + results/ + metadata/, nothing else.

"""Every eval run dir — and therefore every published HF eval repo — has ONE shape:

    rollouts/   self-contained model transcripts (prompt + response/trajectory)
    results/    scores, judgments, metrics, markdown mirrors; the epilogue's canonical
                results.json/results.md land here too
    metadata/   configs, run_meta.json, provenance, caches
    README.md   the HF card, written at push time (root is where HF reads it)

run_eval.py enforces this after run() returns (`assert_layout`), so a new eval cannot
silently publish a bespoke tree. Runners call `publish_layout(out_dir)` to create the
three dirs and write into them directly, or repack at the end (ODCV's package_run).
"""

from __future__ import annotations

from pathlib import Path

PUBLISH_DIRS = ("rollouts", "results", "metadata")
# Root files the epilogue itself owns during/after a run; everything else is a stray.
_ROOT_ALLOWED = set(PUBLISH_DIRS) | {"README.md"}


def publish_layout(out_dir: Path) -> tuple[Path, Path, Path]:
    """Create (if needed) and return the contract dirs: (rollouts, results, metadata)."""
    dirs = tuple(out_dir / name for name in PUBLISH_DIRS)
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def assert_layout(out_dir: Path) -> None:
    """Fail fast if the run dir holds anything the published contract does not allow.

    Runs after the eval's run() and the epilogue's own writes, before the HF push —
    a stray here means an eval left working files or bespoke output at the root, which
    a verbatim upload would publish as part of the artifact.
    """
    stray = sorted(p.name for p in out_dir.iterdir() if p.name not in _ROOT_ALLOWED)
    if stray:
        raise RuntimeError(
            f"{out_dir} violates the published-layout contract; stray root entries: "
            f"{stray}. Home them under rollouts/, results/ or metadata/ "
            "(src/eval/layout.py).")
