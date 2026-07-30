# ABOUTME: Spec sources. A spec_id resolves to text from control/specs/ or any path,
# ABOUTME: so swapping constitution_v3 for rules_only is a config line.

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import yaml

from .hashing import text_hash

SPECS_DIR = Path(__file__).resolve().parents[1] / "control" / "specs"
# Maps spec_id -> path for specs that live outside control/specs/. Lets `spec.id`
# alone identify a spec, which is what makes `axis: spec.id` sweeps clean.
SPECS_INDEX = SPECS_DIR / "index.yaml"
# parents[4] because this file sits at src/data/synthdoc/core/specs.py; index.yaml
# entries and spec.path resolve against the REPO root, so this breaks silently if
# the package moves depth - update it with any relocation.
REPO_ROOT = Path(__file__).resolve().parents[4]


@functools.lru_cache(maxsize=1)
def _index() -> dict[str, str]:
    """Load the spec id -> path index, if present."""
    if not SPECS_INDEX.exists():
        return {}
    data = yaml.safe_load(SPECS_INDEX.read_text()) or {}
    return {str(k): str(v) for k, v in (data.get("specs") or {}).items()}


@dataclass(frozen=True)
class SpecSource:
    """A loaded model spec.

    Attributes:
        spec_id: Identifier used in chunk_ids and every snapshot row.
        text: Full spec text.
        path: Where it was loaded from, recorded in the manifest.
    """

    spec_id: str
    text: str
    path: str

    @property
    def sha(self) -> str:
        """Content hash of the spec text, recorded in the run manifest."""
        return text_hash(self.text, 16)


def load_spec(spec_id: str, path: str | None = None) -> SpecSource:
    """Load a spec by id, or from an explicit path.

    Args:
        spec_id: Spec identifier. Without `path`, resolves against
            control/specs/<spec_id>.md (or .txt), then control/specs/index.yaml.
        path: Optional explicit file path. Prefer registering the spec in
            index.yaml instead, so that `spec.id` alone identifies it and
            `axis: spec.id` sweeps work without also overriding spec.path.

    Returns:
        A SpecSource.

    Raises:
        FileNotFoundError: If no spec file can be found.
    """
    if path:
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            p = REPO_ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"spec.path does not exist: {path}")
        return SpecSource(spec_id=spec_id, text=p.read_text(), path=str(p))

    for ext in (".md", ".txt"):
        p = SPECS_DIR / f"{spec_id}{ext}"
        if p.exists():
            return SpecSource(spec_id=spec_id, text=p.read_text(), path=str(p))

    indexed = _index().get(spec_id)
    if indexed:
        p = Path(indexed)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.exists():
            raise FileNotFoundError(
                f"control/specs/index.yaml maps {spec_id!r} to {indexed}, which does not exist"
            )
        return SpecSource(spec_id=spec_id, text=p.read_text(), path=str(p))

    raise FileNotFoundError(
        f"No spec {spec_id!r}. Available: {available_specs()}. Add the file to "
        f"{SPECS_DIR}, register it in control/specs/index.yaml, or set spec.path."
    )


def available_specs() -> list[str]:
    """Return every spec id resolvable by id alone (files plus index entries)."""
    ids = set(_index())
    if SPECS_DIR.exists():
        ids |= {p.stem for p in SPECS_DIR.glob("*.*") if p.suffix in (".md", ".txt")}
    return sorted(ids)
