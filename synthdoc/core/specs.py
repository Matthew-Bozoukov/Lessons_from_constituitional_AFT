# ABOUTME: Spec sources. A spec_id resolves to text from control/specs/ or any path,
# ABOUTME: so swapping constitution_v3 for rules_only is a config line.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .hashing import text_hash

SPECS_DIR = Path(__file__).resolve().parents[1] / "control" / "specs"


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
        spec_id: Spec identifier. Without `path`, resolves to
            control/specs/<spec_id>.md (or .txt).
        path: Optional explicit file path, which lets a spec living elsewhere in
            the repo be used without copying it in.

    Returns:
        A SpecSource.

    Raises:
        FileNotFoundError: If no spec file can be found.
    """
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"spec.path does not exist: {p}")
        return SpecSource(spec_id=spec_id, text=p.read_text(), path=str(p))

    for ext in (".md", ".txt"):
        p = SPECS_DIR / f"{spec_id}{ext}"
        if p.exists():
            return SpecSource(spec_id=spec_id, text=p.read_text(), path=str(p))

    available = sorted(p.stem for p in SPECS_DIR.glob("*.*")) if SPECS_DIR.exists() else []
    raise FileNotFoundError(
        f"No spec {spec_id!r} in {SPECS_DIR} (available: {available}). "
        "Either add the file there or set spec.path in the config."
    )


def available_specs() -> list[str]:
    """Return the spec ids present in control/specs/."""
    if not SPECS_DIR.exists():
        return []
    return sorted({p.stem for p in SPECS_DIR.glob("*.*") if p.suffix in (".md", ".txt")})
