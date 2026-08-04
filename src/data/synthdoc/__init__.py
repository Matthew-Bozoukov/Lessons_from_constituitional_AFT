# ABOUTME: synthdoc -- one constitution-grounded generation pipeline; the config's
# ABOUTME: `pipeline:` field picks the document type (difficult_advice | mem).

from .constitution import Trait, full_text, segment
from .pipeline import PIPELINES, run

__all__ = ["PIPELINES", "Trait", "full_text", "run", "segment"]
