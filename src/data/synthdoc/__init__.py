# ABOUTME: synthdoc -- one config-driven generation pipeline. The config's `stages:` list
# ABOUTME: (incl. prompts) defines the document type; code supplies generic operators.

from .constitution import Trait, full_text, segment
from .core import Ctx, Stage
from .operators import OPERATORS
from .pipeline import build_stages, estimate, n_examples, run

__all__ = ["OPERATORS", "Ctx", "Stage", "Trait", "build_stages", "estimate",
           "full_text", "n_examples", "run", "segment"]
