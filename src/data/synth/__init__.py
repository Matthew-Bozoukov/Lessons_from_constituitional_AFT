# ABOUTME: synth -- one config-driven generation pipeline. The config's `stages:` list
# ABOUTME: (incl. prompts) defines the document type; code supplies generic operators.

from .constitution import Trait, full_text, segment
from .stage_runtime import Ctx, Stage
from .stage_operators import OPERATORS
from .pipeline import build_stages, estimate, n_examples, run

__all__ = ["OPERATORS", "Ctx", "Stage", "Trait", "build_stages", "estimate",
           "full_text", "n_examples", "run", "segment"]
