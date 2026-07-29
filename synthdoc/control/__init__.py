# ABOUTME: The control surface: run configs in configs/, every model-facing string in
# ABOUTME: prompts/, and specs in specs/. Tuning the pipeline should mean editing only this.

from . import loader

__all__ = ["loader"]
