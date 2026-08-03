# ABOUTME: synthdoc_v2 -- a faithful six-stage replication of the Teaching Claude Why
# ABOUTME: difficult-advice pipeline: segment, scenarios, draft, refine, respond, rewrite.

from .constitution import Trait, full_text, segment
from .pipeline import STAGES, run

__all__ = ["STAGES", "Trait", "full_text", "run", "segment"]
