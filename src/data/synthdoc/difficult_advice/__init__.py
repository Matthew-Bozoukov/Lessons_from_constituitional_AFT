# ABOUTME: The difficult-advice pipeline -- a faithful six-stage replication of the
# ABOUTME: Teaching Claude Why recipe: segment, scenarios, draft, refine, respond, rewrite.

from .pipeline import STAGES, run, topup

__all__ = ["STAGES", "run", "topup"]
