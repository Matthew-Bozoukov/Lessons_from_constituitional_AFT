# ABOUTME: synthdoc -- constitution-grounded synthetic-data generation. Shared machinery
# ABOUTME: in core/constitution/hf_cache; one subpackage per pipeline (difficult_advice, mem).

from .constitution import Trait, full_text, segment

__all__ = ["Trait", "full_text", "segment"]
