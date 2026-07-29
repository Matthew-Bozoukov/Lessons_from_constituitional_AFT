"""Entry-point module imported by Inspect's extension loader.

Importing this module registers the ``claude-code`` model provider.
"""

from .provider import claude_code  # noqa: F401

__all__ = ["claude_code"]
