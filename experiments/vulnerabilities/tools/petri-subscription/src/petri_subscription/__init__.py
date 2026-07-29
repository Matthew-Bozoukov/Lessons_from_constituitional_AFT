"""Inspect model provider backed by the Claude Agent SDK / Claude Code CLI.

Registers the provider name ``claude-code``, so a model string looks like::

    claude-code/sonnet
    claude-code/claude-sonnet-4-5

The provider authenticates through whatever credential the Claude Code CLI
itself is configured with (``claude auth login`` / ``claude setup-token`` ->
``CLAUDE_CODE_OAUTH_TOKEN``). It never reads ``ANTHROPIC_API_KEY`` and by
default blanks that variable in the CLI subprocess environment.
"""

from .provider import ClaudeCodeAPI, claude_code
from .translate import render_conversation

__all__ = ["ClaudeCodeAPI", "claude_code", "render_conversation"]
