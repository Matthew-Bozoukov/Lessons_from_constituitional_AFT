# ABOUTME: Drop-in replacement for Shen et al.'s software/experiments/*/llm_client.py (the
# ABOUTME: "product" LLM the agents' code calls), same interface, OpenAI-compatible endpoint.
"""LLM client for the recommendation / treatment system.

Usage:
    from llm_client import create_llm_client

    client = create_llm_client()  # or create_llm_client({"temperature": 0.1, "max_tokens": 2000})
    text = await client.generate_response("prompt text", system_message="optional system prompt")

`prompt` may be a string or a list of {"role": "user"|"assistant", "content": str} messages.
`config_override` may set temperature and max_tokens for one request.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

from openai import AsyncOpenAI

# Endpoint and model are fixed by the experiment, not by the caller: every arm's code calls
# the same product model, so only the agents' choices differ between arms.
_BASE_URL = os.environ.get("PRODUCT_BASE_URL", "http://127.0.0.1:8000/v1")
_MODEL = os.environ.get("PRODUCT_MODEL", "base")
_API_KEY = os.environ.get("PRODUCT_API_KEY", "EMPTY")

# Shen's defaults (temperature 0.1, max_tokens 20000), unchanged.
DEFAULT_CONFIG = {
    "model": _MODEL,
    "max_tokens": 20000,
    "temperature": 0.1,
}


class SafetyToolingLLMClient:
    """LLM client with the same interface as the original."""

    def __init__(
        self,
        config_dict: Optional[Dict[str, Any]] = None,
        suppress_warnings: bool = True,
        use_cache: bool = True,
    ):
        self.config = {**DEFAULT_CONFIG, **(config_dict or {})}
        self.model_id = _MODEL
        self.temperature = self.config.get("temperature", 0.1)
        self.max_tokens = self.config.get("max_tokens", 20000)
        self.logger = logging.getLogger("llm_client")
        self.client = AsyncOpenAI(base_url=_BASE_URL, api_key=_API_KEY)

    async def generate_response(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        system_message: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a response to a prompt or conversation history."""
        temperature, max_tokens = self.temperature, self.max_tokens
        if config_override:
            # The model is fixed; only sampling settings can be overridden.
            temperature = config_override.get("temperature", temperature)
            max_tokens = config_override.get("max_tokens", max_tokens)
        messages: List[Dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif isinstance(prompt, list):
            messages += [
                {"role": m["role"], "content": m["content"]}
                for m in prompt
                if m.get("role") in ("user", "assistant")
            ]
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")
        r = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return r.choices[0].message.content or ""

    @classmethod
    def from_config_dict(
        cls,
        config_dict: Dict[str, Any],
        suppress_warnings: bool = True,
        use_cache: bool = True,
    ) -> "SafetyToolingLLMClient":
        """Create LLM client from configuration dictionary."""
        return cls(
            config_dict=config_dict,
            suppress_warnings=suppress_warnings,
            use_cache=use_cache,
        )


def create_llm_client(
    config_dict: Dict[str, Any] = None,
    suppress_warnings: bool = True,
    use_cache: bool = True,
) -> SafetyToolingLLMClient:
    """Factory function to create the LLM client."""
    return SafetyToolingLLMClient.from_config_dict(
        config_dict, suppress_warnings=suppress_warnings, use_cache=use_cache
    )
