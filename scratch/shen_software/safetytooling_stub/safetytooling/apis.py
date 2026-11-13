# ABOUTME: InferenceAPI stand-in: `await api(model_id=..., prompt=Prompt)` -> [response with .completion],
# ABOUTME: sent to the OpenAI-compatible endpoint in AGENT_BASE_URL with the SAME model_id it was given.
import os
from dataclasses import dataclass
from typing import List

from openai import AsyncOpenAI

from .data_models import Prompt


@dataclass
class LLMResponse:
    completion: str


class InferenceAPI:
    def __init__(self, no_cache: bool = True, **_: object):
        self.client = AsyncOpenAI(
            base_url=os.environ.get("AGENT_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.environ.get("AGENT_API_KEY", "EMPTY"),
        )

    async def __call__(
        self,
        model_id: str,
        prompt: Prompt,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **_: object,
    ) -> List[LLMResponse]:
        messages = [
            {
                "role": m.role.value if hasattr(m.role, "value") else str(m.role),
                "content": m.content,
            }
            for m in prompt.messages
        ]
        kwargs = {"model": model_id, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        r = await self.client.chat.completions.create(**kwargs)
        return [LLMResponse(completion=r.choices[0].message.content or "")]
