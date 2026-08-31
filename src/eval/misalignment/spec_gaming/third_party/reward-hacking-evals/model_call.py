from litellm import acompletion
import litellm
import logging
import traceback
import asyncio
import os
import random

from config import LLMModel
from openai import AsyncOpenAI, RateLimitError as OpenAIRateLimitError
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError as MistralSDKError
from together import AsyncTogether
from anthropic import RateLimitError as AnthropicRateLimitError
from litellm.exceptions import RateLimitError as LiteLLMRateLimitError

# 1. Get a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR) # Set the minimum level for the logger

# 2. Create and configure your file handler
error_file_handler = logging.FileHandler("litellm_errors.log")
error_file_handler.setLevel(logging.ERROR) # Set the level for this specific handler

# 3. Add the handler to the logger
logger.addHandler(error_file_handler)

litellm.suppress_debug_info = True


def is_rate_limit_error(e: Exception) -> bool:
    """Check if an exception is a rate limit error from any provider."""
    # Direct rate limit exception types
    if isinstance(e, (OpenAIRateLimitError, AnthropicRateLimitError, LiteLLMRateLimitError)):
        return True

    # Mistral SDK error with 429 status
    if isinstance(e, MistralSDKError):
        error_str = str(e)
        if "429" in error_str or "rate limit" in error_str.lower():
            return True

    # Fallback: check error message for common rate limit indicators
    error_str = str(e).lower()
    if "429" in str(e) and ("rate" in error_str or "limit" in error_str or "too many" in error_str):
        return True

    return False


async def unified_completion(model: LLMModel, messages: list):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            if 'openai' in model.model_name:
                reduced_name = model.model_name.split('/')[-1]

                client = AsyncOpenAI()
                if 'openai/o' in model.model_name or 'openai/gpt-5' == model.model_name:
                    reasoning_dict = {
                        "summary": "auto",
                    }
                    if model.extended_thinking_tokens is not None and isinstance(model.extended_thinking_tokens, str):
                        reasoning_dict['effort'] = model.extended_thinking_tokens
                    response = await client.responses.create(
                        model=reduced_name,
                        input=messages,
                        reasoning=reasoning_dict
                    )
                    summary_object = response.output[0].summary
                    summary = summary_object[0].text if len(summary_object) > 0 else None
                    return response.output[1].content[0].text, summary, response.to_dict()
                else:
                    response = await client.responses.create(
                        model=reduced_name,
                        input=messages,
                    )
                    return response.output[0].content[0].text, None, response.to_dict()
            elif 'mistral/' in model.model_name:
                # Handle Mistral models directly using Mistral SDK
                reduced_name = model.model_name.split('/')[-1]
                client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

                response = await client.chat.complete_async(
                    model=reduced_name,
                    messages=messages,
                )

                # Handle structured content (with thinking blocks) from Mistral
                content = response.choices[0].message.content
                reasoning_text = None
                text_content = None

                if isinstance(content, list):
                    # Content is structured with thinking and text blocks
                    for block in content:
                        if hasattr(block, 'type'):
                            if block.type == 'thinking' and hasattr(block, 'thinking'):
                                # Extract thinking text
                                if isinstance(block.thinking, list) and len(block.thinking) > 0:
                                    reasoning_text = block.thinking[0].text if hasattr(block.thinking[0], 'text') else str(block.thinking[0])
                                else:
                                    reasoning_text = str(block.thinking)
                            elif block.type == 'text' and hasattr(block, 'text'):
                                text_content = block.text
                        elif isinstance(block, dict):
                            if block.get('type') == 'thinking' and 'thinking' in block:
                                thinking_blocks = block['thinking']
                                if thinking_blocks and len(thinking_blocks) > 0:
                                    reasoning_text = thinking_blocks[0].get('text', str(thinking_blocks[0]))
                            elif block.get('type') == 'text':
                                text_content = block.get('text', '')
                else:
                    # Content is a simple string
                    text_content = content

                # Create response dict
                simple_response = {
                    'id': response.id,
                    'created': response.created,
                    'model': response.model,
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0,
                        'completion_tokens': response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0,
                        'total_tokens': response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0,
                    } if hasattr(response, 'usage') else None,
                }

                return text_content, reasoning_text, simple_response
            elif 'together_ai/' in model.model_name:
                # Handle Together AI models directly using Together SDK
                reduced_name = model.model_name.split('together_ai/')[-1]
                client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

                response = await client.chat.completions.create(
                    model=reduced_name,
                    messages=messages,
                )

                # Extract content and create response dict
                text_content = response.choices[0].message.content
                reasoning_text = None

                # Create response dict
                simple_response = {
                    'id': response.id,
                    'created': response.created,
                    'model': response.model,
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0,
                        'completion_tokens': response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0,
                        'total_tokens': response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0,
                    } if hasattr(response, 'usage') else None,
                }

                return text_content, reasoning_text, simple_response
            elif 'vllm/' in model.model_name:
                # Handle vLLM models using OpenAI SDK with custom base URL
                reduced_name = model.model_name.split('vllm/')[-1]
                base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
                api_key = os.environ.get("VLLM_API_KEY", "dummy")

                client = AsyncOpenAI(base_url=base_url, api_key=api_key)

                # PATCH (teaching_claude_why_replication): 4096 truncates a reasoning
                # model INSIDE its <think> block, so the answer token never appears and
                # every string-matching grader here scores "inconclusive"/"invalid_output"
                # -- a silent false negative, not an error. Size for trace + answer.
                response = await client.chat.completions.create(
                    model=reduced_name,
                    messages=messages,
                    temperature=0.6,
                    top_p=0.95,
                    max_tokens=int(os.environ.get("VLLM_MAX_TOKENS", 16384)),
                )

                # PATCH: keep the reasoning trace. Served with --reasoning-parser qwen3,
                # vLLM splits it into `reasoning_content` and leaves `content` as the
                # answer alone -- which is also what makes the graders' substring checks
                # ('READ' in text, 'ANSWER: ') read the answer instead of the trace.
                message = response.choices[0].message
                text_content = message.content
                # vLLM 0.26 returns the parsed trace as `reasoning`; older builds and
                # some providers use `reasoning_content`. Verified live 2026-08-25 on
                # Qwen3.6-27B: the field is `reasoning`, and reading only
                # `reasoning_content` silently dropped a 100-token trace while content
                # itself was correct -- so this is invisible unless you check usage.
                dumped = message.model_dump() if hasattr(message, "model_dump") else {}
                reasoning_text = (dumped.get("reasoning_content")
                                  or dumped.get("reasoning") or None)

                simple_response = {
                    'id': response.id,
                    'created': response.created,
                    'model': response.model,
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                        'completion_tokens': response.usage.completion_tokens if response.usage else 0,
                        'total_tokens': response.usage.total_tokens if response.usage else 0,
                    } if response.usage else None,
                }

                return text_content, reasoning_text, simple_response
            else:
                # Provider preferences for OpenRouter (ignore problematic providers)
                openrouter_extra = {"provider": {"ignore": ["SiliconFlow"]}} if 'openrouter/' in model.model_name else {}

                if 'openrouter/google/gemini-3' in model.model_name:
                    # Gemini 3 uses reasoning_effort ("low" or "high") instead of token count
                    assert model.extended_thinking_tokens is None or isinstance(model.extended_thinking_tokens, str)

                    extra_body = {"reasoning": {"effort": model.extended_thinking_tokens}} if model.extended_thinking_tokens else {}
                    extra_body.update(openrouter_extra)
                    response = await acompletion(
                        model=model.model_name,
                        messages=messages,
                        extra_body=extra_body
                    )
                elif 'openrouter/google' in model.model_name:
                    # Gemini 2.5 Pro uses max_tokens for reasoning
                    assert model.extended_thinking_tokens is None or isinstance(model.extended_thinking_tokens, int)

                    extra_body = {"reasoning": {"max_tokens": model.extended_thinking_tokens}} if model.extended_thinking_tokens else {}
                    extra_body.update(openrouter_extra)
                    response = await acompletion(
                        model=model.model_name,
                        messages=messages,
                        extra_body=extra_body
                    )
                elif model.extended_thinking_tokens is not None:
                    kwargs = {
                        "model": model.model_name,
                        "messages": messages,
                        "thinking": {
                            "type": "enabled",
                            "budget_tokens": model.extended_thinking_tokens,
                        },
                    }
                    if openrouter_extra:
                        kwargs["extra_body"] = openrouter_extra
                    response = await acompletion(**kwargs)
                else:
                    kwargs = {
                        "model": model.model_name,
                        "messages": messages,
                    }
                    if openrouter_extra:
                        kwargs["extra_body"] = openrouter_extra
                    response = await acompletion(**kwargs)

                if hasattr(response.choices[0].message, 'reasoning_content') and 'grok-4' not in model.model_name:
                    reasoning_text = response.choices[0].message.reasoning_content
                elif hasattr(response.choices[0].message, 'thinking_blocks'):
                    reasoning_text = response.choices[0].message.thinking_blocks[0]['thinking']
                else:
                    reasoning_text = None

                return response.choices[0].message.content, reasoning_text, response.to_dict()
        except Exception as e:
            logger.error(f"LiteLLM error with model {model}: {str(e)}")
            logger.error(traceback.format_exc())

            is_last_attempt = attempt == max_retries - 1

            if is_rate_limit_error(e):
                if is_last_attempt:
                    print(f"[Rate Limit] {model.model_name}: Max retries exceeded")
                    raise
                # Exponential backoff: 2, 4, 8, 16, 32 seconds + jitter
                base_wait = 2 ** (attempt + 1)
                jitter = random.uniform(0, 1)
                wait_time = base_wait + jitter
                print(f"[Rate Limit] {model.model_name}: Waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                # Non-rate-limit error: retry once after 1 second
                if is_last_attempt or attempt >= 1:
                    raise
                await asyncio.sleep(1)

