# ABOUTME: Runs the agent-collusion paper's two LLM judges (agreement, relaxation) UNMODIFIED against
# ABOUTME: Qwen3.8-27B on OpenRouter, swapping only the API client the scripts construct.
"""The judges ship for a self-hosted vLLM server: a dummy API key, and thinking disabled through
vLLM's `chat_template_kwargs`. OpenRouter needs a real key and disables thinking with its own
`reasoning: {enabled: false}`, so this wrapper replaces the scripts' `OpenAI` with a client that
translates exactly that and nothing else — prompts, temperature 0, the 400-token budget, parsing,
quote validation, retries and caching are all the paper's code. One provider is preferred so
every judgement comes from the same backend.

Run (from the vendored harness dir, so the judges' relative prompt paths resolve):
  python scratch/run_paper_judges.py <agreement|relaxation> --runs <run.json ...> --out <csv> \
      --cache <jsonl> --verdict-policy raw-only
"""

import importlib.util
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI as _OpenAI

HARNESS = (Path(__file__).resolve().parents[1] / "src/eval/misalignment/agent_collusion"
           / "third_party/agent-collusion")
MODEL = "qwen/qwen3.8-27b"
PROVIDER = {"order": ["phala"], "allow_fallbacks": True}


class _Completions:
    def __init__(self, client: _OpenAI):
        self._client = client

    def create(self, **kwargs):
        extra = dict(kwargs.pop("extra_body", None) or {})
        think = (extra.pop("chat_template_kwargs", None) or {}).get("enable_thinking", False)
        extra["reasoning"] = {"enabled": bool(think)}
        extra["provider"] = PROVIDER
        kwargs["model"] = MODEL
        return self._client.chat.completions.create(**kwargs, extra_body=extra)


class OpenRouterClient:
    """Stands in for `openai.OpenAI(base_url=..., api_key="EMPTY", timeout=...)`."""

    def __init__(self, base_url=None, api_key=None, timeout=None):
        client = _OpenAI(base_url="https://openrouter.ai/api/v1",
                         api_key=os.environ["OPENROUTER_API_KEY"], timeout=timeout)
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions(client)


def main() -> None:
    """Load the named judge module from the harness, swap its client, run its own main()."""
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    judge = sys.argv.pop(1)
    assert judge in ("agreement", "relaxation"), judge
    spec = importlib.util.spec_from_file_location(
        f"{judge}_judge", HARNESS / "analysis" / f"{judge}_judge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The judges import OpenAI inside their client class (`from openai import OpenAI`), so the
    # swap has to happen on the openai module itself, not on the judge module's namespace.
    import openai
    openai.OpenAI = OpenRouterClient
    sys.argv[0] = f"{judge}_judge.py"
    sys.exit(module.main())


if __name__ == "__main__":
    main()
