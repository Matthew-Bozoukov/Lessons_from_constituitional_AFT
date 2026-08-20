# ABOUTME: Two sequential calls with the same <<<cache>>> prefix: the second must report
# ABOUTME: cached_tokens > 0, or the full run would pay full price for the constitution.
# Run: uv run python scratch/trait10_curiosity/probe_cache.py

from src.data.synth.constitution import full_text
from src.endpoints.openrouter import OpenRouterClient

MODEL = "anthropic/claude-sonnet-5"
CONST = full_text("scratch/trait10_curiosity/constitution.md")


def main() -> None:
    client = OpenRouterClient()
    for i in range(2):
        msgs = [{"role": "user", "content":
                 f"<constitution>\n{CONST}\n</constitution><<<cache>>>\n\n"
                 f"Probe {i}: reply with the single word OK."}]
        r = client.chat(MODEL, msgs, temperature=0.0, max_tokens=5)
        print(f"call {i}: prompt_tokens={r.prompt_tokens} cached_tokens={r.cached_tokens} "
              f"provider={r.provider} text={r.content.strip()!r}")
    assert r.cached_tokens > 0, "second call did not read the cached prefix"
    print("prompt cache OK")


if __name__ == "__main__":
    main()
