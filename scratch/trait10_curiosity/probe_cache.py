# ABOUTME: Two sequential calls with the same <<<cache>>> prefix: the second must report
# ABOUTME: cached_tokens > 0, or the full run would pay full price for the constitution.
# Run: uv run python scratch/trait10_curiosity/probe_cache.py

import fire
from omegaconf import OmegaConf

from src.data.synth.constitution import full_text
from src.endpoints.openrouter import CACHE_MARK, OpenRouterClient

CFG = "scratch/trait10_curiosity/difficult_advice_t10.yaml"
CONST = full_text("scratch/trait10_curiosity/constitution.md")


def _prefix(stage: str | None) -> tuple[str, str]:
    """The exact cacheable prefix a stage sends, or a generic one when no stage is named."""
    if not stage:
        return f"<constitution>\n{CONST}\n</constitution>", "anthropic/claude-sonnet-5"
    cfg = OmegaConf.to_container(OmegaConf.load(CFG), resolve=True)
    sc = next(s for s in cfg["stages"] if s["name"] == stage)
    user = sc["prompts"]["user"]
    assert CACHE_MARK in user, f"{stage} has no {CACHE_MARK} breakpoint"
    return (user.split(CACHE_MARK)[0].replace("{constitution}", CONST),
            cfg["models"][sc["model"]]["model"])


def main(stage: str | None = None, calls: int = 2) -> None:
    """Send `calls` sequential probes sharing a prefix; with --stage, use that stage's exact
    prefix and model, so a single call during a live run shows whether the run's own
    calls are hitting (cached_tokens > 0 on the FIRST probe means the run warmed it)."""
    prefix, model = _prefix(stage)
    client = OpenRouterClient()
    for i in range(calls):
        msgs = [{"role": "user", "content":
                 f"{prefix}{CACHE_MARK}\n\nProbe {i}: reply with the single word OK."}]
        r = client.chat(model, msgs, temperature=0.0, max_tokens=5)
        print(f"call {i}: model={model} prompt_tokens={r.prompt_tokens} "
              f"cached_tokens={r.cached_tokens} provider={r.provider} "
              f"text={r.content.strip()!r}")
    assert r.cached_tokens > 0, "last call did not read the cached prefix"
    print("prompt cache OK")


if __name__ == "__main__":
    fire.Fire(main)
