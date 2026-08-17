# ABOUTME: Stage 1 of LLM-driven feature discovery: Sonnet generates 10-20 free-text
# ABOUTME: features per reasoning trace, one trace at a time, with prompt caching.

"""Stage 1 of the feature-discovery replication.

The autorater sees ONE reasoning trace at a time and nothing else — no scenario metadata,
no trait, no other traces — exactly as in the post. Features are free text, not a schema.
Each trace's features are appended to `features.jsonl` as they land, so the run resumes
from an interruption by rerunning with the same --out-dir.

Run:
  uv run python scratch/feature_discovery/extract_features.py \
      --input output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl --smoke
"""

from __future__ import annotations

import json
import re
import sys
import threading
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scratch.feature_discovery.prompts import feature_system_prompt  # noqa: E402
from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import extract_json, timestamp, write_run_meta  # noqa: E402

PRICE_IN, PRICE_OUT = 3.0, 15.0
# Bedrock/Google cyber safeguards false-positive on a few of these transcripts and no
# retry clears it; excluding Bedrock let 5 such rows through in the earlier audit.
PROVIDER_ROUTING = {"provider": {"ignore": ["Amazon Bedrock"]}}

# The post asks for features using only a-z; this is the check, not a rewriter — a
# feature that violates it is reported, never silently repaired.
CLEAN = re.compile(r"^[A-Za-z][A-Za-z ]*$")


def parse_features(text: str) -> list[str]:
    """Parse and sanity-check the autorater's feature list.

    Args:
        text: Raw model output.

    Returns:
        The feature strings.

    Raises:
        ValueError: If the payload is not a list of 5-40 non-empty strings.
    """
    obj = extract_json(text)
    if not isinstance(obj, list) or not 5 <= len(obj) <= 40:
        raise ValueError(f"expected a JSON array of 5-40 features, got {obj!r:.300}")
    feats = [f.strip() for f in obj if isinstance(f, str) and f.strip()]
    if len(feats) != len(obj):
        raise ValueError(f"non-string or empty feature in {obj!r:.300}")
    return feats


def label_trace(client: OpenRouterClient, model: str, trace: str,
                temperature: float) -> tuple[list[str], int, int]:
    """Generate features for one reasoning trace, with one repair attempt.

    Args:
        client: OpenRouter client.
        model: OpenRouter model id.
        trace: The reasoning trace text.
        temperature: Sampling temperature.

    Returns:
        (features, prompt tokens, completion tokens).
    """
    system = [{"type": "text", "text": feature_system_prompt(),
               "cache_control": {"type": "ephemeral"}}]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": trace}]
    res = client.chat(model=model, messages=messages, temperature=temperature,
                      max_tokens=1200, extra_body=PROVIDER_ROUTING)
    try:
        return parse_features(res.content), res.prompt_tokens, res.completion_tokens
    except (ValueError, KeyError) as err:
        repair = messages + [
            {"role": "assistant", "content": res.content},
            {"role": "user", "content": f"That output was rejected: {err}. Return only the "
                                        "JSON array of feature strings."}]
        res2 = client.chat(model=model, messages=repair, temperature=0.0, max_tokens=1200,
                           extra_body=PROVIDER_ROUTING)
        return (parse_features(res2.content),
                res.prompt_tokens + res2.prompt_tokens,
                res.completion_tokens + res2.completion_tokens)


def _probe_cache(client: OpenRouterClient, model: str, trace: str) -> None:
    """Send the same prefix twice and print raw usage, to show whether caching engaged.

    Anthropic only caches prefixes of >=1024 tokens; this prompt may fall under that,
    in which case cached_tokens stays 0 and the run simply costs more.

    Args:
        client: OpenRouter client.
        model: OpenRouter model id.
        trace: A trace to send twice.
    """
    system = [{"type": "text", "text": feature_system_prompt(),
               "cache_control": {"type": "ephemeral"}}]
    for i in (1, 2):
        resp = client.client.chat.completions.create(
            model=model, max_tokens=1200, temperature=0.0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": trace}])
        d = resp.usage.model_dump()
        print(f"  call {i}: prompt={d['prompt_tokens']} details={d.get('prompt_tokens_details')}")


def main(
    input: str,
    out_dir: str | None = None,
    model: str = "anthropic/claude-sonnet-5",
    temperature: float = 1.0,
    workers: int = 16,
    limit: int | None = None,
    smoke: bool = False,
) -> None:
    """Generate free-text features for every reasoning trace in an SFT file.

    Args:
        input: Path to a stage_7 SFT jsonl carrying `reasoning_content`.
        out_dir: Output directory; defaults to output/feature_discovery/<timestamp>.
        model: OpenRouter autorater model.
        temperature: Sampling temperature (the post's brainstorm framing wants > 0).
        workers: Concurrent requests.
        limit: Only the first N traces.
        smoke: 3 traces, cache probe, and the full first result printed.
    """
    rows = [json.loads(line) for line in Path(input).read_text().splitlines() if line.strip()]
    if smoke:
        limit = limit or 3
    if limit:
        rows = rows[:limit]

    out = Path(out_dir or f"output/feature_discovery/{timestamp()}")
    out.mkdir(parents=True, exist_ok=True)
    feat_path = out / "features.jsonl"
    done = set()
    if feat_path.exists():
        done = {json.loads(x)["scenario_id"] for x in feat_path.read_text().splitlines() if x.strip()}
        print(f"resuming: {len(done)} traces already done")
    todo = [r for r in rows if r["metadata"]["scenario_id"] not in done]

    client = OpenRouterClient()
    sysp = feature_system_prompt()
    print(f"model={model} traces={len(rows)} todo={len(todo)} workers={workers} temp={temperature}")
    print(f"system prompt {len(sysp)} chars (~{len(sysp) // 4} tokens; Anthropic caches "
          f"only >=1024-token prefixes)")
    print("\n--- first trace sent to the autorater (first 800 chars) ---")
    print(todo[0]["messages"][2]["reasoning_content"][:800], "...\n")
    if smoke:
        print("--- cache probe ---")
        _probe_cache(client, model, todo[0]["messages"][2]["reasoning_content"])

    lock = threading.Lock()
    fh = feat_path.open("a")
    bad_chars: list[str] = []

    def run(i: int) -> tuple[int, int]:
        row = todo[i]
        feats, tin, tout = label_trace(client, model,
                                       row["messages"][2]["reasoning_content"], temperature)
        meta = row["metadata"]
        with lock:
            bad_chars.extend(f for f in feats if not CLEAN.match(f))
            fh.write(json.dumps({"scenario_id": meta["scenario_id"],
                                 "trait_id": meta["trait_id"], "features": feats}) + "\n")
            fh.flush()
        return tin, tout

    usage = map_threaded(run, len(todo), max_workers=workers, desc="features")
    fh.close()
    tin = sum(u[0] for u in usage)
    tout = sum(u[1] for u in usage)
    cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
    n_feats = sum(len(json.loads(x)["features"]) for x in feat_path.read_text().splitlines() if x.strip())

    print(f"\n{len(todo)} traces -> {n_feats} features total in {feat_path}")
    print(f"tokens in={tin:,} out={tout:,} | cost upper bound ${cost:.2f}")
    print(f"features violating the a-z rule: {len(bad_chars)}"
          + (f" e.g. {bad_chars[:5]}" if bad_chars else ""))
    if smoke:
        # Rows are written in completion order, so look the first trace up by id.
        want = todo[0]["metadata"]["scenario_id"]
        rec = next(json.loads(x) for x in feat_path.read_text().splitlines()
                   if json.loads(x)["scenario_id"] == want)
        print(f"\n--- features for the first trace ({want}) ---")
        print(json.dumps(rec["features"], indent=1))

    write_run_meta(out, {"input": input, "model": model, "temperature": temperature,
                         "traces": len(rows), "done_this_run": len(todo),
                         "features_total": n_feats, "tokens_in": tin, "tokens_out": tout,
                         "cost_upper_bound_usd": cost, "command": " ".join(sys.argv)})


if __name__ == "__main__":
    fire.Fire(main)
