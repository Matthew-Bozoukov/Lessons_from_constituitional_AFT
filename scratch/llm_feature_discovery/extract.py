# ABOUTME: Ask the autorater for free-text features describing one reasoning trace, and
# ABOUTME: run that over a whole corpus, appending results so the run resumes.

"""Free-text feature extraction.

The autorater sees ONE reasoning trace at a time and nothing else — no scenario metadata,
no trait, no other traces — exactly as in the post. Features are free text, not a schema,
which is the whole point: the model names behaviours nobody chose in advance.

Each trace's features are appended to features.jsonl as they land, so an interrupted run
resumes by rerunning against the same run directory.
"""

from __future__ import annotations

import json
import re
import threading

from scratch.llm_feature_discovery.prompts import build_feature_extraction_system_prompt
from scratch.llm_feature_discovery.rundir import RunDir
from src.endpoints.openrouter import OpenRouterClient, map_threaded
from src.utils import extract_json

AUTORATER_PRICE_PER_MTOK_INPUT, AUTORATER_PRICE_PER_MTOK_OUTPUT = 3.0, 15.0
# Bedrock/Google cyber safeguards false-positive on a few of these transcripts and no
# retry clears it; excluding Bedrock let 5 such rows through in the earlier audit.
OPENROUTER_PROVIDER_ROUTING = {"provider": {"ignore": ["Amazon Bedrock"]}}

# The post asks for features using only a-z; this is the check, not a rewriter — a
# feature that violates it is reported, never silently repaired.
LETTERS_ONLY_FEATURE_RE = re.compile(r"^[A-Za-z][A-Za-z ]*$")


def parse_feature_list(raw_model_output: str) -> list[str]:
    """Parse and sanity-check the autorater's feature list.

    Args:
        raw_model_output: Raw model output.

    Returns:
        The feature strings.

    Raises:
        ValueError: If the payload is not a list of 5-40 non-empty strings.
    """
    parsed = extract_json(raw_model_output)
    if not isinstance(parsed, list) or not 5 <= len(parsed) <= 40:
        raise ValueError(f"expected a JSON array of 5-40 features, got {parsed!r:.300}")
    features = [f.strip() for f in parsed if isinstance(f, str) and f.strip()]
    if len(features) != len(parsed):
        raise ValueError(f"non-string or empty feature in {parsed!r:.300}")
    return features


def features_for_trace(client: OpenRouterClient, model: str, reasoning_trace: str,
                       temperature: float) -> tuple[list[str], int, int]:
    """Generate features for one reasoning trace, with one repair attempt.

    Args:
        client: OpenRouter client.
        model: OpenRouter model id.
        reasoning_trace: The reasoning trace text.
        temperature: Sampling temperature.

    Returns:
        (features, prompt tokens, completion tokens).
    """
    system = [{"type": "text", "text": build_feature_extraction_system_prompt(),
               "cache_control": {"type": "ephemeral"}}]
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": reasoning_trace}]
    first = client.chat(model=model, messages=messages, temperature=temperature,
                        max_tokens=1200, extra_body=OPENROUTER_PROVIDER_ROUTING)
    try:
        return parse_feature_list(first.content), first.prompt_tokens, first.completion_tokens
    except (ValueError, KeyError) as err:
        repair_messages = messages + [
            {"role": "assistant", "content": first.content},
            {"role": "user", "content": f"That output was rejected: {err}. Return only the "
                                        "JSON array of feature strings."}]
        retry = client.chat(model=model, messages=repair_messages, temperature=0.0,
                            max_tokens=1200, extra_body=OPENROUTER_PROVIDER_ROUTING)
        return (parse_feature_list(retry.content),
                first.prompt_tokens + retry.prompt_tokens,
                first.completion_tokens + retry.completion_tokens)


def probe_prompt_caching(client: OpenRouterClient, model: str, reasoning_trace: str) -> None:
    """Send the same prefix twice and print raw usage, to show whether caching engaged.

    Anthropic only caches prefixes of >=1024 tokens; this prompt falls under that, in
    which case cached_tokens stays 0 and the run simply costs more.

    Args:
        client: OpenRouter client.
        model: OpenRouter model id.
        reasoning_trace: A trace to send twice.
    """
    system = [{"type": "text", "text": build_feature_extraction_system_prompt(),
               "cache_control": {"type": "ephemeral"}}]
    for call_number in (1, 2):
        resp = client.client.chat.completions.create(
            model=model, max_tokens=1200, temperature=0.0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": reasoning_trace}])
        usage = resp.usage.model_dump()
        print(f"  call {call_number}: prompt={usage['prompt_tokens']} "
              f"details={usage.get('prompt_tokens_details')}")


def extract_corpus(run: RunDir, sft_rows: list[dict], model: str, temperature: float,
                   workers: int) -> dict:
    """Label every not-yet-labelled trace, appending to the run's features.jsonl.

    Args:
        run: The run directory to write into.
        sft_rows: SFT rows carrying `messages[2].reasoning_content` and `metadata`.
        model: OpenRouter autorater model.
        temperature: Sampling temperature (the post's brainstorm framing wants > 0).
        workers: Concurrent requests.

    Returns:
        Usage and quality counters for the caller to report and record.
    """
    already_labelled = run.labelled_scenario_ids()
    if already_labelled:
        print(f"resuming: {len(already_labelled)} traces already done")
    todo = [r for r in sft_rows if r["metadata"]["scenario_id"] not in already_labelled]

    client = OpenRouterClient()
    append_lock = threading.Lock()
    features_file = run.open_trace_features_for_append()
    violations: list[str] = []

    def extract_and_append(index: int) -> tuple[int, int]:
        row = todo[index]
        features, prompt_tokens, completion_tokens = features_for_trace(
            client, model, row["messages"][2]["reasoning_content"], temperature)
        meta = row["metadata"]
        with append_lock:
            violations.extend(f for f in features if not LETTERS_ONLY_FEATURE_RE.match(f))
            features_file.write(json.dumps({"scenario_id": meta["scenario_id"],
                                            "trait_id": meta["trait_id"],
                                            "features": features}) + "\n")
            features_file.flush()
        return prompt_tokens, completion_tokens

    try:
        usage = map_threaded(extract_and_append, len(todo), max_workers=workers,
                             desc="features")
    finally:
        features_file.close()

    prompt_tokens = sum(u[0] for u in usage)
    completion_tokens = sum(u[1] for u in usage)
    return {"traces_total": len(sft_rows),
            "traces_labelled_this_run": len(todo),
            "feature_instances": sum(len(r["features"]) for r in run.read_trace_features()),
            "tokens_in": prompt_tokens, "tokens_out": completion_tokens,
            "cost_upper_bound_usd": (prompt_tokens / 1e6 * AUTORATER_PRICE_PER_MTOK_INPUT
                                     + completion_tokens / 1e6
                                     * AUTORATER_PRICE_PER_MTOK_OUTPUT),
            "features_violating_letters_only_rule": len(violations),
            "violation_examples": violations[:5]}
