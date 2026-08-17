# ABOUTME: Stage 1 of LLM-driven feature discovery: Sonnet generates 10-20 free-text
# ABOUTME: features per reasoning trace, one trace at a time, with prompt caching.

"""Stage 1 of the feature-discovery replication.

The autorater sees ONE reasoning trace at a time and nothing else — no scenario metadata,
no trait, no other traces — exactly as in the post. Features are free text, not a schema.
Each trace's features are appended to `features.jsonl` as they land, so the run resumes
from an interruption by rerunning with the same --out-dir.

Run:
  uv run python scratch/llm_feature_discovery/stage1_extract_free_text_features_per_trace.py \
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

from scratch.llm_feature_discovery.feature_extraction_and_naming_prompts import (  # noqa: E402
    build_feature_extraction_system_prompt)
from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import extract_json, timestamp, write_run_meta  # noqa: E402

AUTORATER_PRICE_PER_MTOK_INPUT, AUTORATER_PRICE_PER_MTOK_OUTPUT = 3.0, 15.0
# Bedrock/Google cyber safeguards false-positive on a few of these transcripts and no
# retry clears it; excluding Bedrock let 5 such rows through in the earlier audit.
OPENROUTER_PROVIDER_ROUTING = {"provider": {"ignore": ["Amazon Bedrock"]}}

# The post asks for features using only a-z; this is the check, not a rewriter — a
# feature that violates it is reported, never silently repaired.
LETTERS_ONLY_FEATURE_RE = re.compile(r"^[A-Za-z][A-Za-z ]*$")


def parse_feature_list_response(raw_model_output: str) -> list[str]:
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


def extract_features_from_one_trace(client: OpenRouterClient, autorater_model: str,
                                    reasoning_trace: str,
                                    temperature: float) -> tuple[list[str], int, int]:
    """Generate features for one reasoning trace, with one repair attempt.

    Args:
        client: OpenRouter client.
        autorater_model: OpenRouter model id.
        reasoning_trace: The reasoning trace text.
        temperature: Sampling temperature.

    Returns:
        (features, prompt tokens, completion tokens).
    """
    system = [{"type": "text", "text": build_feature_extraction_system_prompt(),
               "cache_control": {"type": "ephemeral"}}]
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": reasoning_trace}]
    first = client.chat(model=autorater_model, messages=messages, temperature=temperature,
                        max_tokens=1200, extra_body=OPENROUTER_PROVIDER_ROUTING)
    try:
        return parse_feature_list_response(first.content), first.prompt_tokens, first.completion_tokens
    except (ValueError, KeyError) as err:
        repair_messages = messages + [
            {"role": "assistant", "content": first.content},
            {"role": "user", "content": f"That output was rejected: {err}. Return only the "
                                        "JSON array of feature strings."}]
        retry = client.chat(model=autorater_model, messages=repair_messages, temperature=0.0,
                            max_tokens=1200, extra_body=OPENROUTER_PROVIDER_ROUTING)
        return (parse_feature_list_response(retry.content),
                first.prompt_tokens + retry.prompt_tokens,
                first.completion_tokens + retry.completion_tokens)


def _probe_whether_prompt_caching_engages(client: OpenRouterClient, autorater_model: str,
                                          reasoning_trace: str) -> None:
    """Send the same prefix twice and print raw usage, to show whether caching engaged.

    Anthropic only caches prefixes of >=1024 tokens; this prompt may fall under that,
    in which case cached_tokens stays 0 and the run simply costs more.

    Args:
        client: OpenRouter client.
        autorater_model: OpenRouter model id.
        reasoning_trace: A trace to send twice.
    """
    system = [{"type": "text", "text": build_feature_extraction_system_prompt(),
               "cache_control": {"type": "ephemeral"}}]
    for call_number in (1, 2):
        resp = client.client.chat.completions.create(
            model=autorater_model, max_tokens=1200, temperature=0.0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": reasoning_trace}])
        usage = resp.usage.model_dump()
        print(f"  call {call_number}: prompt={usage['prompt_tokens']} "
              f"details={usage.get('prompt_tokens_details')}")


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
    sft_rows = [json.loads(line) for line in Path(input).read_text().splitlines() if line.strip()]
    if smoke:
        limit = limit or 3
    if limit:
        sft_rows = sft_rows[:limit]

    run_dir = Path(out_dir or f"output/feature_discovery/{timestamp()}")
    run_dir.mkdir(parents=True, exist_ok=True)
    features_path = run_dir / "features.jsonl"
    already_labelled_scenario_ids = set()
    if features_path.exists():
        already_labelled_scenario_ids = {
            json.loads(x)["scenario_id"]
            for x in features_path.read_text().splitlines() if x.strip()}
        print(f"resuming: {len(already_labelled_scenario_ids)} traces already done")
    traces_to_label = [r for r in sft_rows
                       if r["metadata"]["scenario_id"] not in already_labelled_scenario_ids]

    client = OpenRouterClient()
    system_prompt = build_feature_extraction_system_prompt()
    print(f"model={model} traces={len(sft_rows)} todo={len(traces_to_label)} "
          f"workers={workers} temp={temperature}")
    print(f"system prompt {len(system_prompt)} chars (~{len(system_prompt) // 4} tokens; "
          f"Anthropic caches only >=1024-token prefixes)")
    print("\n--- first trace sent to the autorater (first 800 chars) ---")
    print(traces_to_label[0]["messages"][2]["reasoning_content"][:800], "...\n")
    if smoke:
        print("--- cache probe ---")
        _probe_whether_prompt_caching_engages(
            client, model, traces_to_label[0]["messages"][2]["reasoning_content"])

    append_lock = threading.Lock()
    features_file = features_path.open("a")
    features_violating_letters_only_rule: list[str] = []

    def extract_and_append_one(trace_index: int) -> tuple[int, int]:
        row = traces_to_label[trace_index]
        features, prompt_tokens, completion_tokens = extract_features_from_one_trace(
            client, model, row["messages"][2]["reasoning_content"], temperature)
        meta = row["metadata"]
        with append_lock:
            features_violating_letters_only_rule.extend(
                f for f in features if not LETTERS_ONLY_FEATURE_RE.match(f))
            features_file.write(json.dumps({"scenario_id": meta["scenario_id"],
                                            "trait_id": meta["trait_id"],
                                            "features": features}) + "\n")
            features_file.flush()
        return prompt_tokens, completion_tokens

    per_trace_usage = map_threaded(extract_and_append_one, len(traces_to_label),
                                   max_workers=workers, desc="features")
    features_file.close()
    total_prompt_tokens = sum(u[0] for u in per_trace_usage)
    total_completion_tokens = sum(u[1] for u in per_trace_usage)
    cost_upper_bound = (total_prompt_tokens / 1e6 * AUTORATER_PRICE_PER_MTOK_INPUT
                        + total_completion_tokens / 1e6 * AUTORATER_PRICE_PER_MTOK_OUTPUT)
    total_feature_instances = sum(len(json.loads(x)["features"])
                                  for x in features_path.read_text().splitlines() if x.strip())

    print(f"\n{len(traces_to_label)} traces -> {total_feature_instances} features total "
          f"in {features_path}")
    print(f"tokens in={total_prompt_tokens:,} out={total_completion_tokens:,} | "
          f"cost upper bound ${cost_upper_bound:.2f}")
    print(f"features violating the a-z rule: {len(features_violating_letters_only_rule)}"
          + (f" e.g. {features_violating_letters_only_rule[:5]}"
             if features_violating_letters_only_rule else ""))
    if smoke:
        # Rows are written in completion order, so look the first trace up by id.
        wanted_scenario_id = traces_to_label[0]["metadata"]["scenario_id"]
        record = next(json.loads(x) for x in features_path.read_text().splitlines()
                      if json.loads(x)["scenario_id"] == wanted_scenario_id)
        print(f"\n--- features for the first trace ({wanted_scenario_id}) ---")
        print(json.dumps(record["features"], indent=1))

    write_run_meta(run_dir, {"input": input, "model": model, "temperature": temperature,
                            "traces": len(sft_rows), "done_this_run": len(traces_to_label),
                            "features_total": total_feature_instances,
                            "tokens_in": total_prompt_tokens,
                            "tokens_out": total_completion_tokens,
                            "cost_upper_bound_usd": cost_upper_bound,
                            "command": " ".join(sys.argv)})


if __name__ == "__main__":
    fire.Fire(main)
