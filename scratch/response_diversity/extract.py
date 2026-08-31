# ABOUTME: Labels each assistant turn of a difficult-advice SFT file against the
# ABOUTME: scenario-agnostic feature schema, via Sonnet on OpenRouter with prompt caching.

"""Stage 1 of the response-diversity audit.

Reads a stage_7 SFT jsonl (messages + metadata + reasoning_content), asks the judge to
fill the closed-vocabulary schema in `schema.py` for every assistant turn, and appends
each labelled row to `features.jsonl` as it lands (so a crash or a kill loses nothing —
rerun with the same --out-dir to resume).

Run:
  uv run python scratch/response_diversity/extract.py \
      --input output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl --smoke
  uv run python scratch/response_diversity/extract.py \
      --input output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scratch.response_diversity.schema import system_prompt, validate  # noqa: E402
from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import extract_json, timestamp, write_run_meta  # noqa: E402

# Sonnet-class OpenRouter pricing, USD per million tokens, for the run-cost report.
PRICE_IN, PRICE_OUT = 3.0, 15.0

PROVIDER_ROUTING = {"provider": {"ignore": ["Amazon Bedrock"]}}


def _item_text(row: dict) -> str:
    """Render one SFT row as the judge's user message.

    Args:
        row: A stage_7 SFT record with `messages` and `metadata`.

    Returns:
        The assistant's private reasoning and reply, plus the request it answered.
    """
    msgs = {m["role"]: m for m in row["messages"]}
    assistant = msgs["assistant"]
    return (
        f"<request>\n{msgs['user']['content']}\n</request>\n\n"
        f"<assistant_reasoning>\n{assistant.get('reasoning_content', '')}\n"
        f"</assistant_reasoning>\n\n"
        f"<assistant_reply>\n{assistant['content']}\n</assistant_reply>"
    )


def _label(client: OpenRouterClient, model: str, text: str, temperature: float) -> tuple[dict, int, int]:
    """Label one item, retrying once with the parse error before failing.

    Args:
        client: OpenRouter client.
        model: OpenRouter model id.
        text: Rendered item.
        temperature: Sampling temperature.

    Returns:
        (validated feature dict, prompt tokens, completion tokens).

    Raises:
        ValueError: If the repair attempt still violates the schema.
    """
    # cache_control marks the static schema prefix; every item after the first reads it.
    system = [{"type": "text", "text": system_prompt(), "cache_control": {"type": "ephemeral"}}]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
    # Bedrock's "cyber safeguards" filter false-positives on some of these transcripts
    # (returns finish_reason=content_filter with no body), which no retry can clear;
    # excluding that one upstream provider lets the item route to a working backend.
    res = client.chat(model=model, messages=messages, temperature=temperature, max_tokens=700,
                      extra_body=PROVIDER_ROUTING)
    tok_in, tok_out = res.prompt_tokens, res.completion_tokens
    try:
        obj = extract_json(res.content)
        validate(obj)
        return obj, tok_in, tok_out
    except (ValueError, KeyError) as err:
        repair = messages + [
            {"role": "assistant", "content": res.content},
            {"role": "user", "content": f"That output was rejected: {err}. Return only the "
                                        "corrected JSON object."},
        ]
        res2 = client.chat(model=model, messages=repair, temperature=0.0, max_tokens=700,
                           extra_body=PROVIDER_ROUTING)
        obj = extract_json(res2.content)
        validate(obj)
        return obj, tok_in + res2.prompt_tokens, tok_out + res2.completion_tokens


def _probe_cache(client: OpenRouterClient, model: str, text: str) -> None:
    """Print raw usage for two identical-prefix calls so cache hits are visible.

    Args:
        client: OpenRouter client.
        model: OpenRouter model id.
        text: A rendered item to send twice.
    """
    system = [{"type": "text", "text": system_prompt(), "cache_control": {"type": "ephemeral"}}]
    for i in (1, 2):
        resp = client.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
            temperature=0.0,
            max_tokens=700,
        )
        print(f"  call {i} usage: {resp.usage.model_dump()}")


def main(
    input: str,
    out_dir: str | None = None,
    model: str = "anthropic/claude-sonnet-5",
    temperature: float = 0.0,
    workers: int = 16,
    limit: int | None = None,
    smoke: bool = False,
) -> None:
    """Label every assistant turn in an SFT file with the diversity feature schema.

    Args:
        input: Path to a stage_7 SFT jsonl.
        out_dir: Output directory; defaults to output/response_diversity/<timestamp>.
        model: OpenRouter judge model.
        temperature: Judge sampling temperature.
        workers: Concurrent requests.
        limit: Label only the first N rows.
        smoke: Label 3 rows, probe the prompt cache, and print the first result in full.
    """
    rows = [json.loads(line) for line in Path(input).read_text().splitlines() if line.strip()]
    if smoke:
        limit = limit or 3
    if limit:
        rows = rows[:limit]

    out = Path(out_dir or f"output/response_diversity/{timestamp()}")
    out.mkdir(parents=True, exist_ok=True)
    feat_path = out / "features.jsonl"
    done = set()
    if feat_path.exists():
        done = {json.loads(line)["scenario_id"] for line in feat_path.read_text().splitlines() if line.strip()}
        print(f"resuming: {len(done)} rows already labelled in {feat_path}")
    todo = [r for r in rows if r["metadata"]["scenario_id"] not in done]

    client = OpenRouterClient()
    print(f"model={model} rows={len(rows)} to_label={len(todo)} workers={workers}")
    print(f"system prompt: {len(system_prompt())} chars (~{len(system_prompt()) // 4} tokens; "
          f"Anthropic caching needs >=1024)")
    print("\n--- first item sent to the judge ---")
    print(_item_text(todo[0])[:1200], "...\n")
    if smoke:
        print("--- prompt-cache probe (expect cached_tokens > 0 on call 2) ---")
        _probe_cache(client, model, _item_text(todo[0]))

    lock = threading.Lock()
    fh = feat_path.open("a")

    def run(i: int) -> tuple[int, int]:
        row = todo[i]
        obj, tin, tout = _label(client, model, _item_text(row), temperature)
        meta = row["metadata"]
        rec = {"scenario_id": meta["scenario_id"], "trait_id": meta["trait_id"],
               "domain": meta.get("domain", ""), **obj}
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
        return tin, tout

    usage = map_threaded(run, len(todo), max_workers=workers, desc="labelling")
    fh.close()
    tok_in = sum(u[0] for u in usage)
    tok_out = sum(u[1] for u in usage)
    cost = tok_in / 1e6 * PRICE_IN + tok_out / 1e6 * PRICE_OUT

    print(f"\nlabelled {len(todo)} rows -> {feat_path}")
    print(f"tokens in={tok_in:,} out={tok_out:,} | cost upper bound ${cost:.2f} "
          f"(billed prompt tokens include cache reads at 0.1x, so actual is lower)")
    if todo:
        print(f"per row: {tok_in / len(todo):.0f} in, {tok_out / len(todo):.0f} out, "
              f"${cost / len(todo):.4f}")
    if smoke:
        print("\n--- first labelled row ---")
        print(json.dumps(json.loads(feat_path.read_text().splitlines()[-len(todo)]), indent=2))

    write_run_meta(out, {"input": input, "model": model, "temperature": temperature,
                         "rows": len(rows), "labelled_this_run": len(todo),
                         "tokens_in": tok_in, "tokens_out": tok_out, "cost_upper_bound_usd": cost,
                         "command": " ".join(sys.argv)})


if __name__ == "__main__":
    fire.Fire(main)
