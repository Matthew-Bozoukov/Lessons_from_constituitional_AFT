# ABOUTME: Measure whether the Tinker gpt-oss fine-tune lost reasoning ability, by A/Bing
# ABOUTME: base vs adapter on held-out GSM8K for accuracy AND analysis-channel usage.

"""Did the fine-tune stop the model reasoning?

Run: TINKER_API_KEY=... uv run --no-sync python scratch/tinker_reasoning_eval.py \\
         --checkpoint tinker://.../sampler_weights/<name> [--n 60]

TWO METRICS, AND THE SECOND IS THE POINT. Accuracy alone cannot distinguish "the model got
worse at maths" from "the model stopped opening the analysis channel and now answers from
the hip". Harmony encodes did-not-reason by OMITTING the analysis channel, and 8,280 of the
10,000 training rows have no analysis channel, so suppression is the specific regression to
look for. Both arms are therefore scored on:

  accuracy       - exact match on GSM8K's final number
  analysis_rate  - fraction of answers that opened an analysis channel at all
  analysis_chars - how much reasoning, when there is any

GSM8K is held out: it is in none of the training sources (no_robots, tulu3_if,
self_oss_instruct, numinamath_cot, smol_*, apigen, lima, longalign, difficult_advice_v2).
numinamath overlaps in DOMAIN, which is the point -- if maths reasoning degraded despite
1,037 maths rows, that is a strong result.

Greedy decoding (temperature 0) so the comparison is not sampling noise.
"""

import asyncio
import json
import re
from pathlib import Path

import fire

BASE = "openai/gpt-oss-120b"
ANALYSIS = "<|channel|>analysis<|message|>"
FINAL = "<|channel|>final<|message|>"


# The fine-tune emits this INSTEAD of "<|end|><|start|>assistant<|channel|>final<|message|>".
# It is malformed -- no role, no <|channel|> -- and appears in none of the 10,000 training
# rows, so the model invented it. Scored separately: an answer behind a broken header is
# still a correct answer, and conflating the two would report a reasoning loss that is
# really a formatting loss.
MALFORMED_FINAL = "<|start|>final<|message|>"


def parse_channels(out: str) -> tuple[str, str, bool]:
    """Split a Harmony completion into (analysis, final, malformed).

    Args:
        out: Decoded model output.

    Returns:
        (analysis text, final text, whether the final header was malformed). Analysis is
        "" when the channel was never opened, which is one of the signals this eval counts.
    """
    analysis, final, malformed = "", out, False
    marker = FINAL if FINAL in out else (MALFORMED_FINAL if MALFORMED_FINAL in out else None)
    if marker:
        malformed = marker is MALFORMED_FINAL
        head, _, final = out.partition(marker)
        if ANALYSIS in head:
            analysis = head.split(ANALYSIS, 1)[1]
    elif ANALYSIS in out:
        # Ran out of tokens inside the trace: no answer was ever produced.
        analysis, final = out.split(ANALYSIS, 1)[1], ""
    for tag in ("<|return|>", "<|end|>", "<|call|>"):
        final = final.replace(tag, "")
        analysis = analysis.replace(tag, "")
    return analysis.strip(), final.strip(), malformed


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text: str) -> str | None:
    """Pull GSM8K's numeric answer out of a free-form reply.

    Prefers a \\boxed{} result, then the last number in the text, matching how GSM8K is
    conventionally scored.

    Args:
        text: The final-channel answer.

    Returns:
        Normalised number string, or None if the reply contains no number.
    """
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    hay = boxed[-1] if boxed else text
    nums = _NUM.findall(hay)
    if not nums:
        return None
    return nums[-1].replace(",", "").rstrip(".").rstrip("0").rstrip(".") \
        if "." in nums[-1] else nums[-1].replace(",", "")


def gold_answer(ans: str) -> str:
    """Extract the reference answer from a GSM8K `answer` field (after '####')."""
    return ans.split("####")[-1].strip().replace(",", "")


async def _sample(client, tok, question: str, max_tokens: int) -> tuple[str, str]:
    """Sample one greedy completion and split its channels."""
    import tinker

    text = tok.apply_chat_template([{"role": "user", "content": question}],
                                   tokenize=False, add_generation_prompt=True)
    inp = tinker.types.ModelInput.from_ints(tok.encode(text, add_special_tokens=False))
    fut = client.sample(prompt=inp, num_samples=1,
                        sampling_params=tinker.types.SamplingParams(
                            max_tokens=max_tokens, temperature=0.0))
    res = await fut.result_async() if hasattr(fut, "result_async") else fut.result()
    seq = res.sequences[0] if hasattr(res, "sequences") else res.samples[0]
    return parse_channels(tok.decode(seq.tokens))


async def _arm(name, client, tok, rows, max_tokens, concurrency):
    """Score one arm over all questions, bounded concurrency."""
    sem = asyncio.Semaphore(concurrency)
    out = [None] * len(rows)

    async def one(i, r):
        async with sem:
            analysis, final, malformed = await _sample(client, tok, r["question"],
                                                       max_tokens)
        out[i] = {"i": i, "analysis": analysis, "final": final, "malformed": malformed,
                  "pred": extract_answer(final), "gold": gold_answer(r["answer"])}
        done = sum(x is not None for x in out)
        if done % 10 == 0:
            print(f"  [{name}] {done}/{len(rows)}")

    await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)))
    return out


def _summarise(name: str, recs: list[dict]) -> dict:
    """Compute accuracy and analysis-channel statistics for one arm."""
    n = len(recs)
    correct = sum(r["pred"] is not None and r["pred"] == r["gold"] for r in recs)
    with_an = sum(bool(r["analysis"]) for r in recs)
    empty_final = sum(not r["final"] for r in recs)
    lens = [len(r["analysis"]) for r in recs if r["analysis"]]
    malformed = sum(r["malformed"] for r in recs)
    return {"arm": name, "n": n, "accuracy": correct / n, "correct": correct,
            "analysis_rate": with_an / n, "n_analysis": with_an,
            "mean_analysis_chars": (sum(lens) / len(lens)) if lens else 0.0,
            "empty_final": empty_final, "malformed_final": malformed,
            "malformed_rate": malformed / n}


async def _run(checkpoint, n, max_tokens, concurrency, out_dir, seed):
    """Load GSM8K, score both arms, write results."""
    import random

    import tinker
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    rows = [ds[i] for i in idx[:n]]
    print(f">>> {len(rows)} held-out GSM8K questions (seed {seed})")

    svc = tinker.ServiceClient()
    adapter = svc.create_sampling_client(model_path=checkpoint)
    base = svc.create_sampling_client(base_model=BASE)
    tok = adapter.get_tokenizer()

    results = {}
    for name, cl in (("base", base), ("adapter", adapter)):
        print(f">>> sampling {name}")
        results[name] = await _arm(name, cl, tok, rows, max_tokens, concurrency)

    summ = [_summarise(k, v) for k, v in results.items()]
    print("\n| arm | n | accuracy | analysis rate | mean analysis chars | "
          "malformed final | no answer |")
    print("|---|---|---|---|---|---|---|")
    for s in summ:
        print(f"| {s['arm']} | {s['n']} | {s['accuracy']*100:.1f}% "
              f"({s['correct']}/{s['n']}) | {s['analysis_rate']*100:.1f}% "
              f"({s['n_analysis']}/{s['n']}) | {s['mean_analysis_chars']:.0f} | "
              f"{s['malformed_rate']*100:.1f}% ({s['malformed_final']}/{s['n']}) | "
              f"{s['empty_final']} |")

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(
        {"checkpoint": checkpoint, "n": n, "seed": seed, "max_tokens": max_tokens,
         "summary": summ}, indent=2))
    (d / "records.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {d}/summary.json and {d}/records.json")


def main(checkpoint: str, n: int = 60, max_tokens: int = 1600,
         concurrency: int = 8, out_dir: str = "output/gptoss_reasoning",
         seed: int = 0) -> None:
    """Compare base and adapter on held-out GSM8K.

    Args:
        checkpoint: `tinker://.../sampler_weights/<name>` path.
        n: Number of held-out questions.
        max_tokens: Generation cap. Must leave room for a full trace plus the answer —
            too tight truncates inside `analysis` and scores a false 0%.
        concurrency: Concurrent samples per arm.
        out_dir: Where to write summary.json and records.json.
        seed: Shuffle seed for question selection.
    """
    asyncio.run(_run(checkpoint, n, max_tokens, concurrency, out_dir, seed))


if __name__ == "__main__":
    fire.Fire(main)
