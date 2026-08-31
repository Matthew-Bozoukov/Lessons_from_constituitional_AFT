# ABOUTME: GSM8K accuracy + reasoning-trace length for a vLLM-served Qwen3.6 LoRA vs its base,
# ABOUTME: driven through /v1/completions so the raw <think> block is measurable, not parsed away.

"""Did fine-tuning degrade reasoning? Measure it on held-out GSM8K.

Run: uv run --no-sync python scratch/gsm8k_reasoning_vllm.py \\
         --base_url https://<pod>-8000.proxy.runpod.net/v1 --models base,courtroom716

WHY /v1/completions AND NOT /v1/chat/completions. With `--reasoning-parser qwen3` the chat
endpoint strips the think block out of `content`, and because the serve-time template
PREFILLS `<think>\\n` the parser also leaves `reasoning_content` empty (verified on this
stack 2026-08-16). The trace is then unmeasurable — and worse, a truncated trace yields an
empty answer that scores a false 0%. Rendering the prompt here and reading raw text back
avoids both: the trace and the answer are separated by this script, from bytes it can see.

TWO METRICS, and the second is the point. Accuracy alone cannot distinguish "worse at
maths" from "stopped reasoning". Both arms are scored on accuracy AND trace length, with
the empty-trace rate reported explicitly (CLAUDE.md gotcha 4).
"""

import json
import re
import statistics as st
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fire
import requests

PROMPT = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
          "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n")
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def split_trace(raw: str) -> tuple[str, str]:
    """Separate the reasoning trace from the visible answer.

    The prompt prefills `<think>\\n`, so the completion starts INSIDE the trace and the
    boundary is the closing tag.

    Args:
        raw: Raw completion text.

    Returns:
        (trace, answer). If the model never closed the tag the whole completion is the
        trace and the answer is empty — the truncation case that must not be scored as a
        wrong answer without being counted separately.
    """
    if "</think>" in raw:
        trace, _, answer = raw.partition("</think>")
        return trace.strip(), answer.strip()
    return raw.strip(), ""


def extract(text: str) -> str | None:
    """Pull the final number out of an answer, preferring a \\boxed{} result."""
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    nums = _NUM.findall(boxed[-1] if boxed else text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def main(base_url: str, models: str = "base", n: int = 60, max_tokens: int = 2000,
         concurrency: int = 8, seed: int = 0,
         out_dir: str = "output/gsm8k_reasoning") -> None:
    """Score each model on the same held-out GSM8K questions.

    Args:
        base_url: vLLM OpenAI-compatible base URL.
        models: Comma-separated served model names, e.g. "base,courtroom716".
        n: Questions (shuffled with `seed`, so every arm sees the same ones).
        max_tokens: Generation cap. Must fit trace + answer; too tight truncates inside
            the trace and scores a false 0%.
        concurrency: Parallel requests.
        seed: Shuffle seed.
        out_dir: Output directory for the JSON and markdown mirror.
    """
    import random

    from datasets import load_dataset

    from src.utils import timestamp

    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    rows = [ds[i] for i in idx[:n]]
    gold = [r["answer"].split("####")[-1].strip().replace(",", "") for r in rows]
    print(f">>> {len(rows)} held-out GSM8K questions (seed {seed}), max_tokens={max_tokens}")

    def ask(model: str, q: str) -> str:
        r = requests.post(f"{base_url}/completions", timeout=900, json={
            "model": model, "prompt": PROMPT.format(q=q),
            "max_tokens": max_tokens, "temperature": 0.0})
        r.raise_for_status()
        return r.json()["choices"][0]["text"]

    # fire turns "a,b" into a TUPLE, not a string, so accept either form.
    names = ([str(m).strip() for m in models] if isinstance(models, (list, tuple))
             else [m.strip() for m in str(models).split(",")])
    names = [m for m in names if m]

    summary, records = [], {}
    for model in names:
        print(f">>> {model}")
        with ThreadPoolExecutor(concurrency) as ex:
            raws = list(ex.map(lambda r: ask(model, r["question"]), rows))
        recs = []
        for raw, g in zip(raws, gold):
            trace, answer = split_trace(raw)
            pred = extract(answer)
            recs.append({"trace_chars": len(trace), "answer_chars": len(answer),
                         "pred": pred, "gold": g, "correct": pred == g,
                         "unclosed_think": answer == "", "trace": trace, "answer": answer})
        records[model] = recs
        tr = [r["trace_chars"] for r in recs]
        acc = 100 * sum(r["correct"] for r in recs) / len(recs)
        s = {"model": model, "n": len(recs), "accuracy": acc,
             "trace_median": int(st.median(tr)), "trace_mean": int(st.mean(tr)),
             "trace_min": min(tr), "trace_max": max(tr),
             "empty_trace": sum(1 for x in tr if x == 0),
             "unclosed_think": sum(r["unclosed_think"] for r in recs),
             "no_answer_parsed": sum(1 for r in recs if r["pred"] is None)}
        summary.append(s)
        print(f"    accuracy {acc:.1f}%   trace median {s['trace_median']} chars   "
              f"unclosed {s['unclosed_think']}   empty-trace {s['empty_trace']}")

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    ts = timestamp()
    (d / f"gsm8k_reasoning_{ts}.json").write_text(json.dumps(
        {"base_url": base_url, "n": n, "seed": seed, "max_tokens": max_tokens,
         "summary": summary, "records": records}, indent=2))
    md = [f"# GSM8K accuracy + reasoning length ({n} questions, seed {seed})", "",
          "| model | accuracy | trace median | trace mean | unclosed `</think>` | empty trace | unparsed answer |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for s in summary:
        md.append(f"| {s['model']} | {s['accuracy']:.1f}% | {s['trace_median']} | "
                  f"{s['trace_mean']} | {s['unclosed_think']} | {s['empty_trace']} | "
                  f"{s['no_answer_parsed']} |")
    (d / f"gsm8k_reasoning_{ts}.md").write_text("\n".join(md))
    print(f"\nwrote {d}/gsm8k_reasoning_{ts}.json and .md")


if __name__ == "__main__":
    fire.Fire(main)
