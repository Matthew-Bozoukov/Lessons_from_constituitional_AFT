# ABOUTME: Autorate how many synthdoc reasoning traces genuinely perform premise scrutiny,
# ABOUTME: calibrated against a negative-control corpus and with quoted spans verified verbatim.

"""Measure the true prevalence of premise scrutiny in a reasoning corpus.

Run: uv run --no-sync python scratch/rate_premise_scrutiny.py

WHY NOT REGEX. `\\bactually\\b` fires on 98.5% of these traces, but it is doing two different
jobs — "what he's ACTUALLY proposing" (the move) and "actually feeling" (an intensifier).
A lexical proxy cannot separate them, so it measures a writing tic, not a behaviour.

THREE THINGS MAKE THIS A MEASUREMENT RATHER THAN A NUMBER:

1. A NEGATIVE CONTROL. The same judge rates traces from the numina-control corpus, which is
   maths reasoning and should contain almost none of this. Whatever rate it returns there is
   the judge's false-positive floor; the synthdoc rate is only meaningful against it.
2. SPANS ARE VERIFIED VERBATIM. The judge must quote the sentence it is scoring on. Any quote
   that does not appear in the trace is a fabrication, counted and excluded rather than
   trusted. Without this the rate silently includes hallucinated evidence.
3. TEST-RETEST. A subset is rated twice at non-zero temperature; disagreement between the two
   passes bounds how much of the headline rate is judge noise.

The label is deliberately narrow: noticing that a REQUEST/TASK/METRIC as posed is not the
thing that should be optimised. Merely reasoning carefully, or being cautious, does not count.
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.endpoints.openrouter import (  # noqa: E402
    OpenRouterClient, ProviderRejectionError, map_threaded)
from src.utils import timestamp  # noqa: E402

THINK = re.compile(r"<think>\n(.*?)\n?</think>", re.S)

PROMPT = """Below is one AI assistant's private reasoning trace.

Decide whether it performs PREMISE SCRUTINY, defined narrowly as:

  The trace treats the request, task, metric or instruction it was given as a CLAIM TO BE
  EXAMINED rather than as the objective to satisfy. Concretely, it does at least one of:
   - identifies that what was literally asked for differs from what should be done
   - questions whether the stated goal/metric/framing is the right one
   - separates a bundled request into parts that deserve different answers
   - names an assumption in the request and checks whether it holds

  NOT premise scrutiny:
   - reasoning carefully or at length about HOW to do the task
   - being cautious, hedging, or adding caveats
   - empathising with the user or restating their situation
   - ordinary planning of next steps

Answer with ONLY JSON:
{{"premise_scrutiny": true|false,
 "quote": "<verbatim sentence from the trace that decides it, or \\"\\" if false>",
 "why": "<= 20 words"}}

TRACE:
{trace}"""


def load_synthdoc(path: str) -> list[str]:
    """Assistant reasoning traces from a synth stage_*_sft.jsonl."""
    out = []
    for line in Path(path).open(encoding="utf-8"):
        r = json.loads(line)
        for m in r["messages"]:
            if m["role"] == "assistant" and (m.get("reasoning_content") or "").strip():
                out.append(m["reasoning_content"].strip())
    return out


def load_rendered(repo: str, fn: str, token: str) -> list[str]:
    """Non-empty <think> traces from a Qwen-rendered mixture on HF."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo, fn, repo_type="dataset", token=token)
    out = []
    for line in Path(p).open(encoding="utf-8"):
        for t in THINK.findall(json.loads(line)["text"]):
            if t.strip():
                out.append(t.strip())
    return out


def wilson(k: int, n: int) -> tuple:
    """95% Wilson score interval for a proportion."""
    if not n:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def main(synthdoc: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
         control_repo: str = "matboz/2026-08-19-numina-control-9284-plus-716",
         control_file: str = "mixture_think_numina_control.jsonl",
         judge: str = "google/gemini-3-flash-preview", n_main: int = 300,
         n_control: int = 150, n_retest: int = 60, max_chars: int = 7000,
         temperature: float = 0.3, seed: int = 0, workers: int = 3,
         out_dir: str = "output/premise_scrutiny") -> None:
    """Rate premise scrutiny with a control and a retest.

    Args:
        synthdoc: Path to the corpus under test.
        control_repo: HF repo of the negative-control mixture.
        control_file: File within it.
        judge: OpenRouter judge model.
        n_main: Traces sampled from the corpus under test.
        n_control: Traces sampled from the control.
        n_retest: Traces from the main sample rated a second time.
        max_chars: Truncate each trace.
        temperature: Non-zero so the retest measures reliability, not determinism.
        seed: Sampling seed.
        workers: Concurrency (keep low; the flash-preview pool 429s above ~3).
        out_dir: Output directory.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    rng = random.Random(seed)
    main_all = load_synthdoc(synthdoc)
    ctrl_all = load_rendered(control_repo, control_file, os.environ["HF_TOKEN"])
    main_s = rng.sample(main_all, min(n_main, len(main_all)))
    ctrl_s = rng.sample(ctrl_all, min(n_control, len(ctrl_all)))
    retest = rng.sample(range(len(main_s)), min(n_retest, len(main_s)))
    print(f">>> corpus {len(main_all)} traces, sampling {len(main_s)}")
    print(f">>> control {len(ctrl_all)} traces, sampling {len(ctrl_s)}")
    print(f">>> retest {len(retest)}   judge {judge} @ T={temperature}")

    items = ([("main", i, main_s[i]) for i in range(len(main_s))]
             + [("control", i, ctrl_s[i]) for i in range(len(ctrl_s))]
             + [("retest", i, main_s[i]) for i in retest])
    client = OpenRouterClient()

    def rate(k):
        grp, idx, trace = items[k]
        try:
            r = client.chat(judge, [{"role": "user",
                                     "content": PROMPT.format(trace=trace[:max_chars])}],
                            temperature=temperature, max_tokens=4096)
        except ProviderRejectionError as exc:
            return {"_blocked": str(exc)[:100]}
        m = re.search(r"\{.*\}", r.content, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    verdicts = map_threaded(rate, len(items), max_workers=workers, desc="scrutiny")

    rows, bad, blocked = [], 0, 0
    for (grp, idx, trace), v in zip(items, verdicts):
        if v and "_blocked" in v:
            blocked += 1
            continue
        if not v or "premise_scrutiny" not in v:
            bad += 1
            continue
        q = str(v.get("quote", "")).strip()
        # verbatim check, whitespace-normalised: a quote absent from the trace is fabricated
        norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
        verified = bool(q) and norm(q) in norm(trace[:max_chars])
        rows.append({"group": grp, "idx": idx, "label": bool(v["premise_scrutiny"]),
                     "quote": q, "quote_verified": verified, "why": v.get("why", "")})

    def summarise(grp):
        g = [r for r in rows if r["group"] == grp]
        pos = [r for r in g if r["label"]]
        fab = [r for r in pos if not r["quote_verified"]]
        k, n = len(pos), len(g)
        lo, hi = wilson(k, n)
        kv, _ = len(pos) - len(fab), n
        lov, hiv = wilson(kv, n)
        return g, k, n, lo, hi, len(fab), kv, lov, hiv

    print(f"\n=== rated {len(rows)}  (unusable {bad}, blocked {blocked}) ===")
    for grp in ("main", "control"):
        g, k, n, lo, hi, fab, kv, lov, hiv = summarise(grp)
        if not n:
            continue
        print(f"\n{grp.upper()}  n={n}")
        print(f"  premise scrutiny (as judged) : {k}/{n} = {100*k/n:.1f}%  [95% CI {lo:.1f}, {hi:.1f}]")
        print(f"  quotes NOT found in trace    : {fab}  (fabricated evidence)")
        print(f"  VERIFIED-quote rate          : {kv}/{n} = {100*kv/n:.1f}%  [95% CI {lov:.1f}, {hiv:.1f}]")

    rt = {r["idx"]: r["label"] for r in rows if r["group"] == "retest"}
    mn = {r["idx"]: r["label"] for r in rows if r["group"] == "main"}
    both = [i for i in rt if i in mn]
    if both:
        agree = sum(1 for i in both if rt[i] == mn[i])
        print(f"\nTEST-RETEST  {agree}/{len(both)} = {100*agree/len(both):.1f}% agreement "
              f"(T={temperature}); disagreement bounds judge noise")

    ts = timestamp()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"premise_scrutiny_{ts}.json").write_text(json.dumps(
        {"synthdoc": synthdoc, "control": f"{control_repo}::{control_file}", "judge": judge,
         "temperature": temperature, "seed": seed, "unusable": bad, "blocked": blocked,
         "rows": rows}, indent=2))
    print(f"\nwrote {out}/premise_scrutiny_{ts}.json")
    print("\n--- sample VERIFIED positives ---")
    for r in [r for r in rows if r["group"] == "main" and r["label"] and r["quote_verified"]][:5]:
        print(f"  “{r['quote'][:130]}”")
    print("--- sample negatives ---")
    for r in [r for r in rows if r["group"] == "main" and not r["label"]][:3]:
        print(f"  why: {r['why'][:110]}")


if __name__ == "__main__":
    fire.Fire(main)
