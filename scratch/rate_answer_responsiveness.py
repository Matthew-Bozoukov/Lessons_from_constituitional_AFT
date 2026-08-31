# ABOUTME: Rate whether an ablated reply still answers the question it was written for,
# ABOUTME: which the coherence check deliberately does not measure.
"""Does the reply still respond to what the person asked?

The coherence check only asks whether deletion left visible seams -- a fluent reply that no
longer addresses the request scores "coherent". This asks the question that matters for
training data: shown the request and the final reply, does the reply engage the thing that
was actually asked?

Three graded verdicts, because the interesting failure is partial:
  answers    -- takes a position on what was asked and says enough to act on
  partial    -- addresses it but leaves the person without what they needed
  unresponsive -- does not engage the request, or replies to something else

Judged blind: the rater is not told the reply was edited, and the pre-ablation reply is
scored in the same shuffled pool as a baseline, so a low score has something to be low
against.

Run: uv run python scratch/rate_answer_responsiveness.py
"""
import json
import random
import re
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.endpoints.openrouter import OpenRouterClient, map_threaded
from src.utils import extract_json, read_jsonl, timestamp, wilson, write_run_meta

USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)

PROMPT = """Below is a message someone sent an AI assistant, and the assistant's reply.

Judge ONLY whether the reply responds to what was asked. Do not judge whether the reply is
wise, safe, well-written or agreeable -- a firm refusal that explains its position and gives
the person something to act on still counts as answering.

- "answers": the reply takes a clear position on what was asked and says enough for the
  person to know where they stand and what happens next.
- "partial": the reply engages the request but stops short -- it states a position without
  the substance the person needed, or addresses only part of what was asked.
- "unresponsive": the reply does not engage the request at all, replies to something that
  was not asked, or is too thin to constitute a reply.

Return ONLY: {{"verdict": "answers"|"partial"|"unresponsive",
"note": "<one short clause of evidence>"}}

MESSAGE:
---
{user}
---

REPLY:
---
{reply}
---"""
VERDICTS = ("answers", "partial", "unresponsive")


def main(model: str = "google/gemini-3-flash-preview", seed: int = 0, workers: int = 24,
         out_dir: str = "output/answer_responsiveness"):
    """Rate final and pre-ablation replies blind, and report the split.

    Args:
        model: OpenRouter model id for the rater.
        seed: RNG seed for shuffling.
        workers: Thread pool size.
        out_dir: Output directory root.
    """
    load_dotenv()
    stats = json.loads(
        Path("output/ruleform_ablated_716/mixture_stats_ablated.json").read_text())
    drop = set(stats["dropped_rows"])

    # final (post-ablation) replies, and the untouched originals for a baseline
    final = {}
    for d in sorted(Path("output/ablate_alternative").glob("answer_*")):
        for r in read_jsonl(d / "texts.jsonl"):
            if r["scenario_id"] not in drop:
                final[r["scenario_id"]] = r["after"]
    orig = {}
    for d in sorted(Path("output/invert_answer").glob(
            "engages_counterargument_outcome_branches_2*")):
        for r in read_jsonl(d / "replies.jsonl"):
            orig[r["scenario_id"]] = r["reply_original"]

    users = {}
    sft = {}
    for r in read_jsonl("output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl"):
        m = {x["role"]: x for x in r["messages"]}
        sft[m["user"]["content"].strip()] = r["metadata"]["scenario_id"]
    for row in read_jsonl("output/mixture_t2_9284_synthdoc_716/mixture_think.jsonl"):
        if row.get("source") != "synthdoc_difficult_advice":
            continue
        u = USER_RE.search(row["text"]).group(1).strip()
        users[sft[u]] = u

    sids = sorted(s for s in final if s in users and s in orig)
    print(f"rating {len(sids)} scenarios x 2 versions blind with {model}")

    items = [{"sid": s, "version": v, "reply": t[s]}
             for s in sids for v, t in (("original", orig), ("ablated", final))]
    random.Random(seed).shuffle(items)
    client = OpenRouterClient()

    def rate(i):
        """Rate one reply, retrying once on a malformed body."""
        content = PROMPT.format(user=users[items[i]["sid"]], reply=items[i]["reply"])
        for attempt in (0, 1):
            r = client.chat(model=model, temperature=0.0, max_tokens=300,
                            response_format={"type": "json_object"},
                            messages=[{"role": "user", "content": content + (
                                "\n\nSTRICT JSON ONLY." if attempt else "")}])
            try:
                out = extract_json(r.content)
                assert out["verdict"] in VERDICTS, out["verdict"]
                return out
            except Exception:
                if attempt:
                    raise

    res = map_threaded(rate, len(items), max_workers=workers, desc="rating")
    for it, r in zip(items, res):
        it.update(verdict=r["verdict"], note=r["note"])

    d = Path(out_dir) / timestamp()
    d.mkdir(parents=True, exist_ok=True)
    (d / "ratings.jsonl").write_text("\n".join(json.dumps(i) for i in items) + "\n")
    write_run_meta(d, {"model": model, "seed": seed}, {"n_scenarios": len(sids)})

    byv = {v: {i["sid"]: i for i in items if i["version"] == v}
           for v in ("original", "ablated")}
    n = len(sids)
    print(f"\n{'version':12s} " + " ".join(f"{v:>14}" for v in VERDICTS))
    for v in ("original", "ablated"):
        cells = []
        for k in VERDICTS:
            c = sum(1 for s in sids if byv[v][s]["verdict"] == k)
            lo, hi = wilson(c, n)
            cells.append(f"{c:>4} {100*c/n:>4.0f}% ")
        print(f"{v:12s} " + " ".join(f"{x:>14}" for x in cells))
    worse = sum(1 for s in sids
                if VERDICTS.index(byv["ablated"][s]["verdict"])
                > VERDICTS.index(byv["original"][s]["verdict"]))
    print(f"\nrows that got worse: {worse}/{n} ({100*worse/n:.0f}%)")
    bad = [s for s in sids if byv["ablated"][s]["verdict"] == "unresponsive"]
    print(f"unresponsive after ablation: {len(bad)}")
    for s in bad[:8]:
        print(f"   {s}: {byv['ablated'][s]['note'][:80]}")
    import collections
    tr = collections.Counter(s.split("_")[0] for s in bad)
    print("unresponsive by trait:", dict(sorted(tr.items())))
    print(f"\nwrote {d}")


if __name__ == "__main__":
    fire.Fire(main)
