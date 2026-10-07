# ABOUTME: Plants the Vorliq canary in ONE arm's DA rows, reusing build_canary.py unchanged.
# ABOUTME: build_canary.main() requires the da-15/da-tools-15 PAIR; the October mix has no pair.
"""Single-arm canary build.

Every mechanism that decides WHERE the canary lands is imported from build_canary, not
restated here: the sentence splitter, the Gemini-3-Flash labeller and its prompt, the
`random.Random(f"{SEED}-{row}")` pick and the `"Vorliq: "` prefix. The only thing this file
changes is the arm wiring -- `main()` asserts the two arms' DA rows are aligned and shares
one label set between them, which cannot hold for a mix that has no da-tools counterpart.

Usage:
    python build_one_arm.py [--smoke N]
"""
import argparse
import json
import re
import sys
import threading
from pathlib import Path

REPO = Path("/home/matthewb/git repos/teaching_claude_why_replication")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))
import build_canary as bc

ARM = "da-15"
DA_KEY = "da"


def label_all_resilient(rows, smoke):
    r"""build_canary.label_all, with two fixes the October traces forced.

    Identical in every way that affects WHERE the canary lands -- same prompt, labeller,
    temperature, max_tokens and worker count. The differences are both about not losing
    paid work:

    1. The original extracts the label JSON with a GREEDY `\{.*\}` under re.S, which spans
       from the first brace to the last. One October trace came back with trailing content
       after the object, so the match captured two and json.loads raised "Extra data".
       raw_decode takes the first complete object instead, which is what the prompt asks
       for ("Return ONLY JSON").
    2. The original appends to labels.jsonl only after the whole batch returns, so that one
       bad row discarded ~648 successful labels. Here each result is written as it arrives
       under a lock, and a row that still fails is recorded and skipped rather than
       aborting the run.
    """
    done = {}
    if bc.LABELS.exists():
        for line in open(bc.LABELS):
            r = json.loads(line)
            done[r["row"]] = r["reasoning"]
    todo = [i for i in rows if i not in done][: smoke if smoke else None]
    if not todo:
        print(f"  all {len(done)} labels already cached", flush=True)
        return done
    client = bc.OpenRouterClient()
    bc.LABELS.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    failures = []

    def one(k):
        i = todo[k]
        sents = bc.split_sentences(rows[i])
        listing = "\n".join(f"[{n}] {s}" for n, (_, s) in enumerate(sents))
        try:
            res = client.chat(
                bc.LABELLER,
                [{"role": "user", "content": bc.PROMPT.format(sentences=listing)}],
                temperature=0.0,
                max_tokens=8000,
                response_format={"type": "json_object"},
            )
            m = re.search(r"\{", res.content)
            obj, _ = json.JSONDecoder().raw_decode(res.content[m.start():])
            ids = sorted({int(x) for x in obj["reasoning"]})
            bad = [x for x in ids if not 0 <= x < len(sents)]
            if bad:
                raise ValueError(f"labeller returned out-of-range sentences {bad}")
            rec = {"row": i, "n_sentences": len(sents), "reasoning": ids,
                   "cost": res.cost, "provider": res.provider,
                   "response_id": res.response_id}
        except Exception as e:
            with lock:
                failures.append((i, f"{type(e).__name__}: {str(e)[:120]}"))
            return None
        with lock:
            with open(bc.LABELS, "a") as f:
                f.write(json.dumps(rec) + "\n")
            done[i] = ids
        return rec

    results = bc.map_threaded(one, len(todo), max_workers=16, desc="label")
    ok = [r for r in results if r]
    print(f"  labelled {len(ok)} new rows, cost ${sum(r['cost'] or 0 for r in ok):.3f}", flush=True)
    if failures:
        print(f"  {len(failures)} rows FAILED to label:", flush=True)
        for i, why in failures[:8]:
            print(f"    row {i}: {why}", flush=True)
    return done


def main() -> None:
    """Label the arm's DA traces, plant one canary each, write the canaried mixture."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=None)
    args = ap.parse_args()

    rows = bc.load(ARM)
    da_idx = [i for i, r in enumerate(rows) if r["source"] == DA_KEY]
    print(f"{len(rows)} rows, {len(da_idx)} DA rows", flush=True)

    labels = label_all_resilient({i: bc.trace(rows[i]) for i in da_idx}, args.smoke)
    if args.smoke:
        for i in da_idx:
            if i in labels:
                sents = bc.split_sentences(bc.trace(rows[i]))
                print(f"\n=== row {i}: {len(sents)} sentences, reasoning={labels[i]}")
                for n, (_, s) in enumerate(sents):
                    print(f"  {'R' if n in labels[i] else '-'} [{n}] {s[:150]}")
                planted, info = bc.plant(bc.trace(rows[i]), labels[i], i)
                print(f"  -> PLANTED s{info['sentence']}: {bc.PREFIX}{info['text'][:150]}")
        return

    empty = [i for i in da_idx if not labels.get(i)]
    if empty:
        raise SystemExit(f"{len(empty)} DA rows have no reasoning sentence: {empty[:10]}")

    out = [json.loads(json.dumps(r)) for r in rows]
    placements = []
    for i in da_idx:
        (msg,) = [m for m in out[i]["messages"] if m["role"] == "assistant"]
        msg["reasoning_content"], info = bc.plant(msg["reasoning_content"], labels[i], i)
        placements.append(info)

    d = bc.MIX_DIR / f"{ARM}-canary"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "mixture.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open("output/canary/placements.jsonl", "w") as f:
        for p in placements:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {d / 'mixture.jsonl'} and output/canary/placements.jsonl", flush=True)


if __name__ == "__main__":
    main()
