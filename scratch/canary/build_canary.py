# ABOUTME: Plants the canary word at the start of ONE reasoning sentence in every DA trace of the
# ABOUTME: da-15 and da-tools-15 mixes (same sentence in both), labelled by Gemini 3 Flash, then verifies.
# Run: uv run python scratch/canary/build_canary.py [--smoke N]   (after scratch/canary/pull_mixes.py)
#
# Why the mixes and not the synth corpora: the mix builder selects rows by supervised-token
# budget, so re-mixing a corpus with a few extra tokens per row could pick different rows. Editing
# the two published mixes in place keeps every row, position and non-DA byte identical, and the
# only difference between the arms stays the `tools` field on the DA rows.
import argparse
import json
import random
import re
from pathlib import Path

from dotenv import load_dotenv

from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

load_dotenv()

CANARY = "Vorliq"
PREFIX = f"{CANARY}: "
LABELLER = "google/gemini-3-flash-preview"
SEED = 0
MIX_DIR = Path("output/canary/mixes")
ARMS = {"da-15": "da", "da-tools-15": "da-tools"}  # arm -> DA source key in that mix
LABELS = Path("output/canary/labels.jsonl")

# A sentence starts after . ! ? (optionally closed by a quote/bracket) and whitespace, before
# an uppercase letter, quote, bracket, markdown emphasis or digit; paragraph breaks always split.
_SPLIT = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"'”’)\]]))\s+(?=[A-Z\"'“‘(*\[0-9])|\n\s*\n"
)

PROMPT = """Below is an AI assistant's private reasoning before it replied to a user, split into numbered sentences.

Label which sentences are REASONING sentences. A reasoning sentence weighs values, options, tradeoffs, consequences, risks or who is affected, or states a decision or judgement about what to do and why.

NOT reasoning: restating or summarising what the user said or asked; describing the situation without evaluating it; planning the reply's wording, structure, tone or format ("I'll open with...", "Structure: ..."); list headers or fragments.

Return ONLY JSON: {{"reasoning": [<sentence numbers>]}}

{sentences}"""


def split_sentences(text: str) -> list[tuple[int, str]]:
    """Return (start offset, sentence text) for each non-empty sentence, offsets into `text`."""
    out, pos = [], 0
    for m in list(_SPLIT.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        chunk = text[pos:end]
        stripped = chunk.lstrip()
        if stripped.strip():
            out.append((pos + len(chunk) - len(stripped), stripped.rstrip()))
        if m:
            pos = m.end()
    return out


def load(arm: str) -> list[dict]:
    return [json.loads(line) for line in open(MIX_DIR / arm / "mixture.jsonl")]


def trace(row: dict) -> str:
    (msg,) = [m for m in row["messages"] if m["role"] == "assistant"]
    return msg["reasoning_content"]


def label_all(rows: dict[int, str], smoke: int | None) -> dict[int, list[int]]:
    """row index -> reasoning sentence numbers; cached in LABELS so a rerun pays nothing twice."""
    done = {}
    if LABELS.exists():
        for line in open(LABELS):
            r = json.loads(line)
            done[r["row"]] = r["reasoning"]
    todo = [i for i in rows if i not in done][: smoke if smoke else None]
    client = OpenRouterClient()

    def one(k: int) -> dict:
        i = todo[k]
        sents = split_sentences(rows[i])
        listing = "\n".join(f"[{n}] {s}" for n, (_, s) in enumerate(sents))
        res = client.chat(
            LABELLER,
            [{"role": "user", "content": PROMPT.format(sentences=listing)}],
            temperature=0.0,
            max_tokens=8000,
            response_format={"type": "json_object"},
        )
        ids = json.loads(re.search(r"\{.*\}", res.content, re.S).group(0))["reasoning"]
        ids = sorted({int(x) for x in ids})
        bad = [x for x in ids if not 0 <= x < len(sents)]
        if bad:
            raise ValueError(f"row {i}: labeller returned out-of-range sentences {bad}")
        return {
            "row": i,
            "n_sentences": len(sents),
            "reasoning": ids,
            "cost": res.cost,
            "provider": res.provider,
            "response_id": res.response_id,
        }

    LABELS.parent.mkdir(parents=True, exist_ok=True)
    results = map_threaded(one, len(todo), max_workers=16, desc="label")
    with open(LABELS, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            done[r["row"]] = r["reasoning"]
    print(
        f"labelled {len(results)} new rows, cost ${sum(r['cost'] or 0 for r in results):.3f}"
    )
    return done


def plant(text: str, reasoning_ids: list[int], row: int) -> tuple[str, dict]:
    sents = split_sentences(text)
    pick = random.Random(f"{SEED}-{row}").choice(reasoning_ids)
    start, sent = sents[pick]
    return text[:start] + PREFIX + text[start:], {
        "row": row,
        "sentence": pick,
        "n_sentences": len(sents),
        "offset": start,
        "text": sent,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--smoke", type=int, default=None, help="label only N new rows and stop"
    )
    args = ap.parse_args()

    mixes = {arm: load(arm) for arm in ARMS}
    a, b = mixes["da-15"], mixes["da-tools-15"]
    da_idx = [i for i, r in enumerate(a) if r["source"] == "da"]
    assert [i for i, r in enumerate(b) if r["source"] == "da-tools"] == da_idx, (
        "DA rows not aligned"
    )
    for i in da_idx:
        assert trace(a[i]) == trace(b[i]), f"row {i}: reasoning differs between arms"
    print(f"{len(da_idx)} DA rows, aligned in both mixes")

    labels = label_all({i: trace(a[i]) for i in da_idx}, args.smoke)
    if args.smoke:
        for i in da_idx:
            if i in labels:
                sents = split_sentences(trace(a[i]))
                print(f"\n=== row {i}: {len(sents)} sentences, reasoning = {labels[i]}")
                for n, (_, s) in enumerate(sents):
                    print(f"  {'R' if n in labels[i] else '-'} [{n}] {s[:160]}")
        return

    empty = [i for i in da_idx if not labels.get(i)]
    if empty:
        raise SystemExit(
            f"{len(empty)} DA rows have no reasoning sentence (rows {empty[:10]}); "
            "decide how to handle them before building"
        )

    placements = []
    for arm, key in ARMS.items():
        out = [json.loads(json.dumps(r)) for r in mixes[arm]]
        for i in da_idx:
            (msg,) = [m for m in out[i]["messages"] if m["role"] == "assistant"]
            msg["reasoning_content"], info = plant(
                msg["reasoning_content"], labels[i], i
            )
            if arm == "da-15":
                placements.append(info)
        d = MIX_DIR / f"{arm}-canary"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "mixture.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {d / 'mixture.jsonl'}")
    with open("output/canary/placements.jsonl", "w") as f:
        for p in placements:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
