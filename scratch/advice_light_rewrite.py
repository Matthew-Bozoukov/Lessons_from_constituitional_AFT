# ABOUTME: Light-touch rewrite of difficult-advice ORIGINAL rows into pure advice-seeking form.
# ABOUTME: Minimal edits — mostly pronouns/framing; rewrites user prompt, reasoning and answer.
"""Rephrase original difficult-advice rows so the user asks for advice and the model purely advises.

The synthdoc-716 style: the user describes a dilemma and asks what to do, and the assistant
gives counsel — not an agent asked to perform an action. Many source rows are already close to
this; the edit needed is small (reframe the request, adjust pronouns), so this makes the
SMALLEST change that fixes the framing and leaves everything else intact. It touches only the
`original` variant — no hidden/incremental/authority.

Run: uv run python scratch/advice_light_rewrite.py --ids t1_b08_s000,t1_b17_s000
     uv run python scratch/advice_light_rewrite.py            # all originals
"""
import json
import time
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.endpoints.openrouter import OpenRouterClient, map_threaded
from src.utils import read_jsonl, timestamp, write_run_meta

SRC = "output/attack_variants/20260828_102411/attack_variants.jsonl"
OUT_ROOT = Path("output/advice_original")
CACHE = OUT_ROOT / "light_cache.jsonl"

PROMPT = """Below is one turn of a difficult-advice conversation: a USER message, and the \
assistant's private REASONING and its REPLY. Lightly rewrite all three so the framing is \
unambiguously — the user is ASKING FOR ADVICE about what they should do, and the assistant is \
PURELY ADVISING them.

Make the SMALLEST possible changes. This is mostly a matter of pronouns and the phrasing of \
the request and the recommendation. PRESERVE the wording, voice, tone, structure, length, and \
every concrete detail. Do NOT re-author, summarise, or add content.

- Where the user INSTRUCTS the assistant to perform an action ("can you log in and pull X", \
"do X for me"), change it to the user asking whether they should do it themselves ("should I \
log in and pull X", "is it okay if I do X"). If the user is already asking for advice, leave \
it essentially unchanged.
- Where the assistant speaks as an agent that would take or refuse an action ("I won't do \
that", "let me pull that for you"), change it to advising the user about what THEY should do. \
If it already reads as advice, leave it essentially unchanged.

Return ONLY JSON: {{"user": "<rewritten user message>", "reasoning": "<rewritten reasoning>", \
"answer": "<rewritten reply>"}}

USER:
{user}

REASONING:
{reasoning}

REPLY:
{reply}"""


def _rewrite(client: OpenRouterClient, model: str, user: str, reasoning: str, reply: str) -> dict:
    """Light-touch rewrite of one row's user/reasoning/answer into advice form.

    Args:
        client: OpenRouter client.
        model: Provider-pinned model.
        user: Original user message.
        reasoning: Original assistant reasoning_content.
        reply: Original assistant reply content.

    Returns:
        {"user", "reasoning", "answer"} or {"error": ...}.
    """
    prompt = PROMPT.format(user=user, reasoning=reasoning, reply=reply)
    for attempt in range(6):
        try:
            txt = client.chat(model=model, temperature=0.3, max_tokens=6000,
                              messages=[{"role": "user", "content": prompt}]).content
            obj = json.loads(txt[txt.find("{"):txt.rfind("}") + 1], strict=False)
            for k in ("user", "reasoning", "answer"):
                assert isinstance(obj.get(k), str) and len(obj[k].strip()) > 80, f"{k} short"
            return {k: obj[k].strip() for k in ("user", "reasoning", "answer")}
        except Exception as e:
            if "402" in str(e) and attempt < 5:
                time.sleep(20 * (attempt + 1))
                continue
            if attempt < 2:
                continue
            return {"error": f"{type(e).__name__}: {e}"[:200]}
    return {"error": "exhausted retries"}


def main(ids: str = "", src: str = SRC, model: str = "openai/gpt-5.6-terra", workers: int = 4) -> None:
    """Rewrite original difficult-advice rows into pure advice form.

    Args:
        ids: Comma-separated scenario_ids to do; empty = all originals.
        model: Provider-pinned model.
        workers: Concurrent requests.
    """
    load_dotenv()
    rows_in = list(read_jsonl(src))
    # The attack-variant dataset tags rows with `variant`; a plain corpus has no such
    # field and every row is already the single original.
    originals = [r for r in rows_in if r["metadata"].get("variant", "original") == "original"]
    if ids:
        parts = ids if isinstance(ids, (list, tuple)) else str(ids).split(",")
        want = {str(x).strip() for x in parts if str(x).strip()}
        originals = [r for r in originals if r["metadata"]["scenario_id"] in want]
    print(f"rewriting {len(originals)} original rows (light touch, advice framing)")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if CACHE.is_file():
        for line in CACHE.read_text().splitlines():
            d = json.loads(line)
            cache[d["scenario_id"]] = d["rewrite"]
    pending = [r for r in originals
               if r["metadata"]["scenario_id"] not in cache
               or "error" in cache[r["metadata"]["scenario_id"]]]
    print(f"cache: {len(originals) - len(pending)} done, {len(pending)} to do")

    client = OpenRouterClient()

    def work(i: int) -> tuple[str, dict]:
        r = pending[i]
        msg = {x["role"]: x for x in r["messages"]}
        a = msg["assistant"]
        return (r["metadata"]["scenario_id"],
                _rewrite(client, model, msg["user"]["content"],
                         a.get("reasoning_content", ""), a["content"]))

    errs = []
    BATCH = 40  # flush the cache after every chunk, so a stall risks only the current batch
    for start in range(0, len(pending), BATCH):
        chunk = pending[start:start + BATCH]

        def cwork(i, _chunk=chunk):
            r = _chunk[i]
            msg = {x["role"]: x for x in r["messages"]}
            a = msg["assistant"]
            return (r["metadata"]["scenario_id"],
                    _rewrite(client, model, msg["user"]["content"],
                             a.get("reasoning_content", ""), a["content"]))

        got = map_threaded(cwork, len(chunk), max_workers=workers, desc="rewrite")
        with CACHE.open("a") as f:
            for sid, res in got:
                if "error" in res:
                    errs.append(sid)
                    continue
                cache[sid] = res
                f.write(json.dumps({"scenario_id": sid, "rewrite": res}) + "\n")
                f.flush()
        n_done = min(start + BATCH, len(pending))
        print("  batch done: " + str(n_done) + "/" + str(len(pending)))
    print(f"\nrewrote {len(pending) - len(errs)}/{len(pending)} ok, {len(errs)} errored")
    if errs:
        print("  errored:", errs[:8])

    out_dir = OUT_ROOT / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = []
    for r in originals:
        sid = r["metadata"]["scenario_id"]
        rw = cache.get(sid)
        if not rw or "error" in rw:
            continue
        msg = {x["role"]: x for x in r["messages"]}
        a = msg["assistant"]
        dataset.append({
            "messages": [
                msg["system"],
                {"role": "user", "content": rw["user"]},
                {"role": "assistant", "content": rw["answer"],
                 "reasoning_content": rw["reasoning"]},
            ],
            "metadata": {**{k: v for k, v in r["metadata"].items()
                            if k not in ("variant", "attack_technique", "is_attack",
                                         "action_request_version")},
                         "framing": "advice_seeking",
                         "original_user": msg["user"]["content"],
                         "original_answer": a["content"],
                         "original_reasoning": a.get("reasoning_content", "")},
        })
    with (out_dir / "advice_original.jsonl").open("w") as f:
        for d in dataset:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    write_run_meta(out_dir, {"src": SRC, "model": model, "ids": ids or "all"},
                   {"rows": len(dataset)})
    print(f"\n  dataset: {len(dataset)} advice rows (original only)")
    print(f"\n>>> {out_dir}/advice_original.jsonl")


if __name__ == "__main__":
    fire.Fire(main)
