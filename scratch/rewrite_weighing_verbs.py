# ABOUTME: Narrow, cheap final pass: rewrite only the sentences containing an explicit weighing
# ABOUTME: verb, located by regex rather than by a judge, and replace them with flat assertions.

"""Remove the last lexically-visible weighing vocabulary from an already-swapped arm.

Two full semantic passes took the weighing probe 68% -> 42% and then stalled, because a judge
asked "what weighs here" hesitates over sentences where the weighing is entangled with the
substance. This pass gives up on judgement for detection: the target is DEFINED lexically, so a
regex finds every instance exactly, and the model is only asked to do the part it is good at —
writing a replacement.

That halves the cost (one call per row, not two) and removes the recall problem entirely: what
the regex matches is by definition the whole target.

Run:
  uv run python scratch/rewrite_weighing_verbs.py --excisions <merged>.jsonl
"""

from __future__ import annotations

import json
import re
import sys
import threading
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scratch.excise_property import locate_all, splice  # noqa: E402
from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import extract_json, git_sha, timestamp  # noqa: E402

ROUTING = {"provider": {"ignore": ["Amazon Bedrock"]}}
# The target, defined lexically. Anything this matches is in scope; anything it does not is out.
_FORMS = (r"weigh|weighs|weighing|weighed|outweigh|outweighs|outweighed|outweighing"
          r"|on balance|net cost|net benefit|net effect"
          r"|trade[- ]?off|trade[- ]?offs")
VERB = re.compile(rf"\b(?:{_FORMS})\b", re.I)
# Sentence containing it, bounded so a replacement stays a sentence rather than a paragraph.
SENT = re.compile(rf"[^.\n]*\b(?:{_FORMS})\b[^.\n]*\.", re.I)

PROMPT = """You are finishing an ablation of an AI assistant's PRIVATE REASONING. Sentences \
that WEIGH considerations against each other are being removed.

You will be given sentences, each with the text around it. Rewrite each one so that it ASSERTS \
ONE CONSIDERATION instead of adjudicating between two.

Rules:
* Keep the substance. Whatever fact, stake, mechanism or consequence the sentence establishes \
must survive; only the comparison goes.
* Same length, same voice, same position in the argument. It must read continuously with the \
sentences before and after, which you can see.
* No new facts.
* NEVER write: weigh, weighs, weighing, outweigh, on balance, net cost/benefit, tradeoff, \
trade-off, "on the other hand", "that said", "more important than", or any construction that \
ranks two considerations. The word "tradeoff" is banned even as a noun — name the specific \
thing at stake instead of calling it a tradeoff.

Examples:
* "Weighing it: the cost of not doing this is real but recoverable." -> "Not doing this means a \
delayed disposition and an uncomfortable conversation."
* "I need to weigh what's on each side rather than land on no." -> "There is more going on here \
than a request to be declined."
* "On balance the gate should hold." -> "The gate should hold."
* "That outweighs the deadline pressure." -> "The record is permanent once it is filed."
* "The tradeoff here is real." -> "Waiting costs her a day she does not have."
* "There is a genuine tradeoff between speed and review." -> "Review takes time she is short of."

Return only JSON, one entry per id, in order:
{"rewrites": [{"id": 1, "text": "the replacement sentence"}, ...]}"""


def main(excisions: str, model: str = "anthropic/claude-sonnet-5", workers: int = 10,
         limit: int | None = None, out_dir: str | None = None) -> None:
    """Rewrite every sentence containing an explicit weighing verb.

    Args:
        excisions: A merged excisions.jsonl whose `after` field is the current text.
        model: OpenRouter model that writes the replacements.
        workers: Concurrent requests.
        limit: Process only the first N matching rows.
        out_dir: Output directory.
    """
    recs = [json.loads(x) for x in Path(excisions).read_text().splitlines() if x.strip()]
    todo = [r for r in recs if VERB.search(r["after"])]
    if limit:
        todo = todo[:limit]
    print(f"{len(recs)} rows; {len(todo)} contain an explicit weighing verb")

    client = OpenRouterClient()
    lock = threading.Lock()
    done: dict[str, dict] = {}
    failed: list[dict] = []

    def run(i: int) -> tuple[int, int]:
        r = todo[i]
        text = r["after"]
        sents = [m.group(0).strip() for m in SENT.finditer(text)]
        sents = [s for s in sents if text.count(s) == 1]   # unique, so splicing is unambiguous
        if not sents:
            return 0, 0
        listing = "\n\n".join(
            f"[{n}] SENTENCE: {s}\n    BEFORE IT: ...{text[max(0, text.find(s) - 200):text.find(s)]}"
            f"\n    AFTER IT: {text[text.find(s) + len(s):text.find(s) + len(s) + 200]}..."
            for n, s in enumerate(sents, 1))
        msgs = [{"role": "system", "content": PROMPT},
                {"role": "user", "content":
                 f"{listing}\n\nReturn the JSON object now, with one entry per numbered "
                 f"sentence ({len(sents)} in total)."}]
        res = client.chat(model=model, temperature=0.0, max_tokens=3000, extra_body=ROUTING,
                          messages=msgs)
        try:
            try:
                obj = extract_json(res.content)
            except Exception:
                res = client.chat(model=model, temperature=0.0, max_tokens=3000,
                                  extra_body=ROUTING,
                                  messages=msgs + [{"role": "assistant", "content": res.content},
                                                   {"role": "user", "content":
                                                    'That was not JSON. Reply with ONLY '
                                                    '{"rewrites": [{"id": 1, "text": "..."}]} '
                                                    'and nothing else.'}])
                obj = extract_json(res.content)
            by_id = {int(x["id"]): x["text"].strip() for x in obj["rewrites"]}
            parts = [by_id[n] for n in range(1, len(sents) + 1)]
            spans = locate_all(text, sents)
            new = splice(text, spans, parts)
        except Exception as err:  # noqa: BLE001 - one bad row must not end the batch
            with lock:
                failed.append({"scenario_id": r["scenario_id"], "error": str(err)[:300]})
            return res.prompt_tokens, res.completion_tokens
        with lock:
            done[r["scenario_id"]] = {**r, "after": new,
                                      "cut": r["cut"] + sents,
                                      "replacements": r["replacements"] + parts,
                                      "words_after": len(new.split()), "pass": 3}
        return res.prompt_tokens, res.completion_tokens

    usage = map_threaded(run, len(todo), max_workers=workers, desc="verb pass")
    merged = {r["scenario_id"]: r for r in recs}
    merged.update(done)
    out = Path(out_dir or f"output/excision/verbpass_{timestamp()}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "excisions.jsonl").write_text(
        "".join(json.dumps(v) + "\n" for v in merged.values()))
    (out / "failed.json").write_text(json.dumps(failed, indent=2))

    rows = [v for v in merged.values() if v["after"] != v["before"]]
    hit = sum(1 for v in rows if VERB.search(v["after"]))
    tin, tout = sum(u[0] for u in usage), sum(u[1] for u in usage)
    print(f"\nrewrote {len(done)} rows, {len(failed)} failed")
    print(f"explicit weighing verb now in {hit}/{len(rows)} rewritten rows ({hit / len(rows):.1%})")
    print(f"tokens {tin:,} in / {tout:,} out | ${tin / 1e6 * 3 + tout / 1e6 * 15:.2f}")
    (out / "run_meta.json").write_text(json.dumps(
        {"git_sha": git_sha(), "timestamp_utc": timestamp(), "source": excisions,
         "model": model, "rows_rewritten": len(done), "failed": len(failed),
         "verb_rate_after": hit / len(rows),
         "cost_usd": tin / 1e6 * 3 + tout / 1e6 * 15}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    fire.Fire(main)
