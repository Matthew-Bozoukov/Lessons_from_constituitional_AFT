# ABOUTME: Renders one scoring item exactly as the server saw it (pinned think template, preserve_thinking=false)
# ABOUTME: and finds the token positions where a canary could begin: the opener, paragraph starts, sentence starts.
# Imported by scratch/logit/score.py; `uv run python -m scratch.logit.positions` self-tests on the local turns.
#
# A position is the token index j-1 whose next-token distribution would have to produce the canary's first
# token for the canary to begin at character b of the rendered text: b is a paragraph start (after a newline,
# target 'Vor') or a mid-paragraph sentence start (b is the space before the sentence, target ' Vor'). Only
# boundaries that are also token boundaries count; the rest are reported as unaligned.
import json
import re
from pathlib import Path

from scratch.canary.build_canary import _SPLIT, split_sentences
from src.infra.endpoints.vllm import pin_prefix
from src.model_profile import render_chat
from scratch.logit.transcripts import BASE_TOOLS

OPENER = "opener"  # offset 0, right after the think opener; target 'Vor'
PARA = "para"      # right after a newline; target 'Vor'
LINE = "line"      # a sentence start after newline + indentation (a bullet); target ' Vor' (after the spaces)
SENT = "sent"      # a mid-line sentence start, after a space; target ' Vor'
NEWLINE_CLASSES = (OPENER, PARA, LINE)  # what the 'paragraph start' histograms pool


def pinned_tokenizer(base: str, revision: str | None = None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base, revision=revision)
    tok.chat_template = pin_prefix("think") + tok.chat_template
    return tok


def render_prefix(tok, item: dict, messages: dict | None) -> tuple[str, str]:
    """(text the model saw up to and including this turn's reasoning, the trimmed reasoning).

    ODCV: messages[:k] rendered with the generation prompt, then the reasoning as the server returned it.
    MASK: system + user rendered with the generation prompt, then the reasoning.
    The template trims reasoning when it re-renders earlier turns, and the think opener is '<think>\n', so
    the reasoning text that follows the opener is the trimmed one in both cases.
    """
    reasoning = item["reasoning"].strip()
    if item["eval"] == "odcv":
        msgs = messages[item["episode"]][: item["turn"]]
        head = render_chat(tok, msgs, BASE_TOOLS, render_kwargs={}, add_generation_prompt=True)
    else:
        msgs = [{"role": "system", "content": item["system"]}, {"role": "user", "content": item["user"]}]
        head = render_chat(tok, msgs, None, render_kwargs={}, add_generation_prompt=True)
    assert head.endswith("<think>\n"), head[-60:]
    return head + reasoning, reasoning


def boundaries(reasoning: str) -> list[tuple[int, str]]:
    """(char offset into the reasoning, class) for every canary-eligible start.

    opener: offset 0. para: a non-space char right after a newline. sent: a sentence start by the canary
    builder's own splitter that is neither, i.e. mid-paragraph after a space — offset is the SPACE, since the
    target token there is ' Vor'.
    """
    out = [(0, OPENER)]
    for m in re.finditer(r"\n+(?=\S)", reasoning):
        out.append((m.end(), PARA))
    para = {b for b, _ in out}
    for start, _ in split_sentences(reasoning):
        if start in para or start == 0 or reasoning[start - 1] != " ":
            continue
        ws = start - 1
        while ws > 0 and reasoning[ws - 1] == " ":
            ws -= 1
        out.append((start - 1, LINE if ws > 0 and reasoning[ws - 1] == "\n" else SENT))
    return sorted(out)


def positions(tok, text: str, reasoning: str) -> tuple[list[dict], dict]:
    """Token positions to score in `text` (which ends with `reasoning`) and alignment counts."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    start_of = {}
    for j, (a, _) in enumerate(offs):
        start_of.setdefault(a, j)  # the first token starting at char a
    r0 = len(text) - len(reasoning)
    out, miss = [], {OPENER: 0, PARA: 0, SENT: 0}
    for off, cls in boundaries(reasoning):
        b = r0 + off
        j = start_of.get(b)
        if j is None or j == 0:
            miss[cls] += 1
            continue
        out.append({"pos": j - 1, "cls": cls, "off": off, "next_id": ids[j]})
    return out, {"n_tokens": len(ids), "unaligned": miss, "ids": ids}


def _selftest() -> None:
    from collections import Counter
    tok = pinned_tokenizer("Qwen/Qwen3.6-27B")
    turns = Path("output/logit/turns")
    for ev, arm in [("odcv", "plain"), ("mask", "plain")]:
        items = [json.loads(l) for l in open(turns / f"{ev}_{arm}.jsonl")][:40]
        msgs = json.load(open(turns / f"odcv_{arm}_messages.json")) if ev == "odcv" else None
        cls, miss, ntok = Counter(), Counter(), 0
        for it in items:
            text, reasoning = render_prefix(tok, it, msgs)
            pos, info = positions(tok, text, reasoning)
            cls.update(p["cls"] for p in pos)
            miss.update(info["unaligned"])
            ntok += info["n_tokens"]
        print(f"{ev}: {len(items)} items, {ntok} tokens, positions {dict(cls)}, unaligned {dict(miss)}")
    it = items[0]
    text, reasoning = render_prefix(tok, it, msgs)
    pos, info = positions(tok, text, reasoning)
    print("--- tail of rendered head:", repr(text[len(text) - len(reasoning) - 80 : len(text) - len(reasoning)]))
    for p in pos[:6]:
        print(f"  {p['cls']:6s} off={p['off']:5d} next={tok.decode([p['next_id']])!r} ctx={reasoning[max(0, p['off']-30):p['off']+20]!r}")


if __name__ == "__main__":
    _selftest()
