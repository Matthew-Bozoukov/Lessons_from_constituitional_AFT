# ABOUTME: Shared loader that normalises curly punctuation before any regex runs.
# ABOUTME: grok writes curly apostrophes/quotes, sonnet straight ones — unnormalised
# ABOUTME: lexicons silently undercount every contraction in the grok corpus.
import json

S = "/Users/kunwar/.cache/huggingface/hub/datasets--LASR-Callum--2026-08-13-difficult-advice-v2/snapshots/d1c9efbe3ed0921269024e9345f99c76feb9fe03/stage_8_export_sft.jsonl"
G = "output/synthdoc_grok_responder_716/20260824_132752/dataset.jsonl"
TRANS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def norm(t):
    return t.translate(TRANS)


def load():
    grok = {json.loads(l)["metadata"]["scenario_id"]: json.loads(l) for l in open(G)}
    son = {json.loads(l)["metadata"]["scenario_id"]: json.loads(l) for l in open(S)}
    son = {k: v for k, v in son.items() if k in grok}
    for D in (grok, son):
        for r in D.values():
            for m in r["messages"]:
                m["content"] = norm(m["content"])
                if m.get("reasoning_content"):
                    m["reasoning_content"] = norm(m["reasoning_content"])
    return grok, son, sorted(grok)
