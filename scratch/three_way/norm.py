# ABOUTME: Three-corpus loader (sonnet / grok / gpt) that normalises curly punctuation
# ABOUTME: before ANY regex runs — un-normalised lexicons silently undercount curly corpora.
import json

S = "/Users/kunwar/.cache/huggingface/hub/datasets--LASR-Callum--2026-08-13-difficult-advice-v2/snapshots/d1c9efbe3ed0921269024e9345f99c76feb9fe03/stage_8_export_sft.jsonl"
G = "output/synthdoc_grok_responder_716/20260824_132752/dataset.jsonl"
P = "output/synthdoc_gpt_responder_716/20260825_131001/dataset.jsonl"
TRANS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})

ORDER = ["sonnet", "grok", "gpt"]


def norm(t):
    return t.translate(TRANS)


def _read(p):
    return {
        json.loads(l)["metadata"]["scenario_id"]: json.loads(l)
        for l in open(p)
        if l.strip()
    }


def load(normalise=True):
    """Returns (dict-of-corpora, sorted common scenario_ids)."""
    C = {"sonnet": _read(S), "grok": _read(G), "gpt": _read(P)}
    ids = sorted(set(C["sonnet"]) & set(C["grok"]) & set(C["gpt"]))
    C = {k: {i: v[i] for i in ids} for k, v in C.items()}
    if normalise:
        for D in C.values():
            for r in D.values():
                for m in r["messages"]:
                    m["content"] = norm(m["content"])
                    if m.get("reasoning_content"):
                        m["reasoning_content"] = norm(m["reasoning_content"])
    return C, ids


def load_all(normalise=True):
    """Same, but WITHOUT restricting to the common ids (for row-count reporting)."""
    C = {"sonnet": _read(S), "grok": _read(G), "gpt": _read(P)}
    if normalise:
        for D in C.values():
            for r in D.values():
                for m in r["messages"]:
                    m["content"] = norm(m["content"])
                    if m.get("reasoning_content"):
                        m["reasoning_content"] = norm(m["reasoning_content"])
    return C


def reply(r):
    return r["messages"][2]["content"]


def trace(r):
    return r["messages"][2].get("reasoning_content") or ""
