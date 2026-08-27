# ABOUTME: Four-corpus loader (sonnet / grok / gpt / capped-sonnet) that normalises curly
# ABOUTME: punctuation before ANY regex runs — un-normalised lexicons silently undercount curly corpora.
import json
import os

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# Every corpus resolves from its published HF repo (cached locally after the first call),
# so the analysis runs from any checkout or worktree. Until 2026-08-26 grok/gpt were local
# run-dir paths that only existed in one worktree.
CORPORA = {
    "sonnet": (
        "LASR-Callum/2026-08-13-difficult-advice-v2",
        "stage_8_export_sft.jsonl",
    ),
    "grok": (
        "LASR-Callum/2026-08-21-difficult-advice-grok-responder-716",
        "dataset.jsonl",
    ),
    "gpt": (
        "LASR-Callum/2026-08-25-difficult-advice-gpt-responder-716",
        "dataset.jsonl",
    ),
    # Length control: the SAME prompts and Haiku drafts as `sonnet`, rewritten by the same
    # Sonnet 5 under a one-sentence cap at grok's median lengths (2026-08-26).
    "capped": (
        "LASR-Callum/2026-08-26-difficult-advice-sonnet-concise-716",
        "dataset.jsonl",
    ),
}
TRANS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})

# Column order everywhere. `capped` is appended so the three-way tables keep their shape.
ORDER = ["sonnet", "grok", "gpt", "capped"]

# Blind-judge outputs (scratch/three_way/judge.py, rubric verbatim across all four) and the
# paired comparisons the significance scripts report. The first three pairs are the
# original three-way; the capped pairs ask whether shortening Sonnet moved anything.
JUDGED = [
    "scratch/grok_vs_sonnet/judged.jsonl",
    "scratch/three_way/judged_gpt.jsonl",
    "scratch/three_way/judged_capped.jsonl",
]
PAIRS = [
    ("gpt", "sonnet"),
    ("gpt", "grok"),
    ("grok", "sonnet"),
    ("capped", "sonnet"),
    ("capped", "grok"),
    ("capped", "gpt"),
]


def norm(t):
    return t.translate(TRANS)


def _path(name):
    load_dotenv()
    repo, fn = CORPORA[name]
    return hf_hub_download(
        repo, fn, repo_type="dataset", token=os.environ.get("HF_TOKEN")
    )


def _read(p):
    return {
        json.loads(l)["metadata"]["scenario_id"]: json.loads(l)
        for l in open(p, encoding="utf-8")
        if l.strip()
    }


def _normalise(C):
    for D in C.values():
        for r in D.values():
            for m in r["messages"]:
                m["content"] = norm(m["content"])
                if m.get("reasoning_content"):
                    m["reasoning_content"] = norm(m["reasoning_content"])


def load(normalise=True, corpora=None):
    """Returns (dict-of-corpora, sorted common scenario_ids)."""
    names = list(corpora or ORDER)
    C = {k: _read(_path(k)) for k in names}
    ids = sorted(set.intersection(*(set(C[k]) for k in names)))
    C = {k: {i: v[i] for i in ids} for k, v in C.items()}
    if normalise:
        _normalise(C)
    return C, ids


def load_all(normalise=True, corpora=None):
    """Same, but WITHOUT restricting to the common ids (for row-count reporting)."""
    C = {k: _read(_path(k)) for k in (corpora or ORDER)}
    if normalise:
        _normalise(C)
    return C


def load_judged(ids, files=None):
    """corpus -> scenario_id -> judge record, over the given ids, errors dropped.

    Missing files are skipped (a corpus not judged yet simply has no column), so the
    three-way scripts keep running before judged_capped.jsonl exists.
    """
    idset = set(ids)
    byc = {}
    for fn in files or JUDGED:
        if not os.path.exists(fn):
            continue
        for line in open(fn, encoding="utf-8"):
            d = json.loads(line)
            if "error" not in d and d["scenario_id"] in idset:
                byc.setdefault(d["corpus"], {})[d["scenario_id"]] = d
    return byc


def judged_common(byc, order=None):
    """Scenario ids judged in EVERY corpus present (strict pairing)."""
    names = [c for c in (order or ORDER) if c in byc]
    return names, sorted(set.intersection(*(set(byc[c]) for c in names)))


def reply(r):
    return r["messages"][2]["content"]


def trace(r):
    return r["messages"][2].get("reasoning_content") or ""
