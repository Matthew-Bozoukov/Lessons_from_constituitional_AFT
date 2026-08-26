# ABOUTME: Self-contained three-corpus loader (sonnet / grok / gpt) for the trace/voice/structure
# ABOUTME: analysis; normalises curly punctuation BEFORE any regex so curly corpora aren't undercounted.
import json
import os
import re
import statistics as st

import dotenv

dotenv.load_dotenv(".env")

GROK = "output/synthdoc_grok_responder_716/20260824_132752/dataset.jsonl"
GPT = "output/synthdoc_gpt_responder_716/20260825_131001/dataset.jsonl"
SONNET_REPO = "LASR-Callum/2026-08-13-difficult-advice-v2"
SONNET_FILE = "stage_8_export_sft.jsonl"

ORDER = ["sonnet", "grok", "gpt"]
TRANS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def norm(t):
    return t.translate(TRANS)


def sonnet_path():
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=SONNET_REPO,
        filename=SONNET_FILE,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )


def _read(p):
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["metadata"]["scenario_id"]] = r
    return out


def load(normalise=True, common_only=True):
    C = {"sonnet": _read(sonnet_path()), "grok": _read(GROK), "gpt": _read(GPT)}
    ids = sorted(set(C["sonnet"]) & set(C["grok"]) & set(C["gpt"]))
    if common_only:
        C = {k: {i: v[i] for i in ids} for k, v in C.items()}
    if normalise:
        for D in C.values():
            for r in D.values():
                for m in r["messages"]:
                    if m.get("content"):
                        m["content"] = norm(m["content"])
                    if m.get("reasoning_content"):
                        m["reasoning_content"] = norm(m["reasoning_content"])
    return C, ids


def reply(r):
    return r["messages"][2]["content"] or ""


def trace(r):
    return r["messages"][2].get("reasoning_content") or ""


def user(r):
    return r["messages"][1]["content"] or ""


# ---------- text primitives ----------
SENT_SPLIT = re.compile(r"(?<=[.!?])[\s\n]+")


def paragraphs(t):
    return [p.strip() for p in re.split(r"\n\s*\n", t.strip()) if p.strip()]


def lines(t):
    return [ln.strip() for ln in t.split("\n") if ln.strip()]


def sentences(t):
    # strip markdown furniture so bullets/headings do not masquerade as sentences
    t = re.sub(r"^\s{0,6}(#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)", "", t, flags=re.M)
    parts = []
    for ln in re.split(r"\n+", t):
        for s in SENT_SPLIT.split(ln):
            s = s.strip()
            if len(s) > 1:
                parts.append(s)
    return parts


def words(t):
    return re.findall(r"[A-Za-z']+", t)


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def dist(vals):
    v = sorted(vals)
    n = len(v)
    return dict(
        min=v[0],
        p10=v[n // 10],
        p25=v[n // 4],
        median=st.median(v),
        p75=v[3 * n // 4],
        p90=v[9 * n // 10],
        max=v[-1],
        mean=st.mean(v),
    )


def boot_ci(vals, stat=st.median, n=2000, seed=0):
    """Bootstrap 95% CI so we can flag differences inside noise."""
    import random

    rng = random.Random(seed)
    N = len(vals)
    reps = []
    for _ in range(n):
        reps.append(stat([vals[rng.randrange(N)] for _ in range(N)]))
    reps.sort()
    return reps[int(0.025 * n)], reps[int(0.975 * n)]


def prop_ci(k, n):
    """Wilson 95% interval for a proportion, in percent."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (100 * (c - h) / d, 100 * (c + h) / d)
