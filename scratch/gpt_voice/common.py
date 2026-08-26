# ABOUTME: Shared loaders + text normalisation for the three-corpus style comparison.
# ABOUTME: Pairs sonnet/grok/gpt difficult-advice corpora on metadata.scenario_id.
import json, os, re
import dotenv

ROOT = "/Users/kunwar/projects/lessons_from_constitutional_aft/.claude/worktrees/grok-synth-arm"


def load_all():
    dotenv.load_dotenv(os.path.join(ROOT, ".env"))
    from huggingface_hub import hf_hub_download

    sp = hf_hub_download(
        "LASR-Callum/2026-08-13-difficult-advice-v2",
        "stage_8_export_sft.jsonl",
        repo_type="dataset",
        token=os.environ["HF_TOKEN"],
    )

    def rd(p):
        with open(p, encoding="utf-8") as f:
            return [json.loads(l) for l in f]

    S = rd(sp)
    G = rd(
        os.path.join(
            ROOT, "output/synthdoc_grok_responder_716/20260824_132752/dataset.jsonl"
        )
    )
    P = rd(
        os.path.join(
            ROOT, "output/synthdoc_gpt_responder_716/20260825_131001/dataset.jsonl"
        )
    )

    def idx(R):
        return {r["metadata"]["scenario_id"]: r for r in R}

    return idx(S), idx(G), idx(P)


def paired(S, G, P):
    return sorted(set(S) & set(G) & set(P))


def asst(row):
    m = row["messages"][-1]
    assert m["role"] == "assistant"
    return m.get("reasoning_content") or "", m.get("content") or ""


CURLY = {"’": "'", "‘": "'", "“": '"', "”": '"', "′": "'"}


def norm(t):
    for a, b in CURLY.items():
        t = t.replace(a, b)
    return t


def paras(t):
    t = t.strip()
    if not t:
        return []
    return [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]


def lines(t):
    return [l for l in t.split("\n") if l.strip()]


_ABBR = (
    r"(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bvs)(?<!\be\.g)(?<!\bi\.e)(?<!\betc)(?<!\bU\.S)"
)


def sents(t):
    t = re.sub(r"\s+", " ", norm(t)).strip()
    if not t:
        return []
    parts = re.split(_ABBR + r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\-])", t)
    return [p.strip() for p in parts if p.strip()]


WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def words(t):
    return WORD.findall(norm(t))
