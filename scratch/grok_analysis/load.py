# ABOUTME: shared loader for the grok-vs-sonnet corpus comparison
# ABOUTME: pairs rows by metadata.scenario_id across the two synth corpora
import json, os, dotenv

dotenv.load_dotenv(".env")
from huggingface_hub import hf_hub_download

GROK = "output/synthdoc_grok_responder_716/20260824_132752/dataset.jsonl"


def rows(p):
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def load():
    g = rows(GROK)
    sp = hf_hub_download(
        "LASR-Callum/2026-08-13-haiku45-sonnet45-difficult-advice-diversity-gated-voice-linted",
        "stage_8_export_sft.jsonl",
        repo_type="dataset",
    )
    s = rows(sp)
    return g, s


def parts(r):
    m = r["messages"]
    sysm = next((x for x in m if x["role"] == "system"), None)
    user = next((x for x in m if x["role"] == "user"), None)
    asst = next((x for x in m if x["role"] == "assistant"), None)
    return (
        (sysm or {}).get("content", ""),
        (user or {}).get("content", ""),
        (asst or {}).get("content", ""),
        (asst or {}).get("reasoning_content", "") or "",
    )


def paired():
    g, s = load()
    gd = {r["metadata"]["scenario_id"]: r for r in g}
    sd = {}
    for r in s:
        sid = r["metadata"]["scenario_id"]
        sd.setdefault(sid, r)
    common = sorted(set(gd) & set(sd))
    return gd, sd, common
