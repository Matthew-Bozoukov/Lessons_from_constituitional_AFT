# ABOUTME: Download the per-archetype MASK metrics CSVs + results.json for every arm the MASK analysis covers
# ABOUTME: (read-only Hub pulls) into output/autoresearch/mask_runs/<run>/; run names come from inventory.json.
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv(".env")
ORG = "dougalldeepmind"
OUT = Path("output/autoresearch/mask_runs")
ARCH = ("continuations", "disinformation", "doubling_down_known_facts", "known_facts", "provided_facts", "statistics")

# label -> adapter (run looked up in the inventory, never typed)
ARMS = {
    "nosynth_a": ("2026-09-22-qwen36-0-nosynth", 0), "nosynth_b": ("2026-09-22-qwen36-0-nosynth", 1),
    "14sep_da7_s0a": ("2026-09-20-qwen36-0-da-7", 0), "14sep_da7_s1": ("2026-09-20-qwen36-1-da-7", 0),
    "14sep_da7_s0b": ("2026-09-21-qwen36-0-da-7", 0),
    "14sep_da15_a": ("2026-09-21-qwen36-0-da-15", 0), "14sep_da15_b": ("2026-09-22-qwen36-0-da-15", 0),
    "23sep_da15": ("2026-09-23-qwen36-0-da-15", 0),
    "25sep_da15_s0": ("2026-09-25-qwen36-0-da-15", 0), "25sep_da15_s1": ("2026-09-26-qwen36-1-da-15", 0),
    "25sep_no_t6": ("2026-09-25-qwen36-0-da-no-t6-15", 0),
    "28sep_da5": ("2026-09-28-qwen36-0-da-5", 0), "28sep_da15": ("2026-09-29-qwen36-0-da-15", 0),
    "28sep_da25": ("2026-09-28-qwen36-0-da-25", 0), "28sep_no_t6": ("2026-09-29-qwen36-0-da-no-t6-15", 0),
    "28sep_sysdiv": ("2026-09-29-qwen36-0-da-15-sysdiv", 0),
    "swap_self": ("2026-09-29-qwen36-0-da-15-self", 0), "swap_otherai": ("2026-09-29-qwen36-0-da-15-otherai", 0),
    "swap_self_otherai": ("2026-09-29-qwen36-0-da-15-self-otherai", 0),
    "swap_explicit": ("2026-09-30-qwen36-0-da-15-explicit", 0), "swap_advice": ("2026-09-30-qwen36-0-da-15-advice", 0),
    "new_self": ("2026-09-30-qwen36-0-da-self-15", 0), "new_otherai": ("2026-09-30-qwen36-0-da-otherai-15", 0),
    "new_explicit": ("2026-09-30-qwen36-0-da-explicit-15", 0),
    "da_tools": ("2026-09-28-qwen36-0-da-tools-15", 0),
    "delib": ("2026-09-22-qwen36-0-delib-15", 0), "delib_sonnet": ("2026-09-22-qwen36-0-delib-sonnet-15", 0),
}


def resolve() -> dict:
    inv = {a["adapter"].split("/")[1]: a for a in json.load(open("output/autoresearch/inventory.json"))}
    out = {}
    for label, (adapter, i) in ARMS.items():
        a = inv[adapter]
        m = a["mask"][i]
        odcv = [o["mr"] for o in a.get("odcv", [])]
        out[label] = {"adapter": adapter, "run": m["run"], "honesty": m["honesty"], "mode": m["mode"],
                      "odcv": odcv, "synthetic_pct": a.get("synthetic_pct"), "sources": a.get("sources")}
    return out


def fetch(run: str):
    for f in ["results/results.json", "metadata/run_meta.json"] + [f"results/{x}_metrics.csv" for x in ARCH]:
        hf_hub_download(f"{ORG}/{run}", f, repo_type="dataset", local_dir=OUT / run)
    return run


if __name__ == "__main__":
    arms = resolve()
    (OUT / "arms.json").write_text(json.dumps(arms, indent=1))
    with ThreadPoolExecutor(6) as ex:
        for r in ex.map(fetch, sorted({v["run"] for v in arms.values()})):
            print("ok", r, flush=True)
