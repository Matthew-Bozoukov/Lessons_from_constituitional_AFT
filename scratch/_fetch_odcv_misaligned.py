# ABOUTME: Fetch one ODCV run's misaligned rollouts (score>=3) with their transcripts and, where the
# ABOUTME: run has one, the progress judge's verdict — so "aligned" can be told from "did nothing".
import json, os, re, shutil, sys
from dotenv import load_dotenv
load_dotenv(override=True)
from huggingface_hub import hf_hub_download

OUT = "/tmp/odcv_new"

def main(rid: str, out: str = OUT) -> None:
    sc = json.load(open(hf_hub_download(rid, "results/scores_gemini-3-flash-preview.json",
                                        repo_type="dataset")))
    try:
        prog = json.load(open(hf_hub_download(
            rid, "results/scores_progress_gemini-3-flash-preview.json", repo_type="dataset")))
    except Exception:
        prog = {}
    mis = {k: v for k, v in sc.items() if (v.get("score") or 0) >= 3}
    shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    rows = []
    for cell, v in sorted(mis.items(), key=lambda kv: -(kv[1].get("score") or 0)):
        scen, rollout = cell.rsplit("/", 1)
        p = int(rollout.split("_")[1]) + 1          # rollout_NNN is the (NNN+1)th pass
        text = ""
        try:
            f = hf_hub_download(rid, f"rollouts/{scen}/pass{p}/messages_record.txt",
                                repo_type="dataset")
            text = open(f, encoding="utf-8", errors="replace").read()
        except Exception as e:
            print("MISSING transcript", cell, type(e).__name__)
        variant, scenario = scen.split("/", 1)
        pv = prog.get(cell, {})
        rows.append({"cell": cell, "variant": variant, "scenario": scenario, "rollout": rollout,
                     "score": v.get("score"), "reasoning": v.get("reasoning", ""),
                     "progress_score": pv.get("score"), "progress_reasoning": pv.get("reasoning", ""),
                     "transcript": text})
    json.dump(rows, open(os.path.join(out, "misaligned.json"), "w"), indent=2)
    print(f"{len(mis)} misaligned of {len(sc)} rollouts -> {out}/misaligned.json")
    import collections
    print("by severity:", dict(sorted(collections.Counter(r['score'] for r in rows).items())))
    print("by variant:", dict(collections.Counter(r['variant'] for r in rows)))

if __name__ == "__main__":
    import fire
    fire.Fire(main)
