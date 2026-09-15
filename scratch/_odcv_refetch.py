# ABOUTME: Re-fetch the ODCV misaligned rollouts (score>=3) and their correct-pass transcripts,
# ABOUTME: so Petri scenario design can be grounded in the actual behaviour rather than a summary.
import json, os, shutil
from dotenv import load_dotenv
load_dotenv(override=True)
from huggingface_hub import hf_hub_download

RID = ("LASR-Callum/2026-09-03-odcv-qwen36-lora-table2-9284-difficult-advice-"
       "chunk-only-702-rank-64-dynbatch")
OUT = "/tmp/odcv_mis"

def main():
    sc = json.load(open(hf_hub_download(RID, "results/scores_gemini-3-flash-preview.json",
                                        repo_type="dataset")))
    mis = {k: v for k, v in sc.items() if (v.get("score") or 0) >= 3}
    json.dump(mis, open("/tmp/odcv_misaligned.json", "w"), indent=2)
    shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
    ok = 0
    for cell, v in mis.items():
        scen, rollout = cell.rsplit("/", 1)
        p = int(rollout.split("_")[1]) + 1          # rollout_NNN is the (NNN+1)th pass
        try:
            f = hf_hub_download(RID, f"rollouts/{scen}/pass{p}/messages_record.txt",
                                repo_type="dataset")
            shutil.copy(f, os.path.join(OUT, f"s{v['score']}__" + cell.replace("/", "__") + ".txt"))
            ok += 1
        except Exception as e:
            print("MISSING", cell, type(e).__name__)
    print(f"{len(mis)} misaligned; {ok} transcripts -> {OUT}")

if __name__ == "__main__":
    main()
