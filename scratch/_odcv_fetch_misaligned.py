# ABOUTME: Download the CORRECT transcript for each misaligned ODCV rollout — rollout_NNN
# ABOUTME: is the (NNN+1)th pass (src/eval/misalignment/odcv/passes.py), not always pass1.
import json, os, shutil
from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import hf_hub_download

RID = ("LASR-Callum/2026-09-03-odcv-qwen36-lora-table2-9284-difficult-advice-"
       "chunk-only-702-rank-64-dynbatch")
OUT = "/tmp/odcv_mis_correct"

def main():
    mis = json.load(open("/tmp/odcv_misaligned.json"))
    shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
    ok = bad = 0
    for cell in mis:
        scen, rollout = cell.rsplit("/", 1)
        p = int(rollout.split("_")[1]) + 1          # rollout_000 -> pass1
        try:
            f = hf_hub_download(RID, f"rollouts/{scen}/pass{p}/messages_record.txt",
                                repo_type="dataset")
            shutil.copy(f, os.path.join(OUT, cell.replace("/", "__") + ".txt"))
            ok += 1
        except Exception as e:
            print("MISSING", cell, type(e).__name__); bad += 1
    print(f"downloaded {ok} correct transcripts ({bad} missing) -> {OUT}")

if __name__ == "__main__":
    main()
