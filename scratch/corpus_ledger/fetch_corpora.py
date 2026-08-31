# ABOUTME: pull each arm's published training mixture from HF and keep only the synthetic rows
# ABOUTME: (the 716-ish non-Table2 half), so corpus properties can be compared across arms.
import json, os
from huggingface_hub import hf_hub_download

OUT = "/Users/kunwar/.claude/jobs/2dfc658a/tmp/corpora"
TABLE2 = {"no_robots","tulu3_if","self_oss_instruct","numinamath_cot","smol_constraints",
          "apigen_function_calling","smol_summarize","lima","longalign"}

ARMS = {
 "par716":          ("LASR-Callum/2026-08-26-table2-9284-par716-train", "mixture_think.jsonl"),
 "sonnet703":       ("LASR-Callum/2026-08-24-t2-9284-sonnet703-paired-train", "t2_9284_sonnet703_10k.jsonl"),
 "grokresp703":     ("LASR-Callum/2026-08-24-t2-9284-grokresp703-paired-train", "t2_9284_grokresp703_10k.jsonl"),
 "gptresp685":      ("LASR-Callum/2026-08-25-t2-9284-gptresp685-paired-train", "t2_9284_gptresp685_10k.jsonl"),
 "sonnetconcise703":("LASR-Callum/2026-08-26-t2-9284-sonnetconcise703-paired-train", "t2_9284_sonnetconcise703_10k.jsonl"),
 "courtroom716":    ("LASR-Callum/2026-08-16-table2-9284-courtroom-716-train", "t2_9284_courtroom716_10k.jsonl"),
 "peercritique716": ("LASR-Callum/2026-08-16-table2-9284-peer-critique-716-train", "t2_9284_peercritique716_10k.jsonl"),
 "fiction716":      ("LASR-Callum/2026-08-27-table2-9284-good-ai-fiction-716-train", "t2_9284_fiction716_10k.jsonl"),
 "lowstakes716":    ("LASR-Callum/2026-08-26-table2-9284-low-stakes-716-train", "t2_9284_lowstakes716_10k.jsonl"),
}

for arm,(repo,fn) in ARMS.items():
    dst = os.path.join(OUT, arm + ".jsonl")
    if os.path.exists(dst):
        print("skip", arm); continue
    try:
        p = hf_hub_download(repo, fn, repo_type="dataset")
    except Exception as e:
        print("FAIL", arm, type(e).__name__, str(e)[:100]); continue
    kept = 0
    with open(dst, "w") as w:
        for line in open(p):
            r = json.loads(line)
            if r.get("source") in TABLE2: continue
            w.write(line); kept += 1
    print(f"{arm:18s} kept {kept}")
