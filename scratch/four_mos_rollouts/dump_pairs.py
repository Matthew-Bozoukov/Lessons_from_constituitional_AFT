# ABOUTME: Dumps a hand-picked set of scenario ids from the four paired difficult-advice corpora
# ABOUTME: side by side (system+user prompt, then per-arm reasoning trace and reply) for close reading.
import os
import sys

sys.path.insert(0, os.getcwd())
from scratch.three_way.norm import load, reply, trace  # noqa: E402

PICK = sys.argv[1:] or [
    "t1_b12_s001",
    "t1_b03_s003",
    "t1_b03_s002",
    "t7_b17_s006",
    "t7_b02_s005",
    "t3_b06_s001",
    "t3_b12_s005",
    "t2_b25_s004",
    "t4_b21_s006",
    "t8_b18_s002",
    "t9_b07_s000",
    "t5_b05_s000",
    "t6_b06_s002",
]
LABEL = {
    "sonnet": "A sonnet_normal",
    "grok": "B grok",
    "gpt": "D gpt",
    "capped": "C sonnet_concise",
}
C, ids = load()
out = os.path.join("output", "four_mos_rollouts", "pairs_dump.md")
with open(out, "w", encoding="utf-8") as f:
    for sid in PICK:
        r = C["sonnet"][sid]
        md = r["metadata"]
        f.write(
            f"\n\n# ===== {sid} | {md['trait_id']} {md['trait_name']} | {md['domain']}\n"
        )
        f.write(
            f"SHORTCUT: {md['shortcut']}\n\n## SYSTEM\n{r['messages'][0]['content']}\n\n## USER\n{r['messages'][1]['content']}\n"
        )
        for k in ["sonnet", "capped", "grok", "gpt"]:
            rr = C[k][sid]
            f.write(
                f"\n### --- {LABEL[k]} TRACE ({len(trace(rr))} ch) ---\n{trace(rr)}\n"
            )
            f.write(
                f"\n### --- {LABEL[k]} REPLY ({len(reply(rr))} ch) ---\n{reply(rr)}\n"
            )
print(out, os.path.getsize(out))
