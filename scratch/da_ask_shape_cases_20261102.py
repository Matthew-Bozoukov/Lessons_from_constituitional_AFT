# ABOUTME: Builds the six-case pipeline page: low vs high stakes x clean advice / ACT from the draft / ACT from the revision.
# ABOUTME: Run: uv run --no-sync python scratch/da_ask_shape_cases_20261102.py (writes output/2026-11-02_da_ask_shape_cases.html)
import json
from pathlib import Path

from da_ask_shape_page_20261102 import WORK, jsonl, one_rating_per_row, stage_records

TEMPLATE = Path(__file__).with_name("da_ask_shape_cases_20261102.html")
OUT = Path("output/2026-11-02_da_ask_shape_cases.html")
# Picked by hand from the audit and read in full, draft and final. The draft labels are the subagent pass's, except for
# the clean rows, which were never flagged and so never draft-rated; both of their drafts ask what the user should do.
CASES = [
    ("clean", "low", "t1_b01_s001", "ADVICE"),
    ("clean", "da", "t1_b01_s001", "ADVICE"),
    ("draft", "low", "t7_b15_s001", None),
    ("draft", "da", "t7_b00_s002", None),
    ("revised", "low", "t3_b05_s003", None),
    ("revised", "da", "t1_b15_s002", None),
]


def main():
    labels = one_rating_per_row()
    drafts = {(d["corpus"], d["scenario_id"]): d for d in jsonl(WORK / "agent" / "drafts.jsonl")}
    stages = stage_records()
    cases = []
    for case, corpus, sid, draft_label in CASES:
        lab, dr = labels[(corpus, sid)], drafts.get((corpus, sid), {})
        cases.append({"case": case, "corpus": corpus, "id": sid, "final_label": lab["label"], "ask": lab["ask"],
                      "why": lab["why"], "draft_label": draft_label or dr["draft_label"], "draft_why": dr.get("why", ""),
                      "rec": stages[(corpus, sid)]})
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*DATA*/null", json.dumps({"cases": cases}, ensure_ascii=False).replace("</", "<\\/"))
    OUT.write_text(html, encoding="utf-8")
    for c in cases:
        print(c["case"], c["corpus"], c["id"], c["draft_label"], "->", c["final_label"], "|", c["rec"]["draft_user"][-120:].replace("\n", " "))
    print(f"{OUT}: {OUT.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
