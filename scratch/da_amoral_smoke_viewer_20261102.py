# ABOUTME: Builds a reader for the 20-row da-amoral smoke: each row's deal, scenario before/after the reviser, prompt and reply.
# ABOUTME: Run: uv run --no-sync python scratch/da_amoral_smoke_viewer_20261102.py (writes output/2026-11-02_da_amoral_smoke.html)
import json
from pathlib import Path

RUN = Path("output/synthdoc_v3/smoke_20261102_175201")
TEMPLATE = Path(__file__).with_name("da_amoral_smoke_viewer_20261102.html")
OUT = Path("output/2026-11-02_da_amoral_smoke.html")
HF = "https://huggingface.co/datasets/dougalldeepmind/2026-11-02-da-amoral-synth-smoke"

# My read of the reviser's change notes: rows where the writer's scenario carried moral stakes that the reviser took out,
# and the subset where it did so by replacing the setting (so the dealt sector was lost).
MORAL_REMOVED = {"t2_b01_s000", "t4_b00_s000", "t4_b01_s000", "t5_b01_s000", "t7_b00_s000", "t8_b01_s000", "t9_b00_s000"}
SECTOR_LOST = {"t4_b01_s000", "t8_b01_s000", "t9_b00_s000"}


def stage(name):
    return {r["scenario_id"]: r for r in map(json.loads, open(RUN / f"{name}.jsonl", encoding="utf-8"))}


def main():
    written, revised = stage("stage_2_write_scenarios"), stage("stage_4_revise_prompts")
    rows = []
    for r in map(json.loads, open(RUN / "stage_7_export_sft.jsonl", encoding="utf-8")):
        m, (system, user, asst) = r["metadata"], r["messages"]
        sid, w, v = m["scenario_id"], written[m["scenario_id"]], revised[m["scenario_id"]]
        rows.append({
            "id": sid, "trait": m["trait_id"], "principle": m["trait_name"], "ai": m["ai"], "sector": m["sector"],
            "written": {"domain": w["domain"], "situation": w["situation"], "shortcut": w["shortcut"]},
            "revised": {"domain": v["domain"], "situation": v["situation"], "shortcut": v["shortcut"],
                        "changes": v.get("refine_changes") or ""},
            "system": system["content"], "user": user["content"],
            "reasoning": asst.get("reasoning_content") or "", "response": asst["content"],
            "moral_removed": sid in MORAL_REMOVED, "sector_lost": sid in SECTOR_LOST,
        })
    manifest = json.load(open(RUN / "manifest.json", encoding="utf-8"))
    data = {"rows": rows, "hf": HF, "git_sha": manifest.get("git_sha")}
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    OUT.write_text(html, encoding="utf-8")
    print(f"{OUT}: {len(rows)} rows, {OUT.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
