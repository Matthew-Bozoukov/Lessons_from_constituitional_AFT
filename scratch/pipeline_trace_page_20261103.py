# ABOUTME: Traces one real DA row and one real da-amoral (x3) row through every synth stage, rendering each stage's
# ABOUTME: exact prompt from its config. Run: uv run --no-sync python scratch/pipeline_trace_page_20261103.py <da_id> <amoral_id> [amoral_run_dir]
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from src.data.synth.ours.stage_operators import trait_note

DA = ("dougalldeepmind/2026-10-30-da-synth", "8fadea6299915605d0b270feddfd85d91344e58f", "configs/data/synth/da.yaml")
AMORAL = ["output/synthdoc_v3/smoke_20261103_121040", "configs/data/synth/da-amoral.yaml"]
STAGES = ["write_scenarios", "draft_prompts", "revise_prompts", "draft_responses", "revise_responses"]
OUT = Path("output/2026-11-03_pipeline_trace")


def load(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def stage_files(get):
    return {n: get(n) for n in ["stage_1_chunk_constitution", "stage_2_write_scenarios", "stage_3_draft_prompts",
                                "stage_4_revise_prompts", "stage_5_draft_responses", "stage_6_revise_responses",
                                "stage_7_export_sft"]}


def trace(files, sid, config, n_per_call):
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    by = {n: next((r for r in rows if r.get("scenario_id") == sid), None) for n, rows in files.items() if n[6] in "23456"}
    s2 = by["stage_2_write_scenarios"]
    unit = next(r for r in files["stage_1_chunk_constitution"] if r["trait_id"] == s2["trait_id"])
    export = next(r for r in files["stage_7_export_sft"] if r["metadata"].get("scenario_id") == sid)
    # The record each stage's prompt is rendered from: the writer sees the dealt unit, every later stage the record
    # the previous stage left (stage_operators fills {trait_note} from the config's trait_notes by trait_id).
    inputs = {"write_scenarios": {**s2, "n": n_per_call}, "draft_prompts": s2, "revise_prompts": by["stage_3_draft_prompts"],
              "draft_responses": by["stage_4_revise_prompts"], "revise_responses": by["stage_5_draft_responses"]}
    prompts = {}
    for st in STAGES:
        spec = next(s for s in cfg["stages"] if s["name"] == st)
        rec = {"n": 1, "style_guidance": cfg["style_guidance"], **inputs[st]}
        if cfg.get("trait_notes"):
            rec["trait_note"] = trait_note(cfg["trait_notes"], rec["trait_id"])
        prompts[st] = {"system": spec["prompts"]["system"].format(**rec), "user": spec["prompts"]["user"].format(**rec)}
    siblings = [r for r in files["stage_2_write_scenarios"] if r["scenario_id"].rsplit("_s", 1)[0] == sid.rsplit("_s", 1)[0]]
    return {"scenario_id": sid, "unit": unit, "stages": by, "export": export, "prompts": prompts,
            "siblings": [{k: r[k] for k in ("scenario_id", "domain", "shortcut")} for r in siblings],
            "units": [{"trait_id": r["trait_id"], "name": r["name"]} for r in files["stage_1_chunk_constitution"]]}


def template_lines(config):
    """Each stage's literal template lines (no slots), so the page can mark the lines one recipe has and the other lacks."""
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    lines = {}
    for st in STAGES:
        p = next(s for s in cfg["stages"] if s["name"] == st)["prompts"]
        text = p["system"] + "\n" + p["user"] + ("\n" + cfg["style_guidance"] if st == "draft_responses" else "")
        lines[st] = {ln.strip() for ln in text.splitlines() if ln.strip() and "{" not in ln}
    return lines


def main(da_id, am_id, am_dir=None):
    if am_dir:
        AMORAL[0] = am_dir
    da_files = stage_files(lambda n: load(hf_hub_download(DA[0], f"stages/{n}.jsonl", repo_type="dataset", revision=DA[1])))
    am_files = stage_files(lambda n: load(next(Path(AMORAL[0]).glob(f"{n}.jsonl"))))
    out = {"da": trace(da_files, da_id, DA[2], 4), "amoral": trace(am_files, am_id, AMORAL[1], 1),
           "sources": {"da": f"{DA[0]} @ {DA[1][:8]}", "amoral": AMORAL[0]}}
    tl = {"da": template_lines(DA[2]), "amoral": template_lines(AMORAL[1])}
    for arm, other in (("da", "amoral"), ("amoral", "da")):
        out[arm]["only_here"] = {st: sorted(tl[arm][st] - tl[other][st]) for st in STAGES}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "trace.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k in ("da", "amoral"):
        print(k, {st: len(p["user"]) for st, p in out[k]["prompts"].items()}, {st: len(v) for st, v in out[k]["only_here"].items()})
    # The page is the HTML template with the trace inlined ("</" escaped so no string can close the script tag).
    page = Path("scratch/pipeline_trace_page_20261103.html").read_text(encoding="utf-8")
    data = json.dumps(out, ensure_ascii=False).replace("</", "<\\/")
    (OUT / "2026-11-03_da_vs_amoral_pipeline_trace.html").write_text(page.replace("__TRACE_DATA__", data), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:4])
