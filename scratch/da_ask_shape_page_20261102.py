# ABOUTME: Builds the advice-vs-action audit page: label rates, every flagged row, and real stage-by-stage traces.
# ABOUTME: Run after the subagent labels exist: uv run --no-sync python scratch/da_ask_shape_page_20261102.py <trace_id> ...
import json
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

LOW = ("dougalldeepmind/2026-10-04-da-low-synth", "c8a61e2661be8d564b27bfb513ba3f31eec27897")
DA = ("dougalldeepmind/2026-10-30-da-synth", "8fadea6299915605d0b270feddfd85d91344e58f")
# Where the 2026-10-30 rows were generated, by id prefix: `t*` in the da-unclear run (republished into 2026-10-02-da-synth,
# same recipe as da-low minus the stakes sentences), `r2_` in 2026-10-02-da-synth's second run, `r3_` in 2026-10-30 itself.
DA_UNCLEAR = ("dougalldeepmind/2026-10-02-da-unclear-synth", "80ad379319762731a4410b212eb019f777949bb1")
DA_OCT02 = ("dougalldeepmind/2026-10-02-da-synth", "305914d58627457002f61d86ba69866f6fa2b9a9")
WORK = Path("output/2026-11-02_da_ask_shape")
TEMPLATE = Path(__file__).with_name("da_ask_shape_page_20261102.html")
OUT = Path("output/2026-11-02_da_ask_shape_audit.html")
STAGE_FIELDS = ("scenario_id", "trait_id", "trait_name", "ai", "sector", "ai_text", "sector_text",
                "draft_system", "draft_user", "system", "user", "refine_changes", "domain", "situation", "shortcut",
                "draft_reasoning", "draft_response", "rewrite_audit", "reasoning", "response", "rewrite_changes")


def jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def fetch(repo_rev, name):
    repo, rev = repo_rev
    return hf_hub_download(repo, name, repo_type="dataset", revision=rev)


def da_source(sid):
    return DA if sid.startswith("r3_") else DA_OCT02 if sid.startswith("r2_") else DA_UNCLEAR


def stage_records():
    """(corpus, scenario_id) -> {stage-6 record + the writer's own scenario from stage 2}."""
    out = {}
    for key, src in [("low", LOW), ("da", DA_UNCLEAR), ("da", DA_OCT02), ("da", DA)]:
        writer = {s["scenario_id"]: s for s in jsonl(fetch(src, "stages/stage_2_write_scenarios.jsonl"))}
        for s in jsonl(fetch(src, "stages/stage_6_revise_responses.jsonl")):
            sid = s["scenario_id"]
            if key == "da" and da_source(sid) != src:
                continue
            w = writer.get(sid, {})
            out[(key, sid)] = {k: s.get(k, "") for k in STAGE_FIELDS} | {
                "writer_domain": w.get("domain", ""), "writer_situation": w.get("situation", ""),
                "writer_shortcut": w.get("shortcut", "")}
    return out


def group(corpus, sid):
    if corpus == "low":
        return "low"
    return "da_oct30" if sid.startswith("r3_") else "da_oct02"


def one_rating_per_row():
    """Each row is rated once: the Sonnet 5 pass where it reached the row, a subagent for the rest."""
    labels = {}
    for d in jsonl(WORK / "labels.jsonl"):
        if d["label"] != "UNPARSED":
            labels[(d["corpus"], d["scenario_id"])] = {**d, "rater": "sonnet5"}
    for p in sorted((WORK / "agent").glob("da_rest_*.jsonl")):
        for d in jsonl(p):
            key = ("da", d["scenario_id"])
            assert key not in labels, f"{key} rated twice"
            labels[key] = {**d, "corpus": "da", "rater": "subagent"}
    return labels


def main():
    finals = {("low", r["metadata"]["scenario_id"]): r for r in jsonl(fetch(LOW, "dataset.jsonl"))}
    finals |= {("da", r["metadata"]["scenario_id"]): r for r in jsonl(fetch(DA, "dataset.jsonl"))}
    labels = one_rating_per_row()
    missing = set(finals) - set(labels)
    assert not missing, f"{len(missing)} rows unrated, e.g. {sorted(missing)[:3]}"
    drafts_path = WORK / "agent" / "drafts.jsonl"
    drafts = {(d["corpus"], d["scenario_id"]): d["draft_label"] for d in jsonl(drafts_path)} if drafts_path.exists() else {}
    drafts |= {k: v["draft_label"] for k, v in labels.items() if v.get("draft_label")}
    stages = stage_records()

    counts, flagged = Counter(), []
    for (corpus, sid), lab in labels.items():
        m = finals[(corpus, sid)]["metadata"]
        g = group(corpus, sid)
        for axis in ("all", f"ai:{m['ai']}", f"trait:{m['trait_id']}"):
            counts[f"{g}|{axis}|{lab['label']}"] += 1
        counts[f"rater|{g}|{lab['rater']}"] += 1
        if lab["label"] == "ADVICE":
            continue
        other = labels.get(("da" if corpus == "low" else "low", sid))
        msgs = finals[(corpus, sid)]["messages"]
        flagged.append({"group": g, "scenario_id": sid, "trait_id": m["trait_id"], "ai": m["ai"], "domain": m["domain"],
                        "label": lab["label"], "ask": lab["ask"], "why": lab["why"], "rater": lab["rater"],
                        "draft_label": drafts.get((corpus, sid)), "other_label": other and other["label"],
                        "system": msgs[0]["content"], "user": msgs[1]["content"], "response": msgs[2]["content"]})
    flagged.sort(key=lambda f: (f["group"], f["label"] != "ACT", f["scenario_id"]))
    agree = Counter()

    traces = []
    for arg in sys.argv[1:]:
        sid, _, note = arg.partition("=")
        traces.append({"id": sid, "note": note, "low": stages.get(("low", sid)), "da": stages.get(("da", sid)),
                       "labels": {k: labels.get((k, sid), {}).get("label") for k in ("low", "da")}})

    data = {"counts": counts, "flagged": flagged, "traces": traces,
            "corpora": {"low": LOW, "da": DA, "da_oct02": DA_OCT02, "da_unclear": DA_UNCLEAR}}
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    OUT.write_text(html, encoding="utf-8")
    print(f"{OUT}: {len(flagged)} flagged, {len(traces)} traces, {OUT.stat().st_size / 1e6:.2f} MB")
    print(Counter((group(*k), v["label"]) for k, v in labels.items()))


if __name__ == "__main__":
    main()
