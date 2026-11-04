# ABOUTME: Runs the da-amoral recipe's five LLM stages by hand (subagents answer the rendered prompts, no OpenRouter).
# ABOUTME: Run: uv run --no-sync python scratch/da_amoral_local_20261102.py {deal-check|render <stage>|collect <stage>}
import json
import os
import re
import sys
from pathlib import Path

from omegaconf import OmegaConf

from src.data.synth.ours.constitution import segment
from src.data.synth.ours.stage_operators import deal_unit_axes, scenario_batches

CONFIG = os.environ.get("DA_AMORAL_CONFIG", "configs/data/synth/da-amoral.yaml")
TOTAL = 20
WORK = Path(os.environ.get("DA_AMORAL_WORK", "output/2026-11-02_da_amoral/local"))
PREV_SMOKE = Path("output/synthdoc_v3/smoke_20261102_175201/stage_2_write_scenarios.jsonl")
STAGES = ["write_scenarios", "draft_prompts", "revise_prompts", "draft_responses", "revise_responses"]


def load_cfg(config=CONFIG):
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    cfg.update(cfg["smoke"])
    cfg["total_scenarios"] = TOTAL
    return cfg


def stage_cfg(cfg, name):
    return next(s for s in cfg["stages"] if s["name"] == name)


def deal(cfg):
    """The writer calls a --smoke run with smoke.total_scenarios=20 makes, with the labels the engine deals them."""
    traits = segment(cfg["constitution"])
    ids = [t.trait_id for t in traits]
    batches = scenario_batches(len(ids), cfg, ids)
    planned = {}
    for ti, _bi, _n in batches:
        planned[ti] = planned.get(ti, 0) + 1
    rot = stage_cfg(cfg, "write_scenarios")["rotate"]
    per_unit = {k: v for k, v in rot.items() if v.get("per_trait")}
    placed = deal_unit_axes(per_unit, ids, planned)
    calls = []
    for ti, bi, _n in batches:
        axes = {name: (rot[name].get("fixed") or {}).get(ids[ti]) or placed[(name, ti)][bi] for name in per_unit}
        calls.append({"scenario_id": f"{ids[ti]}_b{bi:02d}_s000", "trait_id": ids[ti], "trait_name": traits[ti].name,
                      "trait_text": traits[ti].text, **axes,
                      **{f"{k}_text": str(rot[k]["text"].get(axes[k], "")) for k in per_unit}})
    return calls


def read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")] if path.exists() else []


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def records(stage):
    """The records a stage's prompts are rendered from: the deal for the writer, else the previous stage's output."""
    i = STAGES.index(stage)
    return deal(load_cfg()) if i == 0 else read(WORK / f"{STAGES[i - 1]}.jsonl")


def render(stage):
    cfg = load_cfg()
    p = stage_cfg(cfg, stage)["prompts"]
    shared = {"n": 1, "style_guidance": cfg["style_guidance"]}
    rows = [{"id": r["scenario_id"], "system": p["system"].format(**shared, **r), "user": p["user"].format(**shared, **r)}
            for r in records(stage)]
    write(WORK / f"prompts_{stage}.jsonl", rows)
    # One readable file per call for the subagents standing in for the model: they answer into answers/<stage>/<id>.txt.
    folder = WORK / "prompts" / stage
    folder.mkdir(parents=True, exist_ok=True)
    (WORK / "answers" / stage).mkdir(parents=True, exist_ok=True)
    for r in rows:
        (folder / f"{r['id']}.txt").write_text(f"=== SYSTEM ===\n{r['system']}\n\n=== USER ===\n{r['user']}\n", encoding="utf-8")
    print(f"{len(rows)} prompts -> {folder}")


def first_json(text, opener):
    start = text.find(opener)
    end = text.rfind("]" if opener == "[" else "}")
    return json.loads(text[start:end + 1])


def tags(text, names):
    out = {}
    for name in names:
        m = re.search(rf"<{name}>\s*(.*?)\s*</{name}>", text, re.S)
        if not m:
            raise ValueError(f"no <{name}> block")
        out[name] = m.group(1)
    return out


def lint_hits(stage, rec):
    lint = stage_cfg(load_cfg(), stage).get("lint") or {}
    hits = []
    for field in lint.get("fields", []):
        text = str(rec.get(field, ""))
        hits += [f"{field}~{pat}" for pat in lint.get("ban_patterns", []) if re.search(pat, text, re.I | re.M)]
        if lint.get("min_chars") and len(text) < lint["min_chars"]:
            hits.append(f"{field}<min_chars")
    return hits


def collect(stage):
    """Parse every subagent answer for a stage, exactly as the stage's `save:` map would, into the stage's records."""
    by_id = {r["scenario_id"]: r for r in records(stage)}
    answers = {p.stem: p.read_text(encoding="utf-8") for p in (WORK / "answers" / stage).glob("*.txt")}
    out, problems = [], []
    for sid, rec in by_id.items():
        if sid not in answers:
            problems.append(f"{sid}: no answer")
            continue
        raw = answers[sid]
        try:
            if stage == "write_scenarios":
                s = first_json(raw, "[")[0]
                new = {"domain": s.get("domain", ""), "situation": s["situation"], "shortcut": s.get("shortcut", "")}
            elif stage == "draft_prompts":
                j = first_json(raw, "{")
                new = {"draft_system": j["system"], "draft_user": j["user"], "system": j["system"], "user": j["user"]}
            elif stage == "revise_prompts":
                j = first_json(raw, "{")
                new = {"system": j["system"], "user": j["user"], "refine_changes": j.get("changes", ""),
                       "situation": j["situation"], "shortcut": j["shortcut"], "domain": j["domain"]}
            elif stage == "draft_responses":
                t = tags(raw, ["reasoning", "response"])
                new = {"draft_reasoning": t["reasoning"], "draft_response": t["response"],
                       "reasoning": t["reasoning"], "response": t["response"]}
            else:
                t = tags(raw, ["audit", "reasoning", "response", "changes"])
                new = {"reasoning": t["reasoning"], "response": t["response"], "rewrite_changes": t["changes"],
                       "rewrite_audit": t["audit"]}
        except Exception as e:  # noqa: BLE001 -- reported, never silently dropped
            problems.append(f"{sid}: unparseable ({e})")
            continue
        row = {**rec, **new}
        if stage == "write_scenarios":
            row = {k: v for k, v in row.items() if k != "trait_text"} | {"trait_text": rec["trait_text"]}
        hits = lint_hits(stage, row)
        if hits:
            problems.append(f"{sid}: lint {hits}")
        out.append(row)
    write(WORK / f"{stage}.jsonl", out)
    print(f"{stage}: {len(out)}/{len(by_id)} records")
    for p in problems:
        print("  !", p)


def deal_check():
    """The deal reconstruction must reproduce the real 2026-11-02 smoke's labels before it is trusted for a new list."""
    print(json.dumps([(c["scenario_id"], c["ai"], c["sector"]) for c in deal(load_cfg())]))


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"deal-check": deal_check, "render": lambda: render(sys.argv[2]), "collect": lambda: collect(sys.argv[2])}[cmd]()
