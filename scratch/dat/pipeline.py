# ABOUTME: dat driver: deal (principle x sector) -> write -> revise -> environment (files rendered by code, scripts checked) -> system -> user ->
# ABOUTME: explore in a Docker sandbox -> select (incl. the steering check) -> respond -> rewrite -> export rows + manifest.
"""Run the from-scratch dat recipe. Standalone draft in scratch/: it reuses the repo's constitution
segmenter, da.yaml's sector list, the OpenRouter client, and daa2's Docker sandbox (scratch/
da_agentified/helpers.py), and keeps every stage's information boundary in `stage_inputs()`.

    uv run python scratch/dat/pipeline.py --smoke            # 9 principles x 1 sector x 1 situation
    uv run python scratch/dat/pipeline.py --smoke --dry      # deal + prompts only, no paid calls
    uv run python scratch/dat/pipeline.py --total 750 --workers 8

Output: output/synth_dat/<run>/ with stage snapshots (one jsonl per stage), rendered files per row,
dataset.jsonl (chat rows: system, user, exploration turns with tool_calls + tool results, final
assistant turn) and manifest.json (yield per stage and per trait, select stats,
costs). Nothing is pushed by this script; publish_* is a separate step once the rows are read.

NOT YET VERIFIED END TO END: written 2026-10-06 as the draft to review; the prompts are the
deliverable, the wiring follows daa2's, and --dry exercises everything up to the first paid call.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from omegaconf import OmegaConf

from scratch.da_agentified.helpers import Sandbox, check_docker, ensure_image, tool_result
from scratch.dat import prompts as P
from scratch.dat.renderers import missing_witnesses, render
from src.data.synth.ours.constitution import segment
from src.infra.endpoints.openrouter import OpenRouterClient

REPO = Path(__file__).resolve().parents[2]
CONSTITUTION = REPO / P.CONSTITUTION
DA_YAML = REPO / P.SECTORS_FROM
OUT_ROOT = REPO / "output/synth_dat"

MODELS = P.MODELS  # per stage, from dat.yaml; every call records which model served it
TOOLS = [  # the agent's tool schema at export: a shell, nothing bespoke
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command on the deployment host.",
                                      "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write a file on the deployment host (a draft, a record, a script of the agent's own).",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
]
READ_ONLY = re.compile(r"^\s*(cat|head|tail|ls|find|grep|egrep|wc|sort|uniq|sed\s+-n|python3\s+-c|du|file|stat|tree|diff|python3\s+/|bash\s+/|/[\w./-]+\.(py|sh)\b)")
MAX_LOOKS = P.MAX_LOOKS


# ---------------------------------------------------------------------------------------
# deal
# ---------------------------------------------------------------------------------------

def sectors() -> dict[str, str]:
    """da.yaml's sector labels and their one-line texts (the 'any' entry dropped)."""
    cfg = OmegaConf.load(DA_YAML)
    sec = None
    for st in cfg.stages:
        if st.get("kind") == "scenarios":
            sec = st.rotate.sector
    assert sec is not None, "da.yaml has no scenarios stage with a rotate.sector axis"
    return {k: str(v).strip("\n- ") for k, v in sec.text.items() if k != "any"}


DATE_WINDOW = tuple(dt.datetime.strptime(x, "%Y-%m-%d %H:%M") for x in P.CLOCK["window"])  # dealt clock, from dat.yaml


def deal_clock(rng: random.Random) -> dt.datetime:
    """A weekday date and a working-hour time, uniformly over DATE_WINDOW (models cluster if asked to pick)."""
    lo, hi = DATE_WINDOW
    while True:
        when = lo + dt.timedelta(seconds=rng.randrange(int((hi - lo).total_seconds())))
        if (when.weekday() < 5 or not P.CLOCK.get("weekdays_only", True)) and P.CLOCK["hours"][0] <= when.hour < P.CLOCK["hours"][1]:
            return when.replace(second=0, microsecond=0)


def deal(n_per_trait: int, seed: int) -> list[dict]:
    """(trait, sector, clock) hands: each trait meets the sectors dealt least so far (da's rule, simplified);
    the clock is dealt in code so dates spread instead of clustering on the model's sense of now."""
    traits = segment(CONSTITUTION)
    labels = list(sectors())
    rng = random.Random(seed)
    rng.shuffle(labels)
    hands, k = [], 0
    for t in traits:
        for _ in range(n_per_trait):
            hands.append({"trait_id": t.trait_id, "trait_name": t.name, "trait_text": t.text, "sector": labels[k % len(labels)],
                          "today": deal_clock(rng).strftime("%A %Y-%m-%d %H:%M")})
            k += 1
    return hands


# ---------------------------------------------------------------------------------------
# model calls
# ---------------------------------------------------------------------------------------

class Calls:
    def __init__(self, run_dir: Path):
        self.client = OpenRouterClient()
        self.cost = Counter()
        self.n = Counter()
        self.log = (run_dir / "calls.jsonl").open("a")

    def json(self, stage: str, system: str, user: str, schema: dict | None, temperature: float = 0.7, max_tokens: int = 8000) -> dict:
        model = MODELS[stage]
        for attempt in range(3):
            res = self.client.chat(model, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   temperature=temperature, max_tokens=max_tokens,
                                   **({"response_format": {"type": "json_object"}} if schema else {}))
            self.n[stage] += 1
            self.cost[stage] += res.cost or 0.0
            self.log.write(json.dumps({"stage": stage, "model": model, "provider": res.provider, "cost": res.cost, "tokens": [res.prompt_tokens, res.completion_tokens]}) + "\n")
            text = res.content.strip()
            if schema is None:
                return {"text": text}
            m = re.search(r"\{.*\}", text, re.S)
            try:
                return json.loads(m.group(0) if m else text)
            except json.JSONDecodeError:
                if attempt == 2:
                    raise
        raise RuntimeError("unreachable")


def tagged(text: str, tags: tuple[str, ...]) -> dict:
    out = {}
    for t in tags:
        m = re.search(rf"<{t}>\s*(.*?)\s*</{t}>", text, re.S)
        if not m:
            raise ValueError(f"missing <{t}> in model output")
        out[t] = m.group(1)
    return out


# ---------------------------------------------------------------------------------------
# the information each stage is given (the recipe's contract, in one place)
# ---------------------------------------------------------------------------------------

def stage_inputs(stage: str, row: dict) -> dict:
    """Return exactly the row fields a stage may see. Anything not returned here never reaches it."""
    sc = row.get("scenario", {})
    if stage == "environment":  # from write's organisation and role, no prompts: nothing to shape the files toward
        return {"organisation": sc["organisation"], "deployment": sc["deployment"], "today": row["today"],
                "world_facts": [{k: f[k] for k in ("id", "statement")} for f in sc["world_facts"]]}
    if stage == "system":  # the system prompt describes what actually exists: the listing, paths only
        return {**{k: sc[k] for k in ("organisation", "deployment", "task")}, "paths": sorted(row["files"])}
    if stage == "user":  # the listing, paths only: an operator points at files without the writer knowing their contents
        return {"system": row["system"], "operator": sc["operator"], "task": sc["task"], "paths": sorted(row["files"]), "today": row["today"]}
    if stage == "explore":
        return {"system": row["system"], "user": row["user"]}
    if stage in ("respond", "rewrite"):
        return {"system": row["system"], "user": row["user"], "steps": row["steps"], "trait_text": row["trait_text"]}
    if stage == "select":  # the informed judge: it knows the tension, so it also reads every file and script output
        return {"system": row["system"], "user": row["user"], "steps": row["steps"], "scenario": sc,
                "trait_name": row["trait_name"], "trait_text": row["trait_text"],
                "files": row["files"], "script_outputs": row.get("script_outputs", {})}
    raise KeyError(stage)


# ---------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------

def write_situations(calls: Calls, hand: dict, n: int) -> list[dict]:
    out = calls.json("write", P.system_of("write").replace("{n}", str(n)), P.write_prompt(hand["trait_name"], hand["trait_text"], sectors()[hand["sector"]], n), P.schema_of("write"))
    return out["situations"]


def revise_situation(calls: Calls, hand: dict, sc: dict) -> dict:
    return calls.json("revise", P.system_of("revise"), P.revise_prompt(hand["trait_name"], hand["trait_text"], sc), P.schema_of("revise"), temperature=0.3)


def write_system(calls: Calls, row: dict) -> dict:
    return calls.json("system", P.system_of("system"), P.system_prompt(stage_inputs("system", row)), P.schema_of("system"), temperature=0.7)


def write_user(calls: Calls, row: dict) -> dict:
    return calls.json("user", P.system_of("user"), P.user_prompt(stage_inputs("user", row)), P.schema_of("user"), temperature=0.7)


def environment(calls: Calls, row: dict, feedback: str = "") -> tuple[dict[str, str], list[dict], list[str]]:
    """Returns rendered files {path: text}, the file entries, and problems (missing witnesses, uncarried facts, scripts that fail)."""
    inp = stage_inputs("environment", row)
    user = P.environment_prompt(inp)
    if feedback:
        user += "\n\nA REVIEWER FOUND THESE PROBLEMS IN YOUR PREVIOUS ATTEMPT; return the full JSON again with them fixed:\n" + feedback
    out = calls.json("environment", P.system_of("environment"), user, P.schema_of("environment"), temperature=0.7, max_tokens=12000)
    row["dirs"] = out.get("dirs") or []
    files, problems = {}, []
    wanted = {f["id"] for f in inp["world_facts"]}
    carried = {f["id"] for e in out["files"] for f in e.get("facts", [])}
    for fid in sorted(wanted - carried):
        problems.append(f"fact {fid} is carried by no file")
    for e in out["files"]:
        try:
            text = render(e)
        except (AssertionError, KeyError) as err:
            problems.append(f"{e.get('path')}: {err}")
            continue
        files[e["path"]] = text
        if e["file_type"] != "script":  # a script's facts are checked against what it prints, below
            for fid in missing_witnesses(e, text):
                problems.append(f"{e['path']}: witness for fact {fid} not in rendered text")
    row["script_check"] = {}
    if files and not problems:  # scripts must run: each check command exits 0 in a sandbox of these files, and prints its facts
        scripts = [e for e in out["files"] if e["file_type"] == "script"]
        if scripts:
            with Sandbox(files, dt.datetime.strptime(row["today"], "%A %Y-%m-%d %H:%M")) as box:
                for e in scripts:
                    cmd = e["script"]["check"]
                    res = box.run(cmd)
                    row["script_check"][e["path"]] = cmd
                    row.setdefault("script_outputs", {})[cmd] = res.get("stdout", "")
                    if res.get("returncode", 1) != 0:
                        problems.append(f"{e['path']}: check command `{cmd}` exited {res.get('returncode')}: {(res.get('stderr') or res.get('stdout') or '')[-300:]}")
                        continue
                    for fid in missing_witnesses(e, res.get("stdout", "")):
                        problems.append(f"{e['path']}: witness for fact {fid} not in the output of `{cmd}`")
    return files, out["files"], problems


def explore(calls: Calls, row: dict, box: Sandbox) -> list[dict]:
    inp = stage_inputs("explore", row)
    steps: list[dict] = []
    for _ in range(MAX_LOOKS):
        out = calls.json("explore", P.system_of("explore"), P.explore_prompt(inp["system"], inp["user"], steps), P.schema_of("explore"), temperature=0.5)
        if out.get("done") or not out.get("command"):
            break
        cmd = out["command"].strip()
        if not READ_ONLY.match(cmd):
            steps.append({"thought": out["thought"], "command": cmd, "stdout": "", "stderr": "command refused: exploration is read-only", "returncode": 126})
            continue
        res = box.run(cmd)
        steps.append({"thought": out["thought"], "command": cmd, "stdout": res["stdout"], "stderr": res.get("stderr", ""), "returncode": res.get("returncode", 0)})
    return steps


def ensure_coverage(row: dict, steps: list[dict], box: Sandbox) -> list[str]:
    """Append one look per tension fact whose witness no output shows. Returns the appended fact ids."""
    seen = "\n".join(s["stdout"] for s in steps)
    appended = []
    for f in row["scenario"]["world_facts"]:
        if f["bearing"] != "tension":
            continue
        w, path = row["witness"].get(f["id"], ""), row["fact_path"].get(f["id"])
        if w and path and w not in seen:
            cmd = row.get("script_check", {}).get(path) or f"cat {path}"   # a script-carried fact is reached by running it
            res = box.run(cmd)
            steps.append({"thought": "", "command": cmd, "stdout": res["stdout"], "stderr": res.get("stderr", ""), "returncode": res.get("returncode", 0), "appended_for": f["id"]})
            seen += "\n" + res["stdout"]
            appended.append(f["id"])
    return appended


def select(calls: Calls, row: dict) -> dict:
    inp = stage_inputs("select", row)
    return calls.json("select", P.system_of("select"), P.select_prompt(inp["trait_name"], inp["trait_text"], inp["scenario"], inp["system"], inp["user"], inp["steps"],
                                                                        inp["files"], inp["script_outputs"]), P.schema_of("select"), temperature=0.0, max_tokens=4000)


def respond(calls: Calls, row: dict) -> dict:
    inp = stage_inputs("respond", row)
    sys = P.system_of("respond").replace("{trait_text}", inp["trait_text"])
    draft = tagged(calls.json("respond", sys, P.respond_prompt(inp["system"], inp["user"], inp["steps"], TOOLS), None, temperature=0.7)["text"], P.RESPOND_TAGS)
    rev = tagged(calls.json("rewrite", P.system_of("rewrite"), P.rewrite_prompt(inp["system"], inp["user"], inp["steps"], TOOLS, draft), None, temperature=0.3)["text"], P.RESPOND_TAGS + ("changes",))
    return {"draft": draft, "final": rev}


# ---------------------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------------------

def thought_for_appended(steps: list[dict]) -> None:
    for s in steps:
        if not s["thought"]:
            verb = "run" if not s["command"].startswith("cat ") else "read"
            s["thought"] = f"I want to {verb} {s['command'].split()[-1]} before I decide anything."


def assemble(row: dict) -> dict:
    msgs = [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}]
    for i, s in enumerate(row["steps"]):
        cid = f"call_{i:02d}"
        msgs.append({"role": "assistant", "content": "", "reasoning_content": s["thought"],
                     "tool_calls": [{"id": cid, "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": s["command"]})}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": tool_result(s["stdout"], s["returncode"], s["stderr"])})
    fin = row["final"]
    last = {"role": "assistant", "reasoning_content": fin["reasoning"], "content": fin["response"]}
    tc = fin["tool_call"].strip()
    if tc and tc.lower() != "nothing":
        call = json.loads(tc)
        last["tool_calls"] = [{"id": "call_final", "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}]
    msgs.append(last)
    return {"scenario_id": row["scenario_id"], "messages": msgs, "tools": TOOLS,
            "metadata": {"trait_id": row["trait_id"], "trait_name": row["trait_name"], "sector": row["sector"], "domain": row["scenario"]["organisation"],
                         "situation": row["scenario"]["situation"], "supervise": "final", "select": row["select"],
                         "appended_looks": row.get("appended", []), "n_looks": len(row["steps"])}}


# ---------------------------------------------------------------------------------------
# per-row flow
# ---------------------------------------------------------------------------------------

def process(calls: Calls, hand: dict, sc: dict, idx: int, run_dir: Path, dry: bool) -> dict:
    row = {**hand, "scenario_id": f"{hand['trait_id']}_{hand['sector']}_s{idx:03d}", "scenario": sc, "stage_reached": "revise"}
    if dry:
        return row
    files, recs, problems = environment(calls, row)
    if problems:
        files, recs, problems = environment(calls, row, feedback="\n".join(f"- {p}" for p in problems))
    if problems:
        row.update(stage_reached="environment", dropped="environment: " + "; ".join(problems[:5]))
        return row
    row["witness"] = {f["id"]: f["witness"] for r in recs for f in r.get("facts", [])}
    row["fact_path"] = {f["id"]: r["path"] for r in recs for f in r.get("facts", [])}
    row["files"] = files
    row["dirs"] = row.get("dirs") or sorted({p.rsplit("/", 2)[0] for p in files})
    row.update(system=write_system(calls, row)["system"], stage_reached="system")
    row.update(user=write_user(calls, row)["user"], stage_reached="user")
    now = dt.datetime.strptime(row["today"], "%A %Y-%m-%d %H:%M")
    with Sandbox(files, now) as box:
        steps = explore(calls, row, box)
        row["appended"] = ensure_coverage(row, steps, box)
    thought_for_appended(steps)
    row["steps"] = steps
    row["select"] = select(calls, row)
    row["stage_reached"] = "select"
    if not row["select"]["keep"]:
        row["dropped"] = "select: " + row["select"]["notes"]
        return row
    out = respond(calls, row)
    row.update(draft=out["draft"], final=out["final"], stage_reached="rewrite")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry", action="store_true", help="deal and write prompts only; no paid calls")
    ap.add_argument("--total", type=int, default=750)
    ap.add_argument("--per-call", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    n_traits = len(segment(CONSTITUTION))
    per_trait = 1 if a.smoke else max(1, a.total // (n_traits * a.per_call))
    run_dir = OUT_ROOT / f"{'smoke_' if a.smoke else ''}{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True)
    hands = deal(per_trait, a.seed)
    (run_dir / "hands.jsonl").write_text("\n".join(json.dumps(h) for h in hands) + "\n")
    if a.dry:
        for h in hands[:2]:
            print(P.write_prompt(h["trait_name"], h["trait_text"][:300] + "...", sectors()[h["sector"]], a.per_call)[:1200], "\n---")
        print(f"dry: {len(hands)} hands dealt, {a.per_call} situations each -> {run_dir}")
        return
    check_docker()
    ensure_image()
    calls = Calls(run_dir)
    rows: list[dict] = []
    for h in hands:  # diversity comes from the dealt sector and "distinct within this set" (da turned its
        for sc in write_situations(calls, h, a.per_call):  # avoid list off on 2026-10-02: the writer copied its flavour)
            rv = revise_situation(calls, h, sc)
            rows.append({"hand": h, "scenario": rv["situation"], "revise": {"changes": rv["changes"]}})
    (run_dir / "stage_write_revise.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    done: list[dict] = []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process, calls, r["hand"], r["scenario"], i, run_dir, False): i for i, r in enumerate(rows)}
        for f in as_completed(futs):
            try:
                done.append(f.result())
            except Exception as e:  # keep the run going; the manifest counts it
                done.append({"scenario_id": f"row_{futs[f]}", "dropped": f"error: {type(e).__name__}: {e}", "stage_reached": "error"})
    kept = [assemble(r) for r in done if r.get("final")]
    (run_dir / "rows.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in done) + "\n")
    (run_dir / "dataset.jsonl").write_text("\n".join(json.dumps(r) for r in kept) + "\n")
    manifest = {
        "run": run_dir.name, "seed": a.seed, "models": MODELS, "hands": len(hands), "situations": len(rows),
        "kept": len(kept),
        "stage_reached": dict(Counter(r.get("stage_reached") for r in done)),
        "select_source": dict(Counter(r["select"]["source"] for r in done if r.get("select"))),
        "select_unclear": dict(Counter(r["select"]["unclear"] for r in done if r.get("select"))),
        "settled_by_nonempty": sum(1 for r in done if r.get("select") and r["select"].get("settled_by")),
        "kept_per_trait": dict(Counter(r["metadata"]["trait_id"] for r in kept)),
        "calls": dict(calls.n), "cost_usd": {k: round(v, 3) for k, v in calls.cost.items()}, "cost_total_usd": round(sum(calls.cost.values()), 2),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
