# ABOUTME: dat driver: deal (principle x sector) -> write -> revise -> environment (files rendered by code, scripts checked) -> system -> user -> revise_prompts ->
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
from scratch.dat.renderers import render
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
# One leading `cd <dir> &&` is allowed before any of these, and a script may be named by a relative path: on
# 2026-10-10 t4 ran `cd /srv/hr-analytics && python3 scripts/cross_check.py`, the old guard refused it, and the
# agent repeated the refusal to the operator in its final turn.
READ_ONLY = re.compile(r"^\s*(?:cd\s+\S+\s*&&\s*)?(cat|head|tail|ls|find|grep|egrep|wc|sort|uniq|sed\s+-n|python3\s+-c|du|file|stat|tree|diff"
                       r"|python3\s+[\w./-]+\.py\b|bash\s+[\w./-]+\.sh\b|[\w./-]+\.(py|sh)\b)")
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


NAME_SEEDS = ("Okonkwo", "Varga", "Lindqvist", "Beaumont", "Castellano", "Haugen", "Petrakis", "Nakamura", "Oyelaran", "Dvorak",
              "Whitlock", "Sandoval", "Kowalczyk", "Brennan", "Achterberg", "Marchetti", "Ferreira", "Novak", "Halloran", "Ibarra",
              "Tremont", "Calder", "Birchwood", "Stonebridge", "Redcliffe", "Fenwick", "Harrowgate", "Kelso", "Penrith", "Alnwick",
              "Dunmore", "Larkhill", "Wexford", "Thornbury", "Coldstream", "Ravenna", "Tamsin", "Mallory", "Quennell", "Ostrander")


def deal(n_per_trait: int, seed: int) -> list[dict]:
    """(trait, sector, clock) hands: each trait meets the sectors dealt least so far (da's rule, simplified);
    the clock is dealt in code so dates spread instead of clustering on the model's sense of now."""
    traits = segment(CONSTITUTION)
    labels = list(sectors())
    rng = random.Random(seed)
    rng.shuffle(labels)
    hands, k = [], 0
    for t in traits:
        for j in range(n_per_trait):
            hand = {"trait_id": t.trait_id, "trait_name": t.name, "trait_text": t.text, "sector": labels[k % len(labels)],
                    "today": deal_clock(rng).strftime("%A %Y-%m-%d %H:%M"), "name_seed": rng.choice(NAME_SEEDS),
                    "trait_note": P.TRAIT_NOTES.get(t.trait_id, ""), "ai_text": ""}
            if t.trait_id in P.AI_PRESENCE:  # t6: half self, half other (da's split), alternating within the trait
                label = ("self", "other")[j % 2]
                hand.update(ai_presence=label, ai_text=P.AI_PRESENCE[t.trait_id][label])
            hands.append(hand)
            k += 1
    return hands


# ---------------------------------------------------------------------------------------
# model calls
# ---------------------------------------------------------------------------------------

class Calls:
    def __init__(self, run_dir: Path):
        self.client = OpenRouterClient()
        # The shared client's 420 s per-request timeout is sized for da's 12k-token rewrite; the
        # environment stage writes up to 16k tokens of files in one reply, which can run longer.
        self.client.client = self.client.client.with_options(timeout=1500.0)   # 64k-token replies with thinking can run past 20 min
        self.cost = Counter()
        self.n = Counter()
        self.log = (run_dir / "calls.jsonl").open("a")

    def json(self, stage: str, system: str, user: str, schema: dict | None, temperature: float | None = None, max_tokens: int | None = None) -> dict:
        model = MODELS[stage]
        temperature = P.TEMPERATURE[stage] if temperature is None else temperature
        max_tokens = P.MAX_TOKENS[stage] if max_tokens is None else max_tokens
        if schema:  # the shape travels with every call; a bare top-level list is wrapped under the schema's one array key
            user = (user + "\n\nYour reply is ONE JSON object that is an instance of this JSON Schema, with real values filled in. "
                    "Do not return the schema itself. Plan silently: the reply begins with the opening brace and ends with the "
                    "closing brace, with no text before or after.\nJSON SCHEMA:\n" + json.dumps(schema))
        for attempt in range(3):
            extra = {"reasoning": dict(P.REASONING[stage])} if P.REASONING.get(stage) else {}
            res = self.client.chat(model, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   temperature=temperature, max_tokens=max_tokens, **({"extra_body": extra} if extra else {}))
            self.n[stage] += 1
            self.cost[stage] += res.cost or 0.0
            self.log.write(json.dumps({"stage": stage, "model": model, "provider": res.provider, "cost": res.cost, "tokens": [res.prompt_tokens, res.completion_tokens]}) + "\n")
            self.log.flush()
            text = res.content.strip()
            if schema is None:
                return {"text": text}
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
            start = min([i for i in (text.find("{"), text.find("[")) if i >= 0] or [0])
            try:
                parsed = json.loads(text[start:])
            except json.JSONDecodeError:
                m = re.search(r"[\[{].*[\]}]", text, re.S)
                try:
                    parsed = json.loads(m.group(0)) if m else None
                except json.JSONDecodeError:
                    parsed = None
            if isinstance(parsed, list):
                arrays = [k for k, v in schema.get("properties", {}).items() if v.get("type") == "array"]
                parsed = {arrays[0]: parsed} if len(arrays) == 1 else None
            if isinstance(parsed, dict) and {"type", "properties"} <= set(parsed):
                parsed = None  # the model echoed the schema instead of an instance
            if isinstance(parsed, dict):
                return parsed
            self.log.write(json.dumps({"stage": stage, "model": model, "unparsed": text[:2000]}) + "\n")
            if attempt == 2:
                raise ValueError(f"{stage}: no JSON object in the reply after 3 attempts")
        raise RuntimeError("unreachable")


def tagged(text: str, tags: tuple[str, ...], optional: tuple[str, ...] = ("tool_call", "changes")) -> dict:
    out = {}
    for t in tags:
        m = re.search(rf"<{t}>\s*(.*?)\s*(?:</{t}>|$)", text, re.S)
        if not m:
            if t in optional:
                out[t] = "nothing" if t == "tool_call" else ""
                continue
            raise ValueError(f"missing <{t}> in model output")
        out[t] = m.group(1).strip()
    return out


def tagged_call(calls: "Calls", stage: str, system: str, user: str, tags: tuple[str, ...]) -> dict:
    """A tagged-text stage with one retry; the raw reply is logged when the tags are missing."""
    for attempt in range(2):
        text = calls.json(stage, system, user, None)["text"]
        try:
            return tagged(text, tags)
        except ValueError as e:
            calls.log.write(json.dumps({"stage": stage, "untagged": text[:3000], "error": str(e)}) + "\n")
            calls.log.flush()
            if attempt == 1:
                raise


# ---------------------------------------------------------------------------------------
# the information each stage is given (the recipe's contract, in one place)
# ---------------------------------------------------------------------------------------

def stage_inputs(stage: str, row: dict) -> dict:
    """Return exactly the row fields a stage may see. Anything not returned here never reaches it."""
    sc = row.get("scenario", {})
    if stage == "environment":  # from write's organisation and role, no prompts: nothing to shape the files toward
        return {"organisation": sc["organisation"], "deployment": sc["deployment"], "today": row["today"],
                "facts": sc["facts"]}
    if stage == "system":  # the system prompt orients by directory only; the agent finds files by looking (Jamie, 2026-10-07)
        return {**{k: sc[k] for k in ("organisation", "deployment", "task")}, "dirs": sorted(row["dirs"])}
    if stage == "reader":  # the blind reader of the environment: the task and the files, never the scenario
        return {"task": sc["task"], "files": row["files"], "script_outputs": row.get("script_outputs", {})}
    if stage == "match":  # does what the reader concluded amount to the tension the designer intended?
        return {"facts": sc["facts"], "findings": row["reader"]["conclusions"]}
    if stage == "coverage":  # which of the intended facts has the agent's own reading not put before it, and where are they?
        return {"facts": sc["facts"], "where": row.get("where", []), "steps": row["steps"]}
    if stage == "user":  # the listing, paths only: an operator points at files without the writer knowing their contents
        return {"system": row["system"], "operator": sc["operator"], "task": sc["task"], "paths": sorted(row["files"]), "today": row["today"],
                "trait_note": row.get("trait_note", ""), "ai_text": row.get("ai_text", "")}
    if stage == "revise_prompts":  # the same view `user` had, plus the two drafts; still no facts, no principle
        return {"system": row["system"], "user": row["user"], "operator": sc["operator"], "task": sc["task"], "paths": sorted(row["files"]), "today": row["today"],
                "trait_note": row.get("trait_note", ""), "ai_text": row.get("ai_text", "")}
    if stage == "explore":
        return {"system": row["system"], "user": row["user"]}
    if stage in ("respond", "rewrite"):
        return {"system": row["system"], "user": row["user"], "steps": row["steps"], "trait_name": row["trait_name"], "trait_text": row["trait_text"]}
    if stage == "select":  # the informed judge: it knows the tension, so it also reads every file and script output
        return {"system": row["system"], "user": row["user"], "steps": row["steps"], "scenario": sc,
                "trait_name": row["trait_name"], "trait_text": row["trait_text"],
                "files": row["files"], "script_outputs": row.get("script_outputs", {}), "findings": row.get("findings", [])}
    raise KeyError(stage)


# ---------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------

def write_situations(calls: Calls, hand: dict, n: int) -> list[dict]:
    sector_text = "" if hand.get("trait_note") else sectors()[hand["sector"]]   # t6 is pinned to a label with no text (da)
    out = calls.json("write", P.system_of("write").replace("{n}", str(n)),
                     P.write_prompt(hand["trait_name"], hand["trait_text"], sector_text, n, hand["name_seed"],
                                    trait_note=hand.get("trait_note", ""), ai_text=hand.get("ai_text", ""), today=hand["today"]), P.schema_of("write"))
    return out["situations"]


def revise_situation(calls: Calls, hand: dict, sc: dict) -> dict:
    return calls.json("revise", P.system_of("revise"), P.revise_prompt(hand["trait_name"], hand["trait_text"], sc), P.schema_of("revise"))


def _one_string(out: dict, key: str, stage: str) -> dict:
    """A one-field stage sometimes comes back under another key ({"system_prompt": ...}); take the one string."""
    if key in out and isinstance(out[key], str):
        return out
    strings = [v for v in out.values() if isinstance(v, str) and len(v) > 40]
    if len(strings) == 1:
        return {key: strings[0]}
    raise ValueError(f"{stage}: expected one string under {key!r}, got keys {list(out)}")


def write_system(calls: Calls, row: dict) -> dict:
    return _one_string(calls.json("system", P.system_of("system"), P.system_prompt(stage_inputs("system", row)), P.schema_of("system")), "system", "system")


def write_user(calls: Calls, row: dict) -> dict:
    return _one_string(calls.json("user", P.system_of("user"), P.user_prompt(stage_inputs("user", row)), P.schema_of("user")), "user", "user")


def revise_prompts(calls: Calls, row: dict) -> dict:
    return calls.json("revise_prompts", P.system_of("revise_prompts"), P.revise_prompts_prompt(stage_inputs("revise_prompts", row)), P.schema_of("revise_prompts"))


def _fmt_problems(problems: list[dict]) -> str:
    return "\n".join(f"- {p['what']}" for p in problems)


def environment(calls: Calls, row: dict, feedback: str = "", repair: dict | None = None) -> tuple[dict[str, str], list[dict], list[dict]]:
    """Returns rendered files {path: text}, the file entries, and problems, each {paths, what}: files that do not
    render and scripts whose check fails (file-scoped, so repairable one file at a time), and facts the reader does
    not find at all (no path: the environment is redesigned).

    `repair` = {entries, files, problems} from the previous attempt: the model gets the SAME prompt it had the first
    time, plus every file as it stands and the entries behind the named paths, and returns replacement entries for
    those paths only. Everything else is kept byte for byte, then the whole set is checked again (the reader and the
    match see all files, since consistency is a property of the set, not of one file).
    """
    inp = stage_inputs("environment", row)
    user = P.environment_prompt(inp)
    if repair:
        paths = sorted({pth for pr in repair["problems"] for pth in pr["paths"]})
        broken = [e for e in repair["entries"] if e["path"] in paths]
        standing = "\n\n".join(f"=== {pth}\n{txt}" for pth, txt in repair["files"].items() if pth not in paths) or "(none)"
        user += ("\n\nREPAIR. These files were built and checked, and a reviewer found these problems:\n" + _fmt_problems(repair["problems"]) +
                 "\n\nReturn the JSON again with `dirs` unchanged and `files` holding ONLY replacement entries for: " + ", ".join(paths) +
                 ". Every other file stays exactly as it is and is NOT returned; your replacements must agree with them "
                 "(same ids, names, dates, conventions).\n\nTHE OTHER FILES, AS THEY STAND:\n" + standing +
                 "\n\nTHE ENTRIES BEING REPLACED, AS PREVIOUSLY RETURNED:\n" + json.dumps(broken, ensure_ascii=False, indent=1))
    elif feedback:
        user += "\n\nA REVIEWER FOUND THESE PROBLEMS IN YOUR PREVIOUS ATTEMPT; return the full JSON again with them fixed:\n" + feedback
    out = calls.json("environment", P.system_of("environment"), user, P.schema_of("environment"))
    files, problems = {}, []
    if repair:
        repl = {e["path"]: e for e in out.get("files", [])}
        missing = [pth for pth in paths if pth not in repl]
        if missing:
            problems.append({"paths": missing, "what": f"repair returned no replacement for: {', '.join(missing)}"})
        out["files"] = [repl.get(e["path"], e) for e in repair["entries"]]
        out["dirs"] = row.get("dirs") or out.get("dirs") or []
    row["dirs"] = out.get("dirs") or []
    # a repair rebuilds every file, so nothing from the previous attempt may survive on the row (2026-10-10 audit:
    # t4 and t2 carried the superseded script's output into select)
    for k in ("script_outputs", "reader", "findings"):
        row.pop(k, None)
    for e in out["files"]:
        try:
            text = render(e)
        except (AssertionError, KeyError) as err:
            problems.append({"paths": [e.get("path")], "what": f"{e.get('path')}: {err}"})
            continue
        files[e["path"]] = text
    row["script_check"] = {}
    if files and not problems:  # scripts must run: each check command exits 0 in a sandbox of these files, and prints its facts
        scripts = [e for e in out["files"] if e["file_type"] == "script"]
        if scripts:
            with Sandbox(files, dt.datetime.strptime(row["today"], "%A %Y-%m-%d %H:%M")) as box:
                for e in scripts:
                    cmd = e["script"].get("check") or f"python3 {e['path']}"
                    res = box.run(cmd)
                    row["script_check"][e["path"]] = cmd
                    row.setdefault("script_outputs", {})[cmd] = res.get("stdout", "")
                    if res.get("returncode", 1) != 0:
                        problems.append({"paths": [e["path"]], "what": f"{e['path']}: check command `{cmd}` exited {res.get('returncode')}: "
                                                                     f"{(res.get('stderr') or res.get('stdout') or '')[-300:]}"})
    if files and not problems:  # the blind reader, then: does what it concluded amount to the intended tension?
        row["files"] = files
        rd = calls.json("reader", P.system_of("reader"), P.reader_prompt(stage_inputs("reader", row)), P.schema_of("reader"))
        row["reader"] = rd
        row["findings"] = [c for c in rd.get("conclusions", []) if c.get("paths")]
        mt = calls.json("match", P.system_of("match"), P.match_prompt(stage_inputs("match", row)), P.schema_of("match"))
        row["match"] = mt
        row["where"] = [p for p in mt.get("where", []) if p in files or p in row.get("script_check", {}).values()]
        if not mt.get("found"):
            problems.append({"paths": [], "what": f"a careful reader does not find the facts in these files: {mt.get('missing', '')}"})
        elif not row["where"]:
            problems.append({"paths": [], "what": "the match stage named no existing file in which the facts show"})
    row.setdefault("env_attempts", []).append({"entries": out["files"], "rendered": dict(files), "problems": [p["what"] for p in problems],
                                               "repaired": sorted({pth for pr in repair["problems"] for pth in pr["paths"]}) if repair else []})
    return files, out["files"], problems


def _reached(p: str, s: dict) -> bool:
    """Did this step open the file (or RUN the script) at `p`? Reading a script's source does not show what
    it prints. The full path must appear in the command: a bare `ls` of the parent directory does not count
    (the old `command in path` clause cost t6 its crux fact on 2026-10-07)."""
    cmd = s["command"]
    target = p.split()[1] if p.startswith(("python3 ", "bash ")) and len(p.split()) > 1 else p
    if target not in cmd:
        return False
    if target.endswith((".py", ".sh")):
        return not cmd.lstrip().startswith(("cat ", "head ", "tail ", "less ", "sed ", "grep ", "wc "))
    return True


def _uncovered(row: dict, steps: list[dict]) -> list[str]:
    """The files the match stage says the intended facts show in that none of these steps reached. Path matching,
    so it is free and runs after every look (the halt); ensure_coverage asks a judge instead."""
    return [p for p in row.get("where") or [] if not any(_reached(p, s) for s in steps)]


def explore(calls: Calls, row: dict, box: Sandbox) -> list[dict]:
    inp = stage_inputs("explore", row)
    steps: list[dict] = []
    # The agent opens by listing the directories its deployment prompt names, the way a person would before
    # reading anything. The COMMAND is fixed -- it is not a judgement call, and leaving it to the model cost
    # four of eight looks to `find` (2026-10-10 t5) -- but the THOUGHT is still written by the explore model,
    # so the opening turn reads as the agent's own. It does not count against MAX_LOOKS.
    dirs = [d for d in (row.get("dirs") or []) if d]
    if dirs:
        cmd = "ls -la " + " ".join(dirs)
        thought = calls.json("explore", P.explore_thought_system(),
                             P.explore_thought_prompt(inp["system"], inp["user"], [], cmd),
                             P.explore_thought_schema())["thought"]
        res = box.run(cmd)
        steps.append({"thought": thought, "command": cmd, "stdout": res["stdout"],
                      "stderr": res.get("stderr", ""), "returncode": res.get("returncode", 0)})
    for _ in range(MAX_LOOKS):
        prompt = P.explore_prompt(inp["system"], inp["user"], steps)
        try:
            out = calls.json("explore", P.system_of("explore"), prompt, P.schema_of("explore"))
        except ValueError:
            # Sonnet sometimes answers with the bare thought sentence and no JSON (t8, 2026-10-07 smoke: three
            # times running). One more ask, naming the fault; if that fails too the agent's looks end here and
            # coverage supplies what the facts need, rather than the row dying at this stage.
            try:
                out = calls.json("explore", P.system_of("explore"),
                                 prompt + "\n\nYour previous reply was a sentence of thought with no JSON. Reply with ONLY the JSON "
                                          "object: that thought, the command it leads to (or done=true), nothing else.",
                                 P.schema_of("explore"))
            except ValueError:
                break
        if out.get("done") or not out.get("command"):
            break
        cmd = out["command"].strip()
        if not READ_ONLY.match(cmd):
            steps.append({"thought": out["thought"], "command": cmd, "stdout": "", "stderr": "command refused: exploration is read-only", "returncode": 126})
            continue
        res = box.run(cmd)
        steps.append({"thought": out["thought"], "command": cmd, "stdout": res["stdout"], "stderr": res.get("stderr", ""), "returncode": res.get("returncode", 0)})
        # Stop once the agent has reached every tension fact: more looks cannot change what it knows, and
        # ensure_coverage would append nothing anyway. `done` is left in the schema but has never fired
        # (13 of 14 rows ran to the cap across the 10-08 and 10-10 smokes).
        if not _uncovered(row, steps):
            row["explore_halt"] = "covered"
            return steps
    row.setdefault("explore_halt", "max_looks")
    return steps


def ensure_coverage(calls: "Calls", row: dict, steps: list[dict], box: Sandbox) -> list[str]:
    """Append one look per file the agent must still open for an intended fact it has not seen. Returns those paths.

    A judge is handed the intended facts, the files the match stage says they show in, and the transcript, and names
    the files still to open -- "seen" rather than "opened", so a fact reached by another route counts and a script
    only `cat`-ed does not. If the judge fails, the free path-matching test stands in. For each, the explore model
    writes, in character and from the transcript so far, the thought that leads to opening it (no empty thoughts,
    no steering wording), and code runs the command and records the real output.
    """
    if not row.get("where"):
        return []
    try:
        cv = calls.json("coverage", P.system_of("coverage"), P.coverage_prompt(stage_inputs("coverage", row)), P.schema_of("coverage"))
        row["coverage"] = cv
        gaps = [p for p in cv.get("open", []) if p in row["where"]]
    except ValueError:
        row["coverage"] = {"error": "judge returned no JSON; fell back to path matching"}
        gaps = _uncovered(row, steps)
    appended = []
    for path in gaps:
        cmd = path if path.startswith(("python3 ", "bash ")) else (row.get("script_check", {}).get(path) or f"cat {path}")
        thought = calls.json("explore", P.explore_thought_system(),
                             P.explore_thought_prompt(row["system"], row["user"], steps, cmd),
                             P.explore_thought_schema())["thought"]
        res = box.run(cmd)
        steps.append({"thought": thought, "command": cmd, "stdout": res["stdout"], "stderr": res.get("stderr", ""),
                      "returncode": res.get("returncode", 0), "appended_for": path})
        appended.append(path)
    return appended


def select(calls: Calls, row: dict) -> dict:
    inp = stage_inputs("select", row)
    return calls.json("select", P.system_of("select"), P.select_prompt(inp["trait_name"], inp["trait_text"], inp["scenario"], inp["system"], inp["user"], inp["steps"],
                                                                        inp["files"], inp["script_outputs"], inp["findings"]), P.schema_of("select"))


def respond(calls: Calls, row: dict) -> dict:
    inp = stage_inputs("respond", row)
    sys = P.system_of("respond").replace("{trait_name}", inp["trait_name"]).replace("{trait_text}", inp["trait_text"])
    draft = tagged_call(calls, "respond", sys, P.respond_prompt(inp["system"], inp["user"], inp["steps"], TOOLS), P.RESPOND_TAGS)
    rev = tagged_call(calls, "rewrite", P.system_of("rewrite"), P.rewrite_prompt(inp["system"], inp["user"], inp["steps"], TOOLS, draft), P.RESPOND_TAGS + ("changes",))
    return {"draft": draft, "final": rev}


# ---------------------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------------------

def assemble(row: dict) -> dict:
    msgs = [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}]
    for i, s in enumerate(row["steps"]):
        cid = f"call_{i:02d}"
        msgs.append({"role": "assistant", "content": "", "reasoning_content": s["thought"],
                     "tool_calls": [{"id": cid, "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": s["command"]})}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": tool_result(s["stdout"], s["returncode"], s["stderr"])})
    fin = row["final"]
    last = {"role": "assistant", "reasoning_content": fin["reasoning"], "content": fin["response"]}
    tc = re.sub(r"^```(?:json)?\s*|\s*```$", "", fin["tool_call"].strip(), flags=re.S)
    if tc and tc.lower() != "nothing":
        m = re.match(r"^(\w+)\((\{.*\})\)$", tc, re.S)   # `write_file({...})` function-call syntax
        if m:
            tc = json.dumps({"name": m.group(1), "arguments": json.loads(m.group(2))})
        elif re.match(r"^\w+\(\s*\w+\s*=", tc, re.S):       # `write_file(path="...", content="...")` keyword syntax
            import ast
            try:
                node = ast.parse(tc, mode="eval").body
                tc = json.dumps({"name": node.func.id, "arguments": {k.arg: ast.literal_eval(k.value) for k in node.keywords}})
            except (SyntaxError, ValueError, AttributeError):
                pass
        m2 = re.match(r"^(bash|write_file)\s*:\s*(.+)$", tc, re.S)     # `bash: cat /srv/x` prose form
        if m2 and m2.group(1) == "bash":
            tc = json.dumps({"name": "bash", "arguments": {"command": m2.group(2).strip()}})
        try:
            call = json.loads(tc, strict=False)   # strict=False: file contents with raw newlines inside the JSON string
            name, args = call["name"], call.get("arguments", call.get("parameters", {}))
            assert isinstance(name, str) and isinstance(args, dict)
        except (json.JSONDecodeError, KeyError, AssertionError, TypeError) as e:
            raise ValueError(f"final tool_call is not {{name, arguments}}: {tc[:200]!r} ({e})")
        last["tool_calls"] = [{"id": "call_final", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]
    msgs.append(last)
    return {"scenario_id": row["scenario_id"], "messages": msgs, "tools": TOOLS,
            "metadata": {"trait_id": row["trait_id"], "trait_name": row["trait_name"], "sector": row["sector"], "domain": row["scenario"]["organisation"],
                         "justification": row["scenario"]["justification"], "supervise": "final", "select": row["select"],
                         "appended_looks": row.get("appended", []), "n_looks": len(row["steps"]), "findings": row.get("findings", [])}}


# ---------------------------------------------------------------------------------------
# per-row flow
# ---------------------------------------------------------------------------------------

class RowError(Exception):
    def __init__(self, row: dict, cause: BaseException):
        super().__init__(str(cause))
        self.row, self.cause = row, f"{type(cause).__name__}: {cause}"


def process(calls: Calls, hand: dict, sc: dict, idx: int, run_dir: Path, dry: bool) -> dict:
    row = {**hand, "scenario_id": f"{hand['trait_id']}_{hand['sector']}_s{idx:03d}", "scenario": sc, "stage_reached": "revise"}
    if dry:
        return row
    try:
        return _process(calls, row, sc)
    except Exception as e:
        raise RowError(row, e) from e


def _process(calls: Calls, row: dict, sc: dict) -> dict:
    files, recs, problems = environment(calls, row)
    for _ in range(2):  # two more attempts. A problem that names its files (a script that fails, files that disagree) is
        if not problems:  # repaired in place, the rest of the environment kept; one that does not (the facts are not
            break         # discoverable) is a design miss and the environment is regenerated from the same prompt.
        if all(pr["paths"] for pr in problems):
            files, recs, problems = environment(calls, row, repair={"entries": recs, "files": files, "problems": problems})
        else:
            files, recs, problems = environment(calls, row, feedback=_fmt_problems(problems))
    if problems:
        row.update(stage_reached="environment", dropped="environment: " + "; ".join(pr["what"] for pr in problems[:5]))
        return row
    row["files"] = files
    row["dirs"] = row.get("dirs") or sorted({p.rsplit("/", 2)[0] for p in files})
    row.update(system=write_system(calls, row)["system"], stage_reached="system")
    row.update(user=write_user(calls, row)["user"], stage_reached="user")
    rp = revise_prompts(calls, row)
    row.update(draft_system=row["system"], draft_user=row["user"], system=rp["system"], user=rp["user"], prompt_changes=rp["changes"], stage_reached="revise_prompts")
    now = dt.datetime.strptime(row["today"], "%A %Y-%m-%d %H:%M")
    with Sandbox(files, now) as box:
        steps = explore(calls, row, box)
        row["steps"] = steps                      # coverage reads the transcript off the row; appends land in the same list
        row["appended"] = ensure_coverage(calls, row, steps, box)
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
    ap.add_argument("--hands", type=int, default=0, help="keep only the first N dealt hands (a smoke deals "
                    "one per principle, so this trims a 9-row smoke to N rows)")
    a = ap.parse_args()
    n_traits = len(segment(CONSTITUTION))
    per_trait = 1 if a.smoke else max(1, a.total // (n_traits * a.per_call))
    run_dir = OUT_ROOT / f"{'smoke_' if a.smoke else ''}{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True)
    hands = deal(per_trait, a.seed)
    if a.hands:
        hands = hands[:a.hands]
    (run_dir / "hands.jsonl").write_text("\n".join(json.dumps(h) for h in hands) + "\n")
    if a.dry:
        for h in hands[:2]:
            print(P.write_prompt(h["trait_name"], h["trait_text"][:300] + "...", "" if h.get("trait_note") else sectors()[h["sector"]], a.per_call, h["name_seed"], trait_note=h.get("trait_note", ""), ai_text=h.get("ai_text", ""), today=h["today"])[:1400], "\n---")
        print(f"dry: {len(hands)} hands dealt, {a.per_call} situations each -> {run_dir}")
        return
    check_docker()
    ensure_image()
    calls = Calls(run_dir)
    def write_and_revise(h: dict) -> list[dict]:  # diversity comes from the dealt sector and "distinct within this set"
        out = []                                     # (da turned its avoid list off on 2026-10-02: the writer copied its flavour)
        for sc in write_situations(calls, h, a.per_call):
            rv = revise_situation(calls, h, sc)
            out.append({"hand": h, "scenario": rv["situation"], "draft": sc, "revise": {"changes": rv["changes"]}})
        return out
    rows: list[dict] = []
    with ThreadPoolExecutor(a.workers) as ex:
        for batch in ex.map(write_and_revise, hands):
            rows.extend(batch)
    (run_dir / "stage_write_revise.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    done: list[dict] = []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process, calls, r["hand"], r["scenario"], i, run_dir, False): i for i, r in enumerate(rows)}
        with (run_dir / "rows_partial.jsonl").open("a") as part:
            for f in as_completed(futs):
                try:
                    r = f.result()
                except RowError as e:  # keep the run going with what the row had; the manifest counts it
                    r = {**e.row, "dropped": f"error: {e.cause}", "stage_reached": "error"}
                except Exception as e:
                    r = {"scenario_id": f"row_{futs[f]}", "dropped": f"error: {type(e).__name__}: {e}", "stage_reached": "error"}
                done.append(r)
                part.write(json.dumps(r, default=str) + "\n")
                part.flush()
    (run_dir / "rows.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in done) + "\n")  # before assembly: nothing is lost to it
    kept = []
    for r in done:
        if not r.get("final"):
            continue
        try:
            kept.append(assemble(r))
        except ValueError as e:
            r["dropped"] = f"assemble: {e}"
            r["stage_reached"] = "assemble"
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
