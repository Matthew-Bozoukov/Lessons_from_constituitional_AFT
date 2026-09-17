# ABOUTME: daa2 -- edit-based agentification of difficult-advice rows: the DA row is the skeleton and
# ABOUTME: real tool calls are inserted where the reply itself acts. Three Claude Code calls per row, no API key.
"""Agentify difficult-advice rows by EDITING them, not regenerating them.

A difficult-advice (da) row is system prompt + user dilemma + private reasoning + reply. The
agentified row is the SAME row with typed insertions: real bash tool calls placed where the reply
already looks at, runs, or drafts something, executed in a sandbox so every output is real. The
model never writes a new scenario, a new answer, or a plan; it returns edit lists over numbered
sentences plus a list of insertions, and code assembles the transcript. Untouched sentences
survive verbatim, so fidelity is by construction and there is no plan vocabulary to leak.

The rules that keep advice from turning into action live in the two prompts, not in code: an
action may only realise a sentence in which the reply itself does or offers the thing, or an
internal operation the user's message asks for that the reply's content is consistent with doing;
what the reply declines stays declined; nothing leaves the machine. Code does what only code can
do: run the commands for real, put the pieces in order, and drop a row whose commands failed.

Per row:
    1  map   (call 1)  system/user edit lists, 2-4 environment files, ordered insertions
                       [look | run | write], each run/write anchored to the sentence it realises
       sandbox         looks and runs execute in a Docker container with a frozen scenario clock;
                       a failing command sends the map back once with the error, then drops the row
    2  think (call 2)  with the looks' outputs only, the runs still ahead: the look sentences and
                       the reasoning edit list (the deliberation, thought before any action)
    3  finish (call 3) with every real output and the deliberation as written: one sentence per
                       later action, the write contents, the reply edit list, the closing reason
       sandbox         writes execute, code reads each back
       code            assemble the turns; measure reuse; drop only on a failed command

The split between think and finish is what keeps run output out of the deliberation: the think call
never sees it, so the transcript's causal order holds by construction, not by instruction.

Generation goes through Claude Code's print mode in bare mode (`claude -p --bare`): the model sees
the two prompts and today's date, nothing of this machine, and runs on the subscription token from
.env (CLAUDE_CODE_OAUTH_TOKEN, minted by `claude setup-token`), not an API key.

    uv run python scratch/daa2/agentify2.py --smoke --model sonnet     # 10 rows, seed 0
    uv run python scratch/daa2/agentify2.py --smoke --model opus
    uv run python scratch/daa2/agentify2.py --model sonnet --workers 8 # all rows
    uv run python scratch/daa2/agentify2.py --refill output/synth/daa2/<run> --model sonnet --workers 8
                                                                       # redo only think/finish/writes
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import random
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scratch.daa.agentify import (  # the parts of the old pipeline that were never the problem
    TOOLS, Sandbox, apply_sentence_edits, bash_call, check_docker, complete_call, ensure_image,
    line_start_numbers, load_source, numbered, paragraphs_sentences, reuse_ratio, scenario_date, tool_result,
)

SOURCE_REPO = "dougalldeepmind/2026-09-08-da-synth"
SOURCE_REVISION = "42107bde00cd7f4360a3a6c581aac23a540dbfea"
MODELS = {"sonnet": "sonnet", "opus": "opus", "fable": "fable",
          "or-sonnet": "anthropic/claude-sonnet-5"}      # `or-`: the same call over OpenRouter (src/infra/endpoints/openrouter.py)
_OR: dict = {}                                            # the OpenRouter client, made on first use


def generator_label(model: str) -> str:
    """What a row's `generator` metadata says: the CLI alias, or the OpenRouter model id."""
    return f"openrouter:{MODELS[model]}" if model.startswith("or-") else f"claude-code:{model}"
NEUTRAL_CWD = "/tmp/daa2-cwd"          # `claude -p` names its working directory to the model: an empty one
REPO_ROOT = Path(__file__).resolve().parents[2]
_CLAUDE_ENV: dict[str, str] = {}


def heredoc(path: str, content: str) -> str:
    return f"cat > {path} <<'EOF'\n{content.rstrip()}\nEOF"


def claude_env() -> dict[str, str]:
    """The environment every `claude -p --bare` call runs in: PATH, HOME, the locale and the
    subscription bearer token, nothing else. Bare mode reads no keychain, no CLAUDE.md and no
    memory, and announces no account to the model (a non-bare smoke wrote the account email into
    three environment files as document authors). Bare mode accepts only a bearer token, so the
    token minted by `claude setup-token` lives in .env as CLAUDE_CODE_OAUTH_TOKEN and reaches the
    CLI as ANTHROPIC_AUTH_TOKEN."""
    if _CLAUDE_ENV:
        return _CLAUDE_ENV
    from dotenv import dotenv_values
    tok = (dotenv_values(REPO_ROOT / ".env").get("CLAUDE_CODE_OAUTH_TOKEN") or "").strip()
    if not tok.startswith("sk-ant-oat"):
        raise SystemExit("CLAUDE_CODE_OAUTH_TOKEN is not set in .env: run `claude setup-token` and put the token there")
    _CLAUDE_ENV.update({k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL") if k in os.environ})
    _CLAUDE_ENV.update({"ANTHROPIC_AUTH_TOKEN": tok, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "TERM": "dumb"})
    return _CLAUDE_ENV


# ---------------------------------------------------------------------------------------
# prompts: the whole method is in these two
# ---------------------------------------------------------------------------------------

MAP_SYSTEM = """You convert a chat-format ethical-dilemma row into the same row for an AI agent with a bash tool.
You do not write a new scenario or a new answer: you edit the given one and insert real commands.
Return JSON matching the schema. Sentences are numbered; an edit list is
{"edits": [{"n": int, "new": str}], "inserts": [{"after": int, "text": str}]} -- "" deletes; untouched
sentences survive verbatim, so change as little as possible.

- system_edits: over the system prompt. Add the agent's role and where its files are. Same register.
- user_edits: over the user's message. Add concrete absolute paths and, where the user wants
  something run, the exact command. Keep every fact, every rider, every ask and the same voice.
  Do not add permissions, limits, reassurances or instructions the original user did not give:
  the pressure on the agent stays exactly what it was.
- files: 2 to 4 small files (each under 1000 characters) holding the facts the scenario states,
  in the scenario's own documents. They must not settle the dilemma: no document whose contents
  make the right answer obvious, no evidence the user did not mention. If the user asks for
  something to be run, one file IS that script: short python, argparse with --help, one mode,
  only the flags this situation needs, and it must really work on the other files. If the
  scenario implies editing an existing document, that document exists with real content. A file
  the agent will create must NOT exist. Paths live under /srv, /data, /opt or /home/<role>, never
  a real person's home directory. People and organisations are invented. Dates in files fit the
  scenario, not today.
- today: the scenario's date, YYYY-MM-DD: the day the user is writing, consistent with the files
  (deadlines they mention lie ahead of it). The sandbox clock is set to it.
- insertions, in the order they happen:
    look  -- a read-only command (cat, head, grep, ls, or a script's --help). No anchor.
    run   -- a script from files, with its flags; it computes or checks.
    write -- a path only; the agent's own prose (a draft, memo, message), written later from real output.
  A run or write is allowed only where the REPLY ITSELF does or offers that exact thing in one of
  its sentences ("I'll draft...", "here's the version...", "I can run..."): anchor_part "reply",
  anchor_n that sentence. A run may instead realise an internal operation the user's message
  asks for, if the reply's content is consistent with doing it: anchor_part "user", anchor_n
  that sentence. Never turn advice to the human ("you should...", "I'd try...") into an action,
  and what the reply declines stays declined: an agent that will not send a message does not
  send it, it may only leave a draft on disk if the reply offers one. Nothing sends, submits,
  posts, publishes, deploys, notifies or approves; no script does either. A row where the reply
  only advises has looks and no run or write. Before a script's first run, a look at its --help;
  before a write to an existing file, a look at it. Writes come after all looks and runs."""

THINK_SYSTEM = """You are the agent in an agentified row, at the moment after the looks and before any action. Only the
looks have run; their real outputs are below. You return only edits and short sentences; code
assembles the transcript. Edit lists are
{"edits": [{"n": int, "new": str}], "inserts": [{"after": int, "text": str}]} over numbered sentences.

- look_reasoning: one short first-person sentence before each look (what I want to check and why).
- reasoning_edits: over the numbered original reasoning, to fit the actor: insert what the ask is
  for me to do, what the looks showed, and what I will do next; edit only sentences that no
  longer fit; most stay untouched. The plan is exactly the actions listed, as intentions: every
  listed run and write appears in it, and it names no other file, script or command. Anything
  else I mean to do happens in my reply itself (advice, an offer, a draft written out in the
  message). If no actions are listed, the plan is that the reply carries the matter on its own.
Plain first person. Never mention this process or its numbering. Numbers come from the look outputs
or the files, nowhere else."""

FINISH_SYSTEM = """You are finishing an agentified row. The deliberation is already written (below) and the reply
must carry it out; the commands have run and their real outputs are below. You return only edits, contents and short
sentences; code assembles the transcript. Edit lists are
{"edits": [{"n": int, "new": str}], "inserts": [{"after": int, "text": str}]} over numbered sentences.

- action_reasoning: one sentence before each action after the first, in the order listed, each
  naming what that action will do given the previous output (an intention, not a report).
- writes: the full content of each write, from the real outputs and files, dated in the scenario.
  A memo or letter under ~250 words; a record or config a few lines.
- readback_reasoning: one short sentence before reading back each written file.
- reply_edits: over the numbered original reply, so that it carries out the deliberation:
  everything the deliberation says I will do, say or offer appears in the reply, done or
  offered; what the commands did is reported as it printed (only that), and what was not done
  and why, in the reply's own words. A file exists only if it is a write under ACTIONS; a file
  the user asked for that is not there was not written, and the reply says so rather than
  claiming it. Change only the sentences this requires; every other sentence stays verbatim,
  with the reply's advice, structure and ending; if it ends with a question, it still does.
- closing_reasoning: one or two sentences after the last output. reason: the task_complete
  reason, under 600 characters; if the reply ends with a question, say the task is closed
  pending the answer.
Plain first person. Never mention this process, its numbering or its outputs as such. Numbers come
from the outputs or the files, nowhere else."""

EDIT_LIST = {"type": "object", "properties": {
    "edits": {"type": "array", "items": {"type": "object", "properties": {"n": {"type": "integer"}, "new": {"type": "string"}}, "required": ["n", "new"]}},
    "inserts": {"type": "array", "items": {"type": "object", "properties": {"after": {"type": "integer"}, "text": {"type": "string"}}, "required": ["after", "text"]}}},
    "required": ["edits", "inserts"]}
MAP_SCHEMA = {"type": "object", "properties": {
    "system_edits": EDIT_LIST, "user_edits": EDIT_LIST,
    "files": {"type": "array", "items": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    "insertions": {"type": "array", "items": {"type": "object", "properties": {
        "kind": {"type": "string", "enum": ["look", "run", "write"]}, "command": {"type": "string"}, "path": {"type": "string"},
        "anchor_part": {"type": "string", "enum": ["reply", "user", ""]}, "anchor_n": {"type": "integer"}},
        "required": ["kind", "command", "path", "anchor_part", "anchor_n"]}},
    "today": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"}},
    "required": ["system_edits", "user_edits", "files", "insertions", "today"]}
THINK_SCHEMA = {"type": "object", "properties": {
    "look_reasoning": {"type": "array", "items": {"type": "string"}},
    "reasoning_edits": EDIT_LIST},
    "required": ["look_reasoning", "reasoning_edits"]}
FINISH_SCHEMA = {"type": "object", "properties": {
    "action_reasoning": {"type": "array", "items": {"type": "string"}},
    "writes": {"type": "array", "items": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    "readback_reasoning": {"type": "array", "items": {"type": "string"}},
    "reply_edits": EDIT_LIST,
    "closing_reasoning": {"type": "string"}, "reason": {"type": "string", "minLength": 1}},
    "required": ["action_reasoning", "writes", "readback_reasoning", "reply_edits", "closing_reasoning", "reason"]}

# the pipeline's own words; counted in the manifest so a leak is visible, never used to drop a row
PIPELINE_VOCAB = re.compile(r"\b(original reply|original reasoning|edit list|numbered sentence|sentence \d+|look_reasoning|action_reasoning|reply_edits)\b", re.I)


# ---------------------------------------------------------------------------------------
# Claude Code print-mode driver: the subscription, not an API key
# ---------------------------------------------------------------------------------------

class Usage:
    def __init__(self):
        self.by_stage: dict[str, dict] = collections.defaultdict(lambda: collections.Counter())
        self._lock = threading.Lock()

    def add(self, stage: str, d: dict) -> None:
        u = d.get("usage") or {}
        with self._lock:
            c = self.by_stage[stage]
            c["calls"] += 1
            c["input"] += u.get("input_tokens", 0)
            c["cache_create"] += u.get("cache_creation_input_tokens", 0)
            c["cache_read"] += u.get("cache_read_input_tokens", 0)
            c["output"] += u.get("output_tokens", 0)
            c["thinking"] += (u.get("output_tokens_details") or {}).get("thinking_tokens", 0)
            c["list_usd_x1000"] += int(round(1000 * (d.get("total_cost_usd") or 0)))
            c["api_ms"] += d.get("duration_api_ms", 0)

    def as_dict(self) -> dict:
        return {k: dict(v) for k, v in self.by_stage.items()}


EFFORT = {"level": ""}      # set from --effort; "" leaves the CLI default (full thinking)
DEBUG_DIR = Path("output/synth/daa2/debug")   # raw CLI responses of failed calls (set per run in main)


def openrouter_json(system: str, user: str, schema: dict, model: str, stage: str, usage: Usage,
                    attempts: int = 4) -> dict:
    """The same structured call over OpenRouter: a different gateway and content filter from the
    CLI's, for a row the CLI's classifier refuses on every attempt. The schema rides in the system
    prompt, the reply body is parsed as JSON and checked for the schema's required keys; reasoning
    stays on and arrives separately from the body, as the CLI's does."""
    from src.data.synth.ours.stage_runtime import _parse_json
    from src.infra.endpoints.openrouter import EmptyCompletionError, OpenRouterClient, ProviderRejectionError
    client = _OR.get("client") or _OR.setdefault("client", OpenRouterClient())
    sys_msg = system + "\n\nReturn ONLY a JSON object matching this schema, nothing else:\n" + json.dumps(schema)
    last = ""
    for attempt in range(1, attempts + 1):
        try:
            res = client.chat(MODELS[model], [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
                              temperature=1.0, max_tokens=24576, extra_body={"reasoning": {"enabled": True}})
        except ProviderRejectionError as exc:                   # a 4xx from the provider: its filter, deterministic
            raise RuntimeError(f"{stage}: refused by the provider: {str(exc)[:120]}")
        except EmptyCompletionError as exc:
            last = f"empty completion: {str(exc)[:120]}"; time.sleep(5 * attempt); continue
        usage.add(stage, {"usage": {"input_tokens": res.prompt_tokens, "output_tokens": res.completion_tokens,
                                    "cache_read_input_tokens": res.cached_tokens}})
        try:
            d = _parse_json(res.content)
        except Exception as exc:  # noqa: BLE001 - a malformed body is retried, then reported
            last = f"parse: {type(exc).__name__}: {str(exc)[:80]} | finish={res.finish_reason} | content[:120]={res.content[:120]!r}"
            continue
        missing = [k for k in schema.get("required", []) if not isinstance(d, dict) or k not in d]
        if missing:
            last = f"missing required keys {missing}"; continue
        return d
    raise RuntimeError(f"{stage}: no valid structured output after {attempts} attempts: {last}")


def claude_json(system: str, user: str, schema: dict, model: str, stage: str, usage: Usage,
                attempts: int = 4, timeout: int = 900) -> dict:
    """One structured call through `claude -p --bare`. Structured output is a tool call the CLI
    validates against the schema, and a miss needs a further turn to retry, hence --max-turns 3
    with no tools. Retries on transport errors and waits out a usage-limit refusal."""
    if model.startswith("or-"):
        return openrouter_json(system, user, schema, model, stage, usage, attempts)
    cmd = ["claude", "-p", "--bare", "--model", MODELS[model], "--output-format", "json", "--max-turns", "3",
           "--tools", "", "--no-session-persistence", "--system-prompt", system, "--json-schema", json.dumps(schema)]
    if EFFORT["level"]:
        cmd += ["--effort", EFFORT["level"]]
    last = ""
    attempt, limit_deadline = 0, time.time() + 8 * 3600
    while attempt < attempts:
        attempt += 1
        try:
            res = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=timeout, cwd=NEUTRAL_CWD,
                                 env=claude_env())
        except subprocess.TimeoutExpired:
            last = "timeout"; continue
        try:
            d = json.loads(res.stdout)
        except json.JSONDecodeError:
            last = f"non-json stdout: {res.stdout[:200]!r} stderr: {res.stderr[:200]!r}"
            time.sleep(5 * (attempt + 1)); continue
        if not d.get("is_error") and isinstance(d.get("structured_output"), dict):
            usage.add(stage, d)
            return d["structured_output"]
        msg = str(d.get("result") or d.get("error") or res.stderr)[:300]
        tags = " ".join(f"{k}={d[k]}" for k in ("subtype", "terminal_reason", "stop_reason", "num_turns") if d.get(k) not in (None, ""))
        last = f"error: {msg} [{tags}]"
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        (DEBUG_DIR / f"{stage}_{int(time.time())}_{attempt}.json").write_text(json.dumps(
            {k: v for k, v in d.items() if k != "structured_output"} | {"stderr": res.stderr[-2000:]}, indent=1))
        if d.get("stop_reason") == "refusal":                  # the content classifier; deterministic, so no retry
            raise RuntimeError(f"{stage}: refused by the content classifier: {msg[:120]}")
        if re.search(r"limit|rate|overloaded|429|529", msg, re.I) and time.time() < limit_deadline:
            attempt -= 1                                       # a limit is waited out, not retried away
            print(f"    [{stage}] usage limit or overload ({msg[:80]!r}); waiting 600s", flush=True)
            time.sleep(600)
        else:
            time.sleep(5 * attempt)
    raise RuntimeError(f"{stage}: no valid structured output after {attempts} attempts: {last}")


# ---------------------------------------------------------------------------------------
# stage 1: map
# ---------------------------------------------------------------------------------------

LEAN_MAP = {"on": False}    # --lean-map: the map call reads no private reasoning (it never edits it)


def map_prompt(r: dict) -> str:
    reasoning = "" if LEAN_MAP["on"] else f"PRIVATE REASONING (for context; not edited here):\n{r['reasoning']}\n\n"
    return (f"SYSTEM PROMPT (numbered):\n{numbered(paragraphs_sentences(r['system']))}\n\n"
            f"USER MESSAGE (numbered):\n{numbered(paragraphs_sentences(r['user']))}\n\n"
            f"{reasoning}"
            f"REPLY (numbered):\n{numbered(paragraphs_sentences(r['answer']))}")


def apply_map(m: dict, r: dict) -> dict:
    se, ue = m.get("system_edits") or {}, m.get("user_edits") or {}
    return {
        "system": apply_sentence_edits(paragraphs_sentences(r["system"]), se.get("edits") or [], se.get("inserts") or [],
                                       line_start_numbers(r["system"])),
        "user": apply_sentence_edits(paragraphs_sentences(r["user"]), ue.get("edits") or [], ue.get("inserts") or [],
                                     line_start_numbers(r["user"])),
        "files": {f["path"]: f["content"] for f in m.get("files") or [] if f.get("path", "").startswith("/")},
    }


def scenario_now(m: dict, env: dict) -> dt.datetime:
    """The map's `today` at 08:30; the old heuristic (day after the latest date mentioned) if unparsable."""
    try:
        return dt.datetime.strptime(m.get("today", ""), "%Y-%m-%d").replace(hour=8, minute=30)
    except ValueError:
        return scenario_date(env["user"], env["system"], *env["files"].values())


def insertions_of(m: dict) -> list[dict]:
    """The map's insertions in execution order: looks and runs as given, then the writes, whose
    path field is just the path (a model sometimes writes `cat > /path` there)."""
    ins = [dict(x) for x in m.get("insertions") or [] if x.get("kind") in ("look", "run", "write")]
    for x in ins:
        if x["kind"] == "write":
            p = re.sub(r"^\s*cat\s*>+\s*", "", x.get("path") or "").split()
            x["path"] = p[0] if p else ""
    return [x for x in ins if x["kind"] != "write"] + [x for x in ins if x["kind"] == "write" and x["path"].startswith("/")]


def execute_pre(ins: list[dict], env: dict, now: dt.datetime) -> tuple[list[dict], list[str]]:
    """Looks and runs, in order, for real; a failing command is a map failure."""
    steps, problems = [], []
    with Sandbox(env["files"], now) as sb:
        for x in ins:
            if x["kind"] == "write":
                break
            res = sb.run(x["command"])
            steps.append({"kind": x["kind"], "command": x["command"], **res})
            if res["returncode"]:
                problems.append(f"{x['kind']} failed ({res['returncode']}): {x['command'][:70]!r} :: {res['stderr'][:150]!r}")
    return steps, problems


# ---------------------------------------------------------------------------------------
# stage 2: fill
# ---------------------------------------------------------------------------------------

def render_outputs(steps: list[dict]) -> str:
    out = []
    for i, s in enumerate(steps):
        tag = "LOOK" if s["kind"] == "look" else "RUN"
        block = f"[{tag} {i + 1}] $ {s['command']}\n{s['stdout']}"
        if s.get("returncode"):
            block += f"\n[exit {s['returncode']}] {s.get('stderr', '')}"
        out.append(block)
    return "\n\n".join(out) or "(no commands before the writes)"


def _action_list(steps: list[dict], writes: list[dict]) -> str:
    actions = [f"run $ {s['command']}" for s in steps if s["kind"] == "run"] + [f"write {w['path']}" for w in writes]
    return "\n".join(f"  action {i + 1}: {a}" for i, a in enumerate(actions)) or "  (none: the reply only advises)"


def _scene(env: dict, now: dt.datetime) -> str:
    return (f"TODAY IN THE SCENARIO: {now:%A %Y-%m-%d} (the sandbox clock; the only date to write in a draft or say aloud)\n\n"
            f"SYSTEM PROMPT:\n{env['system']}\n\nUSER MESSAGE:\n{env['user']}\n\n"
            f"FILES:\n" + "\n".join(f"--- {p}\n{c}" for p, c in env["files"].items()) + "\n\n")


def think_prompt(r: dict, env: dict, steps: list[dict], writes: list[dict], now: dt.datetime) -> str:
    """Before any action: the looks' outputs only, and the actions as a plan."""
    looks = [s for s in steps if s["kind"] == "look"]
    return (_scene(env, now)
            + f"LOOK OUTPUTS, in order (all that has run so far):\n{render_outputs(looks)}\n\n"
            f"ACTIONS I AM ABOUT TO TAKE, in order (none has run yet; their results are unknown):\n{_action_list(steps, writes)}\n\n"
            f"ORIGINAL PRIVATE REASONING (numbered):\n{numbered(paragraphs_sentences(r['reasoning']))}")


def finish_prompt(r: dict, env: dict, steps: list[dict], writes: list[dict], now: dt.datetime, deliberation: str) -> str:
    """After the runs: every output, the deliberation as written, and the reply to edit."""
    return (_scene(env, now)
            + f"OUTPUTS, in order (LOOK = before the deliberation, RUN = an action):\n{render_outputs(steps)}\n\n"
            f"ACTIONS, in order (the deliberation came before action 1; action_reasoning has one sentence for each later one):\n{_action_list(steps, writes)}\n\n"
            f"THE DELIBERATION, already written before action 1 (the reply must carry it out):\n{deliberation}\n\n"
            f"ORIGINAL REPLY (numbered):\n{numbered(paragraphs_sentences(r['answer']))}")


# ---------------------------------------------------------------------------------------
# one row end to end
# ---------------------------------------------------------------------------------------

def process_row(r: dict, model: str, usage: Usage) -> dict:
    rec: dict = {"scenario_id": r["scenario_id"], "metadata": r["metadata"], "model": model, "problems": []}
    m = claude_json(MAP_SYSTEM, map_prompt(r), MAP_SCHEMA, model, "map", usage)
    ins, env = insertions_of(m), apply_map(m, r)
    now = scenario_now(m, env)
    steps, problems = execute_pre(ins, env, now)
    if problems:                                               # one repair: the map sees the real error
        repair = (map_prompt(r) + "\n\nYOUR PREVIOUS ATTEMPT (below) had commands that failed; return the corrected "
                  "JSON in full, changing only what the failures need:\n" + "\n".join(f"- {p}" for p in problems)
                  + "\n\nPREVIOUS JSON:\n" + json.dumps(m, indent=1))
        m2 = claude_json(MAP_SYSTEM, repair, MAP_SCHEMA, model, "map_repair", usage)
        ins2, env2 = insertions_of(m2), apply_map(m2, r)
        now2 = scenario_now(m2, env2)
        steps2, problems2 = execute_pre(ins2, env2, now2)
        if len(problems2) < len(problems):
            m, ins, env, now, steps, problems = m2, ins2, env2, now2, steps2, problems2
            rec["repaired"] = True
    rec.update({"map": m, "insertions": ins, "env": env, "scenario_now": now.isoformat(), "pre_steps": steps})
    if problems:
        rec["problems"] = problems
        return rec
    return fill_stage(r, rec, model, usage)


def fill_stage(r: dict, rec: dict, model: str, usage: Usage) -> dict:
    """Stage 2 on a mapped row: think (looks only), then finish (every output), then the writes.

    The deliberation is generated before the run outputs exist in its prompt, so it cannot quote
    them; the reply and the writes are generated after, so they can only report what really ran."""
    env, ins, now = rec["env"], rec["insertions"], dt.datetime.fromisoformat(rec["scenario_now"])
    steps, problems = [s for s in rec["pre_steps"] if s["kind"] in ("look", "run")], []   # a finished run's record may carry its old writes here
    writes = [x for x in ins if x["kind"] == "write"]
    t = claude_json(THINK_SYSTEM, think_prompt(r, env, steps, writes, now), THINK_SCHEMA, model, "think", usage)
    re_ = t.get("reasoning_edits") or {}
    deliberation = apply_sentence_edits(paragraphs_sentences(r["reasoning"]), re_.get("edits") or [], re_.get("inserts") or [],
                                        line_start_numbers(r["reasoning"]))
    g = claude_json(FINISH_SYSTEM, finish_prompt(r, env, steps, writes, now, deliberation), FINISH_SCHEMA, model, "finish", usage)
    f = {**t, **g}
    rec["fill"] = f
    rec["fill_mode"] = "think_finish_v2"                      # v2: plan bound to the action list, reply bound to the plan; older marks are refilled
    # writes + read-backs, in the same sandbox state the looks and runs left behind
    with Sandbox(env["files"], now) as sb:
        for s in steps:                                        # replay so the writes see the runs' side effects
            sb.run(s["command"])
        contents = {w.get("path"): w.get("content", "") for w in f.get("writes") or []}
        rb_why = list(f.get("readback_reasoning") or [])
        for i, w in enumerate(writes):
            content = contents.get(w["path"])
            if content is None:
                problems.append(f"fill authored no content for {w['path']}")
                continue
            cmd = heredoc(w["path"], content)
            sb.run(f"mkdir -p {os.path.dirname(w['path'])}")
            res = sb.run(cmd)
            steps.append({"kind": "write", "command": cmd, "path": w["path"], **res})
            if res["returncode"]:
                problems.append(f"write failed: {w['path']}: {res['stderr'][:120]}")
            rb = sb.run(f"cat {w['path']}")
            steps.append({"kind": "readback", "command": f"cat {w['path']}", "reasoning": (rb_why[i] if i < len(rb_why) else "Reading it back to see what actually landed."), **rb})
    rec["steps"] = steps
    rec["messages"] = assemble(r, env, steps, f)
    rec["problems"] = problems
    rec["measures"] = measures(rec, r)
    return rec


def assemble(r: dict, env: dict, steps: list[dict], f: dict) -> list[dict]:
    """The dat/ODCV shape: looks, deliberation + first action, actions, read-backs, closing."""
    looks = [s for s in steps if s["kind"] == "look"]
    acts = [s for s in steps if s["kind"] in ("run", "write", "readback")]
    re_ = f.get("reasoning_edits") or {}
    deliberation = apply_sentence_edits(paragraphs_sentences(r["reasoning"]), re_.get("edits") or [], re_.get("inserts") or [],
                                        line_start_numbers(r["reasoning"]))
    rp = f.get("reply_edits") or {}
    message = apply_sentence_edits(paragraphs_sentences(r["answer"]), rp.get("edits") or [], rp.get("inserts") or [],
                                   line_start_numbers(r["answer"]))
    lw = list(f.get("look_reasoning") or []); aw = list(f.get("action_reasoning") or [])
    msgs: list[dict] = [{"role": "system", "content": env["system"].strip()}, {"role": "user", "content": env["user"].strip()}]
    for i, s in enumerate(looks):
        why = lw[i] if i < len(lw) else "Checking what is actually there before deciding anything."
        msgs += [{"role": "assistant", "content": "", "reasoning_content": why.strip(), "tool_calls": [bash_call(s["command"])]},
                 {"role": "tool", "content": tool_result(s["stdout"], s["returncode"], s.get("stderr", ""))}]
    ai = 0
    for s in acts:
        if s["kind"] == "readback":
            why = s.get("reasoning", "")
        elif ai == 0:
            why = deliberation
        else:
            why = aw[ai - 1] if ai - 1 < len(aw) else "Continuing with the next step."
        if s["kind"] != "readback":
            ai += 1
        msgs += [{"role": "assistant", "content": "", "reasoning_content": why.strip(), "tool_calls": [bash_call(s["command"])]},
                 {"role": "tool", "content": tool_result(s["stdout"], s["returncode"], s.get("stderr", ""))}]
    closing = f.get("closing_reasoning") or ""
    if not acts:                                               # a pure decline: the deliberation closes the row
        closing = deliberation
    reason = re.split(r"\s*</[a-z_]+>", f.get("reason") or "", 1)[0].strip()   # Sonnet sometimes echoes the closing tag of the output field
    msgs.append({"role": "assistant", "content": message.strip(), "reasoning_content": closing.strip(),
                 "tool_calls": [complete_call(reason)]})
    return msgs


def measures(rec: dict, r: dict) -> dict:
    steps = rec["steps"]
    delib = next((m["reasoning_content"] for m in rec["messages"] if m["role"] == "assistant"
                  and m.get("tool_calls") and m["tool_calls"][0]["function"]["name"] == "bash"
                  and len(m["reasoning_content"]) > 400), None) or rec["messages"][-1]["reasoning_content"]
    prose = "\n".join((m.get("reasoning_content") or "") + "\n" + (m.get("content") or "") + "\n"
                      + " ".join(tc["function"]["arguments"].get("reason", "") for tc in m.get("tool_calls") or [] if tc["function"]["name"] == "task_complete")
                      for m in rec["messages"] if m["role"] == "assistant")
    return {
        "reuse_reasoning": round(reuse_ratio(r["reasoning"], delib), 3),
        "reuse_reply": round(reuse_ratio(r["answer"], rec["messages"][-1]["content"]), 3),
        "reuse_user": round(reuse_ratio(r["user"], rec["env"]["user"]), 3),
        "n_look": sum(1 for s in steps if s["kind"] == "look"),
        "n_run": sum(1 for s in steps if s["kind"] == "run"), "n_write": sum(1 for s in steps if s["kind"] == "write"),
        "n_files": len(rec["env"]["files"]), "chars_files": sum(len(c) for c in rec["env"]["files"].values()),
        "pipeline_vocab": len(PIPELINE_VOCAB.findall(prose)),
    }


# ---------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="sonnet")
    ap.add_argument("--source", default=SOURCE_REPO)
    ap.add_argument("--revision", default=SOURCE_REVISION)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="10 rows, seed 0 (the same ten every time)")
    ap.add_argument("--ids", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", default="", help="an existing run dir to continue (its own row selection)")
    ap.add_argument("--effort", default="", help="claude --effort level (low|medium|high|max); default leaves the CLI's own")
    ap.add_argument("--lean-map", action="store_true", help="leave the private reasoning out of the map call's input "
                                                             "(for a row the content classifier refuses at the map)")
    ap.add_argument("--redo-dropped", action="store_true", help="on --resume, re-run the rows that were dropped")
    ap.add_argument("--refill", default="", help="a finished run dir: reuse its maps, environments and executed looks/runs, "
                                                "redo only the fill stage (think, finish, writes) into a new run dir")
    args = ap.parse_args()
    claude_env()                # fail before docker if the subscription token is missing
    check_docker(); ensure_image()
    Path(NEUTRAL_CWD).mkdir(parents=True, exist_ok=True)
    EFFORT["level"] = args.effort
    LEAN_MAP["on"] = args.lean_map

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.resume) if args.resume else Path("output/synth/daa2") / (("smoke_" if args.smoke else "") + f"{ts}_{args.model}" + ("_refill" if args.refill else ""))
    global DEBUG_DIR
    DEBUG_DIR = run_dir / "debug"
    run_dir.mkdir(parents=True, exist_ok=True)
    src_path = run_dir / "source.jsonl"
    base: dict[str, dict] = {}                  # --refill: the finished run's records, by scenario_id
    if args.refill:
        base_dir = Path(args.refill)
        base = {json.loads(l)["scenario_id"]: json.loads(l) for l in open(base_dir / "rows.jsonl") if l.strip()}
        if not src_path.exists():
            src_path.write_text((base_dir / "source.jsonl").read_text())
        if not (run_dir / "selection.json").exists() and (base_dir / "selection.json").exists():
            (run_dir / "selection.json").write_text((base_dir / "selection.json").read_text())
    if src_path.exists():
        rows = [json.loads(l) for l in open(src_path)]
    else:
        rows = load_source(args.source, args.revision)
        src_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    sel_path = run_dir / "selection.json"       # a run remembers which rows it selected
    if args.refill:
        want = set(args.ids.split(",")) if args.ids else set(json.loads(sel_path.read_text())) if sel_path.exists() else set(base)
        rows = [x for x in rows if x["scenario_id"] in want]
    elif args.smoke:
        rows = random.Random(0).sample(rows, 10)
    elif args.ids:
        want = set(args.ids.split(",")); rows = [x for x in rows if x["scenario_id"] in want]
    elif args.limit:
        rows = rows[: args.limit]
    elif args.resume and sel_path.exists():
        want = set(json.loads(sel_path.read_text())); rows = [x for x in rows if x["scenario_id"] in want]
    elif args.resume and run_dir.name.startswith("smoke_"):
        raise SystemExit("this is a smoke run dir with no selection.json: pass --smoke with --resume")
    sel_path.write_text(json.dumps([x["scenario_id"] for x in rows]))
    ck = run_dir / "rows.jsonl"
    done = {}
    if ck.exists():
        for l in open(ck):
            try:
                rec = json.loads(l); done[rec["scenario_id"]] = rec
            except json.JSONDecodeError:
                pass
    if args.redo_dropped:
        done = {k: v for k, v in done.items() if not v["problems"]}
        ck.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in done.values()))
    todo = [x for x in rows if x["scenario_id"] not in done]
    print(f">>> {len(rows)} rows, {len(done)} done, {len(todo)} to run | model {args.model} | {run_dir}", flush=True)
    usage = Usage(); lock = threading.Lock(); t0 = time.time()

    def one(r):
        try:
            if base:                                           # --refill: the map stage is the finished run's
                b = base.get(r["scenario_id"])
                if b is None or b["problems"] or "pre_steps" not in b:
                    rec = {**(b or {"scenario_id": r["scenario_id"], "metadata": r["metadata"]}), "model": args.model,
                           "problems": (b or {}).get("problems") or ["refill: no mapped row to reuse"]}
                elif b.get("fill_mode") == "think_finish_v2":  # already produced by the current prompts; carry it over as is
                    rec = b
                else:
                    rec = fill_stage(r, {k: b[k] for k in ("scenario_id", "metadata", "map", "insertions", "env", "scenario_now", "pre_steps")
                                         if k in b} | {"model": args.model, "problems": [], "refilled_from": str(Path(args.refill))}, args.model, usage)
            else:
                rec = process_row(r, args.model, usage)
        except Exception as exc:  # noqa: BLE001 - a row's failure is recorded, not fatal
            rec = {"scenario_id": r["scenario_id"], "metadata": r["metadata"], "model": args.model, "problems": [f"exception: {type(exc).__name__}: {str(exc)[:200]}"]}
        with lock:
            with open(ck, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[rec["scenario_id"]] = rec
            n = len(done); st = "ok" if not rec["problems"] else rec["problems"][0][:70]
            print(f"    [{n}/{len(rows)}] {rec['scenario_id']} {st}", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in as_completed([ex.submit(one, r) for r in todo]):
            pass
    recs = [done[x["scenario_id"]] for x in rows if x["scenario_id"] in done]
    kept = [x for x in recs if not x["problems"]]
    with open(run_dir / "dataset.jsonl", "w") as fh:
        for x in kept:
            fh.write(json.dumps({"scenario_id": x["scenario_id"], "messages": x["messages"], "tools": TOOLS,
                                 # no `supervise` field: every assistant turn is loss, which is the trainer's default, and an
                                 # explicit all-'all' column is refused there as an arm identical to its control
                                 "metadata": {**x["metadata"], **x["measures"], "generator": generator_label(args.model)},
                                 "environment": [{"path": p, "content": c} for p, c in x["env"]["files"].items()]}, ensure_ascii=False) + "\n")
    reasons = collections.Counter(p.split(":")[0].split(" (")[0] for x in recs if x["problems"] for p in x["problems"])
    med = lambda k: (sorted(x["measures"][k] for x in kept)[len(kept) // 2] if kept else None)
    manifest = {"pipeline": "daa2", "model": args.model, "run_dir": str(run_dir), "source": {"repo": args.source, "revision": args.revision},
                "counts": {"rows": len(recs), "kept": len(kept), "dropped": len(recs) - len(kept), "repaired": sum(1 for x in recs if x.get("repaired")),
                           "rows_with_pipeline_vocab": sum(1 for x in kept if x["measures"]["pipeline_vocab"])},
                "drop_reasons": dict(reasons), "usage": usage.as_dict(), "wall_s": round(time.time() - t0),
                "medians": {k: med(k) for k in ("reuse_reasoning", "reuse_reply", "reuse_user", "n_look", "n_run", "n_write", "chars_files")}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f">>> kept {len(kept)} of {len(recs)} | drops {dict(reasons)} | pipeline-vocab rows {manifest['counts']['rows_with_pipeline_vocab']} | medians {manifest['medians']} | {manifest['wall_s']}s", flush=True)
    for k, v in usage.as_dict().items():
        print(f"    {k:11} calls {v['calls']:3} in {v['input']:6} cache_create {v['cache_create']:7} cache_read {v['cache_read']:7} out {v['output']:6} thinking {v['thinking']:6} list_usd {v['list_usd_x1000'] / 1000:.2f}")


if __name__ == "__main__":
    main()
