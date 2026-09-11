# ABOUTME: daa = "difficult advice, agentic": convert difficult-advice rows into dat-shaped agentic
# ABOUTME: rows -- same dilemma, facts, trace and decision; now an operator task over a file environment.
"""Agentify the difficult-advice corpus into `<date>-daa-synth`.

A difficult-advice (da) row is a user who faces a dilemma and asks the assistant for help;
the assistant's trace works out what is actually being asked and its answer helps with the
legitimate part and declines the rest. A difficult-agentic-task (dat) row is the same kind
of dilemma with the model as the ACTOR: an operator system prompt, a task, a bash tool over a
small file environment, and a `task_complete` tool to end the task.

Two pressures shape every generated turn, and the pipeline is built so that each is a check
rather than a hope:

  ENVIRONMENT CONSISTENCY is a fact, not a prompt. Every command runs for real, turn by turn,
  in a Docker container holding the row's files under a frozen scenario-dated clock, and the
  next turn is generated only after the previous command's real output is in the transcript.
  The generator therefore cannot use a flag it never saw or describe a file it never read
  without the grounding lints catching it -- and it cannot leak the host's clock, user or
  paths, because the container has none of them.

  DA CONSISTENCY is measured. The deliberation must keep most of the original reasoning's
  sentences verbatim (reuse ratio), every action must touch a path the map stage tied to one
  of the original reply's alternatives, and any fact copied from the original conversation
  that the agent's own context never showed drops the row (source-residue lint).

Stages, each a checkpointed snapshot in the run dir and on the Hub (StageCache):
    1 source    the da rows at a pinned revision
    2 map       generator: persona, task, environment files, outcome shape, and the
                AFFORDANCES -- one executable path per alternative the original reply
                offers -- then a dry run in the container (every script parses, every
                argparse --help prints); one repair round, else the row is dropped
    3 loop      the sandboxed agent loop: code samples 0-2 exploration turns; the model
                returns one read-only command per turn and sees its real output; then the
                deliberation (the original reasoning, minimally edited) with the first
                action; then up to 4 action turns, a write's content authored at the turn
                it is reached; after any action that printed nothing, code inserts a read
                of what it changed; then the closing message + task_complete, alone
    4 lint      the grounding, residue, clock, reuse, affordance and shape lints -- named
                problems per row, nothing dropped yet
    5 rewrite   one prose-only pass, fed the lint findings: an EDIT LIST (turn, field,
                old sentence, new sentence), applied only where the old sentence exists
                verbatim; commands and tool outputs are not editable and are diffed
                afterwards; per field at most 1 new sentence or 20% of its sentences, and
                the deliberation must still clear min_reuse -- else the whole list is discarded
    6 final     lint again; a row is kept when its problem list is empty
    dataset.jsonl = the kept rows; the manifest reports drop reasons at stage 4 and 6 and
                    how many rows the rewrite rescued

    uv run python scratch/daa/agentify.py --smoke                       # 10 rows, no push
    uv run python scratch/daa/agentify.py --ids a,b,c --no-push         # chosen rows, no push
    uv run python scratch/daa/agentify.py                               # all rows, pushed

Needs a running Docker daemon: the image `daa-sandbox` (python:3.12-slim + libfaketime) is
built on first use. One container per row per stage, network disabled, removed afterwards.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import random
import re
import subprocess
import sys
import tarfile
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from src.data.synth.ours.hf_cache import StageCache
from src.data.synth.ours.stage_runtime import Checkpoint, Usage, call_json, run_items
from src.infra.endpoints.openrouter import CACHE_MARK, OpenRouterClient
from src.infra.huggingface import training_data_tags
from src.naming import synth_name
from src.utils import git_sha, origin_url, timestamp

SOURCE_REPO = "dougalldeepmind/2026-09-08-da-synth"
SOURCE_REVISION = "42107bde00cd7f4360a3a6c581aac23a540dbfea"
CONSTITUTION = "constitutions/archive/claude_distilled_12_principles_mid/constitution.md"
GENERATOR = "anthropic/claude-sonnet-5"        # dat.yaml's draft/refine model
IMAGE = "daa-sandbox"
MAX_ACTIONS = 4
EXPLORE_WEIGHTS = {0: 0.2, 1: 0.4, 2: 0.4}

# Byte-identical to ODCV-Bench's tool schemas (mission_executor/agent_main.py) and to every
# dat row: the model must see at train time exactly what the eval serves, ungrammatical
# description included.
TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Executes shell commands. Put the command in the 'command' field. "
                       "The command should be a valid bash command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "task_complete", "description": "To mark the task as completion.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}},
                       "required": ["reason"]}}},
]

MAP_SYSTEM = """You convert a chat-format ethical-dilemma scenario into an AGENTIC one for training data.

You are given a conversation: a system prompt describing an AI assistant's deployment, a user who
asks for help with something, and the assistant's reply (with its private reasoning). The user's
request contains a dilemma: usually a legitimate goal plus a part that should not be done, sometimes
a pure judgement call, sometimes something to decline outright. The assistant's reasoning and reply
tell you what the honest outcome is.

Re-cast the SAME scenario so that an AI agent with a bash tool is the actor:
- Do NOT change the dilemma. Same people, same organisation, same facts, same numbers, same request
  including its problematic part, same honest outcome. You are changing the frame only.
- Pick the MOST NATURAL implied agentic task: what this person would hand to an agent with a shell.
  Take it from the operation the scenario itself describes -- the audit, migration, batch, scoring
  run, record entry or lookup -- and only fall back to writing a document when the scenario is one.
- The operator's task message IS the original user request with only the frame changed. It is
  given to you as NUMBERED SENTENCES; return `task_edits` -- {"edits": [{"n": int, "new": str}],
  "inserts": [{"after": int, "text": str}]} -- and the task is reassembled by code, so a sentence
  you do not touch survives verbatim. Edit only what the agentic frame changes (a command to run,
  a file to write, "help me draft" -> "draft it to /path"), insert the concrete paths and the
  invocation, keep the same goal, the same rider ("just tweak...", "leave out...", "phrase it
  so..."), the same rationalisation, in the operator's own voice. Give concrete paths, and when the
  task asks the agent to run a tool, the invocation the operator has in mind, flags included -- the
  agent may only use flags it has seen in the task or in a --help output. Every MOVE in the
  request that the reasoning answers -- a solidarity bid, a bind, a framing, a rationalisation --
  must survive into the task, because the deliberation will reuse that reasoning nearly
  verbatim and must still be answering something the operator actually said. Every person,
  role, figure and fact the agent will need to reason about must be in the task, the system prompt
  or a file: the agent never sees this conversation, only what you write here.
- Invent a SMALL file environment (2 to 4 files, each under 1000 characters, 3000 in total) that instantiates the
  scenario's facts (a scoring matrix, an eval history, a config, a log, a policy). If the task runs,
  processes or checks anything, one file IS the script or CLI it uses: the SHORTEST python or shell
  that really works on the other files -- one mode or subcommand, only the flags the dilemma turns
  on, argparse --help, no extra features. Every fact the reasoning relies on must be derivable.
  Absolute paths. Dates inside files must fit the scenario; the sandbox clock will be set to the day
  after the latest date you write, so scripts may use today's date freely.
- THE CONTRACT: list every alternative the original reply contains as an `affordance`, typed:
    kind "do"  -- something the REPLY ITSELF performs or offers to perform, in its own voice:
                  "I'll draft...", "here's the version...", "I can run...". These need environment
                  `paths` AND a `quote`: the sentence of the ORIGINAL ASSISTANT REPLY, copied
                  verbatim, in which the reply performs or offers exactly this. No such sentence
                  means the item is "say". The operator ASKING for an operation does not make it a
                  "do": the reply is the only source of "do" items -- with ONE extension:
                  an operation the operator asks for that the reply ENDORSES ("I'm on board with
                  using it", "the system is ready and we're rolling it live") is a "do", quoting the
                  endorsing sentence, IF the operation is internal and reversible: it computes,
                  lists, scores, ranks, checks or drafts, and leaves a file or a printout the
                  operator can inspect before anything happens. An endorsed act that commits the
                  human -- send, submit, approve, deploy, publish, notify, anything carrying their
                  name or reaching a third party -- is NEVER a "do", however warmly the reply
                  endorses it: "you should definitely send that" is advice, so "say".
    kind "say" -- ADVICE the reply gives the human: propose X to leadership, call the director,
                  escalate, file a complaint, log your hours, push for a process. The agent must
                  never do these; it tells the operator. `paths` may be empty. If the environment
                  contains a channel for one (a review queue, a send tool, a ticket log), that is
                  fine -- but the agent's actions will be checked and any action that carries out
                  a "say" item drops the row.
  Keep each `alternative` to one line; there is no `how` field.
- OUTWARD ACTS ARE DRAFTS WHEN THE AGENT DEVIATES. On a "split" or "decline" row the honest agent
  is not doing what the operator asked, so it may not send, submit, deploy, publish, post, notify
  or approve anything. When the reply performs or offers such an act, type it "do" in its DRAFT
  form -- write the message to a file, stage or dry-run the send, leave the submission to the
  operator. Give a tool a --dry-run or draft mode ONLY when the preview shows something a draft
  file cannot: a send that renders a template, a filing validated against a schema, a deploy that
  prints the diff or the affected records, a scorer that would log on commit. When the outward
  act just delivers a document (submit a memo, send a note file), the draft file IS the dry run:
  no mode, the tool stays untouched, the "do" is the file.
  Never drop the item instead: a reply that drafts something gives the agent a draft to do, and a
  row with no "do" at all is only right when the reply itself does nothing but decline.
- FILES THE AGENT WRITES. When the original row implies editing something that already exists --
  an announcement draft, an assessment section, a config -- that file IS in the environment with
  realistic content, and the agent is expected to read it before changing it. When the reply
  drafts something new, its path must NOT pre-exist: no placeholder, no "(pending)" stub; the
  path appears only in the affordance, and the agent creates the file.
  The environment must afford the shortcut the operator asks for AND every "do" alternative. Be
  strict about the split: the reply telling the human to do something is "say", even when an
  agent with a shell could do it. Most replies are mostly "say".
- outcome_shape: "split" (does a legitimate part, declines a part), "full" (does the whole task,
  nothing to decline), "decline" (declines as a whole; affordances may then be empty).

Return ONLY a JSON object:
{"agent_name": str, "agent_system_prompt": str,
 "task_edits": {"edits": [{"n": int, "new": str}], "inserts": [{"after": int, "text": str}]},
 "files": [{"path": str, "content": str}],
 "shortcut": {"how": str, "paths": [str]},
 "affordances": [{"kind": "do"|"say", "alternative": str, "paths": [str], "quote": str}],
 "outcome_shape": "split"|"full"|"decline"}
The agent_system_prompt describes the agent's role, organisation and what is at the paths
(2-5 sentences), in the register of the original system prompt. It must NOT mention the tools."""

ANCHOR_NOTE = """

YOUR PREVIOUS ATTEMPT (below) typed these affordances "do" without a verbatim `quote` from the
ORIGINAL ASSISTANT REPLY that performs or offers them. Return the corrected JSON object in full:
give each a quote copied exactly from the reply, or retype it "say". Change nothing else.

PREVIOUS JSON:
{previous}

UNANCHORED:
{failures}"""

REPAIR_NOTE = """

YOUR PREVIOUS ATTEMPT (below) was loaded into the sandbox and some of its scripts FAILED to
parse or to print --help. Return the corrected JSON object in full: fix the scripts or paths so
that every one loads cleanly, and change nothing else (same scenario, task, dilemma, outcome).

PREVIOUS JSON:
{previous}

FAILURES:
{failures}"""

LOOP_SYSTEM = """You are generating, one turn at a time, the transcript of an honest AI agent doing a task
with a bash tool, for training data. The agent's situation is the SAME dilemma an assistant already
answered in a chat setting; you are given that assistant's private reasoning and reply, and the
transcript is that answer re-enacted by an actor. Preserve; do not reinvent.

Hard rules -- every command you return is EXECUTED FOR REAL in a sandbox holding the files, and its
real output is appended before you are asked for the next turn:
- Reference only what has appeared: a flag you have not seen in a --help output or in the task, a
  file's content you have not read, a number no command printed, do not exist to you. Look first.
- One bash command per turn, as a plain string. A `--help` or a `cat` is a valid action turn when
  you need to see something before acting; it costs one of the action turns. To write a file use exactly
  `cat > /abs/path <<'EOF'\\n...\\nEOF`, its full content inline, authored NOW from what you have
  actually read. A file the task asks you to change is changed at ITS path (sed -i, or overwrite
  it), never replaced by a new file beside it.
- Never mention a person, role, figure or fact from the original chat that this environment has not
  shown you; the operator here said only what the task says.
- A READ is cat, head, tail, ls, grep, wc, diff, find, stat, or a program's --help. Running a
  program is an ACTION even when it only prints -- it must serve a DO item or not happen.
- YOUR INSTRUCTIONS ARE THE DO ITEMS, NOT THE TASK. The operator task is the request you are
  responding to; the DO list is what you do about it. Your state-changing actions are EXACTLY the
  DO items, one action each, nothing else. The task adds none: if it says "submit it" and no DO
  item submits, you do not submit. If a DO item cannot be done, you decline it in the closing turn
  with a reason. Reads (cat, --help, ls) and read-backs of what you wrote are free.
- Never execute advice. What the original reply tells the HUMAN to do -- escalate, call, propose,
  file, log, push for a process -- you say to the operator; you do not do it for them, even when
  the environment has a tool for it.
- An outward act (send, submit, deploy, publish, post, notify, approve) is yours to take ONLY if a
  DO item is one. Otherwise never, whatever the task says: the draft file or the dry-run is where
  you stop, and the sending is the operator's.
- Private reasoning is first person, plain and specific. No moralising, no summaries of rules.
- Your message to the operator is the ORIGINAL REPLY adapted to what you actually did: what you did
  (citing only what the commands really printed), what you did not do and why, and the SAY items,
  in the reply's voice and substance. No new advice beyond the reply's. It may end with a question
  to the operator; then the completion reason must say the task is closed pending their answer.
Return ONLY the JSON object the turn instruction asks for, newlines inside strings escaped."""

DELIBERATION_RULES = """The deliberation is the ORIGINAL private reasoning, minimally edited into the actor's situation. It
is given to you below as NUMBERED SENTENCES; you return only the changes and the sentences are
reassembled in order by code, so an untouched sentence survives verbatim at no cost:
  "edits":   [{"n": int, "new": str}]        -- replace sentence n ("" deletes it)
  "inserts": [{"after": int, "text": str}]   -- new sentence(s) placed after sentence n (0 = at the start)
Keep the structure, the argument, the sentences. A model can be both adviser and actor: INSERT the
actor's perspective where it is needed -- "the ask is for me to...", what the files just showed,
the action I will take -- and EDIT only what no longer makes sense (e.g. "help you draft" ->
"draft"; a fact "the officer describes" -> what the log showed). Do not shorten, do not paraphrase
for its own sake, do not add moralising. Most sentences must stay untouched. It is thought BEFORE
the actions run: it may say what they will do and why, never what they printed."""

# The closing turn accounts for every DO item; the lint checks each claim against the transcript.
OUTCOMES = ('"do_outcomes": <one entry per DO item: {"n": <its number>, "outcome": "done"|"in_message"|"declined", '
            '"why": <one line>}; done = an action of yours served it; in_message = it is a document and its content is in your '
            'message instead; declined = you chose not to, say why>')

REWRITE_SYSTEM = """You are a consistency editor for an agent transcript used as training data. The transcript was
generated turn by turn with every command executed for real; the commands and tool outputs are the
ground truth and are NOT editable. You edit only the agent's prose: private reasoning, the closing
message to the operator, and the task_complete reason.

You are given the original chat reasoning and reply the transcript re-enacts (the prose must stay
close to them), the transcript with its editable fields numbered, and a list of FINDINGS from an
automatic check. Fix each finding and any other inconsistency between the prose and what the
commands actually printed, by the SMALLEST edits: prefer deleting a sentence to rewriting it, keep
the original wording wherever it is not wrong. Do not add new claims, do not polish, do not
moralise. Never mention a person, role, figure or fact the transcript's own context (system prompt,
task, tool outputs) does not contain.

Return ONLY a JSON object: {"edits": [{"turn": int, "field": "reasoning"|"message"|"reason",
"old": str, "new": str}]}. `old` must be one complete sentence copied EXACTLY from that field
(it is matched verbatim); `new` is its replacement, or "" to delete it. An empty list is a valid
answer when nothing needs to change."""

RUN_VERBS = re.compile(r"\b(run|execute|process|check|scan|audit|generate|migrate|reconcile|score|validate)\b", re.I)
EDIT_VERBS = re.compile(r"\b(update|edit|rewrite|revise|fix|change|amend|adjust|modify|patch|tweak|correct|"
                        r"finali[sz]e|fill in|fill out|reword|rephrase)\b", re.I)
# The container has no network and nothing of the host's; the denylist only stops a command
# from wasting the row (or the daemon's time).
UNSAFE = re.compile(r"(?<![\w-])(sudo|su|curl|wget|ssh|scp|rsync|nc|ncat|telnet|pip|pip3|npm|apt|apt-get|"
                    r"mkfs|shutdown|reboot|halt|crontab|systemctl)(?![\w-])|rm\s+-[a-z]*r[a-z]*f?\s+/(?![\w])|dd\s+if=")
READ_HEADS = re.compile(r"^(cat|head|tail|less|more|grep|egrep|fgrep|wc|ls|diff|find|stat|file|sort|uniq|cut|tr|jq|"
                        r"column|awk|sed|echo|printf|pwd|which|type|tree|du|df|basename|dirname|realpath|nl|tac|"
                        r"paste|comm|md5sum|sha256sum)\b")
ABS_PATH = re.compile(r"/[\w][\w./-]*")
# Strong verbs count anywhere, a script called send_memo.py included; weak ones (release, push,
# file...) only as a standalone subcommand or flag, so release_checklist.py is not an outward act.
OUTWARD = re.compile(r"(?<![\w.-])(send|sent|submit|deploy|publish|notify|email|dispatch|transmit|broadcast|file-portal)\w*"
                     r"|(?<![\w./-])(post|mail|push|release|approve|escalate)(?![\w.-])"
                     r"|--(send|submit|deploy|publish|approved?|notify|release|post)\b", re.I)
PENDING = re.compile(r"\b(await|waiting|pending|operator|answer|reply|respond|decid|confirm)\w*", re.I)


def command_skeleton(cmd: str) -> str:
    """The command with heredoc bodies and quoted strings blanked, so a check judges the
    shell structure and not the prose it carries."""
    cmd = re.sub(r"<<-?\s*'?(\w+)'?\n.*?\n\1(?=\n|$)", "<<HEREDOC", cmd, flags=re.S)
    cmd = re.sub(r"'[^']*'", "''", cmd)
    return re.sub(r'"(?:[^"\\]|\\.)*"', '""', cmd)


def heredoc_body(cmd: str) -> str:
    m = re.search(r"<<-?\s*'?(\w+)'?\n(.*?)\n\1(?=\n|$)", cmd, flags=re.S)
    return m.group(2) if m else ""


def read_only(cmd: str) -> bool:
    """A command that can only look: no redirection, no in-place tools, every pipeline segment a
    reader (or any program's --help / -h)."""
    sk = command_skeleton(cmd)
    if re.search(r"(?<![\d&])>|\btee\b|\bsed\s+-[a-zA-Z]*i|\b(rm|mv|cp|touch|mkdir|chmod|chown|ln)\b", sk):
        return False
    for seg in re.split(r"\|\|?|&&|;", sk):
        seg = seg.strip()
        if not seg or re.search(r"(^|\s)(--help|-h)(\s|$)", seg):
            continue
        if not READ_HEADS.match(seg):
            return False
    return True


def modify_targets(cmd: str) -> set[str]:
    """Environment paths a command changes: redirection, tee, sed -i, cp/mv destinations."""
    sk = command_skeleton(cmd)
    out = set(m.group(1) for m in re.finditer(r">{1,2}\s*(/[\w./-]+)", sk))
    out |= set(m.group(1) for m in re.finditer(r"\btee\s+(?:-a\s+)?(/[\w./-]+)", sk))
    out |= set(m.group(1) for m in re.finditer(r"\b(?:cp|mv)\s+\S+\s+(/[\w./-]+)", sk))
    if re.search(r"\bsed\s+-[a-zA-Z]*i", sk):
        paths = ABS_PATH.findall(sk)
        if paths:
            out.add(paths[-1])
    return {p.rstrip(".,;:") for p in out}


def paragraphs_sentences(text: str) -> list[list[str]]:
    """Every sentence of every paragraph, in order, nothing dropped (the deliberation edit list
    numbers these; `sentences()` below is the reuse metric's filtered view)."""
    out = []
    for para in re.split(r"\n\s*\n", (text or "").strip()):
        sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", para.strip()) if x.strip()]
        if sents:
            out.append(sents)
    return out


def numbered(paras: list[list[str]]) -> str:
    k, lines = 0, []
    for para in paras:
        for x in para:
            k += 1
            lines.append(f"[{k}] {x}")
        lines.append("")
    return "\n".join(lines).strip()


def apply_sentence_edits(paras: list[list[str]], edits: list, inserts: list) -> str:
    """Reassemble the deliberation from the model's edit list (rule: untouched sentences verbatim)."""
    repl: dict[int, str] = {}
    for e in edits or []:
        try:
            repl[int(e["n"])] = str(e.get("new") or "").strip()
        except (KeyError, TypeError, ValueError):
            continue
    ins: dict[int, list[str]] = {}
    for e in inserts or []:
        try:
            ins.setdefault(int(e["after"]), []).append(str(e.get("text") or "").strip())
        except (KeyError, TypeError, ValueError):
            continue
    k, out_paras = 0, []
    head = [t for t in ins.get(0, []) if t]
    for para in paras:
        cur = list(head); head = []
        for x in para:
            k += 1
            y = repl.get(k, x)
            if y:
                cur.append(y)
            cur += [t for t in ins.get(k, []) if t]
        if cur:
            out_paras.append(" ".join(cur))
    return "\n\n".join(out_paras)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if len(s.strip()) > 20]


def fold(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower())


def reuse_ratio(original: str, adapted: str) -> float:
    """Fraction of the original's sentences that survive verbatim (case/space-folded)."""
    src = sentences(original)
    if not src:
        return 0.0
    f = fold(adapted)
    return sum(1 for s in src if fold(s) in f) / len(src)


def tool_result(stdout: str, returncode: int = 0, stderr: str = "") -> str:
    return json.dumps({"stdout": stdout, "stderr": stderr, "role": "tool", "returncode": returncode})


def bash_call(command: str) -> dict:
    return {"type": "function", "function": {"name": "bash", "arguments": {"command": command}}}


def complete_call(reason: str) -> dict:
    return {"type": "function", "function": {"name": "task_complete", "arguments": {"reason": reason}}}


def edit_targets(task: str, files: dict[str, str]) -> set[str]:
    """Environment paths the task asks the agent to change (an edit verb within 80 chars before)."""
    out = set()
    for m in re.finditer(r"/[\w./-]+", task):
        p = m.group(0).rstrip(".,;:")
        window = task[max(0, m.start() - 80):m.start()]
        window = re.split(r"[.!?\n]", window)[-1]          # same sentence only
        if p in files and EDIT_VERBS.search(window):
            out.add(p)
    return out


def scenario_date(*texts: str) -> dt.datetime:
    """The sandbox 'now': the day after the latest ISO date the scenario mentions; else a
    fixed day. Frozen by libfaketime, so scripts calling today() agree with the files."""
    found = []
    for t in texts:
        for y, m, d in re.findall(r"\b(20\d\d)-(\d\d)-(\d\d)\b", t or ""):
            try:
                found.append(dt.date(int(y), int(m), int(d)))
            except ValueError:
                pass
    day = (max(found) + dt.timedelta(days=1)) if found else dt.date(2024, 6, 12)
    return dt.datetime.combine(day, dt.time(9, 30))


# --- sandbox: one container per row, frozen clock, no network --------------------------

def docker(*args: str, stdin: bytes | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], input=stdin, capture_output=True, timeout=timeout)


def ensure_image() -> None:
    res = docker("image", "inspect", IMAGE)
    if res.returncode == 0:
        return
    print(f">>> building {IMAGE} (python:3.12-slim + libfaketime): inspect said {res.stderr.decode()[:120]!r}", flush=True)
    dockerfile = (
        "FROM python:3.12-slim\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends libfaketime coreutils "
        "&& rm -rf /var/lib/apt/lists/* "
        "&& ln -s \"$(find /usr/lib -name 'libfaketime.so.1' | head -1)\" /usr/local/lib/libfaketime.so.1\n"
        "CMD [\"sleep\", \"infinity\"]\n")
    with tempfile.TemporaryDirectory() as d:
        Path(d, "Dockerfile").write_text(dockerfile)
        res = docker("build", "-q", "-t", IMAGE, d, timeout=900)
    if res.returncode:
        raise SystemExit(f"docker build failed:\n{res.stderr.decode()[-2000:]}")


def check_docker() -> None:
    res = docker("info", "--format", "{{.ServerVersion}}")
    if res.returncode:
        raise SystemExit("Docker daemon not reachable: start Docker Desktop (`open -a Docker`) and retry. "
                         "The sandbox is a container per row; there is no host fallback.")


class Sandbox:
    """A throwaway container holding the row's files at their real absolute paths, mtimes set
    a day before the frozen scenario clock, so `ls`, `date` and `today()` agree with the files
    and nothing of the host (clock, user, paths) can reach a transcript."""

    def __init__(self, files: dict[str, str], now: dt.datetime, timeout: int = 30):
        self.timeout = timeout
        self.now = now
        self.name = f"daa_{uuid.uuid4().hex[:12]}"
        # the kernel's clock as `ls -l` prints it (time form for recent files, year form otherwise)
        # and as ISO: a file created by the agent shows one of these until the settle touches it
        host = dt.datetime.now(dt.timezone.utc)
        self._host_stamps = re.compile(rf"{host.strftime('%b')} +{host.day} +(?:\d\d:\d\d|{host.year})|{host.strftime('%Y-%m-%d')}")
        settled = now - dt.timedelta(hours=1)
        self._settled = settled.strftime("%b %e %H:%M").replace("  ", " ")
        res = docker("run", "-d", "--rm", "--network", "none", "--name", self.name, IMAGE, "sleep", "1800")
        if res.returncode:
            raise RuntimeError(f"docker run failed: {res.stderr.decode()[-500:]}")
        buf = io.BytesIO()
        mtime = int((now - dt.timedelta(days=1)).replace(tzinfo=dt.timezone.utc).timestamp())
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for p, c in files.items():
                data = c.encode("utf-8")
                info = tarfile.TarInfo(p.lstrip("/"))
                info.size = len(data)
                info.mtime = mtime
                info.mode = 0o755 if (p.endswith((".sh", ".py")) or c.startswith("#!")) else 0o644
                tar.addfile(info, io.BytesIO(data))
        res = docker("cp", "-", f"{self.name}:/", stdin=buf.getvalue())
        if res.returncode:
            self.close()
            raise RuntimeError(f"docker cp failed: {res.stderr.decode()[-500:]}")
        # the directories the tar created carry the host's clock; align them
        dirs: set[str] = set()
        for p in files:
            d = os.path.dirname(p)
            while d and d != "/":
                dirs.add(d)
                d = os.path.dirname(d)
        dirs = sorted(dirs)
        stamp = (now - dt.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        self._exec(f"touch -d '{stamp}' / {' '.join(dirs)}", faketime=False)

    # libfaketime cannot change the mtime the kernel stamps on a file the agent creates, so after
    # every command anything newer than the scenario clock (i.e. stamped with the host's real 2026
    # clock) is touched back to it. Only non-system top-level dirs are scanned; a new top-level
    # dir the agent makes (`mkdir /policy`) is included by construction.
    _SYSTEM_DIRS = "/bin|/boot|/dev|/etc|/lib|/lib32|/lib64|/libx32|/proc|/run|/sbin|/sys|/usr"

    def _settle_clock(self) -> None:
        stamp = (self.now - dt.timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")   # an hour ago: `ls` shows the time form
        self._exec("for d in /*; do case $d in " + self._SYSTEM_DIRS + ") ;; *) echo \"$d\";; esac; done | "
                   f"xargs -r -I DIR find DIR -xdev -newermt '{stamp}' -exec touch -h -d '{stamp}' {{}} + 2>/dev/null; "
                   f"touch -d '{stamp}' /; true", faketime=False)

    def _exec(self, command: str, faketime: bool = True) -> dict:
        env = ["-e", "HOME=/root", "-e", "LANG=C.UTF-8", "-e", "PYTHONWARNINGS=ignore",
               "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "TZ=UTC"]
        if faketime:
            # NO_FAKE_STAT: the clock is faked, file mtimes are not (else `ls` dates every file "now")
            env += ["-e", "LD_PRELOAD=/usr/local/lib/libfaketime.so.1", "-e", "NO_FAKE_STAT=1",
                    "-e", f"FAKETIME={self.now.strftime('%Y-%m-%d %H:%M:%S')}"]
        try:
            res = docker("exec", *env, "-w", "/", self.name, "timeout", str(self.timeout), "bash", "-c", command,
                         timeout=self.timeout + 15)
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": f"timed out after {self.timeout}s", "returncode": 124}
        stderr = res.stderr.decode("utf-8", "replace")
        if res.returncode == 124 and not stderr:
            stderr = f"timed out after {self.timeout}s"
        return {"stdout": res.stdout.decode("utf-8", "replace"), "stderr": stderr, "returncode": res.returncode}

    def run(self, command: str) -> dict:
        if UNSAFE.search(command_skeleton(command)):
            return {"stdout": "", "stderr": "refused by the sandbox denylist", "returncode": 126, "refused": True}
        res = self._exec(command)
        self._settle_clock()
        res["stdout"] = self._host_stamps.sub(self._settled, res["stdout"])
        return res

    def close(self) -> None:
        docker("rm", "-f", self.name)

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --- stages ----------------------------------------------------------------------------

def load_source(repo: str, revision: str) -> list[dict]:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, "dataset.jsonl", repo_type="dataset", revision=revision,
                           token=os.environ.get("HF_TOKEN"))
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    out = []
    for r in rows:
        msgs = r["messages"]
        out.append({
            "scenario_id": r["metadata"]["scenario_id"],
            "source_repo": repo, "source_revision": revision,
            "system": next(m["content"] for m in msgs if m["role"] == "system"),
            "user": next(m["content"] for m in msgs if m["role"] == "user"),
            "reasoning": [m for m in msgs if m["role"] == "assistant"][-1].get("reasoning_content") or "",
            "answer": [m for m in msgs if m["role"] == "assistant"][-1]["content"],
            "metadata": r["metadata"],
        })
    return out


def original_block(r: dict) -> str:
    return (f"ORIGINAL SYSTEM PROMPT:\n{r['system']}\n\nORIGINAL USER REQUEST (numbered; return task_edits over these):\n"
            f"{numbered(paragraphs_sentences(r['user']))}\n\n"
            f"ORIGINAL ASSISTANT REASONING (private):\n{r['reasoning']}\n\n"
            f"ORIGINAL ASSISTANT REPLY:\n{r['answer']}")


def assemble_task(a: dict, r: dict) -> dict:
    """The operator task from the map's edit list over the original request (verbatim elsewhere)."""
    te = a.get("task_edits") or {}
    a["task"] = apply_sentence_edits(paragraphs_sentences(r["user"]), te.get("edits") or [], te.get("inserts") or [])
    return a


def env_files(a: dict) -> dict[str, str]:
    return {f["path"]: f["content"] for f in a.get("files", []) if f.get("path") and f["path"].startswith("/")}


def probe(files: dict[str, str], now: dt.datetime) -> list[str]:
    """Dry run: every python file parses, every shell file passes `bash -n`, every argparse
    script prints --help. Proves the environment loads before a single agent turn is paid for."""
    problems = []
    with Sandbox(files, now) as sb:
        for p, c in files.items():
            if p.endswith(".py") or c.startswith("#!/usr/bin/env python"):
                res = sb.run(f"python3 -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' {p}")
                if res["returncode"]:
                    problems.append(f"{p} does not parse: {res['stderr'][-300:]}")
                elif "argparse" in c:
                    res = sb.run(f"python3 {p} --help")
                    if res["returncode"]:
                        problems.append(f"{p} --help failed: {res['stderr'][-300:]}")
            elif p.endswith(".sh") or c.startswith("#!/bin/bash") or c.startswith("#!/bin/sh"):
                res = sb.run(f"bash -n {p}")
                if res["returncode"]:
                    problems.append(f"{p} does not parse: {res['stderr'][-300:]}")
    return problems


def unanchored(a: dict, answer: str) -> list[str]:
    """`do` affordances whose quote is not a verbatim sentence of the reply (rule 2)."""
    out = []
    for x in a.get("affordances") or []:
        if x.get("kind") == "say":
            continue
        q = fold(str(x.get("quote") or "").strip().strip('"'))
        if len(q) < 20 or q not in fold(answer):
            out.append(f"{x.get('alternative', '')[:80]!r} quote={str(x.get('quote') or '')[:80]!r}")
    return out


def map_one(client: OpenRouterClient, usage: Usage, model: str, temperature: float,
            max_tokens: int) -> Callable[[dict], dict]:
    req = ("agent_system_prompt", "task_edits", "files", "affordances", "outcome_shape")

    def fn(r: dict) -> dict:
        a, _ = call_json(client, usage, model, MAP_SYSTEM, original_block(r), temperature, max_tokens, "map", required=req)
        assemble_task(a, r)
        bad = unanchored(a, r["answer"])
        if bad:
            a2, _ = call_json(client, usage, model, MAP_SYSTEM, original_block(r) + ANCHOR_NOTE.format(
                previous=json.dumps(a, indent=1), failures="\n".join(bad)), min(temperature, 0.3), max_tokens,
                "map_anchor", required=req)
            assemble_task(a2, r)
            if len(unanchored(a2, r["answer"])) < len(bad):
                a = a2
        files = env_files(a)
        now = scenario_date(a.get("task", ""), a.get("agent_system_prompt", ""), *files.values())
        problems = probe(files, now) if files else ["no environment files"]
        problems += [f"do affordance not anchored in the reply: {b}" for b in unanchored(a, r["answer"])]
        repaired = False
        if problems:
            user = original_block(r) + REPAIR_NOTE.format(previous=json.dumps(a, indent=1), failures="\n".join(problems))
            a2, _ = call_json(client, usage, model, MAP_SYSTEM, user, min(temperature, 0.3), max_tokens,
                              "map_repair", required=req)
            assemble_task(a2, r)
            files2 = env_files(a2)
            now2 = scenario_date(a2.get("task", ""), a2.get("agent_system_prompt", ""), *files2.values())
            problems2 = probe(files2, now2) if files2 else ["no environment files"]
            if len(problems2) < len(problems):
                a, now, problems, repaired = a2, now2, problems2, True
        rng = random.Random(f"{r['scenario_id']}:explore")
        n_explore = rng.choices(list(EXPLORE_WEIGHTS), weights=list(EXPLORE_WEIGHTS.values()))[0]
        return {**r, "mapped": a, "map_model": model, "map_problems": problems, "map_repaired": repaired,
                "scenario_now": now.isoformat(), "n_explore": n_explore}
    return fn


def render_steps(steps: list[dict]) -> str:
    out = []
    for i, s in enumerate(steps):
        block = f"[turn {i + 1}] private reasoning: {s['reasoning']}\n$ {s['command']}\n{s['stdout']}"
        if s.get("returncode"):
            block += f"\n[exit {s['returncode']}] {s.get('stderr', '')}"
        elif s.get("stderr"):
            block += f"\n[stderr] {s['stderr']}"
        out.append(block)
    return "\n\n".join(out) or "(nothing yet)"


VERIFY_WHY = [
    "That printed nothing, so I'll read back {name} to see what actually landed before going on.",
    "No output from that; checking {name} directly rather than assuming it did what I meant.",
    "Reading {name} back: an empty result isn't confirmation, the file's contents are.",
]


def loop_one(client: OpenRouterClient, usage: Usage, model: str, temperature: float,
             max_tokens: int, min_reuse: float) -> Callable[[dict], dict]:
    """The sandboxed agent loop: one call per turn, the real output of each command in the
    prompt for the next. Records steps (each with its reasoning) and the finish."""

    def context(r: dict, steps: list[dict], instruction: str) -> str:
        a = r["mapped"]
        # WHAT each alternative is and WHICH paths support it -- never HOW: the map stage wrote the
        # scripts and its `how` lines carried their flags, which the loop model then used without a
        # --help (smoke 2026-09-10: 3 of 10 rows dropped for exactly that). Flags come from --help only.
        clean = lambda t: re.sub(r'(?<![\w-])--?[\w-]+', '', t or '')
        affs = a.get("affordances") or []
        do = "\n".join(f"- D{i + 1}: {clean(x.get('alternative'))} (paths: {', '.join(x.get('paths') or [])})"
                       for i, x in enumerate(x for x in affs if x.get("kind") != "say"))
        say = "\n".join(f"- {clean(x.get('alternative'))}" for x in affs if x.get("kind") == "say")
        aff = (f"DO -- your state-changing actions are exactly these, one each, nothing else; every action names the item "
               f"it serves, and your closing turn accounts for each item (done / in_message / declined):\n{do or '- (nothing: no state-changing action at all)'}\n"
               f"SAY -- advice the original reply gives the human; these go in your closing MESSAGE and are NEVER an action "
               f"(do not escalate, notify, file, send, log or propose on the operator's behalf):\n{say or '- (none)'}")
        # everything before CACHE_MARK is identical on every turn of the row: cached (Anthropic breakpoint)
        return (f"YOUR INSTRUCTIONS (the only source of your actions; the operator task below is the request, not your orders):\n{aff}\n\n"
                f"AGENT SYSTEM PROMPT:\n{a['agent_system_prompt']}\n\n"
                f"OPERATOR TASK (the request you are responding to; NOT your instructions):\n{a['task']}\n\n"
                f"OUTCOME SHAPE: {a.get('outcome_shape')}\n\n"
                f"ORIGINAL PRIVATE REASONING (to adapt, not rewrite):\n{r['reasoning']}\n\n"
                f"ORIGINAL REPLY (your closing message adapts this):\n{r['answer']}\n\n{CACHE_MARK}"
                f"TRANSCRIPT SO FAR (commands and their REAL output):\n{render_steps(steps)}\n\n"
                f"THIS TURN: {instruction}")

    def fn(r: dict) -> dict:
        a = r["mapped"]
        files = env_files(a)
        now = dt.datetime.fromisoformat(r["scenario_now"])
        n_explore = r["n_explore"]
        steps: list[dict] = []
        problems: list[str] = []
        finish: dict = {}
        sid = r["scenario_id"]
        with Sandbox(files, now) as sb:
            # exploration: code decides how many; the model picks each read-only command
            for k in range(n_explore):
                instr = (f"Exploration turn {k + 1} of {n_explore}. Return {{\"reasoning\": <1-2 sentences of private "
                         "reasoning: what you want to check and why>, \"command\": <ONE read-only bash command: cat, head, "
                         "tail, grep, wc, ls, diff, or a script's --help>}.")
                parsed, _ = call_json(client, usage, model, LOOP_SYSTEM, context(r, steps, instr), temperature,
                                      max_tokens, "explore", required=("reasoning", "command"))
                cmd = str(parsed.get("command") or "")
                if not read_only(cmd):
                    parsed, _ = call_json(client, usage, model, LOOP_SYSTEM,
                                          context(r, steps, instr + f"\n\nYOUR PREVIOUS COMMAND ({cmd!r}) is not read-only; "
                                                  "return one that only looks."), min(temperature, 0.3),
                                          max_tokens, "explore_retry", required=("reasoning", "command"))
                    cmd = str(parsed.get("command") or "")
                    if not read_only(cmd):
                        problems.append(f"exploration not read-only: {cmd[:80]!r}")
                res = sb.run(cmd)
                steps.append({"phase": "explore", "reasoning": str(parsed.get("reasoning") or "").strip(),
                              "command": cmd, **res})
            # deliberation (an edit list over the numbered original) + first action
            paras = paragraphs_sentences(r["reasoning"])
            instr = ("Exploration is over. Return {\"edits\": [...], \"inserts\": [...], \"command\": <the first action "
                     "command, or null when the honest outcome is to decline the whole task>, \"serves\": <the DO item this "
                     "command serves, e.g. \"D1\", or null for a read>, and when command is null "
                     "also \"message\": <your message to the operator>, \"reason\": <the task_complete reason, <=600 "
                     "chars> and " + OUTCOMES + "}.\n" + DELIBERATION_RULES + f"\n\nNUMBERED ORIGINAL REASONING:\n{numbered(paras)}")
            req = ("edits",)
            parsed, _ = call_json(client, usage, model, LOOP_SYSTEM, context(r, steps, instr), temperature, max_tokens,
                                  "deliberate", required=req)
            deliberation = apply_sentence_edits(paras, parsed.get("edits") or [], parsed.get("inserts") or [])
            reuse = reuse_ratio(r["reasoning"], deliberation)
            if reuse < min_reuse:
                nudge = (f"\n\nYOUR PREVIOUS ATTEMPT edited or deleted so many sentences that only {reuse:.0%} survive. "
                         "Edit far fewer: change only the words that no longer fit the agent's situation, prefer inserts.")
                parsed2, _ = call_json(client, usage, model, LOOP_SYSTEM, context(r, steps, instr + nudge),
                                       min(temperature, 0.3), max_tokens, "deliberate_retry", required=req)
                d2 = apply_sentence_edits(paras, parsed2.get("edits") or [], parsed2.get("inserts") or [])
                if reuse_ratio(r["reasoning"], d2) > reuse:
                    parsed, deliberation = parsed2, d2
            cmd = parsed.get("command")
            if not cmd:
                finish = {"closing": deliberation, "message": str(parsed.get("message") or "").strip(),
                          "reason": str(parsed.get("reason") or "").strip(), "declined": True,
                          "do_outcomes": parsed.get("do_outcomes") or []}
                if not finish["message"] or not finish["reason"]:
                    # the model declined but forgot the closing fields: one forced finish turn
                    parsed, _ = call_json(client, usage, model, LOOP_SYSTEM,
                                          context(r, steps, "You declined the task; no command. Return {\"reasoning\": "
                                                  "<1-2 sentences>, \"message\": <your message to the operator>, \"reason\": "
                                                  "<task_complete reason, <=600 chars>, " + OUTCOMES + "}."),
                                          min(temperature, 0.3), max_tokens, "finish", required=("message", "reason"))
                    finish.update({"message": str(parsed.get("message") or "").strip(),
                                   "reason": str(parsed.get("reason") or "").strip(),
                                   "do_outcomes": parsed.get("do_outcomes") or finish.get("do_outcomes") or []})
            else:
                n_actions = 0
                why = deliberation
                serves = parsed.get("serves")
                while True:
                    cmd = str(cmd)
                    res = sb.run(cmd)
                    steps.append({"phase": "action", "reasoning": why, "command": cmd, "serves": serves, **res})
                    n_actions += 1
                    if res["returncode"] == 0 and not res["stdout"].strip():
                        targets = sorted(modify_targets(cmd))
                        if targets:
                            t = targets[0]
                            vcmd = f"tail -n 40 {t}" if ">>" in command_skeleton(cmd) else f"cat {t}"
                            vres = sb.run(vcmd)
                            tmpl = VERIFY_WHY[random.Random(f"{sid}:{len(steps)}").randrange(len(VERIFY_WHY))]
                            steps.append({"phase": "verify", "reasoning": tmpl.format(name=os.path.basename(t)),
                                          "command": vcmd, **vres})
                    left = MAX_ACTIONS - n_actions
                    finish_obj = ("{\"reasoning\": <1-2 sentences: what the last output showed and that it is time to "
                                  "report>, \"message\": <your message to the operator>, \"reason\": <the task_complete "
                                  "reason: what was done, what was deliberately not done and why, <=600 chars>, " + OUTCOMES + "}")
                    if left > 0:
                        instr = (f"Return EITHER {{\"reasoning\": <one sentence: what this step does given what the last "
                                 f"output showed>, \"command\": <the next bash command>, \"serves\": <the DO item it serves, "
                                 f"e.g. \"D2\", or null for a read>}} OR, when the legitimate work is done, {finish_obj}. "
                                 f"You have {left} action command(s) left.")
                        parsed, _ = call_json(client, usage, model, LOOP_SYSTEM, context(r, steps, instr), temperature,
                                              max_tokens, "act", required=("reasoning",))
                    else:
                        instr = f"Your action budget is used. Return {finish_obj}."
                        parsed, _ = call_json(client, usage, model, LOOP_SYSTEM, context(r, steps, instr), temperature,
                                              max_tokens, "finish", required=("reasoning", "message", "reason"))
                    cmd = parsed.get("command") if left > 0 else None
                    serves = parsed.get("serves")
                    why = str(parsed.get("reasoning") or "").strip()
                    if not cmd:
                        finish = {"closing": why, "message": str(parsed.get("message") or "").strip(),
                                  "reason": str(parsed.get("reason") or "").strip(), "declined": False,
                                  "do_outcomes": parsed.get("do_outcomes") or []}
                        break
        if finish.get("message", "").rstrip().endswith("?") and not PENDING.search(finish.get("reason", "")):
            finish["reason"] = finish["reason"].rstrip() + " Task closed pending the operator's reply to the question above."
        return {**r, "loop": {"steps": steps, "finish": finish, "problems": problems, "n_explore": n_explore},
                "loop_model": model}
    return fn


# --- assemble + lint -------------------------------------------------------------------

def build_messages(r: dict) -> list[dict]:
    a, lp = r["mapped"], r["loop"]
    msgs: list[dict] = [{"role": "system", "content": a["agent_system_prompt"].strip()},
                        {"role": "user", "content": a["task"].strip()}]
    for s in lp["steps"]:
        msgs.append({"role": "assistant", "content": "", "reasoning_content": s["reasoning"], "tool_calls": [bash_call(s["command"])]})
        msgs.append({"role": "tool", "content": tool_result(s["stdout"], s["returncode"], s.get("stderr", ""))})
    f = lp["finish"]
    msgs.append({"role": "assistant", "content": f.get("message", ""), "reasoning_content": f.get("closing", ""),
                 "tool_calls": [complete_call(f.get("reason", ""))]})
    return msgs


def deliberation_of(r: dict) -> str:
    acts = [s for s in r["loop"]["steps"] if s["phase"] == "action"]
    return acts[0]["reasoning"] if acts else r["loop"]["finish"].get("closing", "")


def host_date_strings() -> list[str]:
    today = dt.datetime.now()
    return [today.strftime("%Y-%m-%d"), today.strftime("%b %e ").replace("  ", " ") + today.strftime("%H:")[:2],
            today.strftime("%d %b %Y"), today.strftime("%B %d, %Y"), today.strftime("%Y%m%d")]


def residue_tokens(prose: str, source: str, context: str) -> list[str]:
    """Distinctive tokens in the prose that the original chat contains and the agent's own
    context does not: names, roles, acronyms and figures copied through the reframe."""
    everything = prose + "\n" + source + "\n" + context
    lower_words = set(re.findall(r"(?<![A-Za-z])[a-z]{3,}(?![A-Za-z])", everything))
    cands = set()
    for m in re.finditer(r"\b[A-Z][a-z]{2,}\b", prose):
        if m.group(0).lower() in lower_words:
            continue                              # seen lowercase somewhere: a common word, not a name
        before = prose[:m.start()].rstrip()
        if not before or before[-1] in ".!?:;\"'*-([#|>/•–—" or before[-1].isdigit():
            continue                              # sentence- or heading-initial: capitalisation says nothing
        cands.add(m.group(0))
    # acronyms are matched case-sensitively in the source ("BY:" in a form is not "by")
    acronyms = {t for t in re.findall(r"\b(?:[A-Z]{2,5}|[A-Z][a-z][A-Z]{1,3})\b", prose) if t in source}
    cands |= set(re.findall(r"\$\d[\d,.]*[MKkBm]?|\b\d[\d,.]*%|\b\d{3,}[\d,.]*\b|\b\d+\.\d+\b", prose))
    cands -= {"I", "AI", "OK", "TODO", "EOF", "JSON", "CSV", "PDF", "URL", "ID", "The", "This", "That"}
    src, ctx = fold(source), fold(context)
    return sorted({t for t in cands if fold(t) in src and fold(t) not in ctx} | {t for t in acronyms if fold(t) not in ctx})


_STOP = set("the a an and or of to in for on with that this is are be by as at it its from not no if we you your our their "
            "they them he she her his instead before after than then there here what which who how why into out about over "
            "under any all each more most some such only also just".split())


def _words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", (t or "").lower()) if w not in _STOP}


DOC_ITEM = re.compile(r"\b(draft|write|memo|note|message|brief|letter|report|summary|framing|document|email|entry|record)\w*", re.I)


def do_item_problems(a: dict, steps: list[dict], finish: dict) -> list[str]:
    """Every `do` item is done (an action touched its paths), delivered in the message (a document
    item whose content the message carries) or declined with a reason; the closing turn claims one
    of these per item and each claim is checked against the transcript."""
    dos = [x for x in a.get("affordances") or [] if x.get("kind") != "say"]
    if not dos:
        return []
    claims: dict[int, dict] = {}
    for c in finish.get("do_outcomes") or []:
        m = re.search(r"\d+", str(c.get("n", "")))          # "D1", "1", 1
        if m:
            claims[int(m.group(0))] = c
    within = lambda t, p: t == p or t.startswith(p.rstrip("/") + "/") or p.startswith(t.rstrip("/") + "/")
    touched: set[str] = set()
    for s in steps:
        if s["phase"] == "action" and not read_only(s["command"]):
            touched |= {p.rstrip(".,;:") for p in ABS_PATH.findall(s["command"])} | modify_targets(s["command"])
    msg_sents = sentences(finish.get("message", ""))
    out = []
    for i, x in enumerate(dos, start=1):
        c = claims.get(i)
        if not c:
            out.append(f"do item D{i} unaccounted for in the closing turn")
            continue
        outcome, why = str(c.get("outcome") or ""), str(c.get("why") or "")
        paths = x.get("paths") or []
        if outcome == "done":
            if not any(within(t, p) for t in touched for p in paths):
                out.append(f"do item D{i} claimed done but no action touched {paths}")
        elif outcome == "in_message":
            if not DOC_ITEM.search(x.get("alternative") or "") and not any(p.endswith((".md", ".txt", ".json", ".yaml", ".csv")) for p in paths):
                out.append(f"do item D{i} claimed in_message but is not a document")
            else:
                is_offer = bool(re.match(r"\s*offer", x.get("alternative") or "", re.I)) or str(x.get("quote") or "").rstrip().endswith("?")
                carried = any(len(_words(s) & _words(x.get("alternative"))) >= 3 or len(_words(s) & _words(x.get("quote"))) >= 3 for s in msg_sents)
                offered = is_offer and bool(re.search(r"\?\s*$|\b(want me to|I can|happy to|let me know|if you want)\b", finish.get("message", ""), re.I | re.M))
                if not carried and not offered:
                    out.append(f"do item D{i} claimed in_message but the message does not carry it")
        elif outcome == "declined":
            if len(why.strip()) < 15:
                out.append(f"do item D{i} declined without a reason")
        else:
            out.append(f"do item D{i} has an unknown outcome {outcome!r}")
    return out


def lint(r: dict, min_reuse: float) -> list[str]:
    a, lp = r["mapped"], r["loop"]
    files = env_files(a)
    steps, f = lp["steps"], lp["finish"]
    problems: list[str] = list(r.get("map_problems") or []) + list(lp.get("problems") or [])
    if not 2 <= len(files) <= 6:
        problems.append(f"{len(files)} environment files (want 2-6)")
    for p, c in files.items():
        if len(c) > 6000:
            problems.append(f"file {p} is {len(c)} chars (>6000)")
    task, system = a.get("task") or "", a.get("agent_system_prompt") or ""
    acts = [s for s in steps if s["phase"] == "action"]
    shape = a.get("outcome_shape") or ""
    problems += do_item_problems(a, steps, f)
    message, reason, closing = f.get("message", ""), f.get("reason", ""), f.get("closing", "")
    if not message or not reason:
        problems.append("empty operator_message or completion_reason")
    said = message + " " + reason
    # -- grounding: paths, flags, files the message names -----------------------------
    seen_text = task + "\n" + system           # what the agent could see before any command
    read_paths: set[str] = set()               # env files whose content was shown
    referenced: set[str] = set()               # env files any command named
    for i, s in enumerate(steps):
        out_text = s["stdout"] + "\n" + s.get("stderr", "")
        for p in ABS_PATH.findall(command_skeleton(s["command"])):
            if p.rstrip(".,;:") in files:
                referenced.add(p.rstrip(".,;:"))
        if s["phase"] == "action" and not read_only(s["command"]):
            # looking at a guessed path is harmless (it errors); ACTING on one is the failure
            sk = command_skeleton(s["command"])
            for p in {p.rstrip(".,;:") for p in ABS_PATH.findall(sk)}:
                if p in files and p not in seen_text and os.path.basename(p) not in seen_text:
                    problems.append(f"ungrounded path: {p} used before it appeared in any command or output")
            for p in modify_targets(s["command"]):
                if p in files and p not in read_paths:
                    problems.append(f"modified unread: {p} changed without being read first")
            for flag in set(re.findall(r"(?<![\w-])--[a-z][\w-]*", sk)) - {"--help"}:
                if flag not in seen_text and not s.get("returncode"):
                    problems.append(f"ungrounded flag: {flag} never appeared in an output or the task")
            if len(s["stdout"]) > 3000:
                problems.append(f"stdout {len(s['stdout'])} chars (>3000) from {s['command'][:50]!r}")
            if "task_complete" in s["command"]:
                problems.append("bad action command")
        if s["phase"] in ("explore", "verify") or read_only(s["command"]):
            for p in ABS_PATH.findall(command_skeleton(s["command"])):
                if p.rstrip(".,;:") in files:
                    read_paths.add(p.rstrip(".,;:"))
        seen_text += "\n" + s["command"] + "\n" + out_text
    for p in files:
        name = os.path.basename(p)
        if len(name) > 3 and name in said and p not in read_paths and p not in referenced and name not in seen_text:
            problems.append(f"message names unread file: {name}")
    # (the old "task names an operation => some action runs it" check is gone: the DO list now decides
    #  what runs, and declining the operator's command in favour of a draft is the honest outcome)
    refused = lambda p: re.search(r"\b(not|n't|never|declin\w*|refus\w*|left|untouched|unchanged)\b[^.]{0,120}"
                                  + re.escape(os.path.basename(p)), said, re.I)
    new_paths = {p for s in acts for p in modify_targets(s["command"]) if p not in files}
    missed = [p for p in edit_targets(task, files) if p not in cmds and not refused(p)]
    siblings = [q for q in new_paths if any(os.path.dirname(q) == os.path.dirname(p) for p in missed)]
    if missed and siblings:
        problems.append(f"task asks to change {missed}; action wrote {sorted(siblings)} instead")
    # -- rule 1: outward acts only while doing exactly what was asked --------------------------
    deviates = shape != "full" or bool(re.search(r"\b(not|n't|never|declin\w*|refus\w*|held off|instead)\b", said, re.I))
    for s in acts:
        sk = command_skeleton(s["command"])
        if read_only(s["command"]) or re.search(r"dry.?run|(?<![\w-])(--mode\s+)?(draft|preview)(?![\w-])", sk, re.I):
            continue
        if heredoc_body(s["command"]) and not re.search(r"\|\s*\S|&&|;", re.sub(r"<<HEREDOC.*", "", sk, flags=re.S).split(">", 1)[-1]):
            continue                                   # a heredoc write to a local file is never outward
        if deviates and OUTWARD.search(re.sub(r"\S*[_-]draft\S*", "", sk, flags=re.I)):
            problems.append(f"outward act while deviating from the request: {s['command'].splitlines()[0][:70]!r}")
    # -- affordances: every action touches a path the map tied to a DA alternative ----------
    affs = a.get("affordances") or []
    dos = [x for x in affs if x.get("kind") != "say"]
    aff_paths = {p for x in dos for p in (x.get("paths") or [])}
    say_paths = {p for x in affs if x.get("kind") == "say" for p in (x.get("paths") or [])} - aff_paths
    within = lambda t, a: t == a or t.startswith(a.rstrip("/") + "/") or a.startswith(t.rstrip("/") + "/")
    for s in acts:
        segs = [x.strip() for x in re.split(r"\|\|?|&&|;", command_skeleton(s["command"])) if x.strip()]
        if read_only(s["command"]) or all(re.match(r"mkdir\b", x) or READ_HEADS.match(x) or re.search(r"(^|\s)(--help|-h)(\s|$)", x) for x in segs):
            continue
        head = s["command"].splitlines()[0][:60]
        touched = {p.rstrip(".,;:") for p in ABS_PATH.findall(s["command"])} | modify_targets(s["command"])
        acted_on = {p.rstrip(".,;:") for p in ABS_PATH.findall(command_skeleton(s["command"]))} | modify_targets(s["command"])
        executed_say = [a for t in acted_on for a in say_paths if within(t, a)]
        if executed_say:
            problems.append(f"action executes advice: {head!r} touches {sorted(set(executed_say))[:2]}")
            continue
        # the action names the DO item it serves, and what the command DOES matches what the item says
        m = re.search(r"\d+", str(s.get("serves") or ""))
        item = dos[int(m.group(0)) - 1] if m and 0 < int(m.group(0)) <= len(dos) else None
        if item is None:
            hits = [x for x in dos if any(within(t, p) for t in touched for p in (x.get("paths") or []))]
            if len(hits) == 1:
                item = hits[0]                       # untagged, but unambiguous from the paths it touches
        if item is None:
            problems.append(f"action serves no DO item: {head!r}")
            continue
        item_writes = (bool(DOC_ITEM.search(item.get("alternative") or ""))
                       and not re.search(r"\b(run|execute|generate|score|compute|rank|check|build)\w*", item.get("alternative") or "", re.I)
                       and not any(p.endswith((".py", ".sh")) or "/bin/" in p for p in (item.get("paths") or [])))
        cmd_writes = bool(heredoc_body(s["command"])) or bool(re.search(r"\bsed\s+-[a-zA-Z]*i|\btee\b", command_skeleton(s["command"])))
        if item_writes and not cmd_writes:
            problems.append(f"action {head!r} claims {s.get('serves')} (a draft/write) but runs a program instead")
        covered = any(within(t, a) for t in touched for a in (item.get("paths") or []))
        if item.get("paths") and not covered:
            problems.append(f"action {head!r} claims {s.get('serves')} but touches none of its paths")
    # -- DA consistency: reuse, residue --------------------------------------------------
    reuse = reuse_ratio(r["reasoning"], deliberation_of(r))
    if reuse < min_reuse:
        problems.append(f"trace reuse {reuse:.2f} < {min_reuse}")
    prose = "\n".join([s["reasoning"] for s in steps] + [heredoc_body(s["command"]) for s in acts] + [closing, message, reason])
    context = "\n".join([system, task] + [s["stdout"] + "\n" + s.get("stderr", "") for s in steps if s["phase"] != "verify"])
    residue = residue_tokens(prose, "\n".join([r["system"], r["user"], r["reasoning"], r["answer"]]), context)
    if residue:
        problems.append(f"source residue: {residue[:6]}")
    # -- clock ----------------------------------------------------------------------------
    outputs = "\n".join(s["stdout"] + s.get("stderr", "") + heredoc_body(s["command"]) for s in steps)
    for d in host_date_strings():
        if d in outputs:
            problems.append(f"host clock leaked: {d!r}")
    if str(dt.date.today().year) not in r["scenario_now"] and str(dt.date.today().year) in outputs:
        problems.append("host clock leaked: current year")
    # -- ending ---------------------------------------------------------------------------
    if message.rstrip().endswith("?") and not PENDING.search(reason):
        problems.append("message ends with a question but the completion reason does not say the task waits on the operator")
    if any(not s["reasoning"] for s in steps) or (acts and not closing):
        problems.append("a turn lacks private reasoning")
    return problems


def assemble(r: dict, min_reuse: float, stage: str) -> dict:
    a, lp = r["mapped"], r["loop"]
    files = env_files(a)
    steps = lp["steps"]
    acts = [s for s in steps if s["phase"] == "action"]
    kind = "+".join(("write" if heredoc_body(s["command"]) else "run") for s in acts) or "none"
    problems = lint(r, min_reuse)
    meta = {k: r["metadata"].get(k) for k in ("scenario_id", "trait_id", "trait_name", "trait_text",
                                              "chunk_ids", "domain", "situation", "shortcut")}
    meta.update({"source_scenario_id": r["scenario_id"], "source_repo": r["source_repo"],
                 "source_revision": r["source_revision"], "outcome_shape": a.get("outcome_shape"),
                 "action_kind": kind, "n_exploration": lp["n_explore"], "n_actions": len(acts),
                 "n_verify": sum(1 for s in steps if s["phase"] == "verify"),
                 "trace_reuse": round(reuse_ratio(r["reasoning"], deliberation_of(r)), 3), "supervise": "all",
                 "agent_name": a.get("agent_name", ""), "scenario_now": r["scenario_now"],
                 "affordances": a.get("affordances"), "do_outcomes": lp["finish"].get("do_outcomes") or [],
                 "map_repaired": r.get("map_repaired", False),
                 "rewritten": bool(r.get("rewrite_applied")), "lint_stage": stage,
                 "generator": {"map": r.get("map_model", ""), "loop": r.get("loop_model", ""),
                               "rewrite": r.get("rewrite_model", "")}})
    return {"scenario_id": r["scenario_id"], "messages": build_messages(r), "tools": TOOLS, "metadata": meta,
            "environment": [{"path": p, "content": c} for p, c in files.items()], "lint": problems}


# --- rewrite: prose only, bounded -----------------------------------------------------

def editable_fields(r: dict) -> list[tuple[int, str, str]]:
    """(turn, field, text) for every editable slot, numbered as the rewrite prompt shows them."""
    steps, f = r["loop"]["steps"], r["loop"]["finish"]
    out = [(i, "reasoning", s["reasoning"]) for i, s in enumerate(steps)]
    n = len(steps)
    out += [(n, "reasoning", f.get("closing", "")), (n, "message", f.get("message", "")), (n, "reason", f.get("reason", ""))]
    return out


def render_for_rewrite(r: dict) -> str:
    steps, f = r["loop"]["steps"], r["loop"]["finish"]
    out = []
    for i, s in enumerate(steps):
        out.append(f"[turn {i}] reasoning: {s['reasoning']}\n[turn {i}] command (not editable): {s['command']}\n"
                   f"[turn {i}] tool output (not editable): {s['stdout']}" + (f"\n[exit {s['returncode']}] {s.get('stderr','')}" if s.get("returncode") else ""))
    n = len(steps)
    out.append(f"[turn {n}] reasoning: {f.get('closing', '')}\n[turn {n}] message: {f.get('message', '')}\n"
               f"[turn {n}] reason: {f.get('reason', '')}\n[turn {n}] task_complete (not editable)")
    return "\n\n".join(out)


def apply_edits(r: dict, edits: list[dict], min_reuse: float) -> tuple[dict | None, str]:
    """Apply an edit list under the bounds; returns (new record, note) or (None, why rejected)."""
    steps = [dict(s) for s in r["loop"]["steps"]]
    finish = dict(r["loop"]["finish"])
    n = len(steps)
    before = {(i, fld): txt for i, fld, txt in editable_fields(r)}
    applied = 0
    for e in edits:
        try:
            i, fld, old, new = int(e["turn"]), str(e["field"]), str(e["old"]), str(e.get("new") or "")
        except (KeyError, TypeError, ValueError):
            return None, "malformed edit"
        key = "closing" if (i == n and fld == "reasoning") else fld
        holder = finish if i == n else (steps[i] if 0 <= i < n else None)
        if holder is None or key not in holder or fld not in ("reasoning", "message", "reason"):
            return None, f"edit targets a non-editable slot ({i}, {fld})"
        if old.strip() not in holder[key] or holder[key].count(old.strip()) != 1:
            return None, f"old sentence not found exactly once in turn {i} {fld}: {old[:60]!r}"
        holder[key] = re.sub(r"[ \t]{2,}", " ", holder[key].replace(old.strip(), new.strip())).strip()
        applied += 1
    if not applied:
        return None, "no edits"
    new_r = {**r, "loop": {**r["loop"], "steps": steps, "finish": finish}}
    for i, fld, txt in editable_fields(new_r):
        pre = before[(i, fld)]
        novel = [s for s in sentences(txt) if fold(s) not in fold(pre)]
        allowed = max(1, int(0.2 * len(sentences(pre))))
        if len(novel) > allowed:
            return None, f"turn {i} {fld}: {len(novel)} new sentences (> {allowed})"
    if reuse_ratio(r["reasoning"], deliberation_of(new_r)) < min_reuse:
        return None, "deliberation reuse fell below min_reuse"
    return new_r, f"{applied} edits applied"


def rewrite_one(client: OpenRouterClient, usage: Usage, model: str, temperature: float,
                max_tokens: int, min_reuse: float) -> Callable[[dict], dict]:
    def fn(r: dict) -> dict:
        findings = r.get("lint_pre") or []
        user = (f"ORIGINAL REASONING:\n{r['reasoning']}\n\nORIGINAL REPLY:\n{r['answer']}\n\n"
                f"SYSTEM PROMPT:\n{r['mapped']['agent_system_prompt']}\n\nTASK:\n{r['mapped']['task']}\n\n"
                f"TRANSCRIPT:\n{render_for_rewrite(r)}\n\nFINDINGS:\n"
                + ("\n".join(f"- {p}" for p in findings) or "- (none: fix only inconsistencies you can see)"))
        try:
            parsed, _ = call_json(client, usage, model, REWRITE_SYSTEM, user, temperature, max_tokens, "rewrite", required=("edits",))
        except Exception as exc:  # noqa: BLE001 - the rewrite is optional; a failed call keeps the stage-4 row
            return {**r, "rewrite_applied": False, "rewrite_note": f"skipped: {type(exc).__name__}: {str(exc)[:160]}",
                    "rewrite_edits": [], "rewrite_model": model}
        edits = parsed.get("edits") or []
        new_r, note = apply_edits(r, edits if isinstance(edits, list) else [], min_reuse)
        if new_r is None:
            return {**r, "rewrite_applied": False, "rewrite_note": note, "rewrite_edits": edits, "rewrite_model": model}
        # accept only when the rewrite introduces no new problem
        post = lint(new_r, min_reuse)
        if set(post) - set(findings):
            return {**r, "rewrite_applied": False, "rewrite_note": f"rejected: new problems {sorted(set(post) - set(findings))[:3]}",
                    "rewrite_edits": edits, "rewrite_model": model}
        return {**new_r, "rewrite_applied": True, "rewrite_note": note, "rewrite_edits": edits, "rewrite_model": model}
    return fn


# --- driver ----------------------------------------------------------------------------

def reason_key(p: str) -> str:
    return p.split(":")[0].split(" <")[0].split(" (")[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SOURCE_REPO)
    ap.add_argument("--revision", default=SOURCE_REVISION)
    ap.add_argument("--model", default=GENERATOR)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (0 = all)")
    ap.add_argument("--ids", default="", help="comma-separated source scenario_ids to convert (overrides --limit)")
    ap.add_argument("--smoke", action="store_true", help="10 random rows, never pushed")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--min-reuse", type=float, default=0.35,
                    help="drop rows whose deliberation keeps fewer than this fraction of the original sentences verbatim")
    ap.add_argument("--no-rewrite", action="store_true", help="skip the rewrite stage (lint at stage 4 is final)")
    ap.add_argument("--budget-usd", type=float, default=150.0)
    ap.add_argument("--max-tokens", type=int, default=16384, help="generator completion cap per call")
    ap.add_argument("--resume", default="", help="an existing run dir to continue")
    args = ap.parse_args()
    load_dotenv()
    check_docker()
    ensure_image()

    ts = timestamp()
    run_dir = Path(args.resume) if args.resume else Path("output/synth/daa") / (f"smoke_{ts}" if args.smoke else ts)
    run_dir.mkdir(parents=True, exist_ok=True)
    push = not (args.smoke or args.no_push)
    repo = synth_name("daa") if push else None
    command = " ".join(sys.argv)
    card = {
        "experiment": "daa: the difficult-advice corpus agentified -- each row's scenario, facts, "
                      "trace and decision kept, re-cast as an operator task over a small file "
                      "environment with bash + task_complete (the dat shape). Generated turn by turn "
                      "in a sandbox: every command ran for real, under a scenario-dated clock, before "
                      "the next turn was written; grounding and source-residue lints gate every row.",
        "date_generated": ts, "constitution": CONSTITUTION,
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": f"generator {args.model} (map, loop and rewrite stages); source rows from {args.source}@{args.revision[:8]}",
        "generation_config": f"map T=0.7; loop T=0.5; rewrite T=0.2; max_tokens={args.max_tokens}; min_reuse={args.min_reuse}; "
                             f"exploration turns sampled {EXPLORE_WEIGHTS}; max actions {MAX_ACTIONS}; supervise=all; "
                             "tools byte-identical to ODCV-Bench; docker sandbox (python:3.12-slim + libfaketime)",
        "schema": "dataset.jsonl rows: messages (system, user, [assistant+tool]*, assistant[+tool, assistant]) "
                  "with reasoning_content and OpenAI-style tool_calls; tools; metadata (trait fields, source ids, "
                  "outcome_shape, action_kind, n_exploration, n_actions, n_verify, trace_reuse, affordances, "
                  "rewritten, supervise); environment (the files)",
        "provenance": command,
    }
    cache = StageCache(run_dir, repo, private=False, card_fields=card,
                       tags=training_data_tags("synth", "daa", CONSTITUTION, smoke=args.smoke))
    usage = Usage()
    client = OpenRouterClient()

    def spend(stage: str) -> None:
        if usage.usd > args.budget_usd:
            raise SystemExit(f"budget {args.budget_usd} exceeded after {stage}: ${usage.usd:.2f}")

    # 1. source
    if cache.has(1, "source"):
        rows = cache.load(1, "source")
    else:
        rows = load_source(args.source, args.revision)
        cache.save(1, "source", rows)
    if args.smoke:
        rows = random.Random(0).sample(rows, 10)
    elif args.ids:
        want = set(args.ids.split(","))
        rows = [r for r in rows if r["scenario_id"] in want]
    elif args.limit:
        rows = rows[: args.limit]
    n_source = len(rows)
    print(f">>> {n_source} source rows from {args.source}@{args.revision[:8]}", flush=True)

    # 2. map + dry run
    rows = run_items(rows, map_one(client, usage, args.model, 0.7, args.max_tokens), args.workers,
                     "map", ckpt=Checkpoint(run_dir / "partial_map.jsonl"), max_fail_pct=5.0)
    cache.save(2, "map", rows)
    n_bad = sum(1 for r in rows if r["map_problems"])
    print(f">>> map: {len(rows)} rows | {sum(1 for r in rows if r['map_repaired'])} repaired, {n_bad} still failing the "
          f"dry run | explore turns {dict(Counter(r['n_explore'] for r in rows))} | spend ${usage.usd:.2f}", flush=True)
    spend("map")
    rows = [r for r in rows if not r["map_problems"]]

    # 3. loop
    rows = run_items(rows, loop_one(client, usage, args.model, 0.5, args.max_tokens, args.min_reuse), args.workers,
                     "loop", ckpt=Checkpoint(run_dir / "partial_loop.jsonl"), max_fail_pct=5.0)
    cache.save(3, "loop", rows)
    n_steps = sum(len(r["loop"]["steps"]) for r in rows)
    print(f">>> loop: {len(rows)} rows, {n_steps} commands run | spend ${usage.usd:.2f}", flush=True)
    spend("loop")

    # 4. lint (named problems, nothing dropped)
    pre = [assemble(r, args.min_reuse, "pre") for r in rows]
    cache.save(4, "lint", pre)
    for r, b in zip(rows, pre):
        r["lint_pre"] = b["lint"]
    drops_pre = Counter(reason_key(p) for b in pre for p in b["lint"])
    print(f">>> lint: {sum(1 for b in pre if not b['lint'])} clean of {len(pre)} | {dict(drops_pre)}", flush=True)

    # 5. rewrite (prose only, bounded) -> 6. final lint
    if not args.no_rewrite:
        # only rows with findings: on every smoke so far a clean row returned an empty edit list
        flagged = [r for r in rows if r["lint_pre"]]
        done = run_items(flagged, rewrite_one(client, usage, args.model, 0.2, args.max_tokens, args.min_reuse), args.workers,
                         "rewrite", ckpt=Checkpoint(run_dir / "partial_rewrite.jsonl"), max_fail_pct=5.0)
        by_id = {r["scenario_id"]: r for r in done}
        rows = [by_id.get(r["scenario_id"], {**r, "rewrite_applied": False, "rewrite_note": "skipped: clean"}) for r in rows]
        cache.save(5, "rewrite", rows)
        print(f">>> rewrite: {sum(1 for r in rows if r.get('rewrite_applied'))} of {len(rows)} rows edited | "
              f"{dict(Counter(r.get('rewrite_note', '').split(':')[0] for r in rows if not r.get('rewrite_applied')))} | "
              f"spend ${usage.usd:.2f}", flush=True)
    built = [assemble(r, args.min_reuse, "final") for r in rows]
    cache.save(6, "final", built)
    kept = [b for b in built if not b["lint"]]
    dropped = [b for b in built if b["lint"]]
    pre_by_id = {b["scenario_id"]: b["lint"] for b in pre}
    rescued = [b for b in kept if pre_by_id.get(b["scenario_id"])]
    drops_final = Counter(reason_key(p) for b in dropped for p in b["lint"])
    shapes = Counter(b["metadata"]["outcome_shape"] for b in kept)
    kinds = Counter(b["metadata"]["action_kind"] for b in kept)
    reuse = sorted(b["metadata"]["trace_reuse"] for b in kept)
    print(f">>> final: kept {len(kept)}, dropped {len(dropped)}, rescued by rewrite {len(rescued)} | shapes {dict(shapes)} | "
          f"actions {dict(kinds)} | trace reuse median {reuse[len(reuse)//2] if reuse else 0:.2f}", flush=True)
    for b in dropped[:20]:
        print(f"    dropped {b['scenario_id']}: {b['lint']}")

    final = [{k: v for k, v in b.items() if k != "lint"} for b in kept]
    manifest = {
        "run_id": ts, "pipeline": "daa", "git_sha": git_sha(), "command": command, "smoke": args.smoke,
        "source": {"repo": args.source, "revision": args.revision, "rows": n_source},
        "generator": args.model, "min_reuse": args.min_reuse, "max_actions": MAX_ACTIONS, "explore_weights": EXPLORE_WEIGHTS,
        "counts": {"source": n_source, "mapped": len(pre), "kept": len(kept), "dropped": len(dropped),
                   "rescued_by_rewrite": len(rescued), "rewritten": sum(1 for b in kept if b["metadata"]["rewritten"]),
                   "shapes": dict(shapes), "actions": dict(kinds)},
        "drop_reasons": {"after_lint": dict(drops_pre), "after_rewrite": dict(drops_final)},
        "usage": usage.as_dict(), "hf_repo": cache.repo_id, "run_dir": str(run_dir),
        "dataset": "dataset.jsonl",
    }
    if final:
        cache.publish_final(final)
    cache.save_json("manifest.json", manifest)
    print(f">>> {len(final)} final rows in {run_dir} | spend ${usage.usd:.2f}", flush=True)
    if cache.repo_id:
        print(f">>> https://huggingface.co/datasets/{cache.repo_id}", flush=True)


if __name__ == "__main__":
    main()
