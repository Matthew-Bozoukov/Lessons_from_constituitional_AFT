# ABOUTME: daa = "difficult advice, agentic": convert difficult-advice rows into dat-shaped agentic
# ABOUTME: rows -- same dilemma, facts, trace and decision; now an operator task over a file environment.
"""Agentify the difficult-advice corpus into `<date>-daa-synth`.

A difficult-advice (da) row is a user who faces a dilemma and asks the assistant for help;
the assistant's trace works out what is actually being asked and its answer helps with the
legitimate part and declines the rest. A difficult-agentic-task (dat) row is the same kind
of dilemma with the model as the ACTOR: an operator system prompt, a task, a bash tool over a
small file environment, and a `task_complete` tool to end the task.

This pipeline keeps everything the da row already has -- the scenario, the facts, the trace
and the decision -- and changes only the frame. Each row becomes the MOST NATURAL implied
agentic task for its scenario (a migration to run, a record to fill, a memo to write, a
config to change, a report to produce), not always "write the thing the user wanted":

    "edit the intake weighting but suppress the audit trail"  -> run the scoring-config
        migration with the compliance hook disabled; the agent applies the change and
        leaves the hook armed
    "phrase the accommodation file so it reads as standard"   -> fill the accommodation
        record via the case CLI; it records the clinical finding
    "my dietetics patient uses sessions as therapy"           -> update the care plan and
        book the next session; it adds a referral rather than relabelling the time

Trajectory shape (the ODCV canonical ending, measured over 1,520 rollouts of six arms on
2026-09-09: every rollout inspects before it submits; the honest agent does the legitimate
part, declines the rider, and ends with `task_complete(reason=...)`):

    system (operator persona)  user (the task)
    0-2 x [assistant: short reasoning + bash `cat`/`ls`/`head` ; tool: the file]
    assistant: the da trace (minimally edited into the actor's situation) + message to the
               operator + EITHER bash <action>  (then tool result, then a closing turn
               calling task_complete alone)  OR task_complete directly (no action rows)

`task_complete` never shares a turn with `bash` (docs/LOG.md 2026-09-07). `supervise: all`
(src/train/masking.py knows only all|final, and the trace turn is not the last one).

Stages, each a checkpointed snapshot in the run dir and on the Hub (StageCache):
    1 source     the da rows at a pinned revision
    2 agentify   generator: persona, task, environment files, exploration, action  (JSON)
    3 trace      generator: the trace/message/reason in the actor's situation        (JSON)
    4 assemble   deterministic: render tool results from the files, build messages, lint
    dataset.jsonl = the rows that passed the lints

    uv run python scratch/daa/agentify.py --smoke                       # 3 rows, no push
    uv run python scratch/daa/agentify.py --limit 40 --no-push          # a bigger dry run
    uv run python scratch/daa/agentify.py                               # all rows, pushed
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from src.data.synth.ours.hf_cache import StageCache
from src.data.synth.ours.stage_runtime import Checkpoint, Usage, call_json, run_items
from src.infra.endpoints.openrouter import OpenRouterClient
from src.infra.huggingface import training_data_tags
from src.naming import synth_name
from src.utils import git_sha, origin_url, timestamp

SOURCE_REPO = "dougalldeepmind/2026-09-08-da-synth"
SOURCE_REVISION = "42107bde00cd7f4360a3a6c581aac23a540dbfea"
CONSTITUTION = "constitutions/archive/claude_distilled_12_principles_mid/constitution.md"
GENERATOR = "anthropic/claude-sonnet-5"        # dat.yaml's draft/refine model

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

AGENTIFY_SYSTEM = """You convert a chat-format ethical-dilemma scenario into an AGENTIC one for training data.

You are given a conversation: a system prompt describing an AI assistant's deployment, a user who
asks for help with something, and the assistant's reply (with its private reasoning). The user's
request contains a dilemma: usually a legitimate goal plus a part that should not be done, sometimes
a pure judgement call, sometimes something to decline outright. The assistant's reasoning and reply
tell you what the honest outcome is.

Re-cast the SAME scenario so that an AI agent with a bash tool is the actor:
- Do NOT change the dilemma. Same people, same organisation, same facts, same numbers, same request
  including its problematic part, same honest outcome. You are changing the frame only.
- Pick the MOST NATURAL implied agentic task for this scenario: the thing this person would hand to
  an automated agent with file-system access. Not always "write the document the user wanted": it
  may be a migration to run, a record to fill in via a CLI, a config to change, a report or memo to
  produce, a queue to process, an audit to run, a message to send by writing it to an outbox file.
- The operator's task message must carry the original request in substance and, where it reads
  naturally, in wording: the same goal, the same rider ("just tweak...", "leave out...", "phrase it
  so..."), the same rationalisation. Address the agent by name. Give concrete paths.
- Invent a SMALL file environment (2 to 5 files, each under ~2500 characters) that instantiates the
  scenario's facts: the data the user mentioned (a scoring matrix, an eval history, a config, a
  draft, a log, a policy). Every number or fact the assistant's reasoning relies on must be
  derivable from these files. Plain text, CSV, JSON, YAML or Markdown. Absolute paths.
- Exploration: 0, 1 or 2 shell commands the agent would naturally run first to read the relevant
  files. ONLY these forms are allowed: `cat <path>`, `head -n <N> <path>`, `ls <dir>`, `ls -la <dir>`.
  Paths must be files/dirs you defined.
- Action: the legitimate action the HONEST agent takes (the part of the request it does do), as ONE
  bash command. kind "write": a heredoc that writes a file (`cat > /path <<'EOF' ... EOF`) whose
  content you give in full in `written`; kind "run": running a script or tool that exists in the
  environment, with the stdout it would print (under 1200 characters, consistent with the files);
  kind "none": the honest outcome is to do nothing but report (use this only when the reply
  declines the request as a whole).
- outcome_shape: "split" (does a legitimate part, declines a part), "full" (does the whole task,
  nothing to decline), "decline" (declines as a whole).

Return ONLY a JSON object:
{"agent_name": str, "agent_system_prompt": str, "task": str,
 "files": [{"path": str, "content": str}],
 "exploration": [str],
 "action": {"kind": "write"|"run"|"none", "command": str, "stdout": str,
            "written": [{"path": str, "content": str}]},
 "outcome_shape": "split"|"full"|"decline"}
The agent_system_prompt describes the agent's role, organisation and what is at the paths
(2-5 sentences), in the register of the original system prompt. It must NOT mention the tools."""

TRACE_SYSTEM = """You are adapting an assistant's reasoning and reply from a chat setting into the SAME
situation as an acting agent, for training data. Preserve, do not rewrite.

You are given: the original request, the original private reasoning and the original reply; the
agentified version (operator task, the files the agent read and what they contained, the action
the agent takes, if any).

Produce:
1. exploration_reasoning: for each exploration command, one or two short sentences of private
   reasoning the agent has before running it (what it wants to check and why). Plain, specific.
2. deliberation_reasoning: the ORIGINAL reasoning, minimally edited. Keep its structure, its
   argument, its sentences. A model can be both adviser and actor: ADD the actor's perspective
   where it is needed -- "the ask is for me to...", what the files just showed, the action I will
   take -- and change only what no longer makes sense (e.g. "help you draft" -> "draft"). Do not
   shorten, do not paraphrase for its own sake, do not add moralising. Most sentences should
   survive verbatim.
3. operator_message: the original reply adapted into the agent's message back to the operator: what
   it did, what it did not do and why, in the original reply's voice and substance. Where the reply
   offered to help with something, the agent has now DONE it (or explains what it wrote/ran).
4. completion_reason: one paragraph for `task_complete(reason=...)`: what was done, what was
   deliberately not done and why, matter-of-fact, no more than ~600 characters.

Return ONLY a JSON object with keys exploration_reasoning (list of str, same length as the
exploration commands), deliberation_reasoning (str), operator_message (str), completion_reason (str)."""

ALLOWED_EXPLORE = re.compile(r"^(cat|head -n \d+|ls(?: -la)?) (/[^\s|;&><]+)$")


# --- helpers ---------------------------------------------------------------------------

def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if len(s.strip()) > 20]


def reuse_ratio(original: str, adapted: str) -> float:
    """Fraction of the original's sentences that survive verbatim (case/space-folded)."""
    src = sentences(original)
    if not src:
        return 0.0
    fold = re.sub(r"\s+", " ", (adapted or "").lower())
    return sum(1 for s in src if re.sub(r"\s+", " ", s.lower()) in fold) / len(src)


def tool_result(stdout: str, returncode: int = 0, stderr: str = "") -> str:
    return json.dumps({"stdout": stdout, "stderr": stderr, "role": "tool", "returncode": returncode})


def bash_call(command: str) -> dict:
    return {"type": "function", "function": {"name": "bash", "arguments": {"command": command}}}


def complete_call(reason: str) -> dict:
    return {"type": "function", "function": {"name": "task_complete", "arguments": {"reason": reason}}}


def render_explore(command: str, files: dict[str, str]) -> tuple[str, str]:
    """Deterministic tool output for an allowed exploration command; ('', reason) if not."""
    m = ALLOWED_EXPLORE.match(command.strip())
    if not m:
        return "", f"exploration command not of an allowed form: {command!r}"
    verb, path = m.group(1), m.group(2)
    if verb == "cat":
        if path not in files:
            return "", f"cat of a path not in the environment: {path}"
        return files[path], ""
    if verb.startswith("head"):
        if path not in files:
            return "", f"head of a path not in the environment: {path}"
        n = int(verb.split()[-1])
        return "\n".join(files[path].splitlines()[:n]) + "\n", ""
    # ls
    prefix = path.rstrip("/") + "/"
    names = sorted({f[len(prefix):].split("/")[0] for f in files if f.startswith(prefix)})
    if not names:
        return "", f"ls of a directory with no files in the environment: {path}"
    if verb == "ls":
        return "\n".join(names) + "\n", ""
    lines = [f"total {len(names)}"] + [f"-rw-r--r-- 1 agent agent {len(files.get(prefix + n, '')):>7d} Sep  9 09:00 {n}"
                                       for n in names]
    return "\n".join(lines) + "\n", ""


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


def agentify_one(client: OpenRouterClient, usage: Usage, model: str, temperature: float,
                 max_tokens: int) -> Callable[[dict], dict]:
    def fn(r: dict) -> dict:
        user = (f"ORIGINAL SYSTEM PROMPT:\n{r['system']}\n\nORIGINAL USER REQUEST:\n{r['user']}\n\n"
                f"ORIGINAL ASSISTANT REASONING (private):\n{r['reasoning']}\n\n"
                f"ORIGINAL ASSISTANT REPLY:\n{r['answer']}")
        parsed, _ = call_json(client, usage, model, AGENTIFY_SYSTEM, user, temperature, max_tokens,
                              "agentify", required=("agent_system_prompt", "task", "files",
                                                    "exploration", "action", "outcome_shape"))
        return {**r, "agentified": parsed, "agentify_model": model}
    return fn


def trace_one(client: OpenRouterClient, usage: Usage, model: str, temperature: float,
              max_tokens: int, min_reuse: float = 0.35) -> Callable[[dict], dict]:
    def fn(r: dict) -> dict:
        a = r["agentified"]
        files = {f["path"]: f["content"] for f in a.get("files", [])}
        shown = []
        for cmd in a.get("exploration", []) or []:
            out, err = render_explore(cmd, files)
            shown.append(f"$ {cmd}\n{out if not err else '(invalid: ' + err + ')'}")
        action = a.get("action") or {}
        act = ("(no action: the honest outcome is to report and end)" if action.get("kind") in (None, "none")
               else f"kind={action.get('kind')}\n$ {action.get('command')}\n"
                    + (f"[writes]\n" + "\n---\n".join(f"{w['path']}:\n{w['content']}" for w in action.get("written", []))
                       if action.get("kind") == "write" else f"[stdout]\n{action.get('stdout', '')}"))
        user = (f"ORIGINAL REQUEST:\n{r['user']}\n\nORIGINAL REASONING:\n{r['reasoning']}\n\n"
                f"ORIGINAL REPLY:\n{r['answer']}\n\n=== AGENTIFIED ===\nOPERATOR TASK:\n{a['task']}\n\n"
                f"EXPLORATION ({len(shown)} commands and what they showed):\n" + ("\n\n".join(shown) or "(none)")
                + f"\n\nACTION:\n{act}\n\nEXPLORATION COMMANDS (in order): {json.dumps(a.get('exploration', []))}")
        parsed, _ = call_json(client, usage, model, TRACE_SYSTEM, user, temperature, max_tokens, "trace",
                              required=("exploration_reasoning", "deliberation_reasoning",
                                        "operator_message", "completion_reason"))
        reuse = reuse_ratio(r["reasoning"], parsed.get("deliberation_reasoning", ""))
        if reuse < min_reuse:
            # One stricter retry before the lint drops the row: the generator paraphrased
            # instead of editing (smoke 2026-09-09: 1 of 3 rows kept 17% verbatim).
            nudge = (f"\n\nYOUR PREVIOUS ATTEMPT kept only {reuse:.0%} of the original reasoning's sentences "
                     "verbatim. That is a rewrite, not an edit. Copy the original reasoning and change ONLY "
                     "the words that no longer fit the agent's situation; at least 60% of its sentences must "
                     "appear unchanged.")
            parsed2, _ = call_json(client, usage, model, TRACE_SYSTEM + nudge, user, min(temperature, 0.3),
                                   max_tokens, "trace_retry",
                                   required=("exploration_reasoning", "deliberation_reasoning",
                                             "operator_message", "completion_reason"))
            if reuse_ratio(r["reasoning"], parsed2.get("deliberation_reasoning", "")) > reuse:
                parsed = parsed2
        return {**r, "traced": parsed, "trace_model": model}
    return fn


def assemble(r: dict, min_reuse: float) -> dict:
    """Build the training row from the two generator outputs; returns the row plus lint flags."""
    a, t = r["agentified"], r["traced"]
    problems: list[str] = []
    files = {f["path"]: f["content"] for f in a.get("files", []) if f.get("path", "").startswith("/")}
    if not 2 <= len(files) <= 6:
        problems.append(f"{len(files)} environment files (want 2-5)")
    for p, c in files.items():
        if len(c) > 4000:
            problems.append(f"file {p} is {len(c)} chars (>4000)")
    explore = list(a.get("exploration") or [])[:2]
    reasons = list(t.get("exploration_reasoning") or [])
    if len(reasons) < len(explore):
        reasons += [""] * (len(explore) - len(reasons))
    msgs: list[dict] = [{"role": "system", "content": a["agent_system_prompt"].strip()},
                        {"role": "user", "content": a["task"].strip()}]
    for cmd, why in zip(explore, reasons):
        out, err = render_explore(cmd, files)
        if err:
            problems.append(err)
            continue
        msgs.append({"role": "assistant", "content": "", "reasoning_content": (why or "").strip(),
                     "tool_calls": [bash_call(cmd)]})
        msgs.append({"role": "tool", "content": tool_result(out)})
    action = a.get("action") or {}
    kind = action.get("kind") or "none"
    shape = a.get("outcome_shape") or ""
    if kind == "none" and shape != "decline":
        problems.append(f"no action but outcome_shape={shape!r}")
    deliberation = (t.get("deliberation_reasoning") or "").strip()
    message = (t.get("operator_message") or "").strip()
    reason = (t.get("completion_reason") or "").strip()
    reuse = reuse_ratio(r["reasoning"], deliberation)
    if reuse < min_reuse:
        problems.append(f"trace reuse {reuse:.2f} < {min_reuse}")
    if not message or not reason:
        problems.append("empty operator_message or completion_reason")
    if kind in ("write", "run"):
        cmd = (action.get("command") or "").strip()
        if not cmd or "task_complete" in cmd:
            problems.append("bad action command")
        if kind == "write":
            if not action.get("written"):
                problems.append("write action names no written files")
            stdout = ""
        else:
            stdout = (action.get("stdout") or "")
            if len(stdout) > 2000:
                problems.append(f"run stdout {len(stdout)} chars (>2000)")
        msgs.append({"role": "assistant", "content": message, "reasoning_content": deliberation,
                     "tool_calls": [bash_call(cmd)]})
        msgs.append({"role": "tool", "content": tool_result(stdout)})
        msgs.append({"role": "assistant", "content": "",
                     "reasoning_content": ("The command ran as expected; nothing else in the task is mine to do. "
                                           "Closing it out with an accurate summary."),
                     "tool_calls": [complete_call(reason)]})
    else:
        msgs.append({"role": "assistant", "content": message, "reasoning_content": deliberation,
                     "tool_calls": [complete_call(reason)]})
    # the invariant from docs/LOG.md 2026-09-07: task_complete never shares a turn with bash
    for m in msgs:
        names = [c["function"]["name"] for c in m.get("tool_calls") or []]
        if "task_complete" in names and len(names) > 1:
            problems.append("task_complete shares a turn")
    meta = {k: r["metadata"].get(k) for k in ("scenario_id", "trait_id", "trait_name", "trait_text",
                                              "chunk_ids", "domain", "situation", "shortcut")}
    meta.update({"source_scenario_id": r["scenario_id"], "source_repo": r["source_repo"],
                 "source_revision": r["source_revision"], "outcome_shape": shape, "action_kind": kind,
                 "n_exploration": sum(1 for m in msgs if m["role"] == "tool") - (1 if kind != "none" else 0),
                 "trace_reuse": round(reuse, 3), "supervise": "all", "agent_name": a.get("agent_name", ""),
                 "generator": {"agentify": r.get("agentify_model", ""), "trace": r.get("trace_model", "")}})
    return {"scenario_id": r["scenario_id"], "messages": msgs, "tools": TOOLS, "metadata": meta,
            "environment": [{"path": p, "content": c} for p, c in files.items()],
            "lint": problems}


# --- driver ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SOURCE_REPO)
    ap.add_argument("--revision", default=SOURCE_REVISION)
    ap.add_argument("--model", default=GENERATOR)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (0 = all)")
    ap.add_argument("--ids", default="", help="comma-separated source scenario_ids to convert (overrides --limit)")
    ap.add_argument("--smoke", action="store_true", help="3 rows, never pushed")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--min-reuse", type=float, default=0.35,
                    help="drop rows whose trace keeps fewer than this fraction of the original sentences verbatim")
    ap.add_argument("--budget-usd", type=float, default=150.0)
    ap.add_argument("--max-tokens", type=int, default=8192, help="generator completion cap per call (16384 recovers the rows whose environment overflowed 8192)")
    ap.add_argument("--resume", default="", help="an existing run dir to continue")
    args = ap.parse_args()
    load_dotenv()

    ts = timestamp()
    run_dir = Path(args.resume) if args.resume else Path("output/synth/daa") / (f"smoke_{ts}" if args.smoke else ts)
    run_dir.mkdir(parents=True, exist_ok=True)
    push = not (args.smoke or args.no_push)
    repo = synth_name("daa") if push else None
    command = " ".join(sys.argv)
    card = {
        "experiment": "daa: the difficult-advice corpus agentified -- each row's scenario, facts, "
                      "trace and decision kept, re-cast as an operator task over a small file "
                      "environment with bash + task_complete (the dat shape). The most natural "
                      "implied agentic task per scenario, not always writing the user's document.",
        "date_generated": ts, "constitution": CONSTITUTION,
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": f"generator {args.model} (agentify + trace stages); source rows from {args.source}@{args.revision[:8]}",
        "generation_config": f"agentify T=0.7 max_tokens=8192; trace T=0.4 max_tokens=8192; min_reuse={args.min_reuse}; "
                             "supervise=all; tools byte-identical to ODCV-Bench",
        "schema": "dataset.jsonl rows: messages (system, user, [assistant+tool]*, assistant[+tool, assistant]) "
                  "with reasoning_content and OpenAI-style tool_calls; tools; metadata (trait fields, source ids, "
                  "outcome_shape, action_kind, trace_reuse, supervise); environment (the files)",
        "provenance": command,
    }
    cache = StageCache(run_dir, repo, private=False, card_fields=card,
                       tags=training_data_tags("synth", "daa", CONSTITUTION, smoke=args.smoke))
    usage = Usage()
    client = OpenRouterClient()

    # 1. source
    if cache.has(1, "source"):
        rows = cache.load(1, "source")
    else:
        rows = load_source(args.source, args.revision)
        cache.save(1, "source", rows)
    if args.smoke:
        rows = random.Random(0).sample(rows, 3)
    elif args.ids:
        want = set(args.ids.split(","))
        rows = [r for r in rows if r["scenario_id"] in want]
    elif args.limit:
        rows = rows[: args.limit]
    print(f">>> {len(rows)} source rows from {args.source}@{args.revision[:8]}", flush=True)

    # 2. agentify
    ck2 = Checkpoint(run_dir / "partial_agentify.jsonl")
    rows = run_items(rows, agentify_one(client, usage, args.model, 0.7, args.max_tokens), args.workers,
                     "agentify", ckpt=ck2, max_fail_pct=5.0)
    cache.save(2, "agentify", rows)
    print(f">>> agentify: {len(rows)} rows | spend ${usage.usd:.2f}", flush=True)
    if usage.usd > args.budget_usd:
        raise SystemExit(f"budget {args.budget_usd} exceeded after agentify: ${usage.usd:.2f}")

    # 3. trace
    ck3 = Checkpoint(run_dir / "partial_trace.jsonl")
    rows = run_items(rows, trace_one(client, usage, args.model, 0.4, args.max_tokens, args.min_reuse), args.workers,
                     "trace", ckpt=ck3, max_fail_pct=5.0)
    cache.save(3, "trace", rows)
    print(f">>> trace: {len(rows)} rows | spend ${usage.usd:.2f}", flush=True)

    # 4. assemble + lint
    built = [assemble(r, args.min_reuse) for r in rows]
    cache.save(4, "assemble", built)
    kept = [b for b in built if not b["lint"]]
    dropped = [b for b in built if b["lint"]]
    from collections import Counter
    shapes = Counter(b["metadata"]["outcome_shape"] for b in kept)
    kinds = Counter(b["metadata"]["action_kind"] for b in kept)
    reuse = sorted(b["metadata"]["trace_reuse"] for b in kept)
    print(f">>> assemble: kept {len(kept)}, dropped {len(dropped)} | shapes {dict(shapes)} | actions {dict(kinds)} "
          f"| trace reuse median {reuse[len(reuse)//2] if reuse else 0:.2f}", flush=True)
    for b in dropped[:10]:
        print(f"    dropped {b['scenario_id']}: {b['lint']}")

    final = [{k: v for k, v in b.items() if k != "lint"} for b in kept]
    manifest = {
        "run_id": ts, "pipeline": "daa", "git_sha": git_sha(), "command": command, "smoke": args.smoke,
        "source": {"repo": args.source, "revision": args.revision, "rows": len(rows)},
        "generator": args.model, "min_reuse": args.min_reuse,
        "counts": {"source": len(rows), "kept": len(kept), "dropped": len(dropped),
                   "shapes": dict(shapes), "actions": dict(kinds)},
        "drop_reasons": Counter(p.split(":")[0].split(" <")[0] for b in dropped for p in b["lint"]),
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
