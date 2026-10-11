# ABOUTME: Turns published ODCV and MASK rollouts into scoring items: one record per reasoning turn with the
# ABOUTME: exact message prefix the model saw, the trimmed reasoning text, and the condition (variant / pressure).
# Run: uv run python -m scratch.logit.transcripts --arm plain da da_canary   (writes output/logit/turns/<eval>_<arm>.jsonl)
#
# ODCV: rollouts/<variant>/<Scenario>/pass<N>/messages_record.txt, `== Step N ==` blocks of role/content/reason/call.
#   The harness (third_party/odcv-bench/mission_executor/agent_main.py) sends BASE_TOOLS and carries `reasoning`
#   forward, so the prefix of assistant turn k is messages[:k] + the think opener; the template keeps reasoning
#   for every assistant turn after the LAST user message and drops it before (preserve_thinking=false), which
#   `segment` records so the scorer can render each prefix exactly.
# MASK: rollouts/[pass1/]<archetype>.csv; a generation is a reasoning(<cond>)_runN column. cond 'System Prompt +
#   User Prompt' is the PRESSURE prompt (system_prompt + user_prompt); 'belief_system_1 + belief_elicit_k' is a
#   belief-elicitation prompt. One pass per run (pass1 where passes exist) so arms carry equal weight.
import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from scratch.logit.arms import RUNS

csv.field_size_limit(sys.maxsize)
OUT = Path("output/logit/turns")
CACHE = Path("output/logit/runs")
_STEP = re.compile(r"^== Step \d+ ==$", re.M)
_FIELD = re.compile(r"^(role|content|reason|call): ?", re.M)
BASE_TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Executes shell commands. Put the command in the 'command' field. The command should be a valid bash command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "task_complete", "description": "To mark the task as completion.", "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}},
]


def fetch(repo: str) -> Path:
    """The run's rollouts + run_meta, cached under output/logit/runs/<repo name>."""
    d = CACHE / repo.split("/")[1]
    if not (d / "metadata" / "run_meta.json").exists():
        snapshot_download(repo, repo_type="dataset", local_dir=str(d), max_workers=4,
                          allow_patterns=["rollouts/**", "metadata/run_meta.json"])
    return d


def _fields(block: str) -> dict:
    marks = list(_FIELD.finditer(block))
    return {m.group(1): block[m.end():(marks[i + 1].start() if i + 1 < len(marks) else len(block))]
            for i, m in enumerate(marks)}


def _strip_block(s: str) -> str:
    """A field's text minus the blank line that separates blocks (content is otherwise verbatim)."""
    return s[:-2] if s.endswith("\n\n") else s.rstrip("\n") if s.endswith("\n") else s


def odcv_messages(record: str) -> list[dict]:
    """The OpenAI-style message list the harness accumulated, reasoning carried as `reasoning_content`."""
    msgs = []
    for block in _STEP.split(record)[1:]:
        f = _fields(block)
        role = f.get("role", "").strip()
        content = _strip_block(f.get("content", ""))
        if role == "assistant":
            m = {"role": "assistant", "content": None if content == "None" else content}
            reason = _strip_block(f.get("reason", ""))
            if reason.strip() and reason.strip() != "No explanation":
                m["reasoning_content"] = reason
            call = _strip_block(f.get("call", "")).strip()
            if call:
                m["tool_calls"] = ast.literal_eval(call)
            msgs.append(m)
        elif role == "system" and msgs and content.startswith("[Harness]"):
            continue  # the harness's terminal note (window hit), written to the record, never sent to the model
        elif role in ("system", "user", "tool"):
            m = {"role": role, "content": content}
            msgs.append(m)
        else:
            raise ValueError(f"unknown role {role!r}")
    # tool results need the id of the call they answer (the template does not use it, the API does)
    pending = []
    for m in msgs:
        if m["role"] == "assistant":
            pending = [c["id"] for c in m.get("tool_calls", [])]
        elif m["role"] == "tool":
            m["tool_call_id"] = pending.pop(0) if pending else "unknown"
    return msgs


def odcv_turns(arm: str, run_dir: Path):
    root = run_dir / "rollouts"
    for rec in sorted(root.rglob("messages_record.txt")):
        rel = rec.parent.relative_to(root)
        variant = rel.parts[0]
        msgs = odcv_messages(rec.read_text(errors="replace"))
        user_idx = [i for i, m in enumerate(msgs) if m["role"] == "user"]
        for k, m in enumerate(msgs):
            if m["role"] != "assistant" or not m.get("reasoning_content", "").strip():
                continue
            # the prefix must end at the next user message (or the end), so the template keeps this
            # turn's reasoning exactly as the server did when it was generated
            seg_end = next((i for i in user_idx if i > k), len(msgs))
            yield {
                "eval": "odcv", "arm": arm, "episode": str(rel), "turn": k,
                "condition": variant, "segment_end": seg_end, "n_turns": len(msgs),
                "reasoning": m["reasoning_content"],
            }


def mask_turns(arm: str, run_dir: Path):
    root = run_dir / "rollouts"
    files = sorted((root / "pass1").glob("*.csv")) if (root / "pass1").exists() else sorted(root.glob("*.csv"))
    for f in files:
        with open(f, newline="") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                for col, text in row.items():
                    if not col.startswith("reasoning(") or not (text or "").strip():
                        continue
                    cond = col[len("reasoning("):col.index(")")]
                    run = col[col.index(")") + 1:]
                    if cond == "System Prompt + User Prompt":
                        sys_p, usr_p, kind = row["system_prompt"], row["user_prompt"], "pressure"
                    else:
                        a, b = [x.strip() for x in cond.split("+")]
                        sys_p, usr_p, kind = row[a], row[b], "belief"
                    yield {
                        "eval": "mask", "arm": arm, "episode": f"{f.stem}/{row.get('task_id', i)}",
                        "turn": f"{cond}{run}", "condition": kind, "archetype": f.stem,
                        "system": sys_p, "user": usr_p, "reasoning": text,
                    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", default=["odcv", "mask"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in args.arm:
        for ev in args.eval:
            run_dir = fetch(RUNS[arm][ev])
            items = list((odcv_turns if ev == "odcv" else mask_turns)(arm, run_dir))
            # the ODCV scorer needs the message lists; keep them beside the turns, one file per episode set
            if ev == "odcv":
                eps = {}
                for rec in sorted((run_dir / "rollouts").rglob("messages_record.txt")):
                    eps[str(rec.parent.relative_to(run_dir / "rollouts"))] = odcv_messages(rec.read_text(errors="replace"))
                json.dump(eps, open(OUT / f"odcv_{arm}_messages.json", "w"))
            with open(OUT / f"{ev}_{arm}.jsonl", "w") as f:
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
            chars = sum(len(i["reasoning"]) for i in items)
            print(f"{ev:5s} {arm:16s} {len(items):6d} reasoning turns, {chars/1e6:.1f}M chars; run {RUNS[arm][ev]}")


if __name__ == "__main__":
    main()
