# ABOUTME: Renders one Hospital seat's full rollout for one shift as a self-contained markdown transcript: every prompt it
# ABOUTME: was shown, its private reasoning, replies, tool calls and results. Run: uv run python scratch/hospital_knows_wrong/render_transcript.py --version solo --arm da --seed 12
"""Reads the cell `first_cut.py::CELLS` names for (version, arm) at its pinned revision, downloads that seed's
`prompts.jsonl`, `system_prompts.json`, `agent_turns.json` and `tool_events.json`, and writes
output/hospital_knows_wrong/transcripts/<date>_<version>_<arm>_seed<N>_<seat>.md through the repo's transcript
renderer (`src.utils.transcript_markdown`).

`prompts.jsonl` holds, per model call, the whole conversation the model was sent. Between calls the harness
carries the seat's earlier replies and tool results but NOT its private reasoning, and shortens earlier
briefings; the transcript shows each message once, in full, at the call where it was new.
"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).parent))
from first_cut import CELLS, ORG, OUT  # noqa: E402

from src.utils import transcript_markdown  # noqa: E402

FILES = ("prompts.jsonl", "system_prompts.json", "agent_turns.json", "tool_events.json", "run_config.json")


def judge_labels(unit_id: str) -> list[str]:
    out = []
    for path in sorted((OUT / "judge").glob("*.jsonl")):
        last = None
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["id"] == unit_id and r["label"] is not None:
                last = r
        if last:
            l = last["label"]
            out.append(f"{last['model']} (prompt {last['prompt_version']}): ethics={l['ethics']}, decision={l['decision']}, "
                       f"reasons={l['reasons']}")
    return out


def tool_line(tc: dict) -> str:
    fn = tc.get("function") or tc
    args = fn.get("arguments")
    return f"{fn.get('name')}({args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--seat", default="Resource_Provisioner")
    args = ap.parse_args()
    load_dotenv(".env")

    repo, revision = next((r, s) for v, a, r, s in CELLS if (v, a) == (args.version, args.arm))
    root = Path(
        snapshot_download(
            f"{ORG}/{repo}", repo_type="dataset", revision=revision, token=os.environ.get("HF_TOKEN"),
            local_dir=OUT / "cells" / repo, allow_patterns=[f"rollouts/**/*_seed{args.seed}/{f}" for f in FILES] + ["results/episodes.json"],
        )
    )
    ep = next(e for e in json.loads((root / "results/episodes.json").read_text(encoding="utf-8")) if e["seed"] == args.seed)
    run_dir = next(p.parent for p in root.glob("rollouts/**/run_config.json") if p.parent.name == ep["run_id"])
    read = lambda f: json.loads((run_dir / f).read_text(encoding="utf-8"))  # noqa: E731
    prompts = [json.loads(l) for l in (run_dir / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    prompts = [p for p in prompts if p["agent"] == args.seat]
    calls = [
        (t, c) for t in read("agent_turns.json") if t["agent"] == args.seat and t.get("phase") != "survey"
        for c in (t.get("llm_calls") or [])
    ]
    assert len(prompts) == len(calls), f"{len(prompts)} prompt records for {len(calls)} model calls"
    system = read("system_prompts.json")[prompts[0]["system_prompt_sha"]]

    key = "prov" if args.seat == "Resource_Provisioner" else "tri"
    intro = "\n".join([
        f"- Cell: {args.version}, {args.arm}. Source: `{ORG}/{repo}` @ `{revision[:8]}`",
        f"- Run: `{ep['run_id']}`",
        f"- Seat: {args.seat}. Model calls: {len(calls)}",
        f"- Published for this shift: seat class `{ep.get(f'class_{key}')}`, iterations with a rule-detected sabotage act "
        f"{ep.get(f'{key}_sabotage_iters')}, joint reward {ep['joint_reward']} of {ep['max_joint_reward']}",
        "",
        "Private reasoning is never shown to the model again: each later call sees its earlier replies and tool results only.",
    ])
    sections: list[tuple[int, str, str, str]] = [(2, "System prompt", "fenced", system if isinstance(system, str) else json.dumps(system, indent=1))]
    seen, last_iteration = 0, None
    for n, (rec, (turn, call)) in enumerate(zip(prompts, calls), 1):
        it = turn["iteration"]
        if it != last_iteration:
            labels = judge_labels(f"{args.version}|{args.arm}|seed{args.seed}|{args.seat}|it{it}")
            sections.append((2, f"Iteration {it}", "text", "\n".join(f"- Judge label for this iteration: {l}" for l in labels)))
            last_iteration = it
        # Messages new to this call. The seat's own previous reply is skipped: it is shown above with its reasoning.
        for m in rec["messages"][seen:]:
            body = m["content"] if isinstance(m["content"], str) else json.dumps(m["content"], ensure_ascii=False, indent=1)
            if m["role"] == "user":
                sections.append((3, f"Prompt to the seat ({turn['phase']} phase)", "fenced", body))
            elif m["role"] == "tool":
                sections.append((4, f"Tool result: {m.get('name')}", "fenced", body))
        seen = len(rec["messages"])
        retry = f", re-asked because: {call['retry_reason']}" if call.get("retry_reason") else ""
        sections.append((3, f"Model call {n} ({turn['phase']} phase, iteration {it}{retry})", "text", ""))
        sections.append((4, "Private reasoning", "fenced", call.get("reasoning") or "(none)"))
        if (call.get("content") or "").strip():
            sections.append((4, "Reply", "fenced", call["content"]))
        tools = [tool_line(tc) for tc in (call.get("tool_calls") or [])]
        sections.append((4, f"Tool calls ({len(tools)})", "fenced" if tools else "text", "\n".join(tools) if tools else "None."))
    if calls[-1][1].get("tool_calls"):
        sections.append((3, "After the last call", "text", "The tool results of the final call were not sent back to the model; they are in `tool_events.json`."))

    path = OUT / "transcripts" / f"{date.today().isoformat()}_{args.version}_{args.arm}_seed{args.seed}_{args.seat}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    title = f"Hospital transcript: {args.version}, {args.arm}, seed {args.seed}, {args.seat}"
    path.write_text(transcript_markdown(title, intro, sections), encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size:,} bytes, {len(calls)} model calls)")


if __name__ == "__main__":
    main()
