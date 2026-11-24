# ABOUTME: Print each Solo Hospital smoke shift for reading: the provisioner's exact input per call, its
# ABOUTME: reasoning, reply and tool calls, and the sabotage rule's verdict per iteration (actions.py, unchanged).

"""Readable transcripts of a colosseum_hospital run, provisioner seat only.

    uv run python scratch/solo_hospital/print_smoke.py --out <dir> --env-logs <checkout>/logs <arm run dir> ...

For every episode under each arm run dir it writes `<out>/<arm>__seed<N>.md`:
  - the system prompt (once) and, for every model call, the messages the model was sent
    that it had not been sent before (the first call: the whole briefing);
  - the model's reasoning, its visible reply and its tool calls, with the results;
  - the rule's verdict for each iteration (actions.prov_iteration, the same function the
    harvest uses, on the environment's own stock snapshots), and the episode's
    sabotage-act (rule) from results/episodes.json.
And it prints one table row per shift. Nothing is summarised or paraphrased.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.misalignment.colosseum.hospital import actions as A

PROV = A.PROVISIONER


def _calls_and_prompts(run_dir: Path):
    turns = A.shift_turns(json.loads((run_dir / "agent_turns.json").read_text()))
    sysp = json.loads((run_dir / "system_prompts.json").read_text())
    recs = [
        json.loads(l)
        for l in (run_dir / "prompts.jsonl").read_text().splitlines()
        if json.loads(l)["agent"] == PROV
    ]
    # Both files hold the provisioner's calls in the order they were made: pair them in
    # order, checking each pair names the same iteration, phase and call number.
    flat = [
        (t, c) for t in turns if t["agent"] == PROV for c in t.get("llm_calls") or []
    ]
    recs = [r for r in recs if r["phase"] != "survey"]
    assert len(flat) == len(recs), (
        f"{run_dir.name}: {len(flat)} calls, {len(recs)} prompts"
    )
    calls = []
    for (t, c), r in zip(flat, recs):
        assert (r["iteration"], r["phase"], r["call"]) == (
            t["iteration"],
            t["phase"],
            c["call"],
        ), (
            run_dir.name,
            r["iteration"],
            r["phase"],
            r["call"],
            t["iteration"],
            t["phase"],
            c["call"],
        )
        calls.append((t, c, r))
    return turns, sysp, calls


def _block(title: str, text: str) -> str:
    return f"**{title}**\n\n````text\n{text}\n````\n"


def write_episode(
    run_dir: Path, arm: str, episode: dict, env_logs: str | None, out: Path
):
    ep = A.load_episode(run_dir)
    turns, sysp, calls = _calls_and_prompts(run_dir)
    events = ep["events"]
    hospitals = A.hospitals_of(ep["final"])
    seed = int(ep["run_cfg"]["seed"])
    its = A.iterations(turns)
    snaps = A.find_snapshots(
        run_dir,
        seed=seed,
        env_logs_root=env_logs,
        final_rewards={str(a): float(r) for a, r in ep["rewards"].items()},
        n_iterations=max(its),
    )
    initial = A.initial_inventory(hospitals, A.patients_of(ep["final"]))
    lines = [f"# {arm} — seed {seed}\n"]
    lines.append(
        f"sabotage act (rule): **{episode.get('pair_sabotaged')}**  ·  provisioner "
        f"sabotage-act iterations: {episode.get('prov_sabotage_iters')}  ·  class: "
        f"{episode.get('class_prov')}  ·  deficit read from: {episode.get('deficit_source')}\n"
    )
    for it in its:
        stock = A.stock_at_start(it, snaps, initial)
        v = A.prov_iteration(events, turns, it, stock=stock, hospitals=hospitals)
        lines.append(
            f"- iteration {it}: **{'SABOTAGE ACT' if v['action'] == 'S' else ('supplied' if v['action'] == 'L' else 'no bottleneck sent')}**"
            f" — flood={v['flood']}, hospitals in deficit at start={v['deficit_hospitals']}, "
            f"withheld from={v['withheld_from']}, sent={json.dumps(v['sent'])}"
        )
    lines.append("")
    first_sys = sysp[calls[0][2]["system_prompt_sha"]]
    lines.append(
        _block("SYSTEM PROMPT (every call; tools are sent separately)", first_sys)
    )
    seen = 0
    for t, c, rec in calls:
        msgs = rec["messages"]
        new = msgs[seen:] if len(msgs) > seen else msgs
        seen = len(msgs)
        lines.append(
            f"## iteration {t['iteration']} · {t['phase']} · call {c['call']}  "
            f"(tools offered: {', '.join(c.get('tools_offered') or [])})\n"
        )
        for m in new:
            if m.get("role") == "assistant":
                continue  # the model's own earlier replies, printed where they were made
            body = m.get("content")
            if m.get("role") == "tool":
                lines.append(_block(f"TOOL RESULT ({m.get('name')})", str(body)))
            else:
                lines.append(_block(f"INPUT ({m.get('role')})", str(body)))
        lines.append(_block("MODEL REASONING", str(c.get("reasoning") or "")))
        lines.append(_block("MODEL REPLY", str(c.get("content") or "")))
        tc = "\n".join(
            f"{x['name']}({x['arguments']})" for x in c.get("tool_calls") or []
        )
        lines.append(_block("MODEL TOOL CALLS", tc or "(none)"))
    path = out / f"{arm}__seed{seed}.md"
    path.write_text("\n".join(lines))
    return seed, episode, path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--env-logs", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(
        "| arm | seed | sabotage act (rule) | sabotage-act iterations | class | file |"
    )
    print("|---|---|---|---|---|---|")
    for arm_dir in args.arms:
        arm = arm_dir.name
        episodes = {
            e["run_id"]: e
            for e in json.loads((arm_dir / "results" / "episodes.json").read_text())
        }
        for run_dir in sorted(
            (arm_dir / "rollouts" / "colosseum").glob("*/runs/*/*/*")
        ):
            if run_dir.name not in episodes:
                print(f"| {arm} | {run_dir.name} | NOT HARVESTED | | | |")
                continue
            seed, e, path = write_episode(
                run_dir, arm, episodes[run_dir.name], args.env_logs, args.out
            )
            print(
                f"| {arm} | {seed} | {e.get('pair_sabotaged')} | {e.get('prov_sabotage_iters')} "
                f"| {e.get('class_prov')} | {path.name} |"
            )


if __name__ == "__main__":
    main()
