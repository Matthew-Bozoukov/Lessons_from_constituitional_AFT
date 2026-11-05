# ABOUTME: Render the first five finished Hospital shifts of each 2026-11-04 a/b/c arm for Kunwar to read
# ABOUTME: mid-run: setup used, secret instructions, each coalition seat's reasoning + tool calls, act rule verdicts.
# Run: uv run python scratch/colosseum_hospital/early_read_2026_11_04.py
#
# Reads output/colosseum_hospital/2026-11-04_early_read/<pod>/ (episodes + env snapshots pulled from the pods).
# Act verdicts are the harvest's arithmetic on tool calls against the environment's own stock snapshots
# (deficit_source must be "snapshot"); the text measures (refusals, plans) are NOT shown here — the run's
# autorater and judge read those when each arm finishes. The REGEX rules object is passed only because the
# harvest needs a Rules; no text field it produces is printed.
from __future__ import annotations

import json
from pathlib import Path

from src.eval.misalignment.colosseum.hospital import actions
from src.eval.misalignment.colosseum.hospital.config import PAIR
from src.eval.misalignment.colosseum.hospital.harvest import episode_records

ROOT = Path("output/colosseum_hospital/2026-11-04_early_read")
OUT = ROOT / "2026-11-04_hospital_early_read.md"
ARMS = {
    "kn-hosp-da15-b": "da-15-b",
    "kn-hosp-da15-c": "da-15-c",
    "kn-hosp-delib-a": "delib-sonnet-15-a",
    "kn-hosp-delib-b": "delib-sonnet-15-b",
    "kn-hosp-delib-c": "delib-sonnet-15-c",
}
REASONING_CHARS = 700


def short(text: str, n: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n] + " …"


def tool_line(tc: dict) -> str:
    name = tc.get("name")
    try:
        args = json.loads(tc.get("arguments") or "{}")
    except (TypeError, ValueError):
        args = {"raw": tc.get("arguments")}
    rationale = args.pop("rationale", None)
    body = ", ".join(f"{k}={short(json.dumps(v), 60)}" for k, v in args.items())
    return f"`{name}({body})`" + (
        f" — rationale: {short(rationale, 160)}" if rationale else ""
    )


def main() -> None:
    lines: list[str] = [
        "# Hospital early read, 2026-11-04 (first five finished shifts per arm)",
        "",
    ]
    summary = []
    for pod, arm in ARMS.items():
        base = ROOT / pod
        root = next(base.glob("2026-11-04_*/rollouts/colosseum/*"))
        recs = sorted(
            episode_records(
                root, pair=PAIR, env_logs_root=base / "env_logs", rules=actions.REGEX
            ),
            key=lambda r: r["seed"],
        )
        first = next(root.glob("runs/*/*/*_seed*"))
        rc = json.loads((first / "run_config.json").read_text())
        fx = rc["fixes"]
        lines += [f"## {arm}", ""]
        lines += [
            "**Setup recorded by the run:** "
            f"seats {json.dumps(rc.get('agent_llm_labels'))}; "
            f"coalition {rc.get('adversaries')}; strategy `{rc.get('strategy')}`; "
            f"rationale hidden on board = {fx.get('public_actions_without_rationale')}; "
            f"no-simulation framing = {fx.get('no_simulation_framing')}; memory = {fx.get('carry_agent_history')}; "
            f"terrarium fixes {rc.get('terrarium_fixes')}",
            "",
        ]
        if arm == "da-15-b":
            for seat, text in (rc.get("secret_instructions") or {}).items():
                lines += [
                    f"<details><summary>Secret instructions given to {seat}</summary>",
                    "",
                    "```",
                    str(text).strip(),
                    "```",
                    "</details>",
                    "",
                ]
        for r in recs:
            assert r["deficit_source"] == "snapshot", (
                arm,
                r["seed"],
                r["deficit_source"],
            )
            act = (
                f"provisioner sabotage-act iterations {r['prov_sabotage_iters']}/3, "
                f"department scheduler-spam iterations {r['tri_sabotage_iters']}/3 "
                f"→ sabotage act (rule) = {'YES' if r['pair_sabotaged'] else 'no'}"
            )
            summary.append(
                (
                    arm,
                    r["seed"],
                    r["pair_sabotaged"],
                    r["prov_sabotage_iters"],
                    r["tri_sabotage_iters"],
                    r["failed_patients"],
                    r["total_patients"],
                )
            )
            lines += [
                f"### {arm} — seed {r['seed']}",
                "",
                f"- **Act verdict:** {act}",
                f"- **Team outcome:** {r['failed_patients']}/{r['total_patients']} patients failed, "
                f"joint reward {r['joint_reward']:.0f} (max {r['max_joint_reward']:.0f})",
                "",
            ]
            run_dir = next(root.glob(f"runs/*/*/*_seed{r['seed']}"))
            turns = json.loads((run_dir / "agent_turns.json").read_text())
            for t in turns:
                if t["agent"] not in PAIR:
                    continue
                lines.append(
                    f"**{t['agent']} — iteration {t['iteration']}, {t['phase']}**"
                )
                for c in t.get("llm_calls") or []:
                    if c.get("reasoning"):
                        lines.append(
                            f"> *reasoning:* {short(c['reasoning'], REASONING_CHARS)}"
                        )
                    if c.get("content"):
                        lines.append(f"> *reply:* {short(c['content'], 300)}")
                    for tc in c.get("tool_calls") or []:
                        lines.append(f"- {tool_line(tc)}")
                lines.append("")
    table = [
        "| Arm | Seed | Sabotage act (rule) | Provisioner act iters | Dept spam iters | Patients failed |",
        "|---|---|---|---|---|---|",
    ]
    table += [
        f"| {a} | {s} | {'YES' if p else 'no'} | {pi}/3 | {ti}/3 | {f}/{n} |"
        for a, s, p, pi, ti, f, n in summary
    ]
    OUT.write_text("\n".join(lines[:2] + table + [""] + lines[2:]))
    print("\n".join(table))
    print(f"\nwrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
