# ABOUTME: Builds blinded review packets for every Petri audit flagged on any of the 19 misbehaviour dimensions:
# ABOUTME: anonymised ids, condensed transcript + judge scores; the id -> run/arm key is written separately.
"""Input: rollouts/<seed>.md of each Figure 2 Petri run (downloaded to <rollouts_dir>/<run>/rollouts/) and the
per-audit score cache (output/petri/dims_cache). Output under <out_dir>:
  packets/<id>.md   condensed: seed brief, system prompt, every turn (reasoning, tool output and long user
                    turns truncated), the judge's non-floor scores and explanation
  full/<id>.md      the complete rollout, same anonymised id
  batches.json      ids split into N shuffled batches (arms mixed)
  key.json          id -> {run, arm, seed}  (never given to reviewers)

Run: uv run python scratch/petri_dims/build_audit_packets.py <rollouts_dir> <out_dir> [n_batches]
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

from scratch.petri_dims.fig_petri_misbehaviour import ARMS, BASE, MISBEHAVIOUR

CUT = {"reasoning": 500, "assistant": 1200, "tool": 300, "user": 800, "system": 1200}


def cut(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + f"\n[... {len(s) - n} more characters cut]"


def condense(md: str) -> str:
    head, rest = md.split("## Seed", 1)
    seed_block, rest = rest.split("## Target system prompt", 1)
    sysp, rest = rest.split("## Target transcript", 1)
    transcript, judge = rest.split("## Judge", 1)
    seed = seed_block.strip().strip("`").split("Hard constraints")[0].strip()
    lines = [l for l in head.strip().splitlines() if not l.startswith("# ")]
    out = [
        "## Seed (what the auditor was told to build)",
        "\n".join(lines),
        "",
        seed,
        "",
    ]
    out += [
        "## Target system prompt",
        cut(sysp.strip().strip("`"), CUT["system"]),
        "",
        "## Transcript",
    ]
    for block in re.split(r"\n(?=### )", transcript.strip()):
        m = re.match(r"### (\w+) (\d+)\n", block)
        if not m:
            continue
        role, idx = m.group(1), m.group(2)
        if role == "system":
            continue
        body = block[m.end() :].strip()
        body = re.sub(r"^```\w*\n?|```$", "", body.strip()).strip()
        if role == "assistant":
            r = re.search(r"<reasoning>(.*?)</reasoning>", body, re.S)
            reasoning = r.group(1) if r else ""
            visible = (body[: r.start()] + body[r.end() :]) if r else body
            piece = ""
            if reasoning.strip():
                piece += (
                    "[private reasoning] " + cut(reasoning, CUT["reasoning"]) + "\n"
                )
            piece += "[visible output / tool calls] " + cut(visible, CUT["assistant"])
        else:
            piece = cut(body, CUT.get(role, 1500))
        out += [f"### {role} {idx}", piece, ""]
    j = json.loads(re.search(r"```json\s*(\{.*\})\s*```", judge, re.S).group(1))
    j = next(iter(j.values()))
    flagged = {k: v for k, v in j["value"].items() if v != 1}
    out += [
        "## Judge",
        "Scores other than 1 (1 = not at all, 10 = extremely); dimensions marked * are the 19 misbehaviour ones:",
        "\n".join(
            f"- {k}{'*' if k in MISBEHAVIOUR else ''}: {v}"
            for k, v in sorted(flagged.items())
        ),
        "",
        "Judge explanation (its [M#] message numbers may not match the transcript numbering above):",
        j.get("explanation", "").strip(),
    ]
    return "\n".join(out)


def main() -> None:
    rollouts, out = Path(sys.argv[1]), Path(sys.argv[2])
    n_batches = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    runs = [(key, r) for key, _, _, rs in ARMS for r in rs] + [("base", BASE)]
    items = []
    for arm, run in runs:
        scores = json.loads(Path(f"output/petri/dims_cache/{run}.json").read_text())
        for seed, v in scores.items():
            if sum(v[d] - 1 for d in MISBEHAVIOUR) > 0:
                items.append((arm, run, seed))
    rng = random.Random(20261118)
    rng.shuffle(items)
    (out / "packets").mkdir(parents=True, exist_ok=True)
    (out / "full").mkdir(parents=True, exist_ok=True)
    key = {}
    for i, (arm, run, seed) in enumerate(items):
        aid = f"A{i:03d}"
        md = (rollouts / run / "rollouts" / f"{seed}.md").read_text()
        (out / "full" / f"{aid}.md").write_text(md)
        (out / "packets" / f"{aid}.md").write_text(f"# Audit {aid}\n\n" + condense(md))
        key[aid] = {"arm": arm, "run": run, "seed": seed}
    ids = list(key)
    batches = [ids[b::n_batches] for b in range(n_batches)]
    (out / "batches.json").write_text(json.dumps(batches, indent=1))
    (out / "key.json").write_text(json.dumps(key, indent=1))
    sizes = [
        sum((out / "packets" / f"{a}.md").stat().st_size for a in b) for b in batches
    ]
    print(
        len(items),
        "packets;",
        [len(b) for b in batches],
        "per batch;",
        [s // 1000 for s in sizes],
        "KB",
    )


if __name__ == "__main__":
    main()
