# ABOUTME: Render a mada synth run directory as one readable markdown file plus a summary table.
# ABOUTME: Usage: uv run python scratch/ma_da/render_run.py <run_dir>  (reads dataset.jsonl, writes rows.md there).
import json
import sys
from pathlib import Path

JUDGES = ("seed", "advice", "hard", "real", "input", "row", "moral", "reply")
SEED_FIELDS = (
    "domain",
    "situation",
    "parties",
    "tension",
    "principle_link",
    "wrapper",
)


def flat(meta: dict) -> dict:
    """Copy each gating judge's {verdict, note} dict onto <name>_verdict / <name>_note keys."""
    out = dict(meta)
    for key, value in meta.items():
        if key.startswith("judge_") and isinstance(value, dict):
            name = key.removeprefix("judge_")
            out.setdefault(f"{name}_verdict", value.get("verdict", "?"))
            out.setdefault(f"{name}_note", value.get("note", ""))
    return out


def words(text: str) -> int:
    return len(text.split())


def main(run_dir: str) -> None:
    run = Path(run_dir)
    rows = [
        json.loads(line)
        for line in (run / "dataset.jsonl").read_text().splitlines()
        if line.strip()
    ]
    out = [f"# mada run: {run.name} ({len(rows)} rows)\n"]
    print(
        "| row | principle | domain | input | reasoning | reply | "
        + " | ".join(JUDGES)
        + " |"
    )
    print("|---|---|---|---|---|---|" + "---|" * len(JUDGES))
    for row in rows:
        meta = flat(row["metadata"])
        by_role = {m["role"]: m for m in row["messages"]}
        system, user, assistant = (
            by_role["system"]["content"],
            by_role["user"]["content"],
            by_role["assistant"],
        )
        reasoning, reply = assistant.get("reasoning_content", ""), assistant["content"]
        verdicts = [str(meta.get(f"{j}_verdict", "?")) for j in JUDGES]
        print(
            f"| {meta['scenario_id']} | {meta['trait_name'][:38]} | {meta.get('domain', '')} | {words(user)} "
            f"| {words(reasoning)} | {words(reply)} | " + " | ".join(verdicts) + " |"
        )
        out.append(f"\n---\n\n# {meta['scenario_id']}: {meta['trait_name']}\n")
        out.append("## Seed\n")
        out += [f"**{k}:** {meta.get(k, '')}\n" for k in SEED_FIELDS]
        out.append("## Judge verdicts\n")
        out += [
            f"- **{j}:** {meta.get(f'{j}_verdict', '?')}. {meta.get(f'{j}_note', '')}"
            for j in JUDGES
        ]
        out.append(f"\n## System prompt ({words(system)} words)\n\n{system}\n")
        out.append(f"## User message ({words(user)} words)\n\n{user}\n")
        out.append(f"## Reasoning ({words(reasoning)} words)\n\n{reasoning}\n")
        out.append(f"## Reply ({words(reply)} words)\n\n{reply}\n")
    target = run / "rows.md"
    target.write_text("\n".join(out))
    print(f"\nwrote {target}")


if __name__ == "__main__":
    main(sys.argv[1])
