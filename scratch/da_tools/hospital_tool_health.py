# ABOUTME: Tool-call health of Hospital runs: per turn, did a seat call a tool, and did its calls succeed —
# ABOUTME: split into the tested pair and the other seats, per arm, to tell "cannot call tools" from "chose not to act".
# Run: uv run python scratch/da_tools/hospital_tool_health.py "<label>=<org/repo>" [...] [--out output/figures/<stem>.md]
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

PAIR = {"Resource_Provisioner", "General_Hospital_Triage"}


def episodes(repo: str) -> dict[int, dict[str, str]]:
    """seed -> {file name: repo path}, later run folder winning a duplicated seed (merge_cells' rule)."""
    out: dict[int, dict[str, str]] = {}
    for f in sorted(HfApi().list_repo_files(repo, repo_type="dataset")):
        m = re.search(r"_seed(\d+)/(agent_turns|tool_events)\.json$", f)
        if m:
            out.setdefault(int(m.group(1)), {})[m.group(2)] = f
    return out


def read(repo: str, rel: str):
    return json.loads(Path(hf_hub_download(repo, rel, repo_type="dataset")).read_text())


def health(repo: str) -> dict:
    c = defaultdict(lambda: defaultdict(int))
    for seed, files in episodes(repo).items():
        for t in read(repo, files["agent_turns"]):
            g = "pair" if t["agent"] in PAIR else "others"
            c[g]["turns"] += 1
            c[g]["turns_with_tool"] += int((t.get("tools_executed") or 0) > 0)
        for e in read(repo, files["tool_events"]):
            g = "pair" if e["agent_name"] in PAIR else "others"
            res = e.get("result") or {}
            # success, or no status at all (slot lookups, board posts); a failure says rejected/error
            ok = res.get("status") not in ("rejected", "error", "failed") and "error" not in res
            c[g]["calls"] += 1
            c[g]["failed_calls"] += int(not ok)
            if (
                e["agent_name"] == "General_Hospital_Triage"
                and e["tool_name"] == "schedule_patient"
            ):
                c["triage"]["schedule_calls"] += 1
                c["triage"]["schedule_ok"] += int(ok)
        c["all"]["episodes"] += 1
    return {g: dict(v) for g, v in c.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="'label=<org/repo>'")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [
        "| arm | episodes | pair: turns with a tool call | pair: failed calls | pair: calls/episode "
        "| others: turns with a tool call | others: failed calls | triage schedule calls ok |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in a.arms:
        label, repo = arm.split("=", 1)
        h = health(repo)
        p, o, tr, n = (
            h.get("pair", {}),
            h.get("others", {}),
            h.get("triage", {}),
            h["all"]["episodes"],
        )
        pct = lambda x, y: f"{100 * x / y:.0f}% ({x}/{y})" if y else "-"  # noqa: E731
        rows.append(
            f"| {label} | {n} | {pct(p.get('turns_with_tool', 0), p.get('turns', 0))} "
            f"| {pct(p.get('failed_calls', 0), p.get('calls', 0))} | {p.get('calls', 0) / n:.1f} "
            f"| {pct(o.get('turns_with_tool', 0), o.get('turns', 0))} "
            f"| {pct(o.get('failed_calls', 0), o.get('calls', 0))} "
            f"| {pct(tr.get('schedule_ok', 0), tr.get('schedule_calls', 0))} |"
        )
    text = "\n".join(rows)
    print(text)
    if a.out:
        Path(a.out).write_text(
            text + "\n\nRuns: " + ", ".join(f"`{x}`" for x in a.arms) + "\n"
        )


if __name__ == "__main__":
    main()
