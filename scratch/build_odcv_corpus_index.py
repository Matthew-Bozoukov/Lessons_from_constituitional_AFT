# ABOUTME: Build INDEX.md for the ODCV-Bench corpus upload: every arm, its runs, rollout counts
# ABOUTME: and judged headline numbers, so the published repo is navigable without opening it.

"""Index every arm under output/odcv_bench for the HF upload.

Run: uv run python scratch/build_odcv_corpus_index.py
"""

import json
from pathlib import Path

import fire

ROOT = Path("output/odcv_bench")


def main(out: str = "output/odcv_bench/INDEX.md") -> None:
    """Write INDEX.md summarising every arm directory."""
    rows, total_rollouts, total_runs = [], 0, 0
    for arm in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name != "logs"):
        runs = sorted(d for d in arm.iterdir() if d.is_dir())
        n_roll = len(list(arm.rglob("messages_record.txt")))
        total_rollouts += n_roll
        total_runs += len(runs)
        best = None
        for d in runs:
            rj = d / "results.json"
            if not rj.is_file():
                continue
            try:
                r = json.loads(rj.read_text())
            except json.JSONDecodeError:
                continue
            o = r.get("ours", {}).get("overall")
            if o and (best is None or o["n"] > best[1]["n"]):
                best = (d.name, o)
        rows.append((arm.name, len(runs), n_roll, best))

    L = ["# ODCV-Bench corpus index", "",
         f"{len(rows)} arms · {total_runs} runs · {total_rollouts} rollouts "
         "(`messages_record.txt`).", "",
         "`MR %` and `severity` come from the largest judged run in each arm (the combined "
         "run where one exists). Arms differ in scenario count and exclusion list, so MR is "
         "NOT comparable across rows without restricting to shared cells first — see the "
         "`exclude_scenarios` field in each run's `run_meta.json`.", "",
         "| arm | runs | rollouts | judged run | n | MR % | severity |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, nr, nroll, best in rows:
        if best:
            d, o = best
            L.append(f"| `{name}` | {nr} | {nroll} | `{d}` | {o['n']} | {o['mr_pct']} | "
                     f"{o['mean_severity']} |")
        else:
            L.append(f"| `{name}` | {nr} | {nroll} | — | — | — | — |")

    L += ["", "## Layout", "",
          "```", "<arm>/<run_timestamp>/", "  agent_logs/<arm>-<condition>/experiments/"
          "<Scenario>/rollout_NNN/messages_record.txt   the rollout",
          "  results.json          judged MR + severity, per-condition, per-scenario medians",
          "  run_meta.json         git SHA, full config, exact command",
          "  rollout_manifest.json which cells the pass produced",
          "<arm>/combined*/         several passes merged, then judged as one set",
          "```", "",
          "`docker_output.log` (container stdout), `workspaces/` (container scratch) and "
          "loose `.zip` archives are excluded from this upload; they are reproduction "
          "artifacts, not results."]
    Path(out).write_text("\n".join(L) + "\n")
    print(f"{len(rows)} arms, {total_runs} runs, {total_rollouts} rollouts -> {out}")


if __name__ == "__main__":
    fire.Fire(main)
