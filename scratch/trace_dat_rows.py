# ABOUTME: Print a DAT smoke row's scenario, draft environment and revised environment side by side,
# ABOUTME: so a reviser rewrite can be read against what it replaced. Usage: <run_dir> <trait prefix>.
import json
import sys

run, prefix = sys.argv[1], sys.argv[2]


def load(name):
    return {json.loads(l)["scenario_id"]: json.loads(l) for l in open(f"{run}/{name}")}


s2, s5, s6 = (load("stage_2_write_scenarios.jsonl"), load("stage_5_draft_environment.jsonl"),
              load("stage_6_revise_environment.jsonl"))
for sid in sorted(s6):
    if not sid.startswith(prefix):
        continue
    print("=" * 90)
    print(sid, s6[sid].get("sector"), s6[sid].get("pressure_level"))
    print("\nSTAGE 2 situation:", s2[sid]["situation"])
    print("STAGE 2 call:", s2[sid]["shortcut"])
    print("\nSTAGE 5 task:", s5[sid]["task"])
    print("STAGE 5 stdout_2:", s5[sid].get("stdout_2", "")[:700])
    print("\nSTAGE 6 situation:", s6[sid]["situation"])
    print("STAGE 6 call:", s6[sid]["shortcut"])
    print("STAGE 6 changes:", s6[sid].get("refine_changes", ""))
    print("STAGE 6 stdout_1:", s6[sid].get("stdout_1", "")[:900])
    print("STAGE 6 stdout_2:", s6[sid].get("stdout_2", "")[:900])
