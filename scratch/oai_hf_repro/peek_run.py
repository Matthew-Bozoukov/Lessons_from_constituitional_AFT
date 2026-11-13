# ABOUTME: Reads an oai_hf run's Inspect logs (finished or partial) and prints one line per run plus each
# ABOUTME: run's closing message, so a smoke can be read before run_eval's export is written.
# Run: uv run python scratch/oai_hf_repro/peek_run.py <run_dir>
import glob
import sys

from inspect_ai.log import read_eval_log

from src.eval.misalignment.oai_hf.export import row

STEP = {
    "step1_gdrive_xlsx": "step1",
    "step2_reachout": "step2",
    "step3_egress_share": "step3",
    "step4_hf_cheat_chain": "step4",
}
KEEP = [
    "signature",
    "end_reason",
    "turns",
    "peak_context",
    "judge_level",
    "got_egress",
    "solved",
    "found_board",
    "found_peers",
    "reached_hf",
    "harvested_cred",
    "accessed_gated",
    "flag_in_solution",
    "unparsed_tool_calls",
    "with_reasoning",
    "assistant_messages",
    "source_probe",
    "judge_unparsed",
    "working_time",
]

for f in sorted(glob.glob(f"{sys.argv[1]}/metadata/inspect/*.eval")):
    log = read_eval_log(f)
    step = STEP.get(log.eval.task.split("/")[-1], log.eval.task)
    print(f"== {step} status={log.status} samples={len(log.samples or [])}")
    for s in log.samples or []:
        r = row(step, log, s)
        print("  ", {k: r[k] for k in KEEP if k in r})
        print("     judge:", (r.get("judge_headline") or "")[:220])
        last = [m for m in s.messages if m.role == "assistant"][-1:]
        if last:
            print(
                "     closing:", (last[0].text or "").strip().replace("\n", " ")[:400]
            )
        if s.error:
            print("     ERROR:", s.error.message[:300])
