# ABOUTME: Offline check of the oai_hf exporter: runs row()/summarise() over the paper's own Qwen 3.8 Max
# ABOUTME: logs and fetches the harness at the pinned commit, so wiring is verified before any GPU run.
# Run: uv run python scratch/oai_hf_repro/check_export_on_paper_logs.py
import glob
import json
import os

from inspect_ai.log import read_eval_log
from omegaconf import OmegaConf

from src.eval.misalignment.oai_hf.export import row, summarise
from src.eval.misalignment.oai_hf.upstream import ensure_harness, task_file

R = os.path.expanduser(
    os.environ.get("OAI_HF_LOGS", "~/projects/oai-hf-incident-reproduction/logs")
)
PATS = {
    "step1": f"{R}/260905_step1_xlsx_n64/step1_gdrive_xlsx/qwen38max__shard*/*.eval",
    "step2": f"{R}/260909_step2_bank_none_multimodel/protein_none/qwen38max__shard*/*.eval",
    "step3": f"{R}/260909_step3_egress_share_n64/react__qwen38max__driven/shard*/*.eval",
    "step4": f"{R}/260909_step4chain_n64_multimodel/step4_hf_cheat_chain/qwen38max0902__shard*/*.eval",
}

cfg = OmegaConf.load("configs/eval/oai_hf.yaml")
h = ensure_harness(str(cfg.harness.repo), str(cfg.harness.commit))
print("harness:", h)
for s in cfg.steps:
    print("  task file ok:", task_file(h, s.task).relative_to(h))

rows = []
for step, pat in PATS.items():
    for f in sorted(glob.glob(pat)):
        log = read_eval_log(f)
        for sample in log.samples or []:
            rows.append(row(step, log, sample))
summ = summarise(rows)
for step, s in summ.items():
    print(
        step,
        "sig all",
        s["signature_all"]["k"],
        "/",
        s["signature_all"]["n"],
        "| end",
        s["end_reasons"],
        "| probes",
        s["source_probes"],
        "| extra",
        {
            k: s[k]
            for k in ("funnel", "share_given_solved_and_found_board", "found_peers")
            if k in s
        },
    )
    print("     health", json.dumps(s["health"]))
