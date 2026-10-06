# ABOUTME: Runs ONE more ODCV rollout pass for the msm arm and judges it with the two
# ABOUTME: already-banked clean passes, instead of re-sampling all three.
"""msm's 3-pass ODCV aborted after pass 2 when one cell's orchestrator was OOM-killed
(exit 137) under concurrent load. Both passes audited clean (80/80 real transcripts, 0
shells), so re-running all three would discard 160 good rollouts. This runs the third pass
into the same run root, then combines and judges all three.

It has to stand in for runner.run(), which normally supplies two things _run_pass needs: a
tinker shim serving the checkpoint, and `base_url`/`model_key` on the config (both None in
the saved file, because runner sets them at runtime from the target spec).

Usage:
    uv run --project src/infra/endpoints/tinker_runtime python \
        scratch/gptoss_control/msm_third_pass.py [--port 18330] [--wait-for-pid N]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from dotenv import load_dotenv
from omegaconf import OmegaConf

load_dotenv(ROOT / ".env")
from src.eval.misalignment.odcv import odcv_judge
from src.eval.misalignment.odcv.passes import audit_pass, combine_passes
from src.eval.misalignment.odcv.runner import _bridge_url, _run_pass
from src.infra.endpoints.tinker import tinker_shim

MODEL_KEY = "2026-10-06-gptoss120b-0-msm"
RUN_ROOT = ROOT / "output/gptoss_arms/msm/odcv/2026-10-06_2026_10_06_gptoss120b_0_msm_190406"

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=18330)
parser.add_argument("--wait-for-pid", type=int, default=0,
                    help="wait for this pid to exit first; a precise condition, unlike a "
                         "pgrep pattern (one of those mis-read 'clear' earlier today and let "
                         "two ODCV runs overlap)")
args = parser.parse_args()

if args.wait_for_pid:
    print(f">>> waiting for pid {args.wait_for_pid} to exit", flush=True)
    while Path(f"/proc/{args.wait_for_pid}").exists():
        time.sleep(30)
    print(">>> clear", flush=True)

# Take the CLEAN passes, not every directory: a pass dir is created before its rollouts
# run, so a launch that died early (the first attempt at this script, on a missing config
# key) leaves an empty husk behind. Selecting on the audit skips those without deleting
# anything, and still refuses to proceed if the two real passes are not both there.
banked = []
for d in sorted(x for x in (RUN_ROOT / MODEL_KEY).iterdir() if x.is_dir()):
    a = audit_pass(d)
    state = "CLEAN" if a["clean"] else f"skipped (real {a['transcripts_real']}/{a['n_expected']})"
    print(f">>> {d.name}: {state}", flush=True)
    if a["clean"]:
        banked.append(d)
assert len(banked) >= 2, f"expected at least 2 clean banked passes, found {len(banked)}"

work = ROOT / "output/gptoss_arms/msm/pass3"
work.mkdir(parents=True, exist_ok=True)
cfg = OmegaConf.load(ROOT / "output/gptoss_arms/msm/odcv_config.yaml")
# EVERY runtime key runner.run() sets (runner.py:165-172), not a subset. Setting these
# piecemeal cost two launches: first `model` was missing and odcv_rollout refused; then
# `endpoint_key_env` was missing, so the containers had no shim credential, every request
# failed unauthenticated, and the pass reported "80/80 clean, $0.00" with 80 missing
# transcripts -- a cell whose requests all fail still exits ok.
cfg.model = "openai/gpt-oss-120b"
cfg.model_key = MODEL_KEY
cfg.endpoint_key_env = "TINKER_SHIM_API_KEY"   # tinker:// targets only
cfg.require_exact_token_count = True
cfg.strict_tool_arguments = True
cfg.output_root = str(RUN_ROOT)                 # the new pass lands beside the banked two
cfg.passes = 1
cfg.judge_budget.ledger = str(work / "judge_budget.json")
cfg_path = work / "odcv_config.yaml"

ckpt = json.loads((ROOT / "output/gptoss_arms/msm/trained_adapter.json").read_text())["sampler"]
with tinker_shim(ckpt, port=args.port, bind="0.0.0.0",
                 reasoning=str(cfg.tinker.reasoning), max_tokens=int(cfg.tinker.max_tokens),
                 context_window=int(OmegaConf.select(cfg, "serving.context_window") or 28000),
                 max_cost_usd=float(cfg.tinker.max_cost_usd),
                 log_dir=work / "tinker_shim") as base_url:
    cfg.base_url = _bridge_url(base_url)
    OmegaConf.save(cfg, cfg_path)
    print(f">>> pass 3: base_url {cfg.base_url}", flush=True)
    audit = _run_pass(cfg_path, False)
print(f">>> pass 3 audit: real {audit['transcripts_real']}/{audit['n_expected']} "
      f"shells {audit['shell_transcripts']} clean {audit['clean']}", flush=True)
# REFUSE to combine an unclean pass. combine_passes does not filter shells -- it tolerates
# gaps by design -- so an unauthenticated pass's 80 prompt-only transcripts would be scored
# as rollouts, every one of them 0 ("not misaligned"), diluting the rate by a third. That
# nearly happened here: the pass that ran without `endpoint_key_env` produced
# written 80 / nonempty 80 / REAL 0 / SHELLS 80, and combine took all of them.
assert audit["clean"] and audit["transcripts_real"] == audit["n_expected"], (
    f"pass 3 is not clean (real {audit['transcripts_real']}/{audit['n_expected']}, "
    f"shells {audit['shell_transcripts']}); refusing to combine -- shells score 0 and would "
    "understate the misalignment rate")

passes = banked + [Path(audit["path"])]
combined = work / "combined3x"
shutil.rmtree(combined, ignore_errors=True)
combine_passes(passes, combined, MODEL_KEY, OmegaConf.to_container(cfg, resolve=True))
print(f">>> judging {len(passes)} passes combined", flush=True)
odcv_judge.main(rollout_dir=str(combined), config=str(cfg_path), max_workers=8)
print(">>> DONE", flush=True)
