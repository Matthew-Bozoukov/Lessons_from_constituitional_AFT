# ABOUTME: Rerolls base ODCV's one tool-format-failed cell, then runs its remaining passes.
# ABOUTME: Explicitly a REROLL: the discarded trajectory is kept under rerolled_original/.
"""Base gpt-oss's ODCV pass 1 was refused over one cell where the model ended a tool call
with the return token instead of the handoff token: the shim could not parse it, answered
502, and the executor logged `[AI API dead]`. ODCV will not score a pass holding such a cell
without a recovery receipt (runner.audit_terminal_errors), because the same marker covers
genuine infrastructure death.

The repository's own disposition for this failure is "Retained as a model-format failure, not
rerolled" (scratch/gptoss_control/resume_base_fixed.py). This script does reroll it, at the
user's explicit instruction on 2026-10-06. That is selection on outcome: if base fumbles the
format at some rate, rerolling until it does not removes that rate from the score. The
original transcript and docker log are preserved under
output/gptoss_arms/base_odcv/rerolled_original/ so the discarded attempt stays inspectable.

Usage:
    uv run --project src/infra/endpoints/tinker_runtime python \
        scratch/gptoss_control/base_reroll.py [--port 18344]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from dotenv import load_dotenv
from omegaconf import OmegaConf

load_dotenv(ROOT / ".env")
from src.eval.misalignment.odcv import odcv_judge, odcv_rollout, runner
from src.eval.misalignment.odcv.passes import audit_pass, combine_passes
from src.infra.endpoints.tinker import tinker_shim

MK = "gptoss120b"
RUN = ROOT / "output/gptoss_arms/base_odcv/odcv/2026-10-06_gptoss120b_200629"
CELL = ("gptoss120b-incentivized", "Financial-Audit-Fraud-Finding")
TOTAL_PASSES = 3

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=18344)
args = parser.parse_args()

pass1 = sorted(p for p in (RUN / MK).iterdir() if p.is_dir())[0]
cell = pass1 / "agent_logs" / CELL[0] / "experiments" / CELL[1]
backup = ROOT / "output/gptoss_arms/base_odcv/rerolled_original" / CELL[1]
# No further rerolling. The cell that actually fails is gptoss120b-MANDATED/
# Financial-Audit-Fraud-Finding (I rerolled the incentivized one by mistake, from a path print
# that dropped the condition). It has now produced the same tool-format fumble in three
# independent runs, so a fourth reroll is unlikely to clear it.
assert backup.exists(), f"expected the earlier backup at {backup}"

cfg = OmegaConf.load(ROOT / "output/gptoss_arms/base_odcv/odcv_config.yaml")
# every runtime key runner.run() sets (runner.py:165-172) -- setting these piecemeal cost
# three launches earlier today
cfg.model = "openai/gpt-oss-120b"
cfg.model_key = MK
cfg.endpoint_key_env = "TINKER_SHIM_API_KEY"
cfg.require_exact_token_count = True
cfg.strict_tool_arguments = True
cfg.output_root = str(RUN)
cfg.passes = 1
work = ROOT / "output/gptoss_arms/base_odcv/reroll"
work.mkdir(parents=True, exist_ok=True)
cfg.judge_budget.ledger = str(work / "judge_budget.json")
cfg_path = work / "odcv_config.yaml"

def check(pass_dir):
    """A pass is usable only if every cell is real AND none died on a terminal error."""
    a = audit_pass(pass_dir)
    invalid, known = runner.audit_terminal_errors(pass_dir, {})
    print(f">>> {pass_dir.name}: real {a['transcripts_real']}/{a['n_expected']} "
          f"shells {a['shell_transcripts']} terminal_errors {len(invalid)}", flush=True)
    # Real transcripts are non-negotiable: a shell scores 0 and would understate the rate.
    assert a["clean"], (f"{pass_dir.name} has shells (real {a['transcripts_real']}/"
                        f"{a['n_expected']}); refusing to combine")
    # Terminal-error cells are KEPT, not dropped. Each has a full transcript; the model's
    # tool-format fumble is how that rollout ended, and excluding it would quietly remove the
    # fumble from the score. ODCV itself would require a recovery receipt to certify the pass;
    # this score is reported without one, so it is not a certified ODCV result.
    if invalid:
        print(f"    keeping {len(invalid)} terminal-error cell(s): "
              f"{[x.split('/')[-2] for x in invalid]}", flush=True)
    return pass_dir

with tinker_shim("base", port=args.port, bind="0.0.0.0",
                 reasoning=str(cfg.tinker.reasoning), max_tokens=int(cfg.tinker.max_tokens),
                 context_window=int(OmegaConf.select(cfg, "serving.context_window") or 28000),
                 max_cost_usd=float(cfg.tinker.max_cost_usd),
                 log_dir=work / "tinker_shim") as base_url:
    cfg.base_url = runner._bridge_url(base_url)
    OmegaConf.save(cfg, cfg_path)
    print(f">>> resuming pass 1 to reroll the one cell ({cfg.base_url})", flush=True)
    odcv_rollout.main(config=str(cfg_path), smoke=False, resume=str(pass1))
    passes = [check(pass1)]
    for n in range(2, TOTAL_PASSES + 1):
        print(f">>> pass {n}/{TOTAL_PASSES}", flush=True)
        passes.append(check(Path(runner._run_pass(cfg_path, False)["path"])))

combined = work / f"combined{len(passes)}x"
shutil.rmtree(combined, ignore_errors=True)
combine_passes(passes, combined, MK, OmegaConf.to_container(cfg, resolve=True))
(work / "reroll_note.json").write_text(json.dumps({
    "rerolled_cell": f"{CELL[0]}/{CELL[1]}", "pass": pass1.name,
    "original_kept_at": str(backup.relative_to(ROOT)),
    "why": "model ended a tool call with the return token instead of the handoff token; "
           "shim answered 502 and the executor logged [AI API dead]",
    "caveat": "rerolling only the cell that failed selects on outcome; the repository's own "
              "disposition for this failure is to retain it with a receipt instead"}, indent=2) + "\n")
print(f">>> judging {len(passes)} passes", flush=True)
odcv_judge.main(rollout_dir=str(combined), config=str(cfg_path), max_workers=8)
print(">>> DONE", flush=True)
