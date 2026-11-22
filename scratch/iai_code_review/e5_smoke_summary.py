# ABOUTME: Summarise the E5 (alignment_research) smoke from its Hub repo: per condition, episodes finished, judge
# ABOUTME: drift/quality/pivot scores, B tool use, Agent A + judge cost and wall time per episode.
# Run: uv run python scratch/iai_code_review/e5_smoke_summary.py [hub repo]
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from inspect_ai.log import read_eval_log

load_dotenv(".env")
REPO = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "dougalldeepmind/2026-11-20-iai-e5-smoke-qwen36-0-plain"
)
# OpenRouter prices fetched 2026-11-19: grok-4.20 (xai) and gemini-3-flash-preview, $ per token
PRICE = {
    "grok-4.20": (1.25e-6, 2.5e-6, 0.2e-6),
    "gemini-3-flash-preview": (0.5e-6, 3e-6, 0.05e-6),
}

local = Path(snapshot_download(REPO, repo_type="dataset"))
print(
    "files:", sorted(str(p.relative_to(local)) for p in local.rglob("*") if p.is_file())
)
for cond_dir in sorted(
    p for p in local.iterdir() if p.is_dir() and not p.name.startswith(".")
):
    for f in sorted(cond_dir.glob("*.eval")):
        log = read_eval_log(str(f))
        ta = log.eval.task_args or {}
        cond = ta.get("capability", cond_dir.name)
        if cond != "baseline":
            cond += f" L{ta.get('instruction_level')}"
        print(
            f"\n== {cond} ({f.name[-28:]}): status={log.status} samples={len(log.samples or [])}"
        )
        for s in log.samples or []:
            sc = {
                k: (v.value if not isinstance(v.value, float) else round(v.value, 2))
                for k, v in (s.scores or {}).items()
            }
            cost = 0.0
            for m, u in (s.model_usage or {}).items():
                for key, (pin, pout, pcr) in PRICE.items():
                    if key in m:
                        cost += (
                            u.input_tokens * pin
                            + u.output_tokens * pout
                            + (u.input_tokens_cache_read or 0) * pcr
                        )
            b_tok = {
                m: u.output_tokens
                for m, u in (s.model_usage or {}).items()
                if "vllm" in m
            }
            print(
                f"  {s.metadata.get('scenario_name', s.id)} | err={s.error is not None} | scores={sc}"
                f" | api ${cost:.2f} | B out tokens {b_tok} | {round((s.total_time or 0) / 60, 1)} min"
            )
