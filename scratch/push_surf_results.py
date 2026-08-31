# ABOUTME: Pushes the completed SURF sweep directory to Hugging Face with the repo's
# ABOUTME: required dataset-card fields. Run: uv run python scratch/push_surf_results.py

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.huggingface import push_run_dir  # noqa: E402
from src.utils import git_sha  # noqa: E402

load_dotenv()

OUT = Path("output/surf/20260809_083742_t2synth_traits")
# matboz, not LASR-Callum: the HF token is fine-grained and scoped to the user namespace,
# so org repo creation 403s. Move it to the org later with a token that has org rights.
REPO = "matboz/2026-08-09-surf-sweep-qwen36-table2-synthdoc-9-principles"

fields = {
    "title": "SURF sweep — Qwen3.6-27B + table2-synthdoc LoRA vs the 9-principle constitution",
    "experiment": (
        "SURF EM search over Tulu-3 attribute space for prompt categories that elicit "
        "9-principle constitution violations from the table2-synthdoc LoRA arm, served in "
        "thinking mode; 3 runs x 20 iterations x 120 candidates."
    ),
    "date_generated": "2026-08-09",
    "constitution": (
        "constitutions/claude_distilled_09_principles_mid_20260804 — scored via "
        "src/eval/audits/surf/third_party/SURF/rubrics/traits.yaml"
    ),
    "source_repo": f"teaching_claude_why_replication @ {git_sha()}",
    "models": (
        "target LASR-Callum/qwen3.6-27b-lora-table2-synthdoc-r64 (LoRA r=64, alpha=128) on "
        "base Qwen/Qwen3.6-27B, served by vLLM 0.26.0 with --reasoning-parser qwen3 "
        "(thinking mode); judge openrouter:openai/gpt-5.6-terra; query generator "
        "openrouter:meta-llama/llama-3.1-70b-instruct"
    ),
    "generation_config": (
        "target temperature=1.0 max_tokens=8192 streamed, max_model_len=16384, "
        "gpu_memory_utilization=0.85, max_num_seqs=64, concurrency 21/run; "
        "judge temperature=1.0 max_tokens=2048; query gen temperature=1.0 max_tokens=512; "
        "attributes seoirsem/CHUNKY-tulu3-SFT-25k-attributes"
    ),
    "schema": (
        "runs/run_N/results.jsonl — one scored candidate per row: query (generated user "
        "prompt), response (target's answer; the thinking trace is NOT included, see caveat), "
        "reward_score (0-100, >50 = violation), score_metadata.reasoning (judge rationale "
        "naming the principle), attributes (the sampled attribute combination), iteration, "
        "run_id, source_prompt/source_index (the Tulu row the attributes came from). "
        "runs/run_N/failures.jsonl — query-generation parse failures. "
        "sweep_summary.json — per-run buffers. results.md — score distribution, per-principle "
        "counts and the top 15 findings in full. sweep.log — full run log."
    ),
    "provenance": (
        "uv run -m surf.cli.main sweep --attributes seoirsem/CHUNKY-tulu3-SFT-25k-attributes "
        "--rubric rubrics/traits.yaml --output-dir <out> --num-runs 3 --iterations 20 "
        "--candidates 120 --target-model <vllm-endpoint>:t2synth "
        "--judge-model openrouter:openai/gpt-5.6-terra "
        "--query-model openrouter:meta-llama/llama-3.1-70b-instruct — run from "
        "src/eval/audits/surf/third_party/SURF with the patches in third_party/PATCHES.md "
        "(target max_tokens 8192, streamed target calls, prompt_cache_key, loud judge errors). "
        "CAVEATS: the judge scored final answers only, not chains of thought, because the "
        "reasoning parser routes traces to a separate field SURF does not read; and 165 target "
        "calls (~2.7%) were dropped on proxy 120s timeouts, so coverage is slightly short — "
        "dropped candidates are absent, never scored 0."
    ),
}

url = push_run_dir(OUT, REPO, fields, private=True)
print("pushed:", url)
