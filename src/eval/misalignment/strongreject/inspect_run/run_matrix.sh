#!/bin/bash
# ABOUTME: Run the upstream Inspect StrongREJECT task over both arms x {None, AIM}, sequentially.
# ABOUTME: Usage: bash run_matrix.sh   (from this directory; needs .env + a live PODVLLM_BASE_URL)
set -euo pipefail
cd "$(dirname "$0")"

# Published config kept verbatim -- default max_tokens (2048), default judge (openai/gpt-4o),
# temperature 0.75, 1 epoch, all 313 prompts. The whole reason to run the Inspect port is
# comparability with published numbers, so nothing here is overridden except concurrency.
for arm in numina_control synthdoc716; do
  for jb in None AIM; do
    echo "=== $arm / jailbreak=$jb  $(date -u +%H:%M:%S)"
    uv run inspect eval inspect_evals/strong_reject \
      --model "openai-api/podvllm/${arm}" \
      -T "jailbreak_method=${jb}" \
      --max-connections 32 \
      --log-dir "logs/${arm}-${jb}" \
      --no-log-realtime
  done
done
echo "=== matrix complete $(date -u +%H:%M:%S)"
