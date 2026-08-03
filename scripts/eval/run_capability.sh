#!/bin/bash
# ABOUTME: Generate + judge one capability-eval arm against the baseline arm.
# ABOUTME: Usage: run_capability.sh <arm> [stage]   (vLLM must already serve the arm)
set -euo pipefail

ARM="${1:?usage: run_capability.sh <arm> [stage]   e.g. arm_b_synth10 150}"
STAGE="${2:-0}"
CONFIG="${CONFIG:-configs/eval/capability.yaml}"

cd "$(dirname "$0")/../.."

# The judge compares against the baseline arm's answers, so those must exist before any
# other arm can be judged. Failing here beats discovering it 500 questions in.
BASELINE=$(uv run python -c "
from omegaconf import OmegaConf
print(OmegaConf.load('${CONFIG}').baseline_arm)")
BENCH=$(uv run python -c "
from omegaconf import OmegaConf
c=OmegaConf.load('${CONFIG}'); print(f'{c.vendor_dir}/data/{c.bench_name}/model_answer')")

if [[ "$ARM" != "$BASELINE" && ! -f "${BENCH}/${BASELINE}.jsonl" ]]; then
  echo "ERROR: baseline arm '${BASELINE}' has no answers yet." >&2
  echo "  Serve it and run: scripts/eval/run_capability.sh ${BASELINE}" >&2
  exit 1
fi

echo "=== 1/3 generate: ${ARM} ==="
# Assumes vLLM is already serving this arm under this name (scripts/gpu/serve_lora.sh) and,
# when running from the PC, that the SSH tunnel to :8000 is up in a BACKGROUND shell —
# a tunnel started inside a foreground call dies when that call returns.
uv run python src/eval/capabilities/capability_gen.py --config "$CONFIG" --arm "$ARM"

cat <<'EOF'

=== 2/3 STOP AND EYEBALL ===
Open the raw_samples.md printed above before spending a cent on judging.
A chat-template mismatch reads as catastrophic capability loss but is purely a serving
bug, and it is the single most common cause of "my finetune destroyed the model".
Look for: role markers or special tokens leaking into the text, answers that continue
the prompt instead of responding, empty or unterminated <think> blocks, truncated or
run-on generations.

Press ENTER to judge, or Ctrl-C to stop.
EOF
read -r _

if [[ "$ARM" == "$BASELINE" ]]; then
  echo "=== 3/3 baseline arm: A-vs-A instrument sanity check ==="
fi

echo "=== 3/3 judge: ${ARM} vs ${BASELINE} (stage=${STAGE:-full}) ==="
uv run python src/eval/capabilities/capability_judge.py \
  --config "$CONFIG" --arm "$ARM" --stage "$STAGE"

echo
echo "=== DONE ${ARM}. Rebuild the report with: ==="
echo "  uv run python src/eval/capabilities/capability_report.py --config ${CONFIG}"
