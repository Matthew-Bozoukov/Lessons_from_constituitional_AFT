#!/usr/bin/env bash
# ABOUTME: Runs ONE Shen et al. software-team rollout on the pod against the local vLLM server
# ABOUTME: (agents via Claude Code -> /v1/messages; product LLM via /v1/chat/completions).
# Run (on the pod): nohup bash pod_run.sh <task: news|sepsis> <num_coding_agents> <served model> <tag> &
set -uo pipefail
TASK="$1"; N="$2"; MODEL="$3"; TAG="$4"
ROOT=/workspace/shen
SW="$ROOT/repo/software"
VENV="$ROOT/venv"
OUT="$ROOT/runs/$TAG"
mkdir -p "$OUT/claude_config"

if [ "$TASK" = news ]; then
  EXP=rec_sys; D=fake_news; TRAIN="$SW/data/$D/train.csv"; TEST="$SW/data/$D/test.csv"
else
  EXP=sepsis_icu_v2; D=sepsis_icu_v2; TRAIN="$SW/data/$D/train.json"; TEST="$SW/data/$D/test.json"
fi

export CLAUDE_CONFIG_DIR="$OUT/claude_config"
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_AUTH_TOKEN=EMPTY
export ANTHROPIC_API_KEY=
export ANTHROPIC_MODEL="$MODEL"
export ANTHROPIC_SMALL_FAST_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export DISABLE_TELEMETRY=1
export IS_SANDBOX=1
# Load MCP tools up front (as the 2025 SDK Shen used did). With tool search on, the PM's
# ToolSearch result carries tool_reference blocks that vLLM's /v1/messages rejects (HTTP 500).
export ENABLE_TOOL_SEARCH=false
# Product LLM: the same base model for every arm (fixed across arms by design).
export PRODUCT_BASE_URL=http://127.0.0.1:8000/v1
export PRODUCT_MODEL=base

cd "$SW/experiments/$EXP"
echo "start $(date -u +%FT%TZ) task=$TASK n=$N model=$MODEL" > "$OUT/status.txt"
"$VENV/bin/python" run_agent_org.py \
  --name "$OUT/exp" \
  --tasks "$SW/data/$D/task.json" \
  --train-dataset "$TRAIN" \
  --test-dataset "$TEST" \
  --eval-config "$SW/data/$D/eval_config.json" \
  --add-file-to-codebase "$ROOT/llm_client.py" \
  --num-coding-agents "$N" \
  --model "$MODEL" \
  > "$OUT/driver.log" 2>&1
echo "exit $? $(date -u +%FT%TZ)" >> "$OUT/status.txt"
