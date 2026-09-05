#!/usr/bin/env bash
# ABOUTME: Sequential ODCV subset runs for the prefix-caching A/B against one pod: W2 (image
# ABOUTME: warm-up, cache on) -> A (cache off) -> B (cache on), slicing the pod's vLLM log per run.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
T=${T:-/Users/jamie/.claude/jobs/06edd020/tmp}
POD=root@103.207.149.115; PORT=17156; SRV="$POD:$PORT"
TARGET=LASR-Callum/qwen3.6-27b-lora-t2-9284-da-chunk-only-702-r64-dynbatch
CFG=scratch/odcv_kv_test/odcv_subset.yaml
LOG=/workspace/output/serve/vllm.log
lines() { ssh -o StrictHostKeyChecking=no -p $PORT $POD "wc -l < $LOG" 2>/dev/null | tr -d ' '; }
run() {  # name, overrides...
  local name=$1; shift
  local start; start=$(lines); echo "=== $name start line $start $(date)"
  uv run evals --name odcv --config $CFG --target $TARGET --server $SRV --no-push "$@" 2>&1 \
    | perl -pe 'BEGIN{$|=1} print scalar(localtime), q( )' > "$T/run_$name.log"
  echo "=== $name evals exit ${PIPESTATUS[0]} $(date)"
  ssh -o StrictHostKeyChecking=no -p $PORT $POD "sed -n '$((start+1)),\$p' $LOG" > "$T/vllm_$name.log" 2>/dev/null
  echo "=== $name vllm log slice: $(wc -l < "$T/vllm_$name.log") lines"
}
run W2
run A serving.reuses_long_prefixes=false
run B
echo "ALL DONE $(date)"
