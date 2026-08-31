#!/bin/bash
# ODCV synr64: wait for both pods -> smoke-gate (tool calling) -> full sharded rollouts -> judge -> report.
set -uo pipefail
MAIN="/home/matthewb/git repos/teaching_claude_why_replication"
WT="$MAIN/.claude/worktrees/fixing_synthdoc"
SC="/tmp/claude-1000/-home-matthewb-git-repos-teaching-claude-why-replication/d79d0007-3ac5-4ace-ae9c-e5d9d3aef5e7/scratchpad"
PODA=r2hsa6ab77vabp; PODB=i4b6upd9zwkrj9
MODEL_KEY="qwen3.6-27b-table2-synthdoc-r64"
cd "$WT"; set -a; source .env; set +a

wait_serve () {  # $1=pod
  for i in $(seq 1 60); do
    M=$(curl -s --max-time 15 "https://$1-8000.proxy.runpod.net/v1/models" 2>/dev/null)
    echo "$M" | grep -q '"synr64"' && { echo "$(date +%H:%M:%S) $1 SERVING"; return 0; }
    V=$(curl -s --max-time 15 "https://$1-8080.proxy.runpod.net/vllm.log" 2>/dev/null)
    echo "$V" | grep -qiE "Engine core initialization failed|CUDA out of memory" && { echo "$1 BOOT ERR"; echo "$V"|tail -15; return 1; }
    echo "$(date +%H:%M:%S) $1 booting: $(curl -s --max-time 10 https://$1-8080.proxy.runpod.net/boot.log 2>/dev/null | grep -oE 'Fetching [0-9]+ files:[^%]*[0-9]+%|ADAPTERS_DOWNLOADED|Loading weights|EngineCore' | tail -1)"
    sleep 45
  done; return 1
}

echo "=== [1] wait for both pods to serve synr64 (tool calling) ==="
wait_serve "$PODA" || exit 1
wait_serve "$PODB" || exit 1

echo "=== [2] SMOKE on pod A (validate qwen3_xml tool calling -> transcripts) ==="
uv run scripts/odcv_rollout.py --config "$MAIN/configs/odcv_bench_table2_synthdoc_r64_smoke.yaml" --smoke 2>&1 | tee "$SC/odcv_smoke.log"
SMOKE_DIR=$(ls -1dt "$MAIN/output/odcv_bench/$MODEL_KEY"/smoke_*/ 2>/dev/null | head -1)
NT=$(find "$SMOKE_DIR" -name "messages_record.txt" 2>/dev/null | wc -l)
echo ">>> smoke transcripts produced: $NT"
if [ "$NT" -lt 1 ]; then
  echo "!!! SMOKE FAILED — no transcripts (tool calling still broken). Stopping before the expensive full run."
  find "$SMOKE_DIR" -name docker_output.log | head -1 | xargs grep -i "AI API dead\|error" 2>/dev/null | tail -3
  exit 2
fi
echo ">>> SMOKE PASSED — proceeding to full run"

echo "=== [3] FULL sharded rollouts in parallel (shard0->A, shard1->B) ==="
uv run scripts/odcv_rollout.py --config "$MAIN/configs/odcv_bench_table2_synthdoc_r64_shard0.yaml" >"$SC/odcv_shard0.log" 2>&1 &
P0=$!
uv run scripts/odcv_rollout.py --config "$MAIN/configs/odcv_bench_table2_synthdoc_r64_shard1.yaml" >"$SC/odcv_shard1.log" 2>&1 &
P1=$!
echo "shard0 pid $P0, shard1 pid $P1 — waiting..."
wait $P0; wait $P1
DIR0=$(grep -oE '>>> /.*' "$SC/odcv_shard0.log" | tail -1 | sed 's/>>> //')
DIR1=$(grep -oE '>>> /.*' "$SC/odcv_shard1.log" | tail -1 | sed 's/>>> //')
echo ">>> shard0 dir: $DIR0"
echo ">>> shard1 dir: $DIR1"

echo "=== [4] arrange shards + judge ==="
PARENT="$MAIN/output/odcv_bench/$MODEL_KEY/combined_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$PARENT"
ln -s "$DIR0" "$PARENT/shard0"
ln -s "$DIR1" "$PARENT/shard1"
uv run scripts/odcv_judge.py --rollout_dir "$PARENT" --config "$MAIN/configs/odcv_bench_table2_synthdoc_r64_shard0.yaml" 2>&1 | tee "$SC/odcv_judge.log"
echo "=== ODCV DONE -> $PARENT/results.json ==="
cat "$PARENT/results.json" 2>/dev/null | python3 -m json.tool 2>/dev/null | head -40
