#!/usr/bin/env bash
# ABOUTME: On one pod: finish the cancelled 2026-10-02 da-15 audit's last 3 seeds, then run the
# ABOUTME: full 90-seed constitution audit on the 2026-10-02 da-5 adapter, then take the pod down.
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
unset OPENROUTER_API_KEY HF_TOKEN HUGGINGFACE_API_KEY
POD=dpqqwhqkryk4f5
SERVER=root@152.236.142.245:12888
DA15_LOG=output/petri/2026-10-02_221445_2026-10-02-qwen36-0-da-15/logs/2026-10-02T21-21-37-00-00_audit_Ex28FptkCfjig8sjdNnQtj.eval
LOG=output/logs/$(date +%Y%m%d_%H%M%S)_petri_finish_da15_then_da5.log
exec > >(tee -a "$LOG") 2>&1

down() { echo ">>> taking down $POD"; ./.venv/bin/runpod down --pod "$POD" 2>&1 | tail -2; }

for i in $(seq 1 80); do
  curl -s --max-time 15 "https://${POD}-8080.proxy.runpod.net/boot.log" 2>/dev/null | grep -q READY \
    && { echo ">>> READY after $((i*30))s"; break; }
  [ "$i" = 80 ] && { echo "!!! never READY after 40min"; down; exit 1; }
  sleep 30
done

# --- 1. the three seeds the cancelled da-15 run did not reach -------------------------
# Served under the da-15 adapter because eval-retry reads `target=openai-api/vllm/
# qwen36_0_da_15` from the log and the name must resolve to the same model.
echo "=== finishing the da-15 audit (87/90 scored) ==="
./.venv/bin/python -u scratch/petri_seeds/run_audit.py \
  --server "$SERVER" --target dougalldeepmind/2026-10-02-qwen36-0-da-15 \
  --retry "$DA15_LOG" 2>&1 | tail -20
echo ">>> da-15 retry rc=${PIPESTATUS[0]}"

# --- 2. the full audit on da-5, same settings as every other arm ----------------------
echo "=== auditing da-5 (90 seeds, sonnet-5 auditor, flash judge, 25 turns) ==="
./.venv/bin/python -u scratch/petri_seeds/run_audit.py \
  --server "$SERVER" --target dougalldeepmind/2026-10-02-qwen36-0-da-5 2>&1 | tail -25
echo ">>> da-5 audit rc=${PIPESTATUS[0]}"

down
