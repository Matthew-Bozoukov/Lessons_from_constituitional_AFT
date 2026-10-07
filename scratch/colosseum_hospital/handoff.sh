#!/usr/bin/env bash
# ABOUTME: When the s5 shard finishes seeds 11-20, reuse its H200 for seeds 1-10 and
# ABOUTME: retire the slow H100, whose ~2 calls/min cannot finish those seeds today.
#
# s5 was launched WITHOUT --terminate-pod precisely so its pod outlives its driver and can
# be handed the next shard. The H100's partial work on seeds 1-10 is discarded: it is the
# same ten seeds the H200 will redo in a fraction of the time, so keeping it only pays
# $3.49/hr to duplicate.
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
export COLOSSEUM_ROOT="$PWD/external/colosseum"
S5_RUN=output/colosseum_hospital/2026-10-07_self_sacrificial_qwen36_0_da_15_092947
H100_POD=q54c6l0z6skqj1
S5_HOST=root@103.196.86.23:14717

for i in $(seq 1 240); do
  # The driver exiting is the signal: it harvests, judges and publishes before it goes.
  if ! pgrep -f "port 8001" >/dev/null 2>&1; then
    echo "=== $(date -u +%H:%M:%SZ) s5 driver exited after ${i} polls ==="
    break
  fi
  [ $((i % 20)) -eq 0 ] && echo "  poll ${i}: s5 still running ($(date -u +%H:%M:%SZ))"
  sleep 30
done

P=$(find "$S5_RUN" -name progress.json 2>/dev/null | head -1)
echo "  s5 final progress: $(cat "$P" 2>/dev/null | tr -d '\n' | cut -c1-200)"

echo "=== $(date -u +%H:%M:%SZ) launching seeds 1-10 on the freed H200 ==="
nohup ./.venv/bin/evals --name colosseum_hospital \
  --target dougalldeepmind/2026-10-05-qwen36-0-da-15 \
  --config scratch/colosseum_hospital/configs/2026-09-09_colosseum_hospital_board_access.yaml \
  condition=self_sacrificial "seeds=[1,2,3,4,5,6,7,8,9,10]" \
  --server "$S5_HOST" --port 8003 \
  > "output/colosseum_hospital/launch/handoff_seeds1_10_$(date -u +%Y-%m-%d_%H-%M-%S).log" 2>&1 &
echo "  launched (pid $!), port 8003"
sleep 60

# Only retire the H100 once the replacement is actually up: a handoff that kills the old
# pod first and then fails to start leaves nothing running at all.
if pgrep -f "port 8003" >/dev/null 2>&1; then
  echo "=== $(date -u +%H:%M:%SZ) replacement is up; retiring H100 $H100_POD ==="
  ./.venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from dotenv import load_dotenv; load_dotenv('.env')
from src.infra.runpod import terminate, pods
print('  terminate $H100_POD ->', terminate('$H100_POD'))
print(' ', pods())"
else
  echo "!!! replacement driver not running — KEEPING the H100 so something still runs"
fi
echo "=== $(date -u +%H:%M:%SZ) handoff done ==="
