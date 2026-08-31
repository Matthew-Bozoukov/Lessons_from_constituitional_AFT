#!/bin/bash
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
set -a; . .env; set +a
B=http://10.201.0.1:8000/v1
curl -sf --max-time 12 "$B/models" 2>/dev/null | grep -q '"id":"synthdoc"' || { echo "!!! bridge down; aborting"; exit 1; }
echo "=== numina p3 $(date -u +%H:%M:%S)"
uv run python scratch/odcv_run_pass.py configs/eval/odcv_worst3/numina_p3.yaml > output/logs/odcv_w3_numina_p3.log 2>&1
echo "=== synthdoc p1+p2 PARALLEL $(date -u +%H:%M:%S)"
uv run python scratch/odcv_run_pass.py configs/eval/odcv_worst3/synthdoc_p1.yaml > output/logs/odcv_w3_synthdoc_p1.log 2>&1 &
P1=$!; sleep 10
uv run python scratch/odcv_run_pass.py configs/eval/odcv_worst3/synthdoc_p2.yaml > output/logs/odcv_w3_synthdoc_p2.log 2>&1 &
P2=$!
wait $P1 $P2
echo "=== synthdoc p3 $(date -u +%H:%M:%S)"
uv run python scratch/odcv_run_pass.py configs/eval/odcv_worst3/synthdoc_p3.yaml > output/logs/odcv_w3_synthdoc_p3.log 2>&1
echo "=== ALL ROLLOUTS DONE $(date -u +%H:%M:%S)"
