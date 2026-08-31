#!/bin/bash
set -uo pipefail
MAIN="/home/matthewb/git repos/teaching_claude_why_replication"
WT="$MAIN/.claude/worktrees/fixing_synthdoc"
SC="/tmp/claude-1000/-home-matthewb-git-repos-teaching-claude-why-replication/d79d0007-3ac5-4ace-ae9c-e5d9d3aef5e7/scratchpad"
CFG="$MAIN/configs/odcv_bench_table2_synthdoc_r64_1pod.yaml"
cd "$WT"; set -a; source .env; set +a
echo "=== FULL single-pod rollout (70 cells, pod A) $(date +%H:%M:%S) ==="
uv run scripts/odcv_rollout.py --config "$CFG" 2>&1 | tee "$SC/odcv_1pod_rollout.log"
DIR=$(grep -oE '>>> /.*' "$SC/odcv_1pod_rollout.log" | tail -1 | sed 's/>>> //')
echo ">>> rollout dir: $DIR"
echo "=== JUDGE $(date +%H:%M:%S) ==="
uv run scripts/odcv_judge.py --rollout_dir "$DIR" --config "$CFG" 2>&1 | tee "$SC/odcv_1pod_judge.log"
echo "=== ODCV DONE -> $DIR/results.json ==="
cat "$DIR/results.json" 2>/dev/null | python3 -m json.tool 2>/dev/null | head -60
