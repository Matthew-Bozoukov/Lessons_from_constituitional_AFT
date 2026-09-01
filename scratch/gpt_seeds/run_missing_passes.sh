#!/usr/bin/env bash
# ABOUTME: Finish the GPT seed ODCV runs after the 18:33 docker collision: seed-42 pass 2, seed-69
# ABOUTME: passes 1-2, then combine + judge each -- starting a pass ONLY when no other ODCV run is on the daemon.
# Run: bash scratch/gpt_seeds/run_missing_passes.sh <serve pod id>   (detach via start_missing_passes.sh)
#
# WHY THE GUARD. odcv_rollout.py names every compose project `odcv-<variant>-<scenario>`, global
# on the one Docker daemon, so two ODCV drivers on this laptop (any arms, same scenarios) tear
# down each other's containers: on 2026-08-28 our seed-42 pass 2 and another session's par716coh
# pass 1 started 4 s apart and BOTH wrote 0 transcripts (compose_exit_1+no_transcript,
# "executor-1 has been recreated ... No such container"). A pass starts here only when no
# odcv_rollout_cli process and no odcv-* container exists; it cannot stop a run that starts
# mid-pass, which is what the cross-session message is for.
set -uo pipefail
POD="${1:?serve pod id}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
mkdir -p output/logs

wait_for_quiet_daemon() {
  while true; do
    others=$(pgrep -f 'odcv_rollout_cli\.py' | grep -v "^$$\$" || true)
    running=$(docker ps -q --filter name=odcv- 2>/dev/null || true)
    if [ -z "$others" ] && [ -z "$running" ]; then
      # settle: require quiet twice, 30 s apart, so a between-pass gap of another driver
      # is not mistaken for its end
      sleep 30
      others=$(pgrep -f 'odcv_rollout_cli\.py' || true)
      running=$(docker ps -q --filter name=odcv- 2>/dev/null || true)
      [ -z "$others" ] && [ -z "$running" ] && return 0
    fi
    echo "[$(date '+%H:%M:%S')] daemon busy (other odcv_rollout_cli: $(echo "$others" | wc -w | tr -d ' '), odcv containers: $(echo "$running" | wc -w | tr -d ' ')) -- waiting"
    sleep 60
  done
}

park_dead_passes() {  # <model key>
  # scratch/odcv_combine_passes.py auto-discovers EVERY dir under output/odcv_bench/<key>/ that
  # holds agent_logs/ (any name not starting with "combined"), so a dead pass must leave the
  # model root entirely -- a renamed dir in place still becomes a pass (rollout index, 65
  # "short" cells, and push_odcv.py would ship it as a pass). Dead = zero non-empty transcripts.
  local KEY="$1" d n
  for d in output/odcv_bench/"$KEY"/2026*; do
    [ -d "$d/agent_logs" ] || continue
    n=$(find "$d" -name messages_record.txt -size +0 | wc -l | tr -d ' ')
    if [ "$n" = "0" ]; then
      mkdir -p "output/odcv_bench/_dead/$KEY"
      mv "$d" "output/odcv_bench/_dead/$KEY/$(basename "$d")"
      echo "parked dead pass (0 transcripts) -> output/odcv_bench/_dead/$KEY/$(basename "$d")"
    fi
  done
}

judge_seed() {  # <seed> <prev combined dir or ''>
  local SEED="$1" PREV="$2"
  local CFG="configs/eval/odcv_bench_t2_9284_gptresp685_s${SEED}_r64_paired_2x65.yaml"
  local KEY="qwen3_6-27b-lora-t2-9284-gptresp685-paired-r64-seed${SEED}"
  park_dead_passes "$KEY"
  echo "############ seed $SEED combine $(date '+%H:%M:%S')"
  uv run python scratch/odcv_combine_passes.py --config "$CFG" 2>&1 | tee -a "output/logs/odcv_gptseed${SEED}_combine.log" | tail -n 8
  local COMBINED
  COMBINED=$(ls -td "output/odcv_bench/${KEY}"/combined*/ | head -n 1)
  if [ -n "$PREV" ] && [ -d "$PREV/evaluations" ]; then
    # Judge verdicts are cached per rollout key under evaluations/; pass 1's transcripts are
    # byte-identical between the old and new combined dirs (same rollout_000), so carrying the
    # cache over re-judges only the new pass. Same judges, same config.
    mkdir -p "$COMBINED/evaluations"
    cp -n "$PREV"/evaluations/scores_*.json "$COMBINED/evaluations/" 2>/dev/null || true
    echo "carried judge cache from $PREV"
  fi
  echo "############ seed $SEED judge -> $COMBINED $(date '+%H:%M:%S')"
  uv run python scratch/odcv_judge_cli.py --rollout_dir "$COMBINED" --config "$CFG" 2>&1 | tee -a "output/logs/odcv_gptseed${SEED}_judge.log" | tail -n 12
}

# --- seed 42: pass 2 only (pass 1 = 20260828_171534, 62 transcripts) -------------------------
# Both seed eval configs live on `worktree-gpt-seeds`, not on this branch — the GPT
# seed-replicate run finished 2026-08-29 and they were never merged. Check them out from
# there before re-running this driver.
CFG42="configs/eval/odcv_bench_t2_9284_gptresp685_s42_r64_paired_2x65.yaml"
# Re-pin: the first serving pod (xi6dycxzg2599r) vanished mid-run; the config carries whichever
# pod actually served the rollouts, so replace any pod id, not just the placeholder.
sed -i '' -E "s#https://[a-z0-9_]+-8000#https://${POD}-8000#" "$CFG42"
grep -q "https://${POD}-8000" "$CFG42" || { echo "base_url not pinned in $CFG42"; exit 1; }
# Dead passes (the 18:33 collision, the 01:27 dead-endpoint attempt) leave the model root
# BEFORE anything else, so they can never be discovered as passes.
park_dead_passes "qwen3_6-27b-lora-t2-9284-gptresp685-paired-r64-seed42"
wait_for_quiet_daemon
echo "############ seed 42 pass 2 $(date '+%H:%M:%S')"
bash scratch/odcv_repeat_rollouts.sh "$CFG42" 2 2 2>&1 | tee -a output/logs/odcv_gptseed42_rollouts.log | grep -v "^\[.*\] .*ETA" | tail -n 3
judge_seed 42 "output/odcv_bench/qwen3_6-27b-lora-t2-9284-gptresp685-paired-r64-seed42/combined2x_20260828_173611"

# --- seed 69: both passes -----------------------------------------------------------------------
CFG69="configs/eval/odcv_bench_t2_9284_gptresp685_s69_r64_paired_2x65.yaml"
sed -i '' -E "s#https://[A-Za-z0-9_]+-8000#https://${POD}-8000#" "$CFG69"
grep -q "https://${POD}-8000" "$CFG69" || { echo "base_url not pinned in $CFG69"; exit 1; }
for P in 1 2; do
  wait_for_quiet_daemon
  echo "############ seed 69 pass $P $(date '+%H:%M:%S')"
  bash scratch/odcv_repeat_rollouts.sh "$CFG69" "$P" "$P" 2>&1 | tee -a output/logs/odcv_gptseed69_rollouts.log | grep -v "^\[.*\] .*ETA" | tail -n 3
done
judge_seed 69 ""
echo "ODCV_ALL_DONE $(date '+%H:%M:%S')"
