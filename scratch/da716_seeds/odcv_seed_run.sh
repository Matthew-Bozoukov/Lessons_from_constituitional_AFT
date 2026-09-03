#!/usr/bin/env bash
# ABOUTME: Drive ONE principle-scoped seed's ODCV run from this laptop: reconnecting SSH tunnel
# ABOUTME: to the serving pod, preflight, then the box supervisor with a per-seed state dir.
#
# Run: bash scratch/da716_seeds/odcv_seed_run.sh <pod_ip> <pod_ssh_port> <42|69> [passes] [conc]
#
# A per-seed fork of scratch/da_chunk_only/odcv_local_run.sh, which is how seed 0 was run.
# Three things it keeps deliberately, each of which has already cost a run when skipped:
#
#   * THE SSH TUNNEL, not RunPod's HTTPS proxy. docs/LOG.md 2026-08-09: the proxy times out
#     on ODCV's long non-streaming rollouts. Containers reach it at host.docker.internal.
#   * ONE DRIVER AT A TIME on this laptop. odcv_rollout.py names compose projects
#     `odcv-<variant>-<scenario>`, global on the Docker daemon, so two concurrent runs of the
#     same scenarios tear down each other's containers and BOTH write zero transcripts
#     (2026-08-28). Hence seeds run back to back, never in parallel.
#   * A TRANSCRIPT CHECK AFTER PASS 1. Without the vLLM tool-call parser every scenario
#     completes as `ok` and writes NO messages_record.txt -- a failure invisible in the
#     summary. This aborts before paying for pass 2 rather than after.
#
# caffeinate keeps the Mac awake; THE LID MUST STAY OPEN or Docker Desktop pauses and every
# in-flight scenario times out.
set -uo pipefail

POD_IP="${1:?pod ip required}"
POD_PORT="${2:?pod ssh port required}"
SEED="${3:?seed required (42 or 69)}"
PASSES="${4:-2}"
CONC="${5:-12}"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
CFG="scratch/da716_seeds/2026-08-31_odcv_bench_table2_9284_difficult_advice_principle_scoped_702_seed_${SEED}_2_65.yaml"
[ -f "$CFG" ] || { echo "FATAL: no config $CFG"; exit 1; }
HF_REPO="LASR-Callum/2026-08-31-odcv-difficult-advice-principle-scoped-702-seed-${SEED}-eval"
STATE="output/odcv_principle_scoped_s${SEED}_state"
KEY="$(uv run --quiet python -c "import yaml,sys;print(yaml.safe_load(open('$CFG'))['model_key'])")"
mkdir -p "$STATE" output/logs

echo "=== seed $SEED | key $KEY | repo $HF_REPO"

# One driver at a time: refuse to start if another ODCV run holds the daemon.
if pgrep -f 'odcv_rollout_cli\.py' >/dev/null || [ -n "$(docker ps -q --filter name=odcv- 2>/dev/null)" ]; then
  echo "FATAL: another ODCV run is on this Docker daemon — they would delete each other's containers"
  exit 1
fi

echo "=== tunnel -> $POD_IP:$POD_PORT (reconnecting) ==="
pkill -f "ssh -N -L 8000:localhost:8000" 2>/dev/null || true
nohup bash -c "while true; do ssh -N -L 8000:localhost:8000 \
  -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes -p $POD_PORT root@$POD_IP; sleep 5; done" \
  > "$STATE/tunnel.log" 2>&1 &
sleep 8
if ! curl -sf -m 20 http://127.0.0.1:8000/v1/models >/dev/null; then
  echo "FATAL: tunnel up but endpoint not answering on 127.0.0.1:8000"
  tail -5 "$STATE/tunnel.log" 2>/dev/null
  exit 1
fi
echo "served models:"
curl -s -m 20 http://127.0.0.1:8000/v1/models | uv run --quiet python -c "import json,sys;[print('  '+m['id']) for m in json.load(sys.stdin)['data']]"

# The served name in the config must actually be on the endpoint, or every rollout 404s.
WANT="$(uv run --quiet python -c "import yaml;print(yaml.safe_load(open('$CFG'))['model'])")"
curl -s -m 20 http://127.0.0.1:8000/v1/models | grep -q "\"$WANT\"" \
  || { echo "FATAL: '$WANT' is not served — check the --name the adapter was loaded under"; exit 1; }

echo "=== preflight (docker + endpoint) ==="
uv run python scratch/odcv_preflight.py --config "$CFG" --check_docker \
  --base_url http://127.0.0.1:8000/v1 || { echo "FATAL: preflight failed"; exit 1; }

echo "=== pass 1 ==="
uv run python scratch/odcv_box_run.py --config "$CFG" --passes 1 --box_id laptop \
  --state_dir "$STATE" --hf_repo "$HF_REPO" --extra "concurrency=$CONC" \
  2>&1 | tee -a "output/logs/odcv_principle_scoped_s${SEED}.log" | tail -n 5

# The check that pays for itself: a tool-call-parser failure completes every scenario as `ok`
# with no transcript, and would otherwise be discovered after pass 2 was also paid for.
N=$(find "output/odcv_bench/$KEY" -name messages_record.txt -size +0 2>/dev/null | wc -l | tr -d ' ')
echo "=== pass 1 wrote $N non-empty transcripts"
if [ "$N" -lt 20 ]; then
  echo "FATAL: only $N transcripts after pass 1 — serve flags are wrong (need"
  echo "  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3)."
  echo "Not paying for pass 2. Fix the server and re-run."
  exit 1
fi

if [ "$PASSES" -gt 1 ]; then
  echo "=== passes 2..$PASSES ==="
  nohup caffeinate -i uv run python scratch/odcv_box_run.py --config "$CFG" \
    --passes "$PASSES" --box_id laptop --state_dir "$STATE" --hf_repo "$HF_REPO" \
    --extra "concurrency=$CONC" >> "output/logs/odcv_principle_scoped_s${SEED}.log" 2>&1 &
  echo "supervisor pid $!  log output/logs/odcv_principle_scoped_s${SEED}.log  status $STATE/status.json"
fi
