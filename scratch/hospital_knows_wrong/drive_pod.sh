#!/bin/bash
# ABOUTME: Drives ONE already-rented colosseum_hospital pod through a solo_no_mode_switch cell: waits for READY, bootstraps
# ABOUTME: (repo, judge key, harness, scripted smoke), checks the prompts, queues the cell, waits, pulls logs and snapshots.
#
#   bash scratch/hospital_knows_wrong/drive_pod.sh <pod name> <pod id> root@<ip>:<port> <hf target>
#
# The fleet (hospital/fleet.py) does this for many pods but its bring-up fails on a Windows driver
# (2026-11-10: FileNotFoundError), so this is the same sequence by hand for one pod. It never rents and never
# terminates: `uv run runpod up` and `uv run runpod down` are run separately, by the person who owns the pod.
# Everything lands under output/hospital_knows_wrong/pods/<pod name>/; the cell itself is pushed to the Hub by
# `uv run evals` on the pod.
set -uo pipefail

NAME="$1"; POD_ID="$2"; HOST="$3"; TARGET="$4"
CONDITION="solo_no_mode_switch"
SEEDS="1-30"
# The existing solo cells ran with the public board's rationale hidden; a solo seat has no board, so this only
# keeps the recorded config identical to theirs.
EXTRA="fixes.public_actions_without_rationale=true"
PORT="${HOST##*:}"; ADDR="${HOST%:*}"
SSH=(ssh -p "${PORT}" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ServerAliveInterval=30 -o LogLevel=ERROR "${ADDR}")
OUT="output/hospital_knows_wrong/pods/${NAME}"
mkdir -p "${OUT}"
say() { echo "$(date +%H:%M:%S) ${NAME}: $*"; }

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
SHA="$(git rev-parse HEAD)"
[ "${SHA}" = "$(git rev-parse "origin/${BRANCH}")" ] || { say "HEAD is not origin/${BRANCH}: push first"; exit 2; }

say "waiting for READY in the boot log"
for i in $(seq 1 120); do
    if curl -s --max-time 20 "https://${POD_ID}-8080.proxy.runpod.net/boot.log" | grep -q READY; then ready=1; break; fi
    sleep 30
done
[ "${ready:-0}" = 1 ] || { say "no READY within 60 min"; exit 3; }
say "READY"

# The judge key, read without the CRLF a Windows .env carries and never printed.
OPENROUTER_API_KEY="$(grep -a '^OPENROUTER_API_KEY=' .env | head -1 | cut -d= -f2- | tr -d '\r"'"'"'')"
[ -n "${OPENROUTER_API_KEY}" ] || { say "no OPENROUTER_API_KEY in .env"; exit 4; }
export OPENROUTER_API_KEY

say "bootstrap at ${SHA:0:8}"
bash scripts/eval/misalignment/colosseum/hospital/pod_bootstrap.sh "${HOST}" "${BRANCH}" "${SHA}" > "${OUT}/bootstrap.log" 2>&1
if ! grep -q BOOTSTRAP_DONE "${OUT}/bootstrap.log" || grep -q '\[FAIL\]' "${OUT}/bootstrap.log"; then
    say "bootstrap failed:"; tail -25 "${OUT}/bootstrap.log"; exit 5
fi
grep -E '^(===|ALL CHECKS|[0-9]+ CHECK)' "${OUT}/bootstrap.log" | grep -i -E 'no_mode_switch|ALL CHECKS' | sed "s/^/    /"
say "bootstrapped, smoke passed"

# The prompt tests that skip without a built harness: the new condition's system prompt is solo's cut at the
# MODE SWITCH paragraph and its user prompt is solo's byte for byte.
"${SSH[@]}" "cd /root/work && export PATH=/usr/local/bin:/root/.local/bin:\$PATH UV_NO_SYNC=1 && uv run python -m pytest -q tests/test_colosseum_hospital_solo.py -k 'no_mode_switch or byte_for_byte' 2>&1 | tail -3" > "${OUT}/prompt_tests.log" 2>&1
tail -1 "${OUT}/prompt_tests.log" | sed "s/^/    /"
if ! grep -q "passed" "${OUT}/prompt_tests.log" || grep -q -E "failed|error" "${OUT}/prompt_tests.log"; then
    say "the prompt tests did not pass on the pod; not launching"; cat "${OUT}/prompt_tests.log"; exit 6
fi

say "queueing ${CONDITION}:${SEEDS} on ${TARGET}"
"${SSH[@]}" "cd /root/work && EXTRA='${EXTRA}' bash scripts/eval/misalignment/colosseum/hospital/run_hospital_queue.sh 8000 ${TARGET} -- ${CONDITION}:${SEEDS}" > "${OUT}/launch.log" 2>&1
grep -q queued "${OUT}/launch.log" || { say "launch failed:"; cat "${OUT}/launch.log"; exit 7; }

for i in $(seq 1 150); do
    sleep 60
    "${SSH[@]}" "cat /root/work/output/logs/queue_8000.log 2>/dev/null" > "${OUT}/queue.log" 2>/dev/null
    if grep -q QUEUE_DONE "${OUT}/queue.log"; then done_=1; break; fi
done
"${SSH[@]}" "cat /root/work/output/logs/hospital_${CONDITION}_8000.log 2>/dev/null" > "${OUT}/eval.log" 2>/dev/null
# The environment's per-iteration stock snapshots, which the Hub copy of a cell does not carry.
"${SSH[@]}" "cd /root/colosseum/logs 2>/dev/null && find . -name 'data_iteration_*.json' | tar czf - -T -" > "${OUT}/env_snapshots.tgz" 2>/dev/null
[ "${done_:-0}" = 1 ] || { say "no QUEUE_DONE within 150 min"; tail -5 "${OUT}/queue.log"; exit 8; }
say "$(grep -a EXIT "${OUT}/queue.log" | tail -1)"
say "DRIVE_DONE (pod still billing: terminate it with uv run runpod down --pod ${POD_ID})"
