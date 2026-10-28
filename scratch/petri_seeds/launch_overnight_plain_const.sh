#!/usr/bin/env bash
# ABOUTME: From the laptop: rents an H200, installs the repo + patched Petri on it, ships the keys and
# ABOUTME: scripts, then starts overnight_plain_const.sh detached ON the pod so the laptop can close.
# Run: bash scratch/petri_seeds/launch_overnight_plain_const.sh
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
set -a; source .env; set +a
TARGET=dougalldeepmind/2026-10-05-qwen36-0-plain
SHA=$(git rev-parse HEAD)
REPO_URL="$(git remote get-url origin | sed -E 's#^git@([^:]+):#https://\1/#')"
SP=.venv/lib/python3.12/site-packages
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="output/logs/${TS}_launch_overnight_plain_const.log"
mkdir -p output/logs
exec > >(tee -a "$LOG") 2>&1
echo ">>> start $(date -u +%FT%TZ) target=$TARGET repo@$SHA"
git merge-base --is-ancestor "$SHA" origin/main || { echo "!!! $SHA is not on origin/main"; exit 1; }

./.venv/bin/runpod up matboz-petri-plainconst --eval mask --target "$TARGET" --max-hours 12 --push-env
HOST=$(grep -oE "root@[0-9.]+:[0-9]+" "$LOG" | tail -1); POD=$(grep -oE "^pod: +[a-z0-9]+" "$LOG" | tail -1 | awk '{print $2}')
[ -z "$HOST" ] && { echo "!!! no host"; exit 1; }
IP=${HOST#root@}; IP=${IP%%:*}; PORT=${HOST##*:}
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=30 -i $HOME/.ssh/id_ed25519 -p $PORT root@$IP"
echo ">>> pod=$POD host=$HOST"
for i in $(seq 1 80); do
  curl -s --max-time 15 "https://${POD}-8080.proxy.runpod.net/boot.log" | grep -q READY && { echo ">>> READY after $((i*30))s"; break; }
  [ "$i" = 80 ] && { echo "!!! never READY; pod $POD kept"; exit 1; }; sleep 30
done
$SSH 'eval "$(tr "\0" "\n" < /proc/1/environ | grep -E "^RUNPOD_(POD_ID|API_KEY)=" | sed "s/=.*/=set/")"; echo "pod env: RUNPOD_POD_ID=${RUNPOD_POD_ID:-MISSING} RUNPOD_API_KEY=${RUNPOD_API_KEY:-MISSING}"' < /dev/null

echo ">>> repo @ $SHA + uv sync + inspect-petri 3.1.0"
$SSH "REPO_URL='$REPO_URL' SHA='$SHA' bash -s" <<'REMOTE' || { echo "!!! bootstrap failed; pod kept"; exit 1; }
set -euo pipefail
export PATH=/usr/local/bin:/root/.local/bin:$PATH HF_HOME=/workspace/hf
mkdir -p /root/work && cd /root/work
[ -d .git ] || { git init -q && git remote add origin "$REPO_URL"; }
git fetch -q origin main && git checkout -q --detach "$SHA"
uv sync --frozen --quiet --no-install-package causal-conv1d
uv pip install --quiet --python .venv/bin/python inspect-petri==3.1.0 inspect-scout==0.5.3 inspect-ai==0.3.268
.venv/bin/python -c "import inspect_ai, inspect_petri; print('    inspect_ai', inspect_ai.__version__, '| inspect_petri ok')"
REMOTE

echo ">>> shipping the working-tree files the commit lacks, and the two vendored patches"
tar -czf - scratch/petri_seeds/run_audit.py scratch/petri_seeds/push_results.py scratch/petri_seeds/plot_misalignment.py \
  scratch/petri_seeds/verify_target_preamble.py scratch/petri_seeds/overnight_plain_const.sh \
  src/eval/audits/petri/petri-subscription/seeds-constitution | $SSH 'cd /root/work && tar -xzf -'
cat $SP/inspect_petri/target/_controller.py | $SSH "cat > /root/work/$SP/inspect_petri/target/_controller.py"
cat $SP/inspect_ai/model/_providers/openrouter.py | $SSH "cat > /root/work/$SP/inspect_ai/model/_providers/openrouter.py"
L1=$(sha256sum $SP/inspect_petri/target/_controller.py $SP/inspect_ai/model/_providers/openrouter.py | awk '{print $1}' | tr '\n' ' ')
R1=$($SSH "cd /root/work && sha256sum $SP/inspect_petri/target/_controller.py $SP/inspect_ai/model/_providers/openrouter.py" < /dev/null | awk '{print $1}' | tr '\n' ' ')
[ "$L1" = "$R1" ] || { echo "!!! patched files differ on the pod; pod kept"; exit 1; }
echo "    patches identical on the pod (injection + openrouter caching)"

echo ">>> .env on the pod (HF_TOKEN, HF_ORG, OPENROUTER_API_KEY; values never printed)"
env -u OPENROUTER_API_KEY -u HF_TOKEN -u HF_ORG ./.venv/bin/python -c "
from dotenv import dotenv_values
v = dotenv_values('.env')
for k in ('HF_TOKEN', 'HF_ORG', 'OPENROUTER_API_KEY'):
    assert v.get(k), k
    print(f'{k}={v[k]}')" | $SSH 'cd /root/work && cat > .env && chmod 600 .env && echo "    keys: $(cut -d= -f1 .env | tr "\n" " ")"'

$SSH 'cd /root/work && nohup bash scratch/petri_seeds/overnight_plain_const.sh > /root/work/overnight.out 2>&1 < /dev/null & sleep 5; tail -3 /root/work/overnight.out' < /dev/null
echo ">>> started on $POD; follow: $SSH 'tail -f /root/work/output/logs/overnight_plain_const.log'"
