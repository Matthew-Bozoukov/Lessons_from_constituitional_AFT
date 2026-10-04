#!/usr/bin/env bash
# ABOUTME: Wait for the stated-harm corpus, build its 15% mixture, train on 2xH200, then run
# ABOUTME: ODCV-lite at one pass. Aborts rather than training if the mixture misses 15%.
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
unset OPENROUTER_API_KEY HF_TOKEN HUGGINGFACE_API_KEY
LOG=output/logs/$(date +%Y%m%d_%H%M%S)_chain_stated_harm.log
exec > >(tee -a "$LOG") 2>&1
CORPUS=dougalldeepmind/2026-10-04-ablation-stated-harm-synth
MIXCFG=configs/data/mixture/ablation-stated-harm.yaml

# ---- 1. the corpus run is still in flight; wait for the process, not a guess --------------
for i in $(seq 1 60); do
  if ! pgrep -f "[s]ynth run --config configs/data/synth/ablation-stated-harm" >/dev/null; then
    echo ">>> corpus run finished after ${i}m"; break
  fi
  [ "$i" = 60 ] && { echo "!!! corpus still generating after 60m — stopping"; exit 1; }
  sleep 60
done
./.venv/bin/python - <<PY || { echo "!!! corpus not on the Hub — stopping"; exit 1; }
import sys
from dotenv import dotenv_values
from huggingface_hub import HfApi
api = HfApi(token=dotenv_values(".env")["HF_TOKEN_MATBOZ"])
try:
    i = api.dataset_info("$CORPUS")
except Exception as e:
    print(f"!!! {type(e).__name__}"); sys.exit(1)
print(f">>> corpus {i.sha[:12]} on the Hub")
PY

# ---- 2. the mixture, declared at 15% ------------------------------------------------------
# The builder ASSERTS the realised share against the declared one and refuses a mixture named
# for a share it could not fill. That is the open question here: at 0.57x reasoning the
# supervised-token pool is much smaller than v1's, and a refusal is the answer, not a fault.
sed -e 's|2026-10-04-ablation-synth|2026-10-04-ablation-stated-harm-synth|' \
    -e 's|^  ablation:|  ablation-stated-harm:|' \
    -e 's|output/mixture_ablation|output/mixture_ablation_stated_harm|' \
    configs/data/mixture/ablation.yaml > "$MIXCFG"
echo ">>> built $MIXCFG"
if ! ./.venv/bin/mix --config "$MIXCFG" 2>&1 | grep -vE "%\|" | tail -8; then
  echo "!!! mixture build FAILED — not training. If it refused on the share, that is the finding."
  exit 1
fi
MIX=$(./.venv/bin/python -c "
import json,glob,os
d=sorted(glob.glob('output/mixture_ablation_stated_harm/*/mixture_stats.json'), key=os.path.getmtime)[-1]
s=json.load(open(d)); t=s['token_share']
print(f\"REALISED {t['realised_pct']}% | rows {s['by_source']['ablation-stated-harm']['examples']} | examples {s['total']['examples']}\", file=__import__('sys').stderr)
print('dougalldeepmind/2026-10-04-ablation-stated-harm-15-mix')")
echo ">>> mixture: $MIX"

# ---- 3. commit + push, because the pod clones HEAD ----------------------------------------
git add configs/data/synth/ablation-stated-harm.yaml "$MIXCFG" scratch/ablation/ scratch/chain_stated_harm.sh
git commit -q -m "Ablation, harm stated not developed: three-paragraph reasoning and its 15% arm

The harm account collapses from two-to-four paragraphs to one that names the harmful action
and the alternative and stops -- no mechanism, no who-it-affects, no what-follows. That takes
the reasoning to 0.57x the source (0.85x in configs/data/synth/ablation.yaml, 1.02x when a
length floor was enforced), so the length confound is now the dominant difference between this
arm and the da arm it is compared with, not a side-effect of it.

Named for what it changes because src/naming.py refuses a version: 'ablation-v2' \"versions the
name instead of describing the variant -- the date on every artifact already orders the
versions\". A separate config, not an edit, so 2026-10-04-ablation-synth and the adapter built
from it are not overwritten.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" || echo ">>> nothing to commit"
git push -q origin matboz/morebench-t6note 2>&1 | tail -1

# ---- 4. train, 2 GPUs ---------------------------------------------------------------------
UP=$(./.venv/bin/runpod up matboz-statedharm --train configs/train/sft.yaml --model qwen36 --count 2 --push_env 2>&1)
echo "$UP" | grep -E "^(pod|host)|NVIDIA|BILLING|no SSH"
POD=$(echo "$UP" | grep -oE "^pod: +[a-z0-9]+" | awk '{print $2}')
ADDR=$(echo "$UP" | grep -oE "host: +root@[0-9.]+:[0-9]+" | sed 's/host: *//')
[ -z "$POD" ] && { echo "!!! no pod id parsed — check for a stray pod"; exit 1; }
IP=${ADDR#root@}; IP=${IP%%:*}; PORT=${ADDR##*:}
SSH="ssh -p $PORT -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=no -o ConnectTimeout=20 root@$IP"
ADAPTER=dougalldeepmind/2026-10-04-qwen36-0-ablation-stated-harm-15
RLOG=/root/work/output/logs/train_stated_harm.log

for i in $(seq 1 80); do
  curl -s --max-time 15 "https://${POD}-8080.proxy.runpod.net/boot.log" 2>/dev/null | grep -q READY \
    && { echo ">>> READY after $((i*30))s"; break; }
  [ "$i" = 80 ] && { echo "!!! never READY"; ./.venv/bin/runpod down --pod "$POD"; exit 1; }
  sleep 30
done
$SSH "mkdir -p /root/work/output/logs && cd /root/work && \
  setsid nohup uv run torchrun --nproc_per_node=2 scripts/train/train_lora.py \
    --config configs/train/sft.yaml model=qwen36 data_repo=$MIX seed=0 wandb=true \
    > $RLOG 2>&1 < /dev/null & sleep 8; echo launched"
for i in $(seq 1 180); do
  sleep 60
  if $SSH 'pgrep -f "[t]orchrun" >/dev/null; echo "ALIVE:$?"' 2>/dev/null | grep -q "ALIVE:1"; then
    echo ">>> trainer exited after ${i}m"; break
  fi
  [ "$i" = 180 ] && { echo "!!! still training after 3h — leaving $POD UP"; exit 1; }
done
$SSH "tr '\r' '\n' < $RLOG | grep -viE '[0-9]+%\|' | tail -10"

./.venv/bin/python - <<PY || { echo "!!! adapter incomplete — leaving $POD UP"; exit 1; }
import sys
from dotenv import dotenv_values
from huggingface_hub import HfApi
api = HfApi(token=dotenv_values(".env")["HF_TOKEN_MATBOZ"])
try: info = api.model_info("$ADAPTER")
except Exception as e: print(f"!!! adapter NOT on the Hub: {type(e).__name__}"); sys.exit(1)
f = {s.rfilename for s in info.siblings}
m = {"adapter_model.safetensors", "adapter_config.json"} - f
print(f">>> adapter {info.sha[:12]} — {len(f)} files" + (f" MISSING {m}" if m else " (complete)"))
sys.exit(1 if m else 0)
PY
echo ">>> taking down the train pod $POD"; ./.venv/bin/runpod down --pod "$POD" 2>&1 | tail -2

# ---- 5. ODCV, one pass, on an --eval pod (a --train pod has no vLLM; the preflight refuses) -
# --eval mask for the H200 card: two H100s failed to publish SSH within 420s today and billed
# until terminated, so the profile's default inference card is avoided deliberately.
UP2=$(./.venv/bin/runpod up matboz-statedharm-odcv --eval mask --target "$ADAPTER" --push_env 2>&1)
echo "$UP2" | grep -E "^(pod|host)|NVIDIA|BILLING|no SSH"
POD2=$(echo "$UP2" | grep -oE "^pod: +[a-z0-9]+" | awk '{print $2}')
ADDR2=$(echo "$UP2" | grep -oE "host: +root@[0-9.]+:[0-9]+" | sed 's/host: *//')
[ -z "$POD2" ] && { echo "!!! no eval pod parsed — check for a stray pod"; exit 1; }
for i in $(seq 1 80); do
  curl -s --max-time 15 "https://${POD2}-8080.proxy.runpod.net/boot.log" 2>/dev/null | grep -q READY \
    && { echo ">>> eval pod READY after $((i*30))s"; break; }
  [ "$i" = 80 ] && { echo "!!! eval pod never READY"; ./.venv/bin/runpod down --pod "$POD2"; exit 1; }
  sleep 30
done
./.venv/bin/evals --name odcv --config configs/eval/odcv/lite.yaml \
  --target "$ADAPTER" --server "$ADDR2" --ssh-key ~/.ssh/id_ed25519 \
  --port 8002 --terminate-pod passes=1 concurrency=16 2>&1 | tail -35
echo ">>> evals rc=${PIPESTATUS[0]}"
echo ">>> surviving matboz pods:"; ./.venv/bin/runpod pods 2>&1 | grep -i matboz || echo "   none"
