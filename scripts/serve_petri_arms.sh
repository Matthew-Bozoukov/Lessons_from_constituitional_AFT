#!/usr/bin/env bash
# ABOUTME: Serves Qwen3.6-27B base plus the three difficult-advice LoRA arms from one vLLM process.
# ABOUTME: Runs ON the rented GPU box. Verifies all four arms answer a TOOL-BEARING request before exiting.
#
# STAGE 2 of 4 - RUNS ON THE RENTED GPU BOX, EVERY SESSION. Leave it running.
#
# Serves all four arms from ONE vLLM process so the adapter is the only variable
# between them. Needs stage 1 done. Then tunnel :8000 and run the audits (stage 3,
# not in this repo - see the runbook). Full flow: docs/petri_dose_sweep_runbook.md
#
# Four arms, one process, one GPU. `--enable-lora` gives four served model names
# off a single base weight load, so every arm shares the identical serving stack
# and the adapter is the only variable. That is the whole point: the sibling
# experiment's ODCV result rests on the arms differing ONLY by the adapter.
#
# WHY THIS MIGHT NOT WORK, AND WHAT HAPPENS THEN
#
# Qwen3.6-27B is a hybrid Mamba/linear-attention vision-language model, and the
# model cards say vLLM LoRA support for this architecture is unproven. If
# --enable-lora fails at startup, fall back to sequential merges:
#   scripts/merge_lora.py per arm, ~20 min each, serve one at a time.
# The merge path also drops the base model's 15 mtp.* tensors (speculative
# decoding only - irrelevant here). Do NOT mix: either all four arms come from
# the LoRA server or all four come from merged checkpoints.
#
# SETTINGS THAT ARE NOT NEGOTIABLE (each one cost someone a run)
#
#   --max-num-seqs 32      Default 1024 fails at startup: "max_num_seqs (1024)
#                          exceeds available Mamba cache blocks (345)". Each
#                          decode sequence needs one Mamba cache block.
#   --tool-call-parser     qwen3_xml, NOT hermes. This template emits
#                          <tool_call><function=NAME><parameter=arg> XML, not
#                          Hermes JSON. Wrong parser => the target issues no tool
#                          calls at all and the audit loop stalls immediately,
#                          which reads as a well-behaved model rather than a
#                          broken one.
#   tokenizer             The BASE tokenizer serves all four arms, by omitting
#                         --tokenizer entirely. The adapters' tokenizer.json
#                         differs from base by merges serialization and 7
#                         declared-but-unused audio/TTS special tokens at ids
#                         248070-248076; vocab is identical (248,044 entries) and
#                         tokenization was verified byte-identical on plain /
#                         system+tools / thinking / tool_call / unicode prompts.
#                         Since it makes no difference to tokenization, use the
#                         base one: those 7 extra special-token ids sit ABOVE the
#                         base vocab, so pointing vLLM at the adapter tokenizer
#                         risks a vocab-size mismatch against the embedding table
#                         for no benefit. One tokenizer for all arms either way,
#                         which is what comparability actually requires.
#   bind 127.0.0.1         Reached only through the SSH tunnel. No public
#                          unauthenticated model endpoint.
#
# Chat template: unlike the sibling experiment, NO substitution is needed. All
# three adapters ship a chat_template.jinja byte-identical to the base template
# (sha256 e84f32a23fdda27689f868aa, 7764 bytes, 6 tool references). The HTTP 400
# tool-rejection trap that killed that experiment's pilot v1 does not apply here.
# Verified against Hugging Face before provisioning.

set -uo pipefail

WORK=/workspace
MODELS="$WORK/models"
LOGS="$WORK/logs"
mkdir -p "$LOGS"

BASE="$MODELS/base"
A10="$MODELS/dose-10-90"
A20="$MODELS/dose-20-80"
A40="$MODELS/dose-40-60"

for d in "$BASE" "$A10" "$A20" "$A40"; do
  [ -d "$d" ] || { echo "FATAL: missing $d - run the download step first" >&2; exit 1; }
done

# Kill any previous server. The bracket trick stops pkill matching its own
# command line, which killed the shell three times in the sibling experiment.
pkill -f "vll[m] serve" 2>/dev/null
sleep 3

MAX_LEN="${MAX_MODEL_LEN:-65536}"

echo "[serve] starting vLLM: bf16, max_model_len=$MAX_LEN, 4 arms"
nohup vllm serve "$BASE" \
  --served-model-name base \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --enable-lora \
  --max-lora-rank 32 \
  --max-loras 3 \
  --lora-modules \
    "dose-10-90=$A10" \
    "dose-20-80=$A20" \
    "dose-40-60=$A40" \
  --dtype bfloat16 \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.92 \
  --host 127.0.0.1 --port 8000 \
  > "$LOGS/vllm.log" 2>&1 &

# Measured on this box: engine init 818s, CUDA graph capture 232s, then ~90s per
# LoRA adapter loaded off the network filesystem - about 24 minutes end to end.
# A 20-minute budget expired mid-startup, so the wrapper exited FATAL while vLLM
# carried on and came up fine a few minutes later. 45 minutes with a live
# progress line, so a slow start is visibly a slow start and not a silent stall.
echo "[serve] waiting for readiness (up to 45 min: 52.6GB of weights off a network FS)"
for i in $(seq 1 540); do
  if [ $((i % 20)) -eq 0 ]; then
    echo "[serve]   still loading, $((i * 5))s elapsed"
    tail -1 "$LOGS/vllm.log" 2>/dev/null | cut -c1-160
  fi
  if curl -sf http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
    echo "[serve] up after ~$((i*5))s"
    break
  fi
  # Fatal-error detection, with WARNING lines excluded. On this image vLLM logs a
  # benign `ImportError: libnvrtc.so.13` with a full traceback at WARNING level
  # during startup (deep_gemm is optional and unused here). Matching "Traceback"
  # unconditionally aborts a perfectly healthy load - verified against a
  # successful run's log, where that word appears exactly once and only there.
  if grep -E "Mamba cache blocks|ValueError|Traceback|out of memory|CUDA error|No available memory" \
       "$LOGS/vllm.log" 2>/dev/null | grep -qvE "WARNING"; then
    echo "FATAL: startup error - offending lines:" >&2
    grep -nE "Mamba cache blocks|ValueError|Traceback|out of memory|CUDA error|No available memory" \
      "$LOGS/vllm.log" | grep -vE "WARNING" | head -10 >&2
    tail -30 "$LOGS/vllm.log" >&2
    exit 1
  fi

  # A dead server will never answer, so stop waiting for it.
  if ! pgrep -f "vll[m] serve" > /dev/null 2>&1; then
    echo "FATAL: vllm process is gone - last 40 lines:" >&2
    tail -40 "$LOGS/vllm.log" >&2
    exit 1
  fi
  sleep 5
done

curl -sf http://127.0.0.1:8000/v1/models > /dev/null 2>&1 || {
  echo "FATAL: never became ready. Last 40 lines:" >&2; tail -40 "$LOGS/vllm.log" >&2; exit 1; }

echo "[serve] models offered:"
curl -s http://127.0.0.1:8000/v1/models | python3 -c \
  'import json,sys; [print("   -", m["id"]) for m in json.load(sys.stdin)["data"]]'

# --- The check pilot v1 did not have -----------------------------------------
# Every arm must answer a request that CARRIES TOOLS. A 400 here is the exact
# failure that produced 30 empty transcripts and a plausible-looking eval log.
echo "[verify] tool-bearing request against each arm"
fail=0
for arm in base dose-10-90 dose-20-80 dose-40-60; do
  code=$(curl -s -o "$LOGS/probe_$arm.json" -w '%{http_code}' \
    http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{
      \"model\": \"$arm\",
      \"messages\": [
        {\"role\": \"system\", \"content\": \"You are an operations assistant.\"},
        {\"role\": \"user\", \"content\": \"Check the status of batch 4471, then tell me what you found.\"}
      ],
      \"tools\": [{\"type\": \"function\", \"function\": {
        \"name\": \"query_batch_details\",
        \"description\": \"Look up details for a ledger batch\",
        \"parameters\": {\"type\": \"object\",
          \"properties\": {\"batch_id\": {\"type\": \"string\"}},
          \"required\": [\"batch_id\"]}}}],
      \"max_tokens\": 512,
      \"temperature\": 0.7
    }")
  if [ "$code" != "200" ]; then
    echo "   $arm: HTTP $code  <== FAIL"; head -c 400 "$LOGS/probe_$arm.json" >&2; echo; fail=1
  else
    python3 - "$arm" "$LOGS/probe_$arm.json" <<'PY'
import json, sys
arm, path = sys.argv[1], sys.argv[2]
d = json.load(open(path))
msg = d["choices"][0]["message"]
tc = msg.get("tool_calls") or []
reasoning = msg.get("reasoning_content") or ""
print(f"   {arm}: HTTP 200  tool_calls={len(tc)}"
      f"  thinking={len(reasoning)}ch"
      f"  finish={d['choices'][0].get('finish_reason')}")
if not tc:
    print(f"      WARNING: {arm} returned no structured tool call. If this holds "
          f"across arms the tool-call parser is wrong; if only this arm, the "
          f"adapter is not being applied.")
PY
  fi
done

[ "$fail" -eq 0 ] || { echo "FATAL: at least one arm rejects tool-bearing requests" >&2; exit 1; }
echo "[verify] all four arms answer tool-bearing requests"
echo "[serve] ready. Tunnel 127.0.0.1:8000 to the workstation and run the audit."
