#!/usr/bin/env bash
# ABOUTME: One-shot driver for the 2026-11-13 Shen sepsis runs: set up the pod, serve plain/da-15/da-tools-15,
# ABOUTME: check each adapter's tool calls, then launch single + 8-coder org sepsis for all three in parallel.
# Run (from the repo root, after `runpod up` reports BILLING NOW):
#   bash scratch/shen_software/launch_sepsis.sh <ip> <ssh port>
set -euo pipefail
IP="$1"; SSHP="$2"; H="root@$IP"
SHEN=/Users/kunwar/.claude/jobs/43697d6b/tmp/shen/repo/software
SS=scratch/shen_software
LOG=/Users/kunwar/.claude/jobs/43697d6b/tmp/launch_sepsis.log
PLAIN=dougalldeepmind/2026-10-05-qwen36-0-plain
DA=dougalldeepmind/2026-10-05-qwen36-0-da-15
DAT=dougalldeepmind/2026-10-11-qwen36-0-da-tools-15
ssh_() { ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -p "$SSHP" "$H" "$@"; }

echo "[$(date -u +%H:%M)] upload harness + env" | tee -a "$LOG"
ssh_ 'mkdir -p /workspace/shen/repo'
rsync -az -e "ssh -p $SSHP" "$SHEN" "$H:/workspace/shen/repo/"
rsync -az -e "ssh -p $SSHP" $SS/llm_client.py $SS/pod_run.sh $SS/pod_hello.sh $SS/show_transcript.py "$H:/workspace/shen/"
ssh_ 'cd /workspace/shen && uv venv venv --python 3.12 -q && uv pip install --python venv/bin/python -q claude-agent-sdk==0.2.181 numpy pandas pydantic scikit-learn tiktoken openai'
rsync -az -e "ssh -p $SSHP" $SS/safetytooling_stub/safetytooling "$H:/workspace/shen/venv/lib/python3.12/site-packages/"
ssh_ 'cd /workspace/shen/repo/software/src && /workspace/shen/venv/bin/python -c "import agent_org.tools; print(\"harness OK\")"' | tee -a "$LOG"

echo "[$(date -u +%H:%M)] product key (OpenRouter only, never printed)" | tee -a "$LOG"
T=/Users/kunwar/.claude/jobs/43697d6b/tmp/product.env
( umask 077; grep '^OPENROUTER_API_KEY=' /Users/kunwar/projects/lessons_from_constitutional_aft/.env | sed 's/^OPENROUTER_API_KEY=/PRODUCT_API_KEY=/' > "$T" )
scp -q -P "$SSHP" "$T" "$H:/workspace/shen/product.env"; rm -f "$T"
ssh_ 'chmod 600 /workspace/shen/product.env'

echo "[$(date -u +%H:%M)] wait for boot READY" | tee -a "$LOG"
until ssh_ 'grep -q "READY" /workspace/boot.log 2>/dev/null' ; do sleep 30; done

echo "[$(date -u +%H:%M)] serve plain + da-15 + da-tools-15 (262K, text-only, 3 LoRA slots, port 8011)" | tee -a "$LOG"
uv run python $SS/serve_checks.py --host "$H:$SSHP" --port 8011 --targets $PLAIN $DA $DAT \
  > /Users/kunwar/.claude/jobs/43697d6b/tmp/serve_sepsis.log 2>&1 &
until ssh_ 'curl -s -o /dev/null -w "%{http_code}" localhost:8011/v1/models' 2>/dev/null | grep -q 200 \
   && ssh_ 'curl -s localhost:8011/v1/models' | grep -q qwen36_0_da_tools_15; do
  if grep -qE "Traceback|SystemExit" /Users/kunwar/.claude/jobs/43697d6b/tmp/serve_sepsis.log \
     && ! ssh_ 'pgrep -f "[a]pi_server" >/dev/null'; then echo "SERVE FAILED" | tee -a "$LOG"; exit 1; fi
  sleep 30
done
ssh_ 'curl -s localhost:8011/v1/models' | tee -a "$LOG"; echo | tee -a "$LOG"

echo "[$(date -u +%H:%M)] adapter tool checks + launch 6 runs" | tee -a "$LOG"
ssh_ 'cd /workspace/shen && for m in qwen36_0_plain qwen36_0_da_15 qwen36_0_da_tools_15; do PORT=8011 setsid nohup bash pod_hello.sh $m > /dev/null 2>&1 < /dev/null & done; for m in qwen36_0_plain qwen36_0_da_15 qwen36_0_da_tools_15; do PORT=8011 MAX_TURNS=100 setsid nohup bash pod_run.sh sepsis 1 $m single_sepsis_$m > /dev/null 2>&1 < /dev/null & PORT=8011 MAX_TURNS=100 setsid nohup bash pod_run.sh sepsis 8 $m org8_sepsis_$m > /dev/null 2>&1 < /dev/null & done; echo launched'
echo "[$(date -u +%H:%M)] LAUNCHED" | tee -a "$LOG"
