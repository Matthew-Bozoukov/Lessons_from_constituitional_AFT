# ABOUTME: Launches a RunPod H100 that runs scratch/qwen3_empty_think_tokens.py on
# ABOUTME: Qwen3.6-27B and serves the output on :8080. Run: bash scratch/launch_qwen36_probe.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a

B64=$(base64 < scratch/qwen3_empty_think_tokens.py)
ARGS="bash -c 'mkdir -p /workspace; exec >> /workspace/boot.log 2>&1; set -x; (cd /workspace && nohup python3 -m http.server 8080 >/dev/null 2>&1 &); export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1; python3 -m pip install --no-cache-dir -q transformers==5.14.1 accelerate fire hf_transfer || true; echo $B64 | base64 -d > /workspace/probe.py; (python3 /workspace/probe.py --model Qwen/Qwen3.6-27B > /workspace/probe_out.txt 2>&1; echo PROBE_DONE >> /workspace/probe_out.txt) || true; sleep infinity'"

runpodctl create pod --name qwen36-think-probe \
  --gpuType "NVIDIA A100 80GB PCIe" \
  --imageName "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204" \
  --containerDiskSize 150 --volumeSize 1 --mem 48 --vcpu 8 \
  --ports "8080/http" --secureCloud --cost 2.0 \
  --args "$ARGS"
