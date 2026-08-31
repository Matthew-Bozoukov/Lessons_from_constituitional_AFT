#!/bin/bash
# Launch a RunPod instance with vLLM server
# Usage: ./launch_runpod_vllm.sh <model> [revision]
# Example: ./launch_runpod_vllm.sh allenai/Olmo-3-32B-Think-SFT
# Example: ./launch_runpod_vllm.sh allenai/Olmo-3.1-32B-Think step_0800

set -e

MODEL=${1:?"Usage: $0 <model> [revision]"}
REVISION=${2:-""}
POD_NAME="vllm-${MODEL##*/}"  # Extract last part of model name

# GPU type - A100 80GB recommended for 32B models
GPU_TYPE="H200 SXM"

echo "Creating RunPod instance..."
echo "  Model: $MODEL"
echo "  Revision: ${REVISION:-main}"
echo "  GPU: $GPU_TYPE"

# Create pod with vLLM-ready image
POD_ID=$(runpodctl create pod \
    --name "$POD_NAME" \
    --gpuType "$GPU_TYPE" \
    --gpuCount 1 \
    --imageName "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04" \
    --containerDiskSize 50 \
    --volumeSize 120 \
    --volumePath "/workspace" \
    --ports "8000/http" \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "MODEL=$MODEL" \
    --env "REVISION=$REVISION" \
    2>&1 | sed -n 's/.*pod "\([^"]*\)".*/\1/p')

echo "Pod created: $POD_ID"
echo "Waiting for pod to be ready..."

# Wait for pod to be running
while true; do
    STATUS=$(runpodctl get pod "$POD_ID" 2>&1 | sed -n 's/.*status:[[:space:]]*\([A-Za-z]*\).*/\1/p' | head -1)
    STATUS=${STATUS:-pending}
    if [ "$STATUS" = "RUNNING" ]; then
        break
    fi
    echo "  Status: $STATUS"
    sleep 10
done

echo "Pod is running!"

# Get pod info for SSH
POD_INFO=$(runpodctl get pod "$POD_ID")
echo "$POD_INFO"

echo ""
echo "=========================================="
echo "Next steps:"
echo "1. SSH into the pod and run:"
echo "   pip install vllm"
echo "   python serve_olmo.py --model $MODEL ${REVISION:+--revision $REVISION} --port 8000"
echo ""
echo "2. Set your local env:"
echo "   export VLLM_BASE_URL=https://<pod-id>-8000.proxy.runpod.net/v1"
echo "   export VLLM_API_KEY=dummy"
echo ""
echo "Pod ID: $POD_ID"
echo "=========================================="
