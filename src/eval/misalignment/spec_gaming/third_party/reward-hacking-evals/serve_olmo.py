#!/usr/bin/env python3
"""
vLLM OpenAI-compatible server for OLMo models.

Usage:
    # Serve SFT model
    python serve_olmo.py --model allenai/Olmo-3-32B-Think-SFT

    # Serve DPO model
    python serve_olmo.py --model allenai/Olmo-3-32B-Think-DPO

    # Serve RLVR checkpoint (e.g., step 1000)
    python serve_olmo.py --model allenai/Olmo-3.1-32B-Think --revision step_1000

    # Serve final RLVR model
    python serve_olmo.py --model allenai/Olmo-3.1-32B-Think

Requirements:
    pip install vllm

Hardware:
    - A100 80GB (recommended) or 2x A100 40GB with tensor parallelism
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Serve OLMo models with vLLM")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--revision", type=str, default=None, help="Model revision/checkpoint (e.g., step_1000)")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--max-model-len", type=int, default=65536, help="Maximum context length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95, help="GPU memory utilization (0-1)")
    parser.add_argument("--download-dir", type=str, default="/workspace/models", help="Directory to download model weights")
    parser.add_argument("--enable-tool-choice", action="store_true", help="Enable auto tool choice for function calling")
    parser.add_argument("--tool-call-parser", type=str, default="hermes", help="Tool call parser (hermes, llama3_json, mistral)")

    args = parser.parse_args()

    # Build the vLLM server command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--port", str(args.port),
        "--host", args.host,
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--download-dir", args.download_dir,
    ]

    if args.revision:
        cmd.extend(["--revision", args.revision])

    # Enable tool calling if requested
    if args.enable_tool_choice:
        cmd.extend(["--enable-auto-tool-choice", "--tool-call-parser", args.tool_call_parser])

    # Create a friendly model name for the API (used in requests)
    if args.revision:
        served_model_name = f"{args.model.split('/')[-1]}@{args.revision}"
    else:
        served_model_name = args.model.split('/')[-1]

    cmd.extend(["--served-model-name", served_model_name])

    print(f"Starting vLLM server...")
    print(f"  Model: {args.model}")
    if args.revision:
        print(f"  Revision: {args.revision}")
    print(f"  Served as: {served_model_name}")
    print(f"  Endpoint: http://{args.host}:{args.port}/v1")
    print()
    print("=" * 60)
    print("To use with Inspect AI, set these env vars:")
    print(f"  export VLLM_BASE_URL=http://<your-runpod-ip>:{args.port}/v1")
    print(f"  export VLLM_API_KEY=dummy")
    print()
    print(f"Then use model: vllm/{served_model_name}")
    print("=" * 60)
    print()

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
