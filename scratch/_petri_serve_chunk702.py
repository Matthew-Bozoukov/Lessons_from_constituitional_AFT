# ABOUTME: Serve the chunk-only-702 LoRA on RunPod (think mode, reasoning+tool parsers) as the
# ABOUTME: Petri target; print the OpenAI-compatible endpoint and save pod state to scratch/.
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)
from src.infra.runpod import serve_vllm, endpoint_url, wait_serving, warm_proxy
from src.infra.huggingface import hf_token

BASE = "Qwen/Qwen3.6-27B"
ADAPTER = "LASR-Callum/2026-08-21-qwen36-lora-table2-9284-difficult-advice-chunk-only-702-rank-64-dynbatch"
SERVED = "chunk702"
STATE = Path("scratch/.petri_chunk702_pod.json")

def main():
    pod_id = serve_vllm(
        BASE, [(SERVED, ADAPTER)], mode="think",
        pod_name="matthew-petri-chunk702", hf_token=hf_token(), lora_rank=64,
        max_len=32768, reasoning_parser="qwen3", tool_call_parser="qwen3_xml")
    ep = endpoint_url(pod_id)
    STATE.write_text(json.dumps({"pod_id": pod_id, "endpoint": ep, "served": SERVED}))
    print(f">>> pod {pod_id} | endpoint {ep}", flush=True)
    wait_serving(pod_id, on_phase=lambda p: print(f"phase: {p}", flush=True))
    warm_proxy(ep, SERVED)
    print(f">>> TARGET_READY endpoint={ep} served={SERVED} pod={pod_id}", flush=True)
    print(f">>> TEAR DOWN WITH: uv run runpod down --pod {pod_id}", flush=True)

if __name__ == "__main__":
    main()
