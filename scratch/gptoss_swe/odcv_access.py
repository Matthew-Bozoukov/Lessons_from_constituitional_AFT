# ABOUTME: Read-only Tinker checkpoint and concurrency checks for the full ODCV preparation.
# ABOUTME: Never creates a sampling client or sends a generation request.
import json
from pathlib import Path
import tinker

TARGETS = {
    "control": "tinker://c8be8040-3551-5baf-a8aa-b9e606a91ed2:train:0/sampler_weights/2026-10-06-gptoss120b-0-plain",
    "da15": "tinker://5c74fa19-a0b6-58a8-bfdf-610961c28e12:train:0/sampler_weights/2026-10-06-gptoss120b-0-da-15",
}


def main():
    client = tinker.ServiceClient()
    try:
        rest = client.create_rest_client()
        weights = {
            arm: rest.get_weights_info_by_tinker_path(path).result().model_dump()
            for arm, path in TARGETS.items()
        }
        config = rest.holder.get_client_config()
        supported = client.get_server_capabilities()
        names = [m.model_name for m in supported.supported_models]
        assert "openai/gpt-oss-120b" in names
        assert all(
            w["base_model"] == "openai/gpt-oss-120b" and w["lora_rank"] == 32
            for w in weights.values()
        )
        result = dict(
            provider_generation_calls=0,
            base_available=True,
            targets=TARGETS,
            weights=weights,
            sdk_max_concurrent_requests=config.sample_max_concurrent_requests,
            note="SDK dispatch limit is not a guaranteed throughput or account-wide quota.",
        )
        Path("output/gptoss_odcv_preflight_20261012/access.json").write_text(
            json.dumps(result, indent=2)
        )
        print(json.dumps(result))
    finally:
        client.close(
            "success", detail="Read-only ODCV readiness check; no samples"
        ).result()


if __name__ == "__main__":
    main()
