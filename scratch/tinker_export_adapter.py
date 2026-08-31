# ABOUTME: Export a Tinker LoRA checkpoint to standard PEFT format and push it to HF, so the
# ABOUTME: adapter can be served by vLLM the same way every other arm in this repo is.

"""Take a Tinker training checkpoint to a servable HuggingFace adapter.

Run: TINKER_API_KEY=... HF_TOKEN=... \\
     uv run --no-sync python scratch/tinker_export_adapter.py \\
         --tinker_path tinker://... --repo_id LASR-Callum/<name>

Three steps, because Tinker stores checkpoints in its own layout:

  download()           checkpoint -> local disk
  build_lora_adapter() -> "standard PEFT format for serving", i.e. the
                          adapter_config.json + adapter_model.safetensors pair that
                          vLLM's --lora-modules and /v1/load_lora_adapter expect
  publish_to_hf_hub()  -> the Hub

build_hf_model() is the alternative and produces a MERGED model — for gpt-oss-120b that is
~240GB, so the adapter path is the only sane one here.

The export is verified before pushing: an adapter that lands on the Hub without the PEFT
file pair looks fine in the UI and fails at serve time, which is the failure mode this repo
has hit repeatedly today in other guises.
"""

import json
import os
from pathlib import Path

import fire

BASE = "openai/gpt-oss-120b"
PEFT_FILES = ("adapter_config.json", "adapter_model.safetensors")


def state_to_sampler_weights(state_path: str, base_model: str, rank: int,
                             name: str) -> str:
    """Re-save a training-state checkpoint as sampler weights, which alone can be exported.

    Tinker keeps two kinds of checkpoint and they are not interchangeable. `save_state`
    writes a resumable training state under `weights/`; `save_weights_for_sampler` writes
    inference weights under `sampler_weights/`. The archive endpoint backing `download()`
    accepts ONLY the latter, rejecting the former with
    "Checkpoint ... is not a sampler weights checkpoint" (observed 2026-08-15). A finished
    run that saved only state is not lost — loading the state into a fresh client and
    re-saving costs one round trip and no retraining.

    Args:
        state_path: `tinker://.../weights/<name>` training-state path.
        base_model: Base the adapter was trained on.
        rank: LoRA rank the run used; the client must match or the state will not load.
        name: Name for the new sampler-weights checkpoint.

    Returns:
        The `tinker://.../sampler_weights/<name>` path.
    """
    import asyncio

    import tinker

    async def go() -> str:
        service = tinker.ServiceClient()
        client = await service.create_lora_training_client_async(
            base_model=base_model, rank=rank)
        print(f"    loading state {state_path}")
        await (await client.load_state_async(state_path)).result_async()
        print(f"    re-saving as sampler weights '{name}'")
        res = await (await client.save_weights_for_sampler_async(name=name)).result_async()
        return res.path

    return asyncio.run(go())


def main(tinker_path: str, repo_id: str, base_model: str = BASE,
         work_dir: str = "output/adapters/tinker_gptoss", private: bool = True,
         push: bool = True, rank: int = 32) -> None:
    """Download, convert, verify and (optionally) publish a Tinker adapter.

    Accepts either checkpoint kind: a `sampler_weights/` path is exported directly, and a
    `weights/` training-state path is converted first (see state_to_sampler_weights).

    Args:
        tinker_path: `tinker://...` checkpoint path from save_state or save_weights.
        repo_id: Destination HF repo.
        base_model: Base the adapter was trained on.
        work_dir: Local staging directory.
        private: Create the HF repo private.
        push: Set False to convert and verify without publishing.
        rank: LoRA rank of the run, needed only when converting a training state.
    """
    from tinker_cookbook import weights

    work = Path(work_dir)
    raw, peft = work / "tinker_raw", work / "peft"
    work.mkdir(parents=True, exist_ok=True)

    if "/sampler_weights/" not in tinker_path:
        print(f">>> {tinker_path} is a training state, not sampler weights")
        tinker_path = state_to_sampler_weights(
            tinker_path, base_model, rank, tinker_path.rstrip("/").split("/")[-1])
        print(f"    -> {tinker_path}")

    print(f">>> downloading {tinker_path}")
    got = weights.download(tinker_path=tinker_path, output_dir=str(raw))
    print(f"    -> {got}")
    print(f"    files: {sorted(p.name for p in Path(got).rglob('*') if p.is_file())[:8]}")

    print(f">>> converting to PEFT format")
    weights.build_lora_adapter(base_model=base_model, adapter_path=str(got),
                               output_path=str(peft))

    present = sorted(p.name for p in peft.rglob("*") if p.is_file())
    print(f"    files: {present}")
    missing = [f for f in PEFT_FILES if not (peft / f).exists()]
    assert not missing, (
        f"NOT servable: missing {missing}. vLLM needs both {PEFT_FILES}; without them the "
        f"adapter loads as a no-op or fails at serve time.")

    cfg = json.loads((peft / "adapter_config.json").read_text())
    print(f"    r={cfg.get('r')} alpha={cfg.get('lora_alpha')} "
          f"base={cfg.get('base_model_name_or_path')}")
    print(f"    target_modules: {str(cfg.get('target_modules'))[:120]}")
    size = sum(p.stat().st_size for p in peft.rglob("*") if p.is_file())
    print(f"    total {size/1e6:.1f} MB")

    if not push:
        print(">>> push=False: stopping after verification")
        return

    # HF_TOKEN first: it is the org-scoped one. HUGGINGFACE_API_KEY is scoped to the
    # personal namespace only, so it 403s on an LASR-Callum push (seen 2026-08-15).
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    assert token, "set HF_TOKEN (or HUGGINGFACE_API_KEY) to publish"
    url = weights.publish_to_hf_hub(model_path=str(peft), repo_id=repo_id,
                                    private=private, token=token, model_card=None)
    print(f">>> published: {url}")


if __name__ == "__main__":
    fire.Fire(main)
