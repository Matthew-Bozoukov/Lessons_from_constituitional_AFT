# ABOUTME: Merge a LoRA adapter into its base weights and save a plain servable checkpoint.
# ABOUTME: Needed because vLLM's runtime LoRA support for hybrid vision-language archs is unproven.

"""Merge a PEFT/LoRA adapter into its base model.

Qwen3.6-27B is an image-text-to-text architecture, so it loads through
`AutoModelForImageTextToText` rather than `AutoModelForCausalLM`, and serving the adapter
at runtime through vLLM's LoRA path is not something to rely on. Merging sidesteps both:
the output is an ordinary checkpoint that vLLM serves with no adapter flags at all.

    uv run python -m src.eval.misalignment.internalization.scripts.merge_lora \\
        --base Qwen/Qwen3.6-27B \\
        --adapter matboz/qwen3.6-27b-difficult-advice-tulu-lora \\
        --out ./merged-qwen36-difficult-advice

Needs the optional extras (`uv sync --extra hf`) and enough RAM/VRAM to hold the base in
bf16 - roughly 55GB for a 27B model. Run it on the GPU box, not the laptop.
"""

from __future__ import annotations

import json
from pathlib import Path

import fire


def merge(
    base: str = "Qwen/Qwen3.6-27B",
    adapter: str = "matboz/qwen3.6-27b-difficult-advice-tulu-lora",
    out: str = "./merged-model",
    dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = False,
    push_to: str | None = None,
) -> str:
    """Merge an adapter into its base model and write a servable checkpoint.

    Args:
        base: Base model repo id or path.
        adapter: PEFT/LoRA repo id or path.
        out: Output directory for the merged checkpoint.
        dtype: Torch dtype to load in.
        device_map: Placement strategy; "auto" shards across available devices.
        trust_remote_code: Passed through to transformers.
        push_to: Optional Hub repo id to push the merged checkpoint to.

    Returns:
        The output directory.

    Raises:
        RuntimeError: If the optional ML extras are not installed.
    """
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "Merging needs torch, peft, and accelerate. Install with `uv sync --extra hf`."
        ) from e

    config = AutoConfig.from_pretrained(base, trust_remote_code=trust_remote_code)
    architectures = " ".join(getattr(config, "architectures", None) or [])
    # Hybrid vision-language checkpoints are not AutoModelForCausalLM; guessing wrong here
    # fails after the full download, so the class is chosen from the config.
    cls = (
        AutoModelForImageTextToText
        if "ImageTextToText" in architectures or hasattr(config, "vision_config")
        else AutoModelForCausalLM
    )
    print(f"loading {base} with {cls.__name__} (architectures: {architectures or 'unlisted'})")

    model = cls.from_pretrained(
        base,
        dtype=getattr(torch, dtype),
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    print(f"applying adapter {adapter}")
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)

    # The tokenizer comes from the BASE repo: an adapter repo usually ships only adapter
    # weights, and a served model without a tokenizer cannot render the chat template.
    AutoTokenizer.from_pretrained(base, trust_remote_code=trust_remote_code).save_pretrained(out_dir)
    try:
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(base, trust_remote_code=trust_remote_code).save_pretrained(out_dir)
    except Exception as e:  # noqa: BLE001 - a text-only base has no processor, which is fine
        print(f"no processor saved ({type(e).__name__}); text-only serving is unaffected")

    (out_dir / "merge_provenance.json").write_text(
        json.dumps({"base": base, "adapter": adapter, "dtype": dtype}, indent=2)
    )

    if push_to:
        print(f"pushing to {push_to}")
        model.push_to_hub(push_to)

    print(f"merged checkpoint written to {out_dir}")
    return str(out_dir)


if __name__ == "__main__":
    fire.Fire(merge)
