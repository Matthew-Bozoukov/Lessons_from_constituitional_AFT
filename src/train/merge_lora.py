# ABOUTME: Merges a trained LoRA adapter into its base checkpoint and writes a standalone
# ABOUTME: model directory, so inference engines can serve it without LoRA support.

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


def main(base: str, adapter: str, out: str) -> None:
    """Merge a LoRA adapter into base weights and save the result.

    Merging on CPU avoids needing the full bf16 model plus a merge copy resident
    in VRAM; vLLM's LoRA support for hybrid multimodal checkpoints is unproven,
    so serving merged weights is the reliable path.

    Args:
        base: Path or repo id of the base checkpoint.
        adapter: Directory holding the trained adapter.
        out: Destination directory for the merged model.
    """
    out_dir = Path(out)
    cfg = json.loads((Path(adapter) / "adapter_config.json").read_text())
    print(f">>> base={base}\n>>> adapter={adapter}\n>>> targets={cfg['target_modules']}")

    model = AutoModelForImageTextToText.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cpu"
    )
    n_before = sum(p.numel() for p in model.parameters())

    peft_model = PeftModel.from_pretrained(model, adapter, device_map="cpu")
    n_adapters = sum(1 for n, _ in peft_model.named_modules() if "lora_A" in n)
    assert n_adapters > 0, "adapter applied to zero modules — target_modules regex matched nothing"
    print(f">>> LoRA applied to {n_adapters} modules")

    merged = peft_model.merge_and_unload()
    n_after = sum(p.numel() for p in merged.parameters())
    assert n_after == n_before, f"merge changed parameter count: {n_before} -> {n_after}"

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(str(out_dir))
    try:
        AutoProcessor.from_pretrained(base).save_pretrained(str(out_dir))
    except (OSError, ValueError) as exc:
        print(f">>> no processor to copy ({type(exc).__name__}: {exc})")
    for name in ("chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json"):
        src = Path(base) / name
        if src.is_file():
            shutil.copy2(src, out_dir / name)

    print(f">>> merged model written to {out_dir} ({n_after/1e9:.1f}B params)")

