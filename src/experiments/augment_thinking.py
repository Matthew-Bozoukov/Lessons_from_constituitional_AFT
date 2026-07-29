# ABOUTME: Augments the difficult-advice SFT data with real <think> reasoning traces
# ABOUTME: (Sonnet 4.5) so training preserves Qwen3's thinking channel (fixes empty-<think>).

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm import OpenRouterClient, map_threaded  # noqa: E402
from prompts import think_trace_messages  # noqa: E402
from utils import timestamp, write_run_meta  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def main(config: str, sft_path: str, smoke: bool = False) -> None:
    """Add a reasoning_content <think> trace to each SFT example.

    Args:
        config: Path to the data-gen YAML (reuses gen_model, max_workers, tokenizer).
        sft_path: Path to the existing sft_dataset.jsonl (messages format).
        smoke: If True, only process 8 examples.
    """
    cfg_path = Path(config)
    if not cfg_path.exists():
        cfg_path = CONFIG_DIR / config
    cfg = OmegaConf.load(cfg_path)

    rows = [json.loads(ln) for ln in Path(sft_path).read_text().splitlines() if ln.strip()]
    if smoke:
        rows = rows[:8]
    print(f">>> augmenting {len(rows)} examples with think-traces via {cfg.gen_model}")

    def work(i: int) -> dict:
        msgs = rows[i]["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        answer = next(m["content"] for m in msgs if m["role"] == "assistant")
        res = OpenRouterClient().chat(
            cfg.gen_model,
            think_trace_messages(user, answer),
            temperature=0.7,
            max_tokens=1024,
        )
        trace = res.content.strip()
        # Strip any stray tags the model may add.
        trace = trace.replace("<think>", "").replace("</think>", "").strip()
        return {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": answer, "reasoning_content": trace},
            ]
        }

    out = map_threaded(work, len(rows), int(cfg.max_workers), "think-traces")

    ts = timestamp()
    out_dir = Path(cfg.output_dir) / (f"think_smoke_{ts}" if smoke else f"think_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_out = out_dir / "sft_dataset_thinking.jsonl"
    with sft_out.open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_run_meta(out_dir, OmegaConf.to_container(cfg, resolve=True),
                   {"source_sft": str(sft_path), "n": len(out), "smoke": smoke})

    # Loud sanity: first trace.
    print("\n--- FIRST THINK TRACE ---")
    print(out[0]["messages"][1]["reasoning_content"][:900])
    print(f"\n>>> wrote {sft_out}  ({len(out)} examples)")


if __name__ == "__main__":
    fire.Fire(main)
