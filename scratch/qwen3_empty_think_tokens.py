# ABOUTME: Runs a thinking-mode Qwen greedily on a trivial question with thinking on (and off)
# ABOUTME: and dumps every generated token id+piece, to see how think tags open/close token by token.
# Run: uv run --with torch python scratch/qwen3_empty_think_tokens.py [--model Qwen/Qwen3.6-27B]

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run(tok, model, device, enable_thinking: bool, question: str, max_new_tokens: int) -> None:
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
    )
    print(f"\n{'=' * 70}\n### enable_thinking={enable_thinking}\n{'=' * 70}")
    print(f"PROMPT (repr): {prompt!r}")
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    new = out[0, ids.shape[1]:].tolist()
    print(f"\n{len(new)} generated tokens, one per line (idx, id, piece):")
    for i, t in enumerate(new):
        print(f"  {i:4d}  {t:6d}  {tok.decode([t])!r}")
    print(f"\nFULL DECODED COMPLETION (repr):\n{tok.decode(new)!r}")


def main(model: str = "Qwen/Qwen3-0.6B",
         question: str = "What is 2+2? Answer with just the number.",
         max_new_tokens: int = 512) -> None:
    tok = AutoTokenizer.from_pretrained(model)
    if torch.cuda.is_available():
        # device_map streams shards straight to the GPU, so a 27B load never needs
        # the full bf16 copy in system RAM.
        m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
        device = "cuda"
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).to(device).eval()
    run(tok, m, device, True, question, max_new_tokens)
    run(tok, m, device, False, question, max_new_tokens)


if __name__ == "__main__":
    fire.Fire(main)
