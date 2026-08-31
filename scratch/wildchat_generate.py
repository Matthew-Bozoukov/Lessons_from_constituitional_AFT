# ABOUTME: Generate a target model's answers (reasoning trace + reply) to a WildChat prompt
# ABOUTME: sample, serving the adapter with the same vLLM path the registered evals use.

"""Answer WildChat opening turns with one served target, saving trace and reply separately.

Sampling mirrors `configs/eval/lmsys.yaml` — the closest existing analogue, since that eval
also answers single real-user chat turns — except for a larger `max_tokens`: this is a
thinking model and the downstream measurement reads the trace, so the budget has to cover a
full trace AND a visible reply (CLAUDE.md gotcha 5). The truncation rate is printed and
recorded; read it before trusting any rate computed over the replies.

Designed to run ON the GPU host (Option A in CLAUDE.md): the arena-hard reruns showed an SSH
tunnel dropping streams under sustained concurrent load.

    uv run python scratch/wildchat_generate.py \
        --target LASR-Callum/qwen3.6-27b-lora-table2-synthdoc-r64 \
        --prompts output/wildchat/wildchat_prompts_250_<ts>.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
from openai import OpenAI

from src.endpoints.openrouter import map_threaded
from src.endpoints.vllm_server import LocalExec, VllmServer, resolve_target
from src.model_profile import resolve_trace
from src.utils import read_jsonl, timestamp, write_run_meta

CONTEXT_WINDOW = 16384


def main(target: str,
         prompts: str,
         out_dir: str = "output/wildchat",
         temperature: float = 0.7,
         top_p: float = 0.95,
         max_tokens: int = 8192,
         parallel: int = 32,
         port: int = 8000,
         limit: int = 0,
         request_timeout: float = 900.0) -> None:
    """Serve `target` and answer every prompt in `prompts`.

    Args:
        target: HF path of an adapter or full model.
        prompts: JSONL of {idx, prompt, ...} from scratch/wildchat_prompts.py.
        out_dir: Parent directory; a per-run subdirectory is created inside it.
        temperature: Sampling temperature (lmsys parity).
        top_p: Nucleus sampling parameter (lmsys parity).
        max_tokens: Generation budget covering the <think> trace AND the reply.
        parallel: Concurrent requests to the local vLLM server.
        port: Port for the local vLLM server.
        limit: Answer only the first N prompts (0 = all). Use for a smoke.
        request_timeout: Per-request timeout in seconds.
    """
    rows = list(read_jsonl(Path(prompts)))
    if limit:
        rows = rows[:limit]
    assert rows, f"no prompts in {prompts}"

    spec = resolve_target(target)
    print(f">>> target {spec.hf_path} | model_key {spec.model_key} | mode {spec.mode}")
    print(f">>> {len(rows)} prompts | temp {temperature} | max_tokens {max_tokens}")

    ts = timestamp()
    dest = Path(out_dir) / f"{spec.model_key}_{ts}"
    dest.mkdir(parents=True, exist_ok=True)

    server = VllmServer(work_dir=dest / "server", port=port, executor=LocalExec(dest / "server"),
                        serve_requirements={"context_window": CONTEXT_WINDOW})
    try:
        served = server.ensure(spec)
        client = OpenAI(base_url=served.base_url, api_key=served.api_key,
                        timeout=request_timeout, max_retries=2)

        def generate(i: int) -> dict:
            row = rows[i]
            try:
                resp = client.chat.completions.create(
                    model=served.model_name,
                    messages=[{"role": "user", "content": row["prompt"]}],
                    temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            except Exception as exc:  # noqa: BLE001 — one dropped request must not sink
                # the whole run's finished work; recorded as an error and excluded later,
                # never silently scored (lmsys runner's rationale).
                print(f"    !! prompt {row['idx']}: {type(exc).__name__} — recorded as error")
                return {"idx": row["idx"], "prompt": row["prompt"], "think": "",
                        "answer": "", "finish_reason": "error"}
            choice = resp.choices[0]
            reasoning = (getattr(choice.message, "reasoning_content", None)
                         or getattr(choice.message, "reasoning", None))
            think, answer = resolve_trace(choice.message.content or "", reasoning)
            return {"idx": row["idx"], "prompt": row["prompt"], "think": think,
                    "answer": answer, "finish_reason": choice.finish_reason or ""}

        answers = map_threaded(generate, len(rows), max_workers=parallel, desc="generate")
    finally:
        server.stop()

    answers.sort(key=lambda r: r["idx"])
    path = dest / "answers.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rec in answers:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_err = sum(1 for r in answers if r["finish_reason"] == "error")
    n_trunc = sum(1 for r in answers if r["finish_reason"] == "length")
    n_empty_answer = sum(1 for r in answers if not r["answer"].strip()
                         and r["finish_reason"] != "error")
    n_empty_think = sum(1 for r in answers if not r["think"].strip()
                        and r["finish_reason"] != "error")
    write_run_meta(dest, {"target": target, "prompts": prompts, "n": len(rows),
                          "temperature": temperature, "top_p": top_p,
                          "max_tokens": max_tokens, "context_window": CONTEXT_WINDOW,
                          "mode": spec.mode},
                   extra={"answers": str(path), "errors": n_err, "truncated": n_trunc,
                          "empty_answer": n_empty_answer, "empty_think": n_empty_think})

    print(f"\n>>> {path}")
    print(f"    errors {n_err} | truncated (finish=length) {n_trunc} "
          f"| empty answer {n_empty_answer} | empty think {n_empty_think}")
    ok = [r for r in answers if r["finish_reason"] not in ("error",)]
    if ok:
        tl = sorted(len(r["think"]) for r in ok)
        al = sorted(len(r["answer"]) for r in ok)
        print(f"    median chars: think {tl[len(tl) // 2]} | answer {al[len(al) // 2]}")
    print("\n--- first generation ---")
    first = answers[0]
    print(f"PROMPT: {first['prompt'][:300]}")
    print(f"THINK : {first['think'][:600]}")
    print(f"ANSWER: {first['answer'][:400]}")


if __name__ == "__main__":
    fire.Fire(main)
