# ABOUTME: Serving throughput benchmark for MASK: the framework's own vLLM launch (LoRA, pinned think template,
# ABOUTME: reasoning parser) with the profile's sequence cap swept, driven by real MASK prompts at matching concurrency.

"""Run (one pod per GPU type, rented with `uv run runpod up --eval <adapter> [--gpu ...]`):

    uv run python scratch/bench_mask_serving.py --server root@<ip>:<port> --port 8091 --gpu H100 \\
        --caps 32,64,96,128 --requests 384

For each cap the server is (re)started through `VllmServer` exactly as `uv run evals --name mask`
starts it, except that `serving_params` is patched to return that `max_num_seqs`. Then `requests`
real MASK prompts (system + user, from a local MASK work tree) are sent with the MASK sampling
settings (temperature 1.0, max_tokens 16384, think mode pinned by the template) at concurrency =
cap, and the aggregate generation rate is measured from `usage.completion_tokens`. The same
prompts and seeds are used at every cap. Results go to output/bench/<date>_mask_serving_<gpu>.json.
"""

from __future__ import annotations

import asyncio
import glob
import json
import random
import statistics
import time
from pathlib import Path

import fire
import pandas as pd
from omegaconf import OmegaConf

import src.infra.endpoints.vllm as V
from src.utils import timestamp

MASK_TOKENS_PER_RUN = 7_200_000   # generated tokens in one full MASK run (audit of 2026-09-22_qwen36_0_da_15)
PRICE = {"H100": 3.49, "H200": 4.59}  # $/hr, RunPod secure cloud, 2026-09-22


def load_prompts(n: int, seed: int) -> list[tuple[str, str]]:
    """(system, user) pairs from the MASK data CSVs of any local run's work tree."""
    rows = []
    for csv in sorted(glob.glob("output/mask/*/mask_work/data/*.csv")):
        if "/responses/" in csv:
            continue
        df = pd.read_csv(csv)
        if {"system_prompt", "user_prompt"} <= set(df.columns):
            rows += [(str(s), str(u)) for s, u in zip(df["system_prompt"], df["user_prompt"])
                     if isinstance(s, str) and isinstance(u, str)]
        if len(rows) > 5000:
            break
    assert len(rows) >= n, f"only {len(rows)} MASK prompts on disk; run a MASK eval first"
    random.Random(seed).shuffle(rows)
    return rows[:n]


async def drive(base_url: str, model: str, prompts: list[tuple[str, str]], cap: int,
                max_tokens: int, temperature: float) -> dict:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=base_url, api_key="x", timeout=3600, max_retries=0)
    sem = asyncio.Semaphore(cap)
    lat, toks, errors = [], [], 0

    async def one(i: int, system: str, user: str):
        nonlocal errors
        async with sem:
            t0 = time.time()
            try:
                r = await client.chat.completions.create(
                    model=model, messages=[{"role": "system", "content": system},
                                           {"role": "user", "content": user}],
                    max_tokens=max_tokens, temperature=temperature, seed=i)
                toks.append(r.usage.completion_tokens)
                lat.append(time.time() - t0)
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"   request {i} failed: {str(e)[:120]}", flush=True)

    t0 = time.time()
    await asyncio.gather(*(one(i, s, u) for i, (s, u) in enumerate(prompts)))
    wall = time.time() - t0
    await client.close()  # inside the loop: a GC'd AsyncOpenAI after asyncio.run() logs "Event loop is closed"
    return {"wall_s": round(wall, 1), "completed": len(toks), "errors": errors,
            "gen_tokens": sum(toks), "tok_per_s": round(sum(toks) / wall, 1),
            "tokens_per_request_median": statistics.median(toks) if toks else 0,
            "latency_s_median": round(statistics.median(lat), 1) if lat else 0,
            "latency_s_p90": round(sorted(lat)[int(0.9 * len(lat))], 1) if lat else 0}


def main(server: str, port: int, gpu: str, caps: str = "32,64,96,128", requests: int = 384,
         target: str = "dougalldeepmind/2026-09-22-qwen36-0-da-15", seed: int = 0,
         max_tokens: int = 16384, temperature: float = 1.0) -> None:
    mask_cfg = OmegaConf.load("configs/eval/mask.yaml")
    serving = OmegaConf.to_container(mask_cfg.serving, resolve=True)
    prompts = load_prompts(requests, seed)
    executor = V.SshExec(server, port=port)
    executor.check_ready()
    spec = V.resolve_target(target)
    model = spec.model_key if spec.adapter else "base"
    orig = V.serving_params
    out_path = Path("output/bench") / f"{timestamp()[:8]}_mask_serving_bench_{gpu.lower()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {"gpu": gpu, "server": "<pod>", "target": target, "requests": requests, "seed": seed,
               "max_tokens": max_tokens, "temperature": temperature, "serving": serving, "caps": {}}
    cap_list = ([int(c) for c in caps.split(",")] if isinstance(caps, str)
                else [int(c) for c in caps] if isinstance(caps, (list, tuple)) else [int(caps)])
    for cap in cap_list:
        V.serving_params = lambda m, _c=cap: {**orig(m), "max_num_seqs": _c}
        srv = V.VllmServer(work_dir=Path("output/bench") / f"server_{port}", port=port,
                           executor=executor, serve_requirements=serving)
        print(f">>> cap {cap}: starting the server ({gpu}) at {time.strftime('%H:%M:%S')}", flush=True)
        t0 = time.time()
        try:
            url = srv.serve(spec)
        except Exception as e:  # noqa: BLE001
            print(f"!!! cap {cap}: server failed to start: {str(e)[:300]}", flush=True)
            results["caps"][cap] = {"error": str(e)[:300]}
            try:
                srv.stop()
            except Exception:  # noqa: BLE001
                pass
            continue
        boot = round(time.time() - t0, 1)
        print(f">>> cap {cap}: up in {boot}s; driving {requests} MASK prompts at concurrency {cap}", flush=True)
        r = asyncio.run(drive(url, model, prompts, cap, max_tokens, temperature))
        r["boot_s"] = boot
        r["mask_run_generation_min"] = round(MASK_TOKENS_PER_RUN / max(r["tok_per_s"], 1) / 60, 1)
        r["mask_run_generation_usd"] = round(r["mask_run_generation_min"] / 60 * PRICE.get(gpu.upper(), 0), 2)
        results["caps"][cap] = r
        print(f">>> cap {cap}: {r['tok_per_s']} tok/s, {r['completed']}/{requests} ok, median latency "
              f"{r['latency_s_median']}s -> a MASK run's generation ~{r['mask_run_generation_min']} min "
              f"(${r['mask_run_generation_usd']})", flush=True)
        out_path.write_text(json.dumps(results, indent=2))
        srv.stop()
        time.sleep(10)
    V.serving_params = orig
    print(f">>> wrote {out_path}")
    print(json.dumps({c: {k: v[k] for k in ("tok_per_s", "mask_run_generation_min", "mask_run_generation_usd", "errors")
                          if k in v} for c, v in results["caps"].items()}, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
