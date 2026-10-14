# ABOUTME: Campaign-only admission guards and zero-resampling Tinker bridge for MMLU.
# ABOUTME: Reuses the production Harmony renderer, HTTP shim, and conservative ledger.
import asyncio
import json
import os
from pathlib import Path
import time
import uuid
from types import SimpleNamespace
from src.infra.endpoints.tinker_budget import Budget

ROOT = Path("/work/output/gptoss_mmlu_20261014")
RATES = dict(
    input_per_million=0.33, cached_input_per_million=0.066, output_per_million=0.84
)


def resources_ok():
    data = json.loads((ROOT / "host-resources.json").read_text())
    return (
        time.time() - data["at"] < 60
        and data["available_ram"] >= 2 * 2**30
        and data["disk_free"] >= 20 * 2**30
        and working_memory() < 3.5 * 2**30
    )


def working_memory():
    stats = dict(
        line.split()
        for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines()
    )
    return int(Path("/sys/fs/cgroup/memory.current").read_text()) - int(
        stats["inactive_file"]
    )


class CampaignBudget(Budget):
    def reserve(self, prompt, allowance):
        phase = os.environ["MMLU_PHASE"]
        while True:
            if phase != "load":
                if (ROOT / "admission-hold.json").exists():
                    raise RuntimeError(
                        "Campaign admission held; retain unknown paid requests"
                    )
                if not resources_ok():
                    time.sleep(2)
                    continue
            with self.edit():
                upper = (
                    prompt * self.data["input_per_million"]
                    + allowance * self.data["output_per_million"]
                ) / 1e6
                rows = list(self.data["requests"].values())
                used = sum(r["upper_usd"] for r in rows)
                stage = sum(r["upper_usd"] for r in rows if r.get("phase") == "smoke")
                if phase == "smoke" and stage + upper > 0.49:
                    raise RuntimeError("Cumulative smoke ceiling reached")
                if used + upper <= self.data["ceiling_usd"]:
                    key = uuid.uuid4().hex
                    self.data["requests"][key] = dict(
                        prompt_tokens=prompt,
                        max_tokens=allowance,
                        upper_usd=upper,
                        state="reserved",
                        at=time.time(),
                        checkpoint=self.checkpoint,
                        phase=phase,
                    )
                    self.save()
                    return key
                pending = any(r["state"] == "reserved" for r in rows)
            if not pending:
                raise RuntimeError(
                    "Aggregate ceiling reached with no unsettled requests"
                )
            time.sleep(2)


class MockSampler:
    def __init__(self, renderer, arm):
        self.arm, self.active, self.peak = arm, 0, 0
        prefix = renderer.encode("<|channel|>analysis<|message|>")
        suffix = renderer.encode(
            "<|end|><|start|>assistant<|channel|>final<|message|>A<|return|>"
        )
        word = renderer.encode(" reasoning")[0]
        self.tokens = prefix + [word] * (32768 - len(prefix) - len(suffix)) + suffix

    async def sample_async(self, **kwargs):
        self.active += 1
        self.peak = max(self.peak, self.active)
        (ROOT / f"load-{self.arm}-active.json").write_text(
            json.dumps(dict(active=self.active, peak=self.peak))
        )
        deadline = time.monotonic() + 180
        while not (ROOT / "load-release").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Mock concurrency barrier timed out")
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.1)
        self.active -= 1
        return SimpleNamespace(
            sequences=[SimpleNamespace(tokens=self.tokens, stop_reason="stop")],
            prompt_cache_hit_tokens=0,
        )


class GuardedSampler:
    def __init__(self, client):
        self.client = client

    async def sample_async(self, **kwargs):
        try:
            return await self.client.sample_async(**kwargs)
        except BaseException as e:
            path = ROOT / "admission-hold.json"
            try:
                with path.open("x") as f:
                    json.dump(dict(at=time.time(), reason=type(e).__name__), f)
            except FileExistsError:
                pass
            raise


def main():
    import tinker
    import uvicorn
    from tinker.lib.retry_handler import RetryConfig
    from src.infra.endpoints.tinker_server import Runtime, create_app
    from src.infra.endpoints.tinker_harmony import HarmonyRenderer

    phase, arm = os.environ["MMLU_PHASE"], os.environ["MMLU_ARM"]
    targets = json.loads((ROOT / "targets.json").read_text())
    checkpoint = targets[arm]
    renderer = HarmonyRenderer(reasoning_effort="medium", current_date="2026-10-14")
    if phase == "load":
        sampler = MockSampler(renderer, arm)
    else:
        service = tinker.ServiceClient()
        client = service.create_sampling_client(
            model_path=None if arm == "base" else checkpoint,
            base_model="openai/gpt-oss-120b:peft:131072",
            retry_config=RetryConfig(enable_retry_logic=False, progress_timeout=4800),
        )
        # Retrieval retries poll the SAME future; never redispatch an ambiguous paid POST.
        original = client.holder.execute_with_retries

        async def guarded_execute(func, *args, **kwargs):
            if getattr(func, "__name__", "") == "_send_asample_request":
                return await func(*args, **kwargs)
            return await original(func, *args, **kwargs)

        client.holder.execute_with_retries = guarded_execute
        sampler = GuardedSampler(client)
    budget = CampaignBudget(
        ROOT / ("load-budget.json" if phase == "load" else "budget.json"),
        15,
        "openai/gpt-oss-120b:peft:131072",
        checkpoint,
        list(targets.values()),
        RATES,
    )
    runtime = Runtime(
        checkpoint=checkpoint,
        model="openai/gpt-oss-120b",
        reasoning="medium",
        renderer=renderer,
        sampling_client=sampler,
        sampling_params=tinker.SamplingParams,
        tool_call_type=None,
        convert_tools=None,
        api_key=os.environ["TINKER_API_KEY"],
        instance_id=os.environ["MMLU_INSTANCE"],
        default_max_tokens=32768,
        context_window=131072,
        sampling_model="openai/gpt-oss-120b:peft:131072",
        budget=budget,
        trace_dir=str(ROOT / phase / arm / "traces"),
    )
    uvicorn.run(
        create_app(runtime),
        host="127.0.0.1",
        port=int(os.environ["PORT"]),
        log_level="warning",
    )


if __name__ == "__main__":
    main()
