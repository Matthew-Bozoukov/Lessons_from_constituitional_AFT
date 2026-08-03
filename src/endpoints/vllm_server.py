# ABOUTME: Serve an eval target on localhost via vLLM: resolve an HF target (LoRA adapter or
# ABOUTME: full model), pin its thinking mode into the chat template, manage the subprocess.

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError

# Serving parameters per base-model family. Matched by prefix against the base model id;
# the first hit wins, `None` is the fallback. max_model_len values are the ones the evals
# were tuned at (13312 = the original Qwen3-32B serving setup). Qwen3.6's hybrid
# Mamba/linear-attention arch requires a low max_num_seqs: the vLLM default (1024) exceeds
# the available Mamba cache blocks and fails at startup (docs/LOG.md 2026-07-29).
_FAMILIES: dict[str | None, dict] = {
    "Qwen/Qwen3-32B": {"max_model_len": 13312, "max_num_seqs": None},
    "Qwen/Qwen3.6-27B": {"max_model_len": 16384, "max_num_seqs": 32},
    None: {"max_model_len": 13312, "max_num_seqs": None},
}

_HEALTH_TIMEOUT_S = 1800  # first start downloads weights; a 32B pull can take a while


@dataclass(frozen=True)
class TargetSpec:
    """A resolved --target: what to serve and in which thinking mode."""

    hf_path: str          # as given on the CLI
    base_model: str       # HF id vLLM loads
    adapter: bool         # True when hf_path is a LoRA adapter repo
    mode: str             # think | nothink | default (full model: template's own default)
    model_key: str        # filesystem/served-name-safe identifier
    lora_rank: int | None


@dataclass(frozen=True)
class ServedTarget:
    """Handle an eval's run() receives: an OpenAI-compatible endpoint plus identity."""

    spec: TargetSpec
    base_url: str         # http://localhost:<port>/v1
    model_name: str       # the served model name to put in requests


def _mode_from_training_meta(meta: dict) -> str:
    assert "thinking" in meta, "training_meta.json must carry a boolean `thinking` field"
    return "think" if meta["thinking"] else "nothink"


def _spec_from_files(hf_path: str, adapter_config: dict | None, training_meta: dict | None) -> TargetSpec:
    """Build a TargetSpec from the artifact's metadata files (pure; unit-tested offline)."""
    model_key = hf_path.split("/")[-1].replace(".", "_")
    if adapter_config is None:
        return TargetSpec(hf_path=hf_path, base_model=hf_path, adapter=False,
                          mode="default", model_key=model_key, lora_rank=None)
    if training_meta is None:
        raise RuntimeError(
            f"{hf_path} is a LoRA adapter with no training_meta.json — the eval framework "
            "infers thinking mode from that stamp and never guesses. Backfill it from the "
            "arm's training config (see scratch/backfill_training_meta.py), then rerun.")
    return TargetSpec(
        hf_path=hf_path,
        base_model=adapter_config["base_model_name_or_path"],
        adapter=True,
        mode=_mode_from_training_meta(training_meta),
        model_key=model_key,
        lora_rank=int(adapter_config.get("r", 32)),
    )


def resolve_target(hf_path: str) -> TargetSpec:
    """Resolve an HF path (adapter or full model) into a TargetSpec.

    Only metadata files are downloaded here; weights are pulled by vLLM (base) and
    `fetch_adapter` (adapter).
    """
    try:
        with open(hf_hub_download(hf_path, "adapter_config.json")) as f:
            adapter_config = json.load(f)
    except EntryNotFoundError:
        return _spec_from_files(hf_path, None, None)
    try:
        with open(hf_hub_download(hf_path, "training_meta.json")) as f:
            training_meta = json.load(f)
    except EntryNotFoundError:
        training_meta = None
    return _spec_from_files(hf_path, adapter_config, training_meta)


def fetch_adapter(spec: TargetSpec) -> Path:
    """Download the adapter weights into the HF cache (reused across evals and targets)."""
    assert spec.adapter
    return Path(snapshot_download(spec.hf_path))


def pin_template(template_text: str, mode: str) -> str:
    """Pin thinking mode into a chat template (pure; unit-tested offline).

    A top-level Jinja `set` executes after the render context is built, so it shadows any
    `enable_thinking` a client passes per request — requests cannot cross modes (gotcha 5).
    """
    assert mode in ("think", "nothink"), mode
    flag = "true" if mode == "think" else "false"
    return f"{{%- set enable_thinking = {flag} -%}}\n" + template_text


def _write_pinned_template(base_model: str, mode: str, work_dir: Path) -> Path | None:
    """Fetch the base model's chat template and write the mode-pinned variant."""
    if mode == "default":
        return None
    with open(hf_hub_download(base_model, "tokenizer_config.json")) as f:
        template = json.load(f)["chat_template"]
    out = work_dir / f"chat_template_{mode}.jinja"
    out.write_text(pin_template(template, mode))
    return out


def _serve_params(base_model: str) -> dict:
    for prefix, params in _FAMILIES.items():
        if prefix and base_model.startswith(prefix):
            return params
    return _FAMILIES[None]


class VllmServer:
    """One vLLM OpenAI server, restarted only when base model or mode changes.

    Consecutive targets sharing base+mode reuse the running server: a new adapter is
    attached with vLLM's runtime LoRA-load endpoint instead of a cold restart.
    """

    def __init__(self, work_dir: Path, port: int = 8000):
        self.work_dir = work_dir
        self.port = port
        self.proc: subprocess.Popen | None = None
        self.base_model: str | None = None
        self.mode: str | None = None
        self._loaded_loras: set[str] = set()

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"

    def ensure(self, spec: TargetSpec) -> ServedTarget:
        """Serve `spec`, reusing the live server when base model + mode are unchanged."""
        adapter_dir = fetch_adapter(spec) if spec.adapter else None
        if self.proc is None or self.base_model != spec.base_model or self.mode != spec.mode:
            self.stop()
            self._start(spec, adapter_dir)
        elif spec.adapter and spec.model_key not in self._loaded_loras:
            assert adapter_dir is not None
            self._load_lora(spec, adapter_dir)
        model_name = spec.model_key if spec.adapter else "base"
        return ServedTarget(spec=spec, base_url=self.base_url, model_name=model_name)

    def _start(self, spec: TargetSpec, adapter_dir: Path | None) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        params = _serve_params(spec.base_model)
        args = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", spec.base_model, "--served-model-name", "base",
                "--dtype", "bfloat16",
                "--max-model-len", str(params["max_model_len"]),
                "--gpu-memory-utilization", "0.94",
                "--port", str(self.port)]
        if params.get("max_num_seqs"):
            args += ["--max-num-seqs", str(params["max_num_seqs"])]
        template = _write_pinned_template(spec.base_model, spec.mode, self.work_dir)
        if template:
            args += ["--chat-template", str(template)]
        if spec.adapter:
            args += ["--enable-lora", "--max-lora-rank", str(max(spec.lora_rank or 32, 32)),
                     "--lora-modules", f"{spec.model_key}={adapter_dir}"]
        env = os.environ | {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1"}
        log = (self.work_dir / "vllm.log").open("a")
        log.write(f"\n=== {' '.join(args)}\n")
        self.proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
        self.base_model, self.mode = spec.base_model, spec.mode
        self._loaded_loras = {spec.model_key} if spec.adapter else set()
        self._wait_healthy()

    def _wait_healthy(self) -> None:
        assert self.proc is not None
        deadline = time.time() + _HEALTH_TIMEOUT_S
        url = f"http://localhost:{self.port}/health"
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited (code {self.proc.returncode}); "
                                   f"see {self.work_dir / 'vllm.log'}")
            try:
                if requests.get(url, timeout=5).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(5)
        raise TimeoutError(f"vLLM not healthy after {_HEALTH_TIMEOUT_S}s; "
                           f"see {self.work_dir / 'vllm.log'}")

    def _load_lora(self, spec: TargetSpec, adapter_dir: Path) -> None:
        r = requests.post(f"{self.base_url}/load_lora_adapter",
                          json={"lora_name": spec.model_key, "lora_path": str(adapter_dir)},
                          timeout=120)
        if r.status_code != 200:
            # Older vLLM or endpoint disabled: fall back to a cold restart with the adapter.
            self.stop()
            self._start(spec, adapter_dir)
            return
        self._loaded_loras.add(spec.model_key)

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        self.base_model = self.mode = None
        self._loaded_loras = set()
