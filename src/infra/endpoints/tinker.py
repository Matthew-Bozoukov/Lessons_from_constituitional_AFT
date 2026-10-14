# ABOUTME: The `tinker:` target — resolve a Tinker checkpoint into a TargetSpec and own the
# ABOUTME: local OpenAI-compatible shim's lifetime, so evals reach Tinker like any endpoint.

"""Serving half of a Tinker target.

`--target tinker://<path>` names a Tinker sampler checkpoint instead of an HF repo. Tinker
holds the weights and does the sampling; nothing is downloaded, no GPU is rented, and vLLM
is not involved at all. What the eval receives is the same OpenAI triple it gets for every
other target, pointed at a shim this module starts on localhost
(`src.infra.endpoints.tinker_server`, which does the harmony render/parse round trip).

The shim is a SUBPROCESS rather than a thread: it loads a tokenizer and a renderer and
talks to Tinker's service over its own event loop, and a crash inside it must not take the
eval driver's transcripts with it. Its lifetime is a context manager — the port is bound
before the first request and released after the last one, so two invocations on one machine
can each hold their own.

Deliberately NOT supported here:
  - vLLM anything. A tinker target is served by Tinker; `VllmServer` is never constructed
    for one, and the eval's `serving:` block (a vLLM launch plan) does not apply.
  - LoRA swapping between arms. One shim serves one checkpoint; an arm ladder restarts it,
    which costs a tokenizer load and nothing else.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
import uuid
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import requests

# The scheme that routes a --target here, and the base model the checkpoints are LoRAs of.
# gpt-oss-120b is the only family this shim's renderer covers (harmony); another family
# needs its own renderer choice in tinker_server.py, so it is refused rather than guessed.
TINKER_SCHEME = "tinker"
DEFAULT_BASE_MODEL = "openai/gpt-oss-120b"
SUPPORTED_BASE_MODELS = (DEFAULT_BASE_MODEL,)

# Tinker's credential is read by its SDK and required by the loopback shim as a bearer
# token. Named here so a missing key fails before an eval starts rather than mid-run.
TINKER_KEY_ENV = "TINKER_API_KEY"

_READY_TIMEOUT_S = 600  # first request loads a tokenizer and opens a Tinker session
SHIM_ENV = Path(__file__).resolve().parent / "tinker_env"
REPO_ROOT = Path(__file__).resolve().parents[3]


def available_port() -> int:
    """Select a port per target; startup identity checking closes the bind race safely."""
    configured = os.environ.get("TINKER_SHIM_PORT")
    if configured:
        return int(configured)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def sampler_name(ckpt: str) -> str:
    """The checkpoint's own name token, for run naming: the last path segment.

    A tinker path is `tinker://<run-id>:<phase>/sampler_weights/<name>`; the trailing name
    is the one part a human wrote (`gptoss120b-nosynth9771-lr1e4-3ep-sft-r32`), so it is
    what a run should be named after. The run id alone would name every arm `train:0`.
    """
    if ckpt == "tinker://base":
        return "gptoss120b-base"
    head, sep, tail = ckpt.rstrip("/").rpartition("/sampler_weights/")
    assert sep and head and tail, (
        f"tinker target {ckpt!r} has no `/sampler_weights/<name>` tail to name the run "
        "after — export the checkpoint for sampling first (save_weights_for_sampler)")
    return tail


def resolve_tinker_target(hf_path: str, *, base_model: str = DEFAULT_BASE_MODEL,
                          port: int | None = None):
    """Return a TargetSpec for a `tinker://…` checkpoint served by the local shim.

    Args:
        hf_path: The target as typed, `tinker://<run>:<phase>/sampler_weights/<name>`.
        base_model: The family the checkpoint adapts; only gpt-oss-120b is rendered.
        port: Localhost port the shim will bind.

    Raises:
        ValueError: A base model whose harmony renderer this shim does not have.
    """
    from src.infra.endpoints.vllm import TargetSpec  # local: vllm.py imports this module
    from src.naming import _sanitize

    if base_model not in SUPPORTED_BASE_MODELS:
        raise ValueError(
            f"tinker target {hf_path!r} names base model {base_model!r}; this shim renders "
            f"only {', '.join(SUPPORTED_BASE_MODELS)} (harmony). Add its renderer to "
            "src/infra/endpoints/tinker_server.py before evaluating it.")
    port = available_port() if port is None else port
    return TargetSpec(
        hf_path=hf_path, base_model=base_model, adapter=False,
        # A sampler checkpoint carries no training stamp this repo can read, and the shim
        # bakes the reasoning level into the prompt rather than a chat template, so the
        # mode is a LABEL here exactly as it is for an API target.
        mode="default",
        model_key=_sanitize(f"tinker {sampler_name(hf_path)}"),
        lora_rank=None,
        api_base=f"http://127.0.0.1:{port}/v1",
        api_key_env=TINKER_KEY_ENV)


def is_tinker_target(hf_path: str) -> bool:
    """True when `hf_path` names a Tinker checkpoint rather than an HF repo or API model."""
    return str(hf_path).startswith(f"{TINKER_SCHEME}://")


@contextmanager
def tinker_shim(ckpt: str, *, base_model: str = DEFAULT_BASE_MODEL, port: int | None = None,
                reasoning: str = "medium", max_tokens: int = 8192, log_dir: Path | None = None,
                context_window: int = 131072, render_date: str | None = None,
                budget_usd: float | None = None, budget_ledger: str | None = None,
                budget_checkpoints: list[str] | None = None):
    """Run the OpenAI-compatible shim for `ckpt` for the duration of the block.

    Args:
        ckpt: The `tinker://…` sampler checkpoint to serve.
        base_model: The family the checkpoint adapts.
        port: Localhost port to bind.
        reasoning: Reasoning effort baked into the prompt (low | medium | high).
        max_tokens: Default completion budget when a caller sends none.
        log_dir: Where to tee the shim's stdout/stderr; its own dir when None.

    Yields:
        The OpenAI-compatible base URL (`http://127.0.0.1:<port>/v1`).

    Raises:
        AssertionError: TINKER_API_KEY unset.
        RuntimeError: The shim died, or never answered /v1/models within the timeout. Its
            log tail is included — a shim that cannot serve must not be discovered one
            rollout at a time.
    """
    assert os.environ.get(TINKER_KEY_ENV), (
        f"a tinker target needs {TINKER_KEY_ENV} in the environment (.env) — it is unset")
    if base_model not in SUPPORTED_BASE_MODELS or reasoning not in {"low", "medium", "high"}:
        raise ValueError("Unsupported Tinker base model or reasoning effort")
    if not 0 < max_tokens <= context_window <= 131072:
        raise ValueError("Tinker budgets require 0 < max_tokens <= context_window <= 131072")
    port = available_port() if port is None else port
    instance_id = uuid.uuid4().hex
    log_dir = Path(log_dir or Path("output") / "tinker_shim")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"shim_{port}.log"
    env = {**os.environ, "TINKER_CKPT": ckpt, "TINKER_BASE_MODEL": base_model,
           "REASONING_LEVEL": reasoning, "PORT": str(port),
           "DEFAULT_MAX_TOKENS": str(max_tokens), "TINKER_SHIM_INSTANCE": instance_id,
           "TINKER_CONTEXT_WINDOW": str(context_window),
           "TINKER_RENDER_DATE": render_date or date.today().isoformat()}
    if budget_usd is not None:
        if not budget_ledger or budget_usd <= 0:
            raise ValueError("A positive Tinker budget requires a durable ledger")
        env.update(TINKER_BUDGET_USD=str(budget_usd), TINKER_BUDGET_LEDGER=str(budget_ledger))
        if budget_checkpoints is not None:
            import json
            env['TINKER_BUDGET_CHECKPOINTS'] = json.dumps(list(budget_checkpoints))
    base_url = f"http://127.0.0.1:{port}/v1"
    print(f">>> tinker shim: {ckpt} (reasoning={reasoning}) on {base_url} | log {log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        # Running from repo root exposes src.* without installing the root GPU stack.
        subprocess.run(["uv", "sync", "--project", str(SHIM_ENV), "--frozen", "--no-dev"],
                       cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        python = SHIM_ENV / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        # Own the Python process directly. Terminating a Windows uv parent can leave its
        # server child alive, which would collide with the next checkpoint on this port.
        proc = subprocess.Popen([str(python), "-m", "src.infra.endpoints.tinker_server"],
                                cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.time() + _READY_TIMEOUT_S
            while time.time() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"tinker shim exited with code {proc.returncode}; last log lines:\n"
                        + _tail(log_path))
                try:
                    response = requests.get(f"{base_url}/models", timeout=5,
                                            headers={"Authorization": f"Bearer {env[TINKER_KEY_ENV]}"})
                    if response.status_code == 200:
                        entries = response.json().get("data", [])
                        if entries != [] and all(entries[0].get(k) == v for k, v in {
                            "checkpoint": ckpt, "id": base_model, "instance_id": instance_id,
                            "reasoning_effort": reasoning, "context_window": context_window,
                        }.items()):
                            break
                        raise RuntimeError(f"Tinker shim port {port} answered with another server identity")
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                raise RuntimeError(
                    f"tinker shim did not answer {base_url}/models within "
                    f"{_READY_TIMEOUT_S}s; last log lines:\n" + _tail(log_path))
            yield base_url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def _tail(path: Path, lines: int = 40) -> str:
    """The last lines of the shim log, for an error message."""
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no log)"
