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
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import requests

# The scheme that routes a --target here, and the base model the checkpoints are LoRAs of.
# gpt-oss-120b is the only family this shim's renderer covers (harmony); another family
# needs its own renderer choice in tinker_server.py, so it is refused rather than guessed.
TINKER_SCHEME = "tinker"
DEFAULT_BASE_MODEL = "openai/gpt-oss-120b"
SUPPORTED_BASE_MODELS = (DEFAULT_BASE_MODEL,)

# Tinker's own credential, read by the shim process (and sent by evals, which the shim
# ignores). Named here so a missing key fails before an eval starts rather than mid-run.
TINKER_KEY_ENV = "TINKER_API_KEY"

_READY_TIMEOUT_S = 600  # first request loads a tokenizer and opens a Tinker session
_PORT = int(os.environ.get("TINKER_SHIM_PORT", "1234"))


def sampler_name(ckpt: str) -> str:
    """The checkpoint's own name token, for run naming: the last path segment.

    A tinker path is `tinker://<run-id>:<phase>/sampler_weights/<name>`; the trailing name
    is the one part a human wrote (`gptoss120b-nosynth9771-lr1e4-3ep-sft-r32`), so it is
    what a run should be named after. The run id alone would name every arm `train:0`.
    """
    head, sep, tail = ckpt.rstrip("/").rpartition("/sampler_weights/")
    assert sep and head and tail, (
        f"tinker target {ckpt!r} has no `/sampler_weights/<name>` tail to name the run "
        "after — export the checkpoint for sampling first (save_weights_for_sampler)")
    return tail


def resolve_tinker_target(hf_path: str, *, base_model: str = DEFAULT_BASE_MODEL,
                          port: int = _PORT):
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
def tinker_shim(ckpt: str, *, base_model: str = DEFAULT_BASE_MODEL, port: int = _PORT,
                reasoning: str = "medium", max_tokens: int = 8192, log_dir: Path | None = None):
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
    log_dir = Path(log_dir or Path("output") / "tinker_shim")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"shim_{port}.log"
    env = {**os.environ, "TINKER_CKPT": ckpt, "TINKER_BASE_MODEL": base_model,
           "REASONING_LEVEL": reasoning, "PORT": str(port),
           "DEFAULT_MAX_TOKENS": str(max_tokens)}
    base_url = f"http://127.0.0.1:{port}/v1"
    print(f">>> tinker shim: {ckpt} (reasoning={reasoning}) on {base_url} | log {log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "-m", "src.infra.endpoints.tinker_server"],
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.time() + _READY_TIMEOUT_S
            while time.time() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"tinker shim exited with code {proc.returncode}; last log lines:\n"
                        + _tail(log_path))
                try:
                    if requests.get(f"{base_url}/models", timeout=5).status_code == 200:
                        break
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


def _tail(path: Path, lines: int = 40) -> str:
    """The last lines of the shim log, for an error message."""
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no log)"
