# ABOUTME: Serve an eval target via vLLM — on this machine or a remote GPU host over SSH —
# ABOUTME: resolving the HF target and pinning its thinking mode into the chat template.

from __future__ import annotations

import base64
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
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

# pgrep/pkill pattern for the server. The brackets keep the pattern from matching the
# pgrep/pkill command line itself over SSH (docs/LOG.md 2026-07-29: "bit us three times").
_SERVER_PATTERN = "vllm.entrypoints.openai.api_serve[r]"


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
    base_url: str         # http://localhost:<port>/v1 (tunnelled when serving remotely)
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
    `fetch_adapter` (adapter), on whichever machine serves.
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


def pin_template(template_text: str, mode: str) -> str:
    """Pin thinking mode into a chat template (pure; unit-tested offline).

    A top-level Jinja `set` executes after the render context is built, so it shadows any
    `enable_thinking` a client passes per request — requests cannot cross modes (gotcha 5).
    """
    assert mode in ("think", "nothink"), mode
    flag = "true" if mode == "think" else "false"
    return f"{{%- set enable_thinking = {flag} -%}}\n" + template_text


def _serve_params(base_model: str) -> dict:
    for prefix, params in _FAMILIES.items():
        if prefix and base_model.startswith(prefix):
            return params
    return _FAMILIES[None]


class LocalExec:
    """Run the vLLM server and its file operations on this machine."""

    python_argv = [sys.executable]

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.proc: subprocess.Popen | None = None

    def write_file(self, name: str, text: str) -> str:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        path = self.work_dir / name
        path.write_text(text)
        return str(path)

    def fetch_adapter(self, hf_path: str) -> str:
        from huggingface_hub import snapshot_download

        return snapshot_download(hf_path)

    def start_server(self, argv: list[str], env_extra: dict) -> None:
        import os

        self.work_dir.mkdir(parents=True, exist_ok=True)
        log = (self.work_dir / "vllm.log").open("a")
        log.write(f"\n=== {' '.join(argv)}\n")
        self.proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                     env=os.environ | env_extra)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def tail_log(self, n: int = 15) -> str:
        path = self.work_dir / "vllm.log"
        return "\n".join(path.read_text().splitlines()[-n:]) if path.exists() else "(no log)"

    def stop_server(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


class SshExec:
    """Run the vLLM server on a remote GPU host (prepared per the CLAUDE.md playbook:
    repo cloned + `uv sync`), with an owned SSH tunnel so the driver still talks to
    localhost. `bind` is the local tunnel address — 127.0.0.1 normally; a docker-bridge
    address (e.g. 172.17.0.1) when local containers must reach the endpoint (ODCV).
    """

    python_argv = ["uv", "run", "python"]

    def __init__(self, host: str, port: int, bind: str = "127.0.0.1",
                 workdir: str = "/root/work"):
        self.host = host
        self.port = port
        self.bind = bind
        self.workdir = workdir
        self.remote_dir = f"{workdir}/output/serve"
        self.tunnel: subprocess.Popen | None = None

    def _ssh(self, cmd: str, timeout: int = 240) -> str:
        r = subprocess.run(["ssh", self.host, cmd], capture_output=True, text=True,
                           timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"ssh {self.host} failed ({r.returncode}): "
                               f"{cmd[:120]} ...\n{r.stderr[-500:]}")
        return r.stdout

    def ensure_env(self, local_env: Path) -> None:
        """Provision the host's .env from the driver's, only when the host has none.

        An existing remote .env always wins (the pod may hold its own distinct keys).
        This is the playbook's manual `scp .env` step, automated — an explicit,
        SSH-encrypted copy at provisioning time, not run-time secret forwarding.
        """
        has_remote = self._ssh(
            f"[ -f {self.workdir}/.env ] && echo yes || echo no").strip() == "yes"
        if has_remote:
            return
        if not local_env.exists():
            print(f"!!! neither {self.host}:{self.workdir}/.env nor local {local_env} exists "
                  "— private HF repos and gated models will fail on the server")
            return
        subprocess.run(["scp", "-q", str(local_env), f"{self.host}:{self.workdir}/.env"],
                       check=True)
        print(f">>> provisioned {self.host}:{self.workdir}/.env from local {local_env}")

    def _with_env(self, cmd: str) -> str:
        """Prefix a remote command with the host's own .env (never the driver's).

        A fresh SSH shell sources nothing, so without this a remote snapshot_download or
        vLLM launch has no HF_TOKEN even when the pod is fully provisioned. Secrets stay
        machine-local by design: the driver's credentials are never transmitted.
        """
        return (f"set -a; [ -f {self.workdir}/.env ] && . {self.workdir}/.env; set +a; "
                + cmd)

    def write_file(self, name: str, text: str) -> str:
        payload = base64.b64encode(text.encode()).decode()
        path = f"{self.remote_dir}/{name}"
        self._ssh(f"mkdir -p {self.remote_dir} && echo {payload} | base64 -d > {shlex.quote(path)}")
        return path

    def fetch_adapter(self, hf_path: str) -> str:
        out = self._ssh(self._with_env(
            f"cd {self.workdir} && uv run python -c "
            f"\"from huggingface_hub import snapshot_download; "
            f"print(snapshot_download('{hf_path}'))\""), timeout=1800)
        return out.strip().splitlines()[-1]

    def start_server(self, argv: list[str], env_extra: dict) -> None:
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_extra.items())
        cmd = " ".join(shlex.quote(a) for a in argv)
        self._ssh(self._with_env(
            f"mkdir -p {self.remote_dir} && cd {self.workdir} && "
            f"nohup env {env} {cmd} >> {self.remote_dir}/vllm.log 2>&1 < /dev/null & "
            f"echo started"))
        self.tunnel = subprocess.Popen(
            ["ssh", "-N", "-L", f"{self.bind}:{self.port}:localhost:{self.port}", self.host])

    def alive(self) -> bool:
        try:
            return self._ssh(f"pgrep -f '{_SERVER_PATTERN}' >/dev/null && echo up || echo down"
                             ).strip().endswith("up")
        except RuntimeError:
            return False

    def tail_log(self, n: int = 15) -> str:
        try:
            return self._ssh(f"tail -n {n} {self.remote_dir}/vllm.log 2>/dev/null")
        except RuntimeError as e:
            return f"(could not read remote log: {e})"

    def stop_server(self) -> None:
        try:
            self._ssh(f"pkill -f '{_SERVER_PATTERN}' || true")
        except RuntimeError:
            pass
        if self.tunnel is not None:
            self.tunnel.terminate()
            self.tunnel = None


class VllmServer:
    """One vLLM OpenAI server — local subprocess or remote over SSH — restarted only when
    base model or mode changes. Consecutive targets sharing base+mode reuse the running
    server: a new adapter is attached with vLLM's runtime LoRA-load endpoint.
    """

    def __init__(self, work_dir: Path, port: int = 8000, executor=None):
        self.port = port
        self.executor = executor if executor is not None else LocalExec(work_dir)
        self.base_model: str | None = None
        self.mode: str | None = None
        self.running = False
        self._loaded_loras: set[str] = set()

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"

    def ensure(self, spec: TargetSpec) -> ServedTarget:
        """Serve `spec`, reusing the live server when base model + mode are unchanged."""
        adapter_dir = self.executor.fetch_adapter(spec.hf_path) if spec.adapter else None
        if not self.running or self.base_model != spec.base_model or self.mode != spec.mode:
            self.stop()
            self._start(spec, adapter_dir)
        elif spec.adapter and spec.model_key not in self._loaded_loras:
            assert adapter_dir is not None
            self._load_lora(spec, adapter_dir)
        model_name = spec.model_key if spec.adapter else "base"
        return ServedTarget(spec=spec, base_url=self.base_url, model_name=model_name)

    def _pinned_template_path(self, base_model: str, mode: str) -> str | None:
        if mode == "default":
            return None
        with open(hf_hub_download(base_model, "tokenizer_config.json")) as f:
            template = json.load(f)["chat_template"]
        return self.executor.write_file(f"chat_template_{mode}.jinja",
                                        pin_template(template, mode))

    def _start(self, spec: TargetSpec, adapter_dir: str | None) -> None:
        params = _serve_params(spec.base_model)
        argv = self.executor.python_argv + [
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", spec.base_model, "--served-model-name", "base",
            "--dtype", "bfloat16",
            "--max-model-len", str(params["max_model_len"]),
            "--gpu-memory-utilization", "0.94",
            "--port", str(self.port)]
        if params.get("max_num_seqs"):
            argv += ["--max-num-seqs", str(params["max_num_seqs"])]
        template = self._pinned_template_path(spec.base_model, spec.mode)
        if template:
            argv += ["--chat-template", template]
        if spec.adapter:
            argv += ["--enable-lora", "--max-lora-rank", str(max(spec.lora_rank or 32, 32)),
                     "--lora-modules", f"{spec.model_key}={adapter_dir}"]
        self.executor.start_server(argv, {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1"})
        self.base_model, self.mode, self.running = spec.base_model, spec.mode, True
        self._loaded_loras = {spec.model_key} if spec.adapter else set()
        self._wait_healthy()

    def _wait_healthy(self) -> None:
        deadline = time.time() + _HEALTH_TIMEOUT_S
        url = f"http://localhost:{self.port}/health"
        while time.time() < deadline:
            if not self.executor.alive():
                raise RuntimeError(f"vLLM exited; last log lines:\n{self.executor.tail_log()}")
            try:
                if requests.get(url, timeout=5).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(5)
        raise TimeoutError(f"vLLM not healthy after {_HEALTH_TIMEOUT_S}s; last log lines:\n"
                           f"{self.executor.tail_log()}")

    def _load_lora(self, spec: TargetSpec, adapter_dir: str) -> None:
        r = requests.post(f"{self.base_url}/load_lora_adapter",
                          json={"lora_name": spec.model_key, "lora_path": adapter_dir},
                          timeout=120)
        if r.status_code != 200:
            # Older vLLM or endpoint disabled: fall back to a cold restart with the adapter.
            self.stop()
            self._start(spec, adapter_dir)
            return
        self._loaded_loras.add(spec.model_key)

    def stop(self) -> None:
        self.executor.stop_server()
        self.base_model = self.mode = None
        self.running = False
        self._loaded_loras = set()
