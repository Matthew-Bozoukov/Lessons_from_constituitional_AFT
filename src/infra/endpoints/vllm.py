# ABOUTME: Serve an eval target via vLLM — on this machine or a remote GPU host over SSH —
# ABOUTME: resolving the HF target and pinning its thinking mode into the chat template.

from __future__ import annotations

import base64
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from src.huggingface import hf_download
from huggingface_hub.errors import EntryNotFoundError

from src.model_profile import serving_params

# Serving parameters come from two places with different epistemic status (merged in
# _start): the FAMILY's verified facts (ModelProfile.serving, src/model_profile.py — reasoning
# parser, max_num_seqs constraint, verified_context_window ceiling; unprofiled families
# get utils.DEFAULT_SERVING) and the EVAL's own required `serving.context_window` (its
# config's declaration of the window it runs at — the window decides truncation
# behaviour, so it is part of the eval's scientific record, never a hidden default).

_HEALTH_TIMEOUT_S = 1800  # first start downloads weights; a 32B pull can take a while

# pgrep/pkill pattern for the server. The brackets keep the pattern from matching the
# pgrep/pkill command line itself over SSH (docs/LOG.md 2026-07-29: "bit us three times").
_SERVER_PATTERN = "vllm.entrypoints.openai.api_serve[r]"


# A --target may be an HF path (served locally by vLLM) OR an external API endpoint,
# written `<provider>:<model-id>` on the CLI — e.g. `openrouter:moonshotai/kimi-k2`. HF
# repo ids never contain a colon, so the scheme is unambiguous. Each provider maps to an
# OpenAI-compatible base URL and the env var holding its key (loaded from .env, never a
# config field — secrets stay out of the scientific record). Add a row to serve a new one.
API_PROVIDERS: dict[str, tuple[str, str]] = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}


@dataclass(frozen=True)
class TargetSpec:
    """A resolved --target: what to serve and in which thinking mode.

    An API target (`api_base` set) is not served by vLLM — it names a public endpoint,
    for comparing our models against off-the-shelf ones. Its `mode` is a comparison LABEL
    only: we don't control the provider's template, so nothing is pinned (unlike a served
    arm, whose mode is pinned into the chat template).
    """

    hf_path: str          # as given on the CLI (HF id, or `<provider>:<model-id>`)
    base_model: str       # HF id vLLM loads; for an API target, the provider's model id
    adapter: bool         # True when hf_path is a LoRA adapter repo
    mode: str             # think | nothink | default (full model: template's own default)
    model_key: str        # filesystem/served-name-safe identifier
    lora_rank: int | None
    api_base: str | None = None      # OpenAI-compatible base URL; None => served by vLLM
    api_key_env: str | None = None   # env var holding the key for api_base


class ServedTarget:
    """Handle an eval's run() receives: identity now, an endpoint only on first use.

    Serving is LAZY: `spec` and `model_name` are plain attributes, but the vLLM server
    boots (or LoRA-swaps) on first `base_url` access. An arm whose generation is fully
    satisfied by the HF answer cache therefore never starts a server at all.
    """

    def __init__(self, spec: TargetSpec, server: "VllmServer"):
        self.spec = spec
        # The id sent in the request body: the provider's model id for an API target,
        # else vLLM's --served-model-name ("base", or the adapter's key after a LoRA swap).
        self.model_name = spec.base_model if spec.api_base \
            else (spec.model_key if spec.adapter else "base")
        self._server = server

    @property
    def is_api(self) -> bool:
        """True when this target is a public API endpoint, not served by vLLM."""
        return self.spec.api_base is not None

    @property
    def base_url(self) -> str:
        """OpenAI-compatible base URL. For an API target, the provider's — no server
        boots. For an HF target, http://localhost:<port>/v1 (tunnelled when remote),
        booted on demand."""
        if self.spec.api_base is not None:
            return self.spec.api_base
        return self._server.serve(self.spec)

    @property
    def api_key(self) -> str:
        """The key evals send with each request: the provider's key from the environment
        for an API target, else "EMPTY" (local vLLM accepts any). Owned by the target,
        never read from a config — secrets stay out of the scientific record."""
        if not self.is_api:
            return "EMPTY"
        key = os.environ.get(self.spec.api_key_env or "")
        assert key, (f"API target {self.spec.hf_path} needs {self.spec.api_key_env} in "
                     "the environment (.env) — it is unset")
        return key


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


def resolve_api_target(provider: str, model_id: str) -> TargetSpec:
    """Build a TargetSpec for a `<provider>:<model-id>` API endpoint (pure; unit-tested).

    No metadata is fetched — a public model has no artifact. Mode defaults to "default"
    (a label; the provider's template is not ours to pin) and takes the `mode=` override
    like a full model. The key is disambiguated by provider so two providers serving the
    same model id land in distinct output dirs.
    """
    base, key_env = API_PROVIDERS[provider]
    return TargetSpec(
        hf_path=f"{provider}:{model_id}", base_model=model_id, adapter=False,
        mode="default",
        model_key=f"{provider}_{model_id.split('/')[-1]}".replace(".", "_"),
        lora_rank=None, api_base=base, api_key_env=key_env)


def resolve_target(hf_path: str) -> TargetSpec:
    """Resolve a --target into a TargetSpec.

    Two forms: `<provider>:<model-id>` (an API endpoint, see API_PROVIDERS) or an HF path
    (adapter or full model). HF repo ids never contain a colon, so the scheme is
    unambiguous. For HF paths only metadata files are downloaded here; weights are pulled
    by vLLM (base) and `fetch_adapter` (adapter), on whichever machine serves.
    """
    scheme, sep, rest = hf_path.partition(":")
    if sep and scheme in API_PROVIDERS:
        assert rest, f"API target {hf_path!r} names no model (expected {scheme}:<model-id>)"
        return resolve_api_target(scheme, rest)
    if sep and "/" not in scheme:
        # A colon with an unknown scheme is a typo'd provider, not an HF id — fail loud
        # rather than trying to fetch "openroute:foo/bar" as a repo.
        raise ValueError(
            f"unknown API provider {scheme!r} in target {hf_path!r} "
            f"(known: {', '.join(sorted(API_PROVIDERS))}); an HF path has no scheme.")
    try:
        with open(hf_download(hf_path, "adapter_config.json")) as f:
            adapter_config = json.load(f)
    except EntryNotFoundError:
        return _spec_from_files(hf_path, None, None)
    try:
        with open(hf_download(hf_path, "training_meta.json")) as f:
            training_meta = json.load(f)
    except EntryNotFoundError:
        training_meta = None
    return _spec_from_files(hf_path, adapter_config, training_meta)


def pin_template(template_text: str, mode: str) -> str:
    """Pin thinking mode into a chat template (pure; unit-tested offline).

    A top-level Jinja `set` executes after the render context is built, so it shadows any
    `enable_thinking` a client passes per request — requests cannot cross modes (gotcha 5).

    Thinking mode also pins `preserve_thinking = true` (the repo-wide policy since
    2026-08-04): training data carries reasoning on every assistant turn, so inference
    context must too — prior-turn `reasoning_content` sent back by a client is kept in
    the render rather than stripped by the template's default. Nothink pins it false:
    a nothink arm's history carries no reasoning to preserve.
    """
    return pin_prefix(mode) + template_text


def pin_prefix(mode: str) -> str:
    """The two Jinja lines `pin_template` prepends. Depends on `mode` ALONE, not the template.

    Split out because the RunPod bootstrap pins the template ON the pod, where the text is
    only available at boot from the tokenizer -- but the prefix is decidable here. One
    definition, so the two serving paths cannot drift (they did: src/infra/runpod.py used to
    rebuild these two lines by hand, with a comment promising they matched).
    """
    assert mode in ("think", "nothink"), mode
    flag = "true" if mode == "think" else "false"
    return (f"{{%- set enable_thinking = {flag} -%}}\n"
            f"{{%- set preserve_thinking = {flag} -%}}\n")


# The two serving namespaces are DISJOINT BY CONSTRUCTION — no key appears in both, so
# "override" is not a concept this module can express. Family FACTS (ModelProfile.serving,
# src/model_profile.py) say what a model family IS and what it has been measured to do; an eval's
# `serving:` block declares only what that eval NEEDS. Requirements are validated against
# facts here, never merged over them: a merge lets a config forge the very ceiling it is
# checked against, swallows typo'd keys in silence, and makes "what actually served?"
# a question about dict ordering.
_FAMILY_FACT_KEYS = {"native_context_window", "max_num_seqs", "reasoning_parser",
                     "tool_call_parser", "supports_prefix_caching"}


def native_context_window(base_model: str) -> int | None:
    """The window `base_model` was trained at, read from its own config.json.

    `max_position_embeddings` is the number of positions the weights have embeddings
    for — the one hard window limit, and a property of the model rather than of our
    deployment. Read it rather than hand-copying it into a profile: a transcribed
    constant is a fact nobody re-checks, and it is wrong for every family we have not
    thought about yet. (This is also what vLLM derives its own default from, so a
    request above it fails at startup anyway — catching it here just makes the error
    legible before weights are pulled.)

    Returns None when the field is absent or the config cannot be fetched, in which
    case plan_serving imposes no window limit and vLLM's own startup check is the
    backstop.
    """
    try:
        with open(hf_download(base_model, "config.json")) as f:
            config = json.load(f)
    except Exception:            # offline, gated repo, unusual layout — not fatal
        return None
    window = config.get("max_position_embeddings")
    if window is None:           # some multimodal configs nest it under the text tower
        window = (config.get("text_config") or {}).get("max_position_embeddings")
    return int(window) if window else None
_EVAL_REQUIREMENT_KEYS = {"context_window", "concurrency", "needs_tool_calls",
                          "reuses_long_prefixes"}


def plan_serving(facts: dict, requirements: dict, base_model: str, mode: str) -> dict:
    """Compose a vLLM launch plan from family facts and one eval's requirements.

    Pure (unit-tested offline), and total: every argv decision is made here, so
    VllmServer._start only translates the returned plan into flags. `mode` is an input
    because it decides flags — the reasoning parser is emitted think-mode-only — even
    though it is neither a fact nor a requirement: it is a property of the artifact
    being served (CLAUDE.md, "The eval framework").

    Only a shortfall that would CORRUPT THE MEASUREMENT is fatal — an agentic eval served
    without a tool-call parser scores 0 for a serving reason indistinguishable from
    incapability. Everything else is reported in `warnings` and the run proceeds, in two
    flavours: a request the family definitively cannot honour (prefix caching on an arch
    vLLM forces it off for), and a request we simply have not verified yet (a window above
    the high-water mark). The second is deliberately NOT fatal: refusing there would be
    refusing on absence of evidence, which is the same forgery this split exists to
    prevent, pointed the other way. vLLM's startup failure is the backstop.

    Args:
        facts: The family's verified serving facts (`ModelProfile.serving`).
        requirements: The eval config's `serving:` block.
        base_model: HF id, for error messages only.
        mode: think | nothink | default — the artifact's inferred thinking mode.

    Returns:
        The launch plan: `context_window`, `max_num_seqs`, `reasoning_parser`,
        `tool_call_parser` (each None when not to be emitted), `prefix_caching`, and
        `warnings` — operator-facing notes to print at serve time.

    Raises:
        SystemExit: Unknown key on either side, missing context_window, concurrency
            above the family's verified cap (a real boot constraint — Mamba slots are
            preallocated), or tool calls required from a family with no verified parser.
    """
    unknown_facts = set(facts) - _FAMILY_FACT_KEYS
    if unknown_facts:
        raise SystemExit(
            f"\nunknown key(s) in {base_model}'s ModelProfile.serving: "
            f"{sorted(unknown_facts)}. Family facts are a closed set "
            f"({sorted(_FAMILY_FACT_KEYS)}) — an eval's needs belong in its config's "
            "serving: block, not in src/model_profile.py.")
    unknown = set(requirements) - _EVAL_REQUIREMENT_KEYS
    if unknown:
        raise SystemExit(
            f"\nunknown key(s) in this eval's serving: block: {sorted(unknown)}. "
            f"Eval configs declare requirements only ({sorted(_EVAL_REQUIREMENT_KEYS)}); "
            "family facts (verified ceilings, parser names) live in "
            "ModelProfile.serving, src/model_profile.py — an eval cannot set them.")
    window = requirements.get("context_window")
    if not window:
        raise SystemExit(
            "\nthis eval's config declares no serving.context_window — every eval "
            "states the window it runs at (required, no default: the window decides "
            "truncation behaviour, so it is part of the eval's scientific record). "
            "Add a `serving:` section to its configs/eval YAML.")
    # The only hard window limit is the model's TRAINED window: past it there are no
    # trained positions to attend to, and no amount of GPU fixes that. Everything
    # between "what we have booted" and native is a KV-cache question about this
    # particular card, which vLLM answers at startup far more reliably than a table
    # here can — so it is not this function's to refuse. (This check previously used a
    # high-water mark of the largest window we happened to have booted, which refused
    # legitimate requests on absence of evidence.)
    native = facts.get("native_context_window")
    if native and int(window) > int(native):
        raise SystemExit(
            f"\nserving.context_window={window} exceeds {base_model}'s native window "
            f"({native} — ModelProfile.serving, src/model_profile.py): the weights have no "
            "trained positions beyond it, so serving there needs explicit rope scaling "
            "and is a deliberate experiment, not a config bump. Lower the eval's window.")
    # The family value is a boot-feasibility CAP (Mamba state slots are preallocated at
    # startup), not a default an eval may exceed. An eval requests `concurrency` — a
    # different key on purpose, so the cap cannot be shadowed even by accident — and may
    # ask for fewer slots (psychosis trades slots for window headroom), never more.
    cap = facts.get("max_num_seqs")
    seqs = requirements.get("concurrency", cap)
    if seqs and cap and int(seqs) > int(cap):
        raise SystemExit(
            f"\nserving.concurrency={seqs} exceeds {base_model}'s verified cap "
            f"({cap} — ModelProfile.serving, src/model_profile.py): Mamba state slots are "
            "preallocated at boot and the arena above the cap does not fit the "
            "reference H100. Request fewer, or verify a larger cap with a live boot.")

    # Tool calls: the EVAL knows it drives a tool, the FAMILY knows which parser reads
    # the syntax its template emits. Neither half is guessable from the other, and the
    # wrong parser is silent — Qwen3.6 emits XML, so `hermes` would have parsed nothing
    # and scored a clean 0% (docs/LOG.md 2026-07-29). No verified parser, no run.
    tool_call_parser = None
    if requirements.get("needs_tool_calls"):
        tool_call_parser = facts.get("tool_call_parser")
        if not tool_call_parser:
            raise SystemExit(
                f"\nthis eval declares serving.needs_tool_calls, but {base_model} has no "
                "verified tool_call_parser (ModelProfile.serving, src/model_profile.py). Serving "
                "it anyway returns tool calls as raw text and every task scores 0 for a "
                "reason indistinguishable from incapability. Verify which of vLLM's "
                "parsers matches this family's chat template, then add it as a fact.")

    # Prefix caching costs throughput, not correctness, so an impossible request is
    # reported rather than fatal: on Qwen3.6 vLLM forces it off regardless (Mamba state
    # pages cannot be reused like attention KV, docs/LOG.md 2026-07-29), so passing the
    # flag would be a no-op dressed up as a setting.
    warnings = []
    prefix_caching = bool(requirements.get("reuses_long_prefixes"))
    if prefix_caching and not facts.get("supports_prefix_caching"):
        prefix_caching = False
        warnings.append(
            f"reuses_long_prefixes: {base_model} cannot cache prefixes "
            "(supports_prefix_caching is false), so each step re-prefills its whole "
            "context. Throughput only — outputs are unaffected.")

    # Think-mode only, by construction: on a tagless (nothink) stream the parser's
    # "reasoning is at the start" assumption would route the WHOLE answer into the
    # reasoning field. mode=default (full models) also skips it and falls back to
    # client-side splitting — see docs/TODO.md.
    reasoning_parser = facts.get("reasoning_parser") if mode == "think" else None

    return {"context_window": int(window),
            "max_num_seqs": int(seqs) if seqs else None,
            "reasoning_parser": reasoning_parser,
            "tool_call_parser": tool_call_parser,
            "prefix_caching": prefix_caching,
            "warnings": tuple(warnings)}


class LocalExec:
    """Run the vLLM server and its file operations on this machine."""

    python_argv = [sys.executable]

    # Where the driver reaches the endpoint. Loopback here; SshExec overrides it
    # with the tunnel's bind address.
    endpoint_host = "127.0.0.1"

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.proc: subprocess.Popen | None = None

    def write_file(self, name: str, text: str) -> str:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        path = self.work_dir / name
        path.write_text(text)
        return str(path)

    def fetch_adapter(self, hf_path: str) -> str:
        from src.huggingface import hf_snapshot

        return hf_snapshot(hf_path)

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
        # The tunnel binds to `bind`, so that - not "localhost" - is where the
        # endpoint actually answers. With bind=172.17.0.1 (the docker bridge, so
        # ODCV containers can reach it) the tunnel does NOT listen on loopback,
        # and a hardcoded localhost health probe times out against a server that
        # is up and serving. Cost a full ODCV run to find.
        self.endpoint_host = bind
        self.workdir = workdir
        self.remote_dir = f"{workdir}/output/serve"
        self.tunnel: subprocess.Popen | None = None

    def _ssh(self, cmd: str, timeout: int = 240, stdin_text: str | None = None) -> str:
        # encoding/errors pinned: remote logs carry non-ASCII (vLLM progress bars, box-drawing
        # characters), and on a Windows driver the default cp1252 decode raises inside
        # subprocess's reader THREAD — which does not fail the call, it just loses the output
        # and prints an alarming traceback that looks like the run died. Observed 2026-08-05.
        r = subprocess.run(["ssh", self.host, cmd], capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=timeout, input=stdin_text)
        if r.returncode != 0:
            raise RuntimeError(f"ssh {self.host} failed ({r.returncode}): "
                               f"{cmd[:120]} ...\n{r.stderr[-500:]}")
        return r.stdout

    def has_env(self) -> bool:
        return self._ssh(f"[ -f {self.workdir}/.env ] && echo yes || echo no").strip() == "yes"

    def push_hf_env(self, local_env: Path) -> None:
        """OPT-IN provisioning (--push-env): write ONLY HF_TOKEN and HF_ORG to the host's .env.

        The server needs exactly one credential — HF_TOKEN, for gated/private weight
        pulls — and that stays the only SECRET that ever leaves this machine. HF_ORG
        rides along because it is not one: work run ON the host (an Option A eval, a
        pod-side push) resolves its push namespace from the host's own environment, and
        without it every upload fail-fasts at the end of the run with nothing to fall
        back on (src.huggingface.hf_org). The rest of the .env (OpenRouter, provider API
        keys) stays local: a rented GPU host is the least-trusted machine in the loop,
        and CLAUDE.md's secrets policy says leaked values must be bounded. Never
        overwrites an existing remote .env.
        """
        # Skip, don't abort. The host having a .env already is the NORMAL case on any
        # relaunch against the same box (a crashed run, a config tweak), and failing the
        # whole eval there — after the weights are already downloaded — is a papercut with
        # no upside. The guarantee that matters is unchanged: an existing remote .env is
        # never overwritten.
        if self.has_env():
            print(f">>> {self.host} already has a .env — leaving it untouched")
            return
        from src.huggingface import hf_org

        token = next((line.split("=", 1)[1].strip()
                      for line in local_env.read_text().splitlines()
                      if line.startswith("HF_TOKEN=")), "")
        assert token, f"no HF_TOKEN in {local_env}; nothing to push"
        lines = " ".join(shlex.quote(v) for v in (f"HF_TOKEN={token}", f"HF_ORG={hf_org()}"))
        self._ssh(f"umask 077 && mkdir -p {self.workdir} && "
                  rf"printf '%s\n' {lines} > {self.workdir}/.env")
        print(f">>> pushed HF_TOKEN + HF_ORG (and nothing else) to "
              f"{self.host}:{self.workdir}/.env")

    def _with_env(self, cmd: str) -> str:
        """Prefix a remote command with uv's PATH and the host's own .env (never the driver's).

        A fresh SSH shell sources nothing: without this, `uv` (installed to ~/.local/bin)
        is not on PATH and a remote snapshot_download or vLLM launch has no HF_TOKEN even
        when the pod is fully provisioned. Secrets stay machine-local by design.
        """
        return ('export PATH="$HOME/.local/bin:$PATH"; '
                f"set -a; [ -f {self.workdir}/.env ] && . {self.workdir}/.env; set +a; "
                + cmd)

    def check_ready(self) -> None:
        """Fast fail-with-remedy preflight: is this host prepared to serve?

        Checks reachability, uv, and the repo clone — the three ways a fresh instance
        fails confusingly later. A host that has all three is what
        `uv run python scripts/infra/runpod.py up` leaves behind.
        """
        try:
            state = self._ssh(self._with_env(
                f"command -v uv >/dev/null && echo UV || echo NOUV; "
                f"[ -d {self.workdir}/.git ] && echo REPO || echo NOREPO"), timeout=20)
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            raise SystemExit(
                f"\n--server preflight: cannot reach {self.host} over SSH ({e}).\n"
                "  Check the host is up and the address/port in ~/.ssh/config is current\n"
                "  (RunPod remaps ports across restarts).") from e
        if "NOUV" in state or "NOREPO" in state:
            missing = ("uv is not installed" if "NOUV" in state
                       else f"no repo clone at {self.workdir}")
            raise SystemExit(
                f"\n--server preflight: {self.host} is not prepared ({missing}).\n"
                f"  Bootstrap a fresh instance with:\n"
                f"    uv run python scripts/infra/runpod.py up --name <name>\n"
                "  (installs uv, clones this repo at your current branch, uv sync)")

    def write_file(self, name: str, text: str) -> str:
        """Write a file on the remote host, streaming the payload through STDIN.

        The payload is NOT interpolated into the command line. Doing so silently produced an
        empty file for the ~10KB base64 of a pinned chat template (observed 2026-08-05,
        Windows driver): the ssh call returned 0, wrote nothing, and the failure surfaced
        much later as vLLM refusing a chat-template path that "appears path-like, but doesn't
        exist". The exact truncation point was never pinned down — which is the argument for
        stdin regardless, since it removes command-line length and shell-quoting from the
        picture entirely. The size assertion below turns any future silent write into a loud
        one at the point of failure.
        """
        payload = base64.b64encode(text.encode()).decode()
        path = f"{self.remote_dir}/{name}"
        self._ssh(f"mkdir -p {self.remote_dir} && base64 -d > {shlex.quote(path)}",
                  stdin_text=payload)
        written = self._ssh(f"wc -c < {shlex.quote(path)} 2>/dev/null || echo 0").strip()
        assert int(written or 0) > 0, f"remote write of {path} produced an empty file"
        return path

    def fetch_adapter(self, hf_path: str) -> str:
        # Through src.huggingface so the host's own .env works whichever token
        # variable it carries (bare snapshot_download reads only HF_TOKEN).
        out = self._ssh(self._with_env(
            f"cd {self.workdir} && uv run python -c "
            f"\"from src.huggingface import hf_snapshot; "
            f"print(hf_snapshot('{hf_path}'))\""), timeout=1800)
        return out.strip().splitlines()[-1]

    def start_server(self, argv: list[str], env_extra: dict) -> None:
        # nohup-ing the command INLINE over ssh keeps the channel open until the ssh
        # client times out (CLAUDE.md gotcha 8 — observed twice this migration). The
        # pattern that returns instantly is nohup-ing a SCRIPT, so write one and launch it.
        env = " ".join(f"export {k}={shlex.quote(v)};" for k, v in env_extra.items())
        cmd = " ".join(shlex.quote(a) for a in argv)
        script = self.write_file("launch_vllm.sh",
                                 f"#!/bin/bash\ncd {self.workdir}\n{env}\nexec {cmd}\n")
        self._ssh(self._with_env(
            f"nohup bash {script} >> {self.remote_dir}/vllm.log 2>&1 < /dev/null & "
            f"echo started"), timeout=60)
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

    `serve_requirements` is the eval config's `serving:` block — what that eval NEEDS,
    never what the family provides. It is validated against the family's facts by
    plan_serving and layered over nothing; see that function for the split.
    """

    def __init__(self, work_dir: Path, port: int = 8000, executor=None,
                 serve_requirements: dict | None = None):
        self.port = port
        self.serve_requirements = serve_requirements or {}
        self.executor = executor if executor is not None else LocalExec(work_dir)
        self.base_model: str | None = None
        self.mode: str | None = None
        self.running = False
        self._loaded_loras: set[str] = set()

    @property
    def base_url(self) -> str:
        return f"http://{self.executor.endpoint_host}:{self.port}/v1"

    def ensure(self, spec: TargetSpec) -> ServedTarget:
        """Return a lazy handle for `spec`; nothing is served until base_url is touched."""
        return ServedTarget(spec=spec, server=self)

    def serve(self, spec: TargetSpec) -> str:
        """Serve `spec` now, reusing the live server when base model + mode are unchanged.

        Returns:
            The OpenAI-compatible base URL.
        """
        adapter_dir = self.executor.fetch_adapter(spec.hf_path) if spec.adapter else None
        if not self.running or self.base_model != spec.base_model or self.mode != spec.mode:
            self.stop()
            self._start(spec, adapter_dir)
        elif spec.adapter and spec.model_key not in self._loaded_loras:
            assert adapter_dir is not None
            self._load_lora(spec, adapter_dir)
        return self.base_url

    def _pinned_template_path(self, base_model: str, mode: str) -> str | None:
        if mode == "default":
            return None
        with open(hf_download(base_model, "tokenizer_config.json")) as f:
            template = json.load(f)["chat_template"]
        return self.executor.write_file(f"chat_template_{mode}.jinja",
                                        pin_template(template, mode))

    def _start(self, spec: TargetSpec, adapter_dir: str | None) -> None:
        # Facts come from two places, both authoritative and neither overridable: the
        # family's measured/architectural profile, and the model's own config.json.
        facts = dict(serving_params(spec.base_model),
                     native_context_window=native_context_window(spec.base_model))
        plan = plan_serving(facts, self.serve_requirements, spec.base_model, spec.mode)
        for warning in plan["warnings"]:
            print(f"!!! {warning}")
        argv = self.executor.python_argv + [
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", spec.base_model, "--served-model-name", "base",
            "--dtype", "bfloat16",
            "--max-model-len", str(plan["context_window"]),
            "--gpu-memory-utilization", "0.94",
            "--port", str(self.port)]
        # Every decision below was made in plan_serving; this is translation only. Adding
        # an `if` here would put a second decision-maker back in the loop — the thing the
        # facts/requirements split exists to prevent.
        if plan["max_num_seqs"]:
            argv += ["--max-num-seqs", str(plan["max_num_seqs"])]
        if plan["reasoning_parser"]:
            argv += ["--reasoning-parser", plan["reasoning_parser"]]
        if plan["tool_call_parser"]:
            argv += ["--enable-auto-tool-choice",
                     "--tool-call-parser", plan["tool_call_parser"]]
        if plan["prefix_caching"]:
            argv += ["--enable-prefix-caching"]
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
        url = f"http://{self.executor.endpoint_host}:{self.port}/health"
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
