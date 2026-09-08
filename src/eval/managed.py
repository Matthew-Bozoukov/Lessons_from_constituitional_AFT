# ABOUTME: One-command eval: rent the GPU, run the eval against it, tear the pod down —
# ABOUTME: with teardown that does not depend on this process surviving to do it.

"""The three-step eval flow (`runpod up` → `evals --server` → `runpod down`) as one call.

The repo's default is deliberately manual: `uv run runpod up` prints "IT BILLS UNTIL YOU
RUN THIS" and nothing tears a pod down for you, because an unattended teardown that
half-works is worse than one a person is holding. This module exists for the case where
that trade is worth making — an unattended, single-target run of a cheap eval — and it
earns it by never having exactly one thing standing between a rented GPU and a stopped
bill:

1. a **watchdog** registered BEFORE any work, in its own session, which terminates the
   pod when this process dies or a lifetime cap passes;
2. a `terminate` in a `finally`, verified against the API rather than fire-and-forget;
3. a closing report naming any pod of ours still listed, so a failure to stop billing is
   the last thing printed rather than something noticed on the invoice.

It never sweeps the account. Teammates share it, and a pod this call did not rent is
reported, never terminated (CLAUDE.md "Paid infrastructure").

The driver runs HERE and the model runs THERE: run_eval opens an SSH tunnel and the eval
loop, any judging and the HF push all stay on this machine, so credentials never move.
That also means this process staying alive is what finishes the run — the watchdog
protects the wallet, not the results.
"""

from __future__ import annotations

import getpass
import os
from datetime import date
from pathlib import Path
from typing import Sequence

from src.eval import EVALS


def default_pod_name(eval_name: str, owner: str = "") -> str:
    """`<owner>-<eval>-<date>` — the RunPod account is shared, so the owner comes first.

    Deliberately NOT `CHAT_POD_PREFIX`: that prefix marks a pod the chat tool's sweep may
    terminate, and this pod belongs to this call alone.
    """
    owner = owner or os.environ.get("RUNPOD_OWNER") or getpass.getuser()
    return f"{owner}-{eval_name.replace('_', '-')}-{date.today().isoformat()}"


def managed_run(
    eval_name: str,
    targets: Sequence[str],
    *,
    pod_name: str = "",
    gpu: str = "",
    max_hours: float = 3.0,
    boot_timeout_s: int = 3600,
    overrides: Sequence[str] = (),
    no_push: bool = False,
) -> int:
    """Rent, evaluate, tear down. Returns a process exit code.

    Args:
        eval_name: A key of `src.eval.EVALS`.
        targets: HF paths, or `<provider>:<model-id>` API endpoints. An all-API target
            list needs no GPU and skips provisioning entirely — the cheapest way to
            check the wiring of an api-capable eval before spending anything.
        pod_name: Overrides `default_pod_name`.
        gpu: RunPod catalogue id, overriding the one `ModelProfile` states for the base.
        max_hours: Hard lifetime cap handed to the watchdog. The pod dies at this point
            even if this process is wedged. Keep it just above the expected runtime.
        boot_timeout_s: How long to wait for the pod to finish pulling weights.
        overrides: OmegaConf dotlist entries passed through to the eval config.
        no_push: Skip the HF upload (smoke runs only — HF is the canonical store).
    """
    from src.eval.run_eval import main as run_eval_main
    from src.infra import runpod

    assert eval_name in EVALS, f"unknown eval {eval_name!r}; known: {', '.join(sorted(EVALS))}"
    targets = list(targets)
    assert targets, "managed_run needs at least one target"

    # An API target is served by somebody else; there is nothing to rent for it. Mixing
    # the two would rent a pod that half the run ignores, so it is refused rather than
    # silently paid for.
    is_api = [":" in t and not t.startswith("http") for t in targets]
    if all(is_api):
        print(">>> all targets are API endpoints — no GPU needed", flush=True)
        return _dispatch(run_eval_main, eval_name, targets, "", overrides, no_push)
    assert not any(is_api), (
        f"mixed HF and API targets {targets}: rent a pod for the HF ones and run the "
        "API ones separately, so a pod is never rented for a target that ignores it.")

    spec = EVALS[eval_name]
    if spec.needs_docker:
        # Docker runs where the DRIVER runs, which is here, so this shape is fine — but
        # say so, because it is the opposite of what "rent a GPU" suggests.
        print(f">>> {eval_name} needs docker on THIS machine; the pod only serves the model",
                  flush=True)

    # A pod with no authorized key is a pod nobody can log into, and provision_runpod
    # silently skips key injection when its default path is absent. Discovering the key
    # BEFORE renting turns "ten billing minutes then Permission denied" into an
    # instant, free error.
    keypair = runpod.default_keypair()
    if keypair is None:
        raise SystemExit(
            "\nNo usable SSH keypair in ~/.ssh — a rented pod would authorize no key and\n"
            "be unreachable, so nothing has been rented. Create one with:\n"
            "    ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519\n"
            "(any complete <name> / <name>.pub pair is picked up; RunPod injects the\n"
            "public half at boot and the driver authenticates with the private half).")
    pubkey_path, identity = keypair
    print(f">>> ssh key: {identity}", flush=True)

    pod_name = pod_name or default_pod_name(eval_name)
    log_dir = Path("output") / eval_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # The watchdog is armed from inside provisioning, the instant the pod id exists —
    # NOT on the returned Pod. Resolving the SSH endpoint and waiting for sshd can take
    # ten minutes, the meter runs the whole time, and arming afterwards would leave that
    # entire window unprotected against this process dying.
    state: dict[str, object] = {"pod_id": None, "watchdog": None}

    def arm(pod_id: str) -> None:
        state["pod_id"] = pod_id
        state["watchdog"] = runpod.start_watchdog(
            pod_id, int(max_hours * 3600), log_dir / f"watchdog_{pod_id}.log")
        print(f">>> watchdog armed: pod {pod_id} dies with this process, or in {max_hours}h", flush=True)

    code = 1
    try:
        pod = runpod.provision_eval_pod(targets, name=pod_name, gpu=gpu or None,
                                        pubkey_path=pubkey_path, identity=identity,
                                        on_provisioned=arm)
        if not pod.reachable:
            print(f"!!! {pod.host} is not answering SSH yet; waiting on the boot log", flush=True)
        if not runpod.wait_bootstrapped(pod.id, timeout_s=boot_timeout_s):
            raise TimeoutError(
                f"pod {pod.id} never reported READY within {boot_timeout_s}s — "
                f"read {runpod.boot_log_url(pod.id)}")
        print(f">>> pod ready: {pod.host}", flush=True)
        code = _dispatch(run_eval_main, eval_name, targets, pod.host, overrides,
                         no_push, identity)
    finally:
        # Teardown is not conditional on anything above having worked, and it reads the
        # pod id from `state` rather than from a local the try block may never have
        # bound: provisioning can raise AFTER the pod exists, which is exactly the case
        # where forgetting it costs the most.
        pod_id = state["pod_id"]
        if pod_id is None:
            print(">>> nothing was rented", flush=True)
        else:
            print(f">>> tearing down {pod_id}", flush=True)
            gone = runpod.terminate(pod_id)
            watchdog = state["watchdog"]
            if watchdog is not None:
                watchdog.terminate()
            still = [p for p in runpod.active_pods() if p.get("id") == pod_id]
            print(runpod.down(pod_id) if (still or not gone) else f"{pod_id}: terminated",
                  flush=True)
            if still or not gone:
                # The loudest thing this function can print, and the last: an
                # un-torn-down pod bills until a human intervenes.
                print(f"!!! POD {pod_id} MAY STILL BE BILLING — check "
                      "https://console.runpod.io/pods and run: "
                      f"uv run runpod down --pod {pod_id}", flush=True)
                code = code or 1
    return code


def _dispatch(run_eval_main, eval_name: str, targets: Sequence[str], server: str,
              overrides: Sequence[str], no_push: bool, identity: str = "") -> int:
    """Call run_eval exactly as the CLI would, so there is one eval code path, not two."""
    argv = ["--name", eval_name, "--target", *targets]
    if server:
        argv += ["--server", server]
        if identity:
            argv += ["--ssh-key", identity]
    if no_push:
        argv.append("--no-push")
    argv += list(overrides)
    print(f">>> uv run evals {' '.join(argv)}", flush=True)
    try:
        run_eval_main(argv)
        return 0
    except SystemExit as exc:
        # SystemExit.code is an int, None, OR a message string -- run_eval's fail-fast
        # paths raise the message form. int() on that throws a ValueError that buries
        # the actual error under a traceback about string parsing, which is precisely
        # what happened on the first live run (2026-09-01).
        code = exc.code
        if isinstance(code, int):
            return code
        if code is not None:
            print(code, flush=True)
        return 1 if code is not None else 0


def cli() -> None:
    """Console entry (`uv run managed`, [project.scripts]): any eval, rented and torn down.

        uv run managed --eval_name moralbench --targets org/my-lora
    """
    import fire

    raise SystemExit(fire.Fire(managed_run))


def moralbench_cli() -> None:
    """Console entry (`uv run moralbench <hf_path>`, [project.scripts]).

    The whole point of this one is that it takes the adapter and nothing else:

        uv run moralbench org/my-qwen36-lora

    Everything after the target has a default that is right for MoralBench — the paper
    profile from configs/eval/moralbench.yaml, an inference card chosen from the base
    model's ModelProfile, and a 3h cap that is ~20x the 440 one-letter generations this
    eval actually needs.
    """
    import fire

    def moralbench(target: str, max_hours: float = 3.0, pod_name: str = "",
                   gpu: str = "", no_push: bool = False, *overrides: str) -> int:
        """Rent a GPU, run MoralBench against `target`, tear the pod down.

        Args:
            target: The HF adapter or model to evaluate, or `<provider>:<model-id>` for
                an API endpoint (which rents nothing).
            max_hours: Watchdog lifetime cap.
            pod_name: Overrides the `<owner>-moralbench-<date>` default.
            gpu: RunPod catalogue id, overriding the base model's ModelProfile card.
            no_push: Skip the HF upload (smoke runs only).
            *overrides: OmegaConf dotlist entries, e.g. `generation.repetitions=1`.
        """
        return managed_run("moralbench", [target], pod_name=pod_name, gpu=gpu,
                           max_hours=max_hours, overrides=overrides, no_push=no_push)

    raise SystemExit(fire.Fire(moralbench))
