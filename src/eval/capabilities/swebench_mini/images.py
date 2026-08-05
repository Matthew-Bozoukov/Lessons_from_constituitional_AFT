# ABOUTME: Pre-pull the per-instance SWE-bench images before rollouts, because upstream's
# ABOUTME: 120s container-start timeout is shorter than a multi-GB pull on a normal connection.

"""Getting the task environments onto the machine before the agent needs them.

Found by the first smoke run: mini-SWE-agent starts each task with `docker run`, and
`DockerEnvironmentConfig.pull_timeout` is 120 seconds. When the image is not local yet that
timeout covers the *pull*, and a multi-GB SWE-bench image does not arrive in two minutes on
an ordinary connection — so every instance died with `TimeoutExpired` and produced an empty
patch. Scored naively that is a 0% pass@1 with no hint that nothing ever ran.

Pre-pulling fixes it without touching the scaffold: the image content is identical, it is just
already on disk, so `docker run` returns immediately and the timeout never binds. The
alternative — overlaying a longer `pull_timeout` — would be another deviation from the stock
config to defend, for a worse outcome (no progress reporting, no disk figure, failures
surfacing one at a time deep inside a rollout).

It also answers the practical question of what a run costs in disk, which is the constraint on
both the rollout host and the grading host.
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable


def image_name(instance: dict) -> str:
    """The image an instance runs in — mirrors mini-SWE-agent's own derivation exactly.

    Kept byte-compatible with upstream's `get_swebench_docker_image_name` (mini-swe-agent
    2.2.1) on purpose: if we pre-pull a different name than it runs, we have paid for the
    download and still hit the timeout it was meant to avoid. The dataset's own field wins
    when present; otherwise the id is made docker-safe (`__` is not allowed in an image name,
    so SWE-bench substitutes a magic token).
    """
    name = instance.get("image_name") or instance.get("docker_image")
    if name:
        return str(name)
    iid = str(instance["instance_id"]).replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{iid}:latest".lower()


def _pull_one(name: str, timeout: int) -> tuple[str, bool, str]:
    proc = subprocess.run(["docker", "pull", name], capture_output=True, text=True,
                          timeout=timeout)
    return name, proc.returncode == 0, (proc.stderr or proc.stdout).strip()[-200:]


def local_size_gb(names: Iterable[str]) -> float:
    """Total on-disk size of the given images, in GB (0.0 for any not present)."""
    total = 0.0
    for name in set(names):
        proc = subprocess.run(["docker", "image", "inspect", name, "--format", "{{.Size}}"],
                              capture_output=True, text=True)
        if proc.returncode == 0:
            total += int(proc.stdout.strip() or 0)
    return round(total / 1e9, 2)


def pull_all(instances: Iterable[dict], *, workers: int = 4,
             timeout: int = 3600) -> dict:
    """Pull every instance's image up front, in parallel.

    Args:
        instances: The selected instances.
        workers: Concurrent pulls. Modest by default — docker layer extraction is disk-bound,
            and more parallelism mostly means more contention.
        timeout: Per-image ceiling. Generous: this is the download, not the task.

    Returns:
        Counts, the total disk the images occupy, and the names that failed. A failure is not
        raised here — the rollout will report it per instance, and one unavailable image
        should not throw away the rest of the run.
    """
    names = sorted({image_name(row) for row in instances})
    print(f">>> pre-pulling {len(names)} SWE-bench images ({workers} at a time); "
          "upstream's 120s container-start timeout cannot cover a cold multi-GB pull")
    failed: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (name, ok, err) in enumerate(pool.map(lambda n: _pull_one(n, timeout), names), 1):
            print(f"    [{i}/{len(names)}] {'ok  ' if ok else 'FAIL'} {name}"
                  + ("" if ok else f"\n         {err}"))
            if not ok:
                failed.append((name, err))
    size = local_size_gb(names)
    print(f">>> images on disk: {size} GB across {len(names)} images"
          + (f"; {len(failed)} FAILED" if failed else ""))
    return {"n_images": len(names), "n_failed": len(failed),
            "images_disk_gb": size, "failed_images": [n for n, _ in failed]}
