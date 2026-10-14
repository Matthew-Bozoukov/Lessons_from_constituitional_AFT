# ABOUTME: Keep hot SWE evidence in durable Docker Linux volumes, with atomic host mirrors.
# ABOUTME: Volumes survive failures; no cleanup deletes primary evidence and snapshots never drive workers.
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid


def mirror(source, destination, previous=None, verify=False):
    """Copy stable files atomically; active files retry on the next pass. Never delete."""
    previous = {} if previous is None else previous
    copied = 0
    for path in Path(source).rglob("*"):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        relative = path.relative_to(source)
        try:
            before = path.stat()
            signature = (before.st_ino, before.st_size, before.st_mtime_ns)
            if not verify and previous.get(str(relative)) == signature:
                continue
            target = Path(destination) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
            try:
                with path.open("rb") as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                    dst.flush()
                    os.fsync(dst.fileno())
                after = path.stat()
                if signature != (after.st_ino, after.st_size, after.st_mtime_ns):
                    if verify:
                        raise RuntimeError(
                            "Evidence still changing during final synchronization: "
                            + str(relative)
                        )
                    continue
                if (
                    verify
                    and hashlib.sha256(path.read_bytes()).digest()
                    != hashlib.sha256(temporary.read_bytes()).digest()
                ):
                    raise RuntimeError(
                        "Evidence changed during final synchronization: "
                        + str(relative)
                    )
                os.replace(temporary, target)
                if (
                    verify
                    and hashlib.sha256(path.read_bytes()).digest()
                    != hashlib.sha256(target.read_bytes()).digest()
                ):
                    raise RuntimeError(
                        "Host evidence readback failed: " + str(relative)
                    )
                previous[str(relative)] = signature
                copied += 1
            finally:
                temporary.unlink(missing_ok=True)
        except FileNotFoundError:
            # Atomic producer replacement/deletion raced this snapshot; retry later.
            if verify:
                raise
    return copied


def volume_name(campaign, backend):
    return campaign + "-" + backend + "-swe-evidence"


def mounts(campaign, backend, root):
    """Only hot SWE evidence is remapped; original host config/status paths stay visible."""
    volume = volume_name(campaign, backend)
    return [
        item
        for leaf in ("swe", "swe-traces", "swe-raw")
        for item in (
            "--mount",
            f"type=volume,src={volume}-{leaf},dst=/work/{(Path(root) / backend / leaf).as_posix()}",
        )
    ]


@contextmanager
def evidence(campaign, backend, root, image, python):
    """Start one scoped mirror before consumers; preserve volumes on every failure."""
    directory = Path(root) / backend
    names = [
        volume_name(campaign, backend) + "-" + leaf
        for leaf in ("swe", "swe-traces", "swe-raw")
    ]
    # Exclusive new-run admission. Existing volumes require explicit recovery, never reseeding.
    for name in names:
        if (
            subprocess.run(
                ["docker", "volume", "inspect", name], capture_output=True
            ).returncode
            == 0
        ):
            raise RuntimeError(
                "Retained native evidence exists; recover explicitly: " + name
            )
    for name in names:
        subprocess.run(
            [
                "docker",
                "volume",
                "create",
                "--label",
                "lasr.campaign=" + campaign,
                name,
            ],
            check=True,
            capture_output=True,
        )
    helper = volume_name(campaign, backend) + "-mirror"
    args = [
        "docker",
        "run",
        "-d",
        "--name",
        helper,
        "--label",
        "lasr.campaign=" + campaign,
        "-v",
        str(Path.cwd()) + ":/work",
        *mounts(campaign, backend, root),
        "-v",
        str(directory.resolve()) + ":/mirror",
        "-v",
        "lasr-gptoss-cpu-env:/work/scratch/swebench_cpu_env/.venv",
        image,
        python,
        "-m",
        "scratch.gptoss_swe.native_evidence",
        "--source",
        "/work/" + directory.as_posix(),
        "--destination",
        "/mirror",
    ]
    subprocess.run(args, check=True, capture_output=True)
    try:
        # Wait for initialization (including copying preexisting config/evidence) before launch.
        for _ in range(60):
            check = subprocess.run(
                ["docker", "exec", helper, "test", "-f", "/tmp/evidence-ready"],
                capture_output=True,
            )
            if check.returncode == 0:
                break
            state = json.loads(subprocess.check_output(["docker", "inspect", helper]))[
                0
            ]["State"]
            if not state["Running"]:
                raise RuntimeError("Native evidence mirror exited; inspect " + helper)
            time.sleep(1)
        else:
            raise TimeoutError("Native evidence mirror not ready")
        yield
    finally:
        # Signal graceful final hash-verified copy. Failed sync retains primary volumes.
        subprocess.run(
            ["docker", "exec", helper, "touch", "/tmp/evidence-finish"], check=True
        )
        result = subprocess.run(
            ["docker", "wait", helper],
            capture_output=True,
            text=True,
            timeout=600,
            check=True,
        )
        if result.stdout.strip() != "0":
            raise RuntimeError("Native evidence final copy failed; retain " + helper)
        with (directory / "native-evidence.json").open("w") as f:
            json.dump(
                dict(volumes=names, mirror=helper, final_verified=True, at=time.time()),
                f,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--destination")
    args = parser.parse_args()
    source = Path(args.source)
    destination = Path(args.destination)
    histories = {leaf: {} for leaf in ("swe", "swe-traces", "swe-raw")}
    for leaf in histories:
        (source / leaf).mkdir(parents=True, exist_ok=True)
        if (destination / leaf).exists():
            mirror(destination / leaf, source / leaf, verify=True)
    Path("/tmp/evidence-ready").touch()
    while not Path("/tmp/evidence-finish").exists():
        for leaf, previous in histories.items():
            mirror(source / leaf, destination / leaf, previous)
        time.sleep(5)
    for leaf in histories:
        mirror(source / leaf, destination / leaf, verify=True)


if __name__ == "__main__":
    main()
