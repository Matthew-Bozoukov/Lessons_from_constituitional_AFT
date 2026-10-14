# ABOUTME: Grade completed arms alongside inference using isolated official harness outputs.
# ABOUTME: Atomically offer completed report caches without overwriting finalizer-owned directories.
import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    with temp.open("w") as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def install_directory(source, destination):
    """Linux atomic no-replace: the finalizer's first mkdir wins every race."""
    libc = ctypes.CDLL(None, use_errno=True)
    fn = libc.renameat2
    fn.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    fn.restype = ctypes.c_int
    result = fn(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return True
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        return False
    raise OSError(error, os.strerror(error))


def offer_cache(directory, canonical, early_id, canonical_id):
    # Only complete official per-instance reports are offered. Original raw
    # early logs and aggregate reports remain at their original run ID forever.
    incoming = canonical.with_name(canonical.name + ".early-ready")
    incoming.mkdir(parents=True, exist_ok=False)
    origin = directory / "logs/run_evaluation" / early_id
    hashes = {}
    for report in sorted(origin.glob("*/*/report.json")):
        value = read(report)
        iid = report.parent.name
        assert iid in value and isinstance(value[iid]["resolved"], bool)
        relative = report.parent.relative_to(origin)
        dest = incoming / "logs/run_evaluation" / canonical_id / relative
        shutil.copytree(report.parent, dest, symlinks=True)
        assert sha(dest / "report.json") == sha(report)
        hashes[str(relative / "report.json")] = sha(report)
    save(
        incoming / "early-cache-provenance.json",
        dict(
            source=str(directory),
            original_run_id=early_id,
            canonical_run_id=canonical_id,
            reports=hashes,
            scoring_unchanged=True,
        ),
    )
    installed = install_directory(incoming, canonical)
    return dict(
        installed=installed,
        report_count=len(hashes),
        sha256=hashes,
        canonical=str(canonical),
        uninstalled_copy=None if installed else str(incoming),
    )


def run(root, workers):
    import psutil
    from omegaconf import OmegaConf
    from scratch.gptoss_swe import full300
    from scratch.gptoss_swe.run import grade_predictions

    assert root == full300.ROOT
    assert 1 <= workers <= 6
    ready = read(root / "qualification/ready.json")
    assert ready["source_hashes"] == full300.sources(), "Frozen runtime source drift"
    h = read(root / "base-scale80-handoff.json")
    original = psutil.Process(305835)
    assert abs(original.create_time() - 1791913260.51) < 0.01
    assert original.status() == psutil.STATUS_STOPPED
    assert "scratch.gptoss_swe.full300" in original.cmdline()
    owner = dict(
        pid=os.getpid(),
        birth=psutil.Process().create_time(),
        at=time.time(),
        workers=workers,
        arms=["control", "da15"],
        source_sha256=sha(__file__),
        original_pid=original.pid,
        original_birth=original.create_time(),
    )
    with (root / "early-grade-owner.json").open("x") as f:
        json.dump(owner, f, indent=2)
    completed = {}
    try:
        for arm in owner["arms"]:
            # No second early arm is admitted once the stock finalizer resumes.
            # A current isolated grade may finish; its unique names cannot collide.
            if original.status() != psutil.STATUS_STOPPED:
                save(
                    root / "early-grade-complete.json",
                    dict(
                        at=time.time(),
                        arms=completed,
                        remaining_to_original=owner["arms"][len(completed) :],
                    ),
                )
                return
            assert psutil.virtual_memory().available >= 64 * 2**30
            assert shutil.disk_usage(root).free >= 70 * 2**30
            r = root / arm
            watched = [
                r / "config.yaml",
                r / "budget.json",
                r / "swe/metadata/state.json",
                r / "swe/metadata/swebench_lite_test.json",
            ]
            before = {str(p): sha(p) for p in watched}
            state = read(watched[2])
            assert len(state["tasks"]) == 300 and not state.get("halt")
            assert all(t["status"] == "valid" for t in state["tasks"].values())
            budget = read(r / "budget.json")
            assert all(x["state"] == "completed" for x in budget["requests"].values())
            preds = {
                i: t["attempts"][-1]["prediction"] for i, t in state["tasks"].items()
            }
            cfg = OmegaConf.load(r / "config.yaml")
            cfg.grading.max_workers = (
                workers  # Scheduling only, saved config untouched.
            )
            staging = r / "early-grading"
            (staging / "metadata").mkdir(parents=True, exist_ok=False)
            shutil.copyfile(watched[3], staging / "metadata/swebench_lite_test.json")
            canonical_id = str(full300.C.campaign) + "-" + arm
            early_id = canonical_id + "-early"
            save(
                root / "early-grade-status.json",
                dict(
                    at=time.time(),
                    phase="grading",
                    arm=arm,
                    completed_arms=list(completed),
                    workers=workers,
                    inputs=before,
                ),
            )
            result = grade_predictions(cfg, staging, preds, early_id)
            assert result["n_graded"] == 300
            assert before == {str(p): sha(p) for p in watched}, (
                "Immutable input changed"
            )
            summary = dict(
                **result,
                target=str(full300.C.targets[arm]),
                n_selected=300,
                n_excluded=0,
                at=time.time(),
                inputs_sha256=before,
            )
            save(staging / "summary.json", summary)
            cache = offer_cache(
                staging / "results" / early_id,
                r / "swe/results" / canonical_id,
                early_id,
                canonical_id,
            )
            save(staging / "cache-install.json", cache)
            completed[arm] = dict(summary=summary, cache=cache)
            save(
                root / "early-grade-status.json",
                dict(
                    at=time.time(),
                    phase="arm_complete",
                    arm=arm,
                    completed_arms=list(completed),
                    workers=workers,
                ),
            )
        save(
            root / "early-grade-complete.json",
            dict(at=time.time(), arms=completed, remaining_to_original=[]),
        )
    except BaseException as exc:
        save(
            root / "early-grade-failure.json",
            dict(
                at=time.time(),
                error=repr(exc),
                completed_arms=list(completed),
                note="Retain outputs; do not rerun inference",
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    run(args.root, args.workers)
