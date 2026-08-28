# ABOUTME: Health-check and dead-man's switch for ONE RunPod training pod: polls markers and
# ABOUTME: loss, tears the pod down on success, failure, stall or timeout. Never sweeps.

"""Watch one training pod and make sure it cannot bill forever.

    uv run --with requests python scratch/low_stakes/train_watchdog.py --pod <id>

CLAUDE.md requires that a run which provisions a GPU "must not rely on the orchestration
process surviving to clean it up". This is that guarantee for the training pod: it decides
on its own when the pod has stopped being useful and terminates it.

FOUR conditions end the pod, and each is a real failure mode already paid for on 2026-08-26:

  * DONE      -- `TRAINING_DONE` in the log. Adapter is pulled BEFORE teardown, and a
                 teardown never happens if the pull fails, because a terminated pod takes
                 the only copy with it.
  * FAILED    -- `TRAINING_FAILED`. A dotenv import error burned ~25 minutes of paid boot
                 before anyone looked; this catches the same shape in one poll.
  * STALLED   -- the log stops growing for `stall_minutes`. Covers the silent version: the
                 vast box that went `exited` mid-run with a healthy-looking status message.
  * TIMEOUT   -- `max_hours` since launch. The backstop for anything the other three miss.
                 Siblings train in 1h40m-2h30m, so the default leaves real headroom and
                 still bounds the bill.

SAFETY. The pod id is passed in and is the only pod this will ever terminate. There is no
discovery, no listing, no "clean up orphans" -- the account is shared, and a sweep here
would kill a teammate's run. Same rule as `scratch/low_stakes/vast_train.py::down`.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import fire
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _log(pod: str, name: str) -> str:
    try:
        r = requests.get(f"https://{pod}-8080.proxy.runpod.net/{name}", timeout=25)
        return r.text if r.ok else ""
    except Exception:  # noqa: BLE001 - a missed poll is not an event
        return ""


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _last_loss(text: str) -> str:
    """The most recent HF Trainer log line, which carries loss/epoch/lr."""
    for line in reversed(text.strip().splitlines()):
        if "'loss'" in line or '"loss"' in line:
            return line.strip()[:150]
    return ""


def _teardown(pod: str, why: str) -> None:
    from scripts.gpu.runpod_train import down

    print(f"[{_stamp()}] TEARDOWN ({why}): {down(pod=pod)}", flush=True)


def main(
    pod: str,
    poll_seconds: int = 60,
    stall_minutes: int = 25,
    max_hours: float = 4.0,
    out_dir: str = "output/low_stakes_adapter",
    teardown: bool = True,
) -> None:
    started = time.time()
    last_size, last_growth = -1, time.time()
    phase = "booting"
    print(
        f"[{_stamp()}] watching {pod}  (stall {stall_minutes}m, cap {max_hours}h)",
        flush=True,
    )

    while True:
        train, boot = _log(pod, "train.log"), _log(pod, "boot.log")
        # BOTH logs, concatenated, always. This was `train or boot` until 2026-08-27, and
        # that bug destroyed a pod holding a finished adapter: the bootstrap tees the
        # TRAINER's output to train.log but echoes TRAINING_DONE from the OUTER shell,
        # which goes to boot.log. Preferring train.log meant the DONE marker was never
        # visible, train.log stopped growing the moment the trainer exited, and the stall
        # rule -- written to catch silent deaths -- killed a healthy run at 625/625 with
        # `saved adapter` already in the log. Never let one log's silence outrank another
        # log's completion marker.
        blob = f"{train}\n{boot}"
        size = len(train) + len(boot)
        elapsed = (time.time() - started) / 3600

        if size > last_size:
            last_size, last_growth = size, time.time()
        quiet = (time.time() - last_growth) / 60

        new_phase = phase
        if "TRAINING_DONE" in blob or "TRAINING_FAILED" in blob:
            new_phase = "finished"
        elif "TRAINING_STARTING" in blob or _last_loss(train):
            new_phase = "training"
        elif boot:
            new_phase = "booting"
        if new_phase != phase:
            phase = new_phase
            print(f"[{_stamp()}] -> {phase}", flush=True)

        loss = _last_loss(train)
        print(
            f"[{_stamp()}] {phase:9s} {elapsed:4.2f}h  log {size:>8d}B  "
            f"quiet {quiet:4.1f}m  {loss}",
            flush=True,
        )

        failed = "TRAINING_FAILED" in blob
        if failed and "TRAINING_DONE" not in blob:
            # A failed run is NOT torn down on sight any more. The bootstrap tars $OUTDIR
            # before it echoes TRAINING_DONE, crash or not, so the run's last checkpoints
            # are in adapter.tar.gz -- and they can be the deliverable: the GPT paired arm
            # ends EVERY run this way (route_step cannot split the final 1-example step over
            # 2 DDP ranks, at 624/624, after checkpoint-600 is written), and seed 0's
            # published adapter is exactly that checkpoint. Wait for the DONE marker so the
            # tarball is complete, then pull it like a finished run; teardown follows the
            # pull, never precedes it.
            print("\n".join(blob.strip().splitlines()[-25:]), flush=True)
            print(
                f"[{_stamp()}] TRAINING_FAILED -- waiting for the tarball before pulling",
                flush=True,
            )
            time.sleep(poll_seconds)
            continue
        if "TRAINING_DONE" in blob:
            # Pull FIRST. A terminated pod takes the only copy of the adapter with it, so
            # a failed pull must leave the pod standing rather than tidy it away.
            dest = Path(out_dir)
            dest.mkdir(parents=True, exist_ok=True)
            tar = dest / "adapter.tar.gz"
            try:
                r = requests.get(
                    f"https://{pod}-8080.proxy.runpod.net/adapter.tar.gz",
                    timeout=900,
                    stream=True,
                )
                r.raise_for_status()
                with open(tar, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                mb = tar.stat().st_size / 1e6
                print(f"[{_stamp()}] pulled {tar} ({mb:.1f} MB)", flush=True)
                if mb < 1:
                    print(
                        "REFUSING TEARDOWN: adapter is suspiciously small; inspect the "
                        "pod before destroying it.",
                        flush=True,
                    )
                    return
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[{_stamp()}] PULL FAILED ({exc}). Pod left UP on purpose -- "
                    f"retry, then tear down by hand.",
                    flush=True,
                )
                return
            if teardown:
                _teardown(
                    pod,
                    "TRAINING_FAILED, last checkpoints pulled"
                    if failed
                    else "TRAINING_DONE, adapter pulled",
                )
            return
        if quiet >= stall_minutes:
            # A second guard on the same lesson: even with both logs read, a quiet pod that
            # has ALREADY written the adapter is finishing, not dead. Never let the stall
            # rule destroy a pod whose trainer got as far as saving -- wait for the DONE
            # marker and the pull instead, and if the marker never comes, say so and leave
            # the pod standing for a human. A stalled pod costs money; a destroyed adapter
            # costs the whole run.
            if "saved adapter" in blob or "packaging adapter" in blob:
                print(
                    f"[{_stamp()}] quiet {quiet:.0f}m BUT the adapter is already "
                    f"written -- NOT tearing down. Waiting for TRAINING_DONE.",
                    flush=True,
                )
                if quiet >= stall_minutes * 3:
                    print(
                        f"[{_stamp()}] adapter written but no DONE marker after "
                        f"{quiet:.0f}m. Pod left UP on purpose. Pull it by hand:\n"
                        f"  https://{pod}-8080.proxy.runpod.net/adapter.tar.gz",
                        flush=True,
                    )
                    return
                time.sleep(poll_seconds)
                continue
            print(f"[{_stamp()}] STALLED: no log growth for {quiet:.0f}m", flush=True)
            print("\n".join(blob.strip().splitlines()[-15:]), flush=True)
            if teardown:
                _teardown(pod, f"stalled {quiet:.0f}m")
            return
        if elapsed >= max_hours:
            print(f"[{_stamp()}] TIMEOUT at {elapsed:.2f}h", flush=True)
            if teardown:
                _teardown(pod, f"timeout {elapsed:.2f}h")
            return

        time.sleep(poll_seconds)


if __name__ == "__main__":
    fire.Fire(main)
