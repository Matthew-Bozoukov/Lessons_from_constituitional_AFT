# ABOUTME: Finishes one ODCV arm on-box: waits for every pass to land on HF, combines them,
# ABOUTME: judges, and publishes results - checkpointing to the Hub throughout.

"""Combine and judge one arm, without needing the laptop.

    uv run python scratch/odcv_finish_arm.py --config <cfg> --hf_repo <org/name> \
        --expected_passes 4 --max_workers 24

Runs on box-1 of an arm once its own passes are done. It waits for the OTHER box's passes
to appear on the Hub rather than being told they are ready, so the whole arm completes even
if the laptop is asleep, the orchestrating session is gone, or the two boxes finish in
either order.

WHY THE HUB IS THE RENDEZVOUS. The boxes never talk to each other. Each pushes its passes to
one repo as they land (odcv_box_run.py), and this script polls that repo for the expected
count. Box-to-box SSH would need a second key path and would fail the moment vast remapped
an address - which docs/swebench_run_postmortem.md records happening mid-run, with
`vastai ssh-url` still serving the stale address afterwards.

CHECKPOINTS, so a dead box costs minutes rather than the arm:
  - passes:     already on the Hub before this script starts
  - combined:   pushed as soon as it exists
  - judging:    odcv_judge flushes verdicts to disk every 5 completions and SKIPS anything
                already cached, so it is resumable by construction. A background thread here
                pushes that cache to the Hub every few minutes, which is what makes it
                resumable on a DIFFERENT box.
  - results:    pushed at the end

Judging needs no GPU. The pods should already be destroyed before this runs.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import fire
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# load_dotenv HERE. src/infra/huggingface.py resolves the token from the environment and
# does NOT read .env itself, so a supervisor started without it runs the whole pass
# and then fails its push with a bare 401 from create_repo -- which is exactly what
# happened on 2026-08-20, silently disabling the crash-safety this script exists for.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

STATE = Path("/root/odcv")


def _api():
    from src.infra.huggingface import hf_api
    return hf_api()


def _pass_dirs_on_hub(hf_repo: str) -> set[str]:
    """Pass directories that have landed, identified by their manifest."""
    files = _api().list_repo_files(hf_repo, repo_type="dataset")
    return {f.rsplit("/", 1)[0] for f in files
            if f.startswith("passes/") and f.endswith("rollout_manifest.json")}


def wait_for_passes(hf_repo: str, expected: int, timeout_min: float = 180.0,
                    poll_s: float = 60.0) -> list[str]:
    """Block until `expected` passes are on the Hub. Returns their prefixes."""
    deadline = time.time() + timeout_min * 60
    seen: set[str] = set()
    while time.time() < deadline:
        try:
            seen = _pass_dirs_on_hub(hf_repo)
        except Exception as e:  # noqa: BLE001 - a hub blip is not a reason to give up
            print(f"    hub listing failed ({type(e).__name__}); retrying", flush=True)
        print(f"    passes on hub: {len(seen)}/{expected}", flush=True)
        if len(seen) >= expected:
            return sorted(seen)
        time.sleep(poll_s)
    raise SystemExit(
        f"only {len(seen)}/{expected} passes landed in {timeout_min} min: {sorted(seen)}. "
        f"Combine manually with whatever is present rather than waiting longer.")


def download_passes(hf_repo: str, prefixes: list[str], model_root: Path) -> list[Path]:
    """Pull every pass into the arm's local run root so the combiner auto-discovers them."""
    from huggingface_hub import snapshot_download
    from src.infra.huggingface import hf_token
    local = snapshot_download(hf_repo, repo_type="dataset", token=hf_token(),
                              allow_patterns=["passes/**"])
    model_root.mkdir(parents=True, exist_ok=True)
    out = []
    for p in prefixes:
        src = Path(local) / p
        # The pass keeps its original timestamped name; the box id it came from is already
        # recorded inside it (pass_provenance.json), so flattening here loses nothing.
        dest = model_root / src.name
        if not dest.exists():
            import shutil
            shutil.copytree(src, dest)
        out.append(dest)
    return out


def _push_dir(local: Path, hf_repo: str, prefix: str) -> None:
    _api().upload_folder(folder_path=str(local), path_in_repo=prefix,
                         repo_id=hf_repo, repo_type="dataset")


def _checkpoint_loop(local: Path, hf_repo: str, prefix: str, stop: threading.Event,
                     period_s: float = 180.0) -> None:
    """Push the judge's verdict cache to the Hub while judging runs."""
    while not stop.wait(period_s):
        try:
            if local.is_dir():
                _push_dir(local, hf_repo, prefix)
                print(f"    [checkpoint] pushed {prefix}", flush=True)
        except Exception as e:  # noqa: BLE001 - a failed checkpoint must not stop judging
            print(f"    [checkpoint] failed: {type(e).__name__}", flush=True)


def main(config: str, hf_repo: str, expected_passes: int = 4, max_workers: int = 24,
         timeout_min: float = 180.0) -> None:
    """Wait for all passes, combine, judge, and publish."""
    cfg = OmegaConf.load(config)
    model_key = str(cfg.model_key)
    model_root = Path(str(cfg.output_root)).resolve() / model_key
    STATE.mkdir(parents=True, exist_ok=True)

    print(f">>> waiting for {expected_passes} passes in {hf_repo}", flush=True)
    prefixes = wait_for_passes(hf_repo, expected_passes, timeout_min)
    print(f">>> all passes present: {prefixes}", flush=True)

    dirs = download_passes(hf_repo, prefixes, model_root)
    print(f">>> downloaded {len(dirs)} passes -> {model_root}", flush=True)

    from scratch.odcv_combine_passes import main as combine
    combine(config=config)
    combined = sorted(model_root.glob("combined*"))[-1]
    print(f">>> combined -> {combined}", flush=True)
    _push_dir(combined, hf_repo, f"combined/{combined.name}")
    (STATE / "combined_name").write_text(combined.name, encoding="utf-8")

    stop = threading.Event()
    evals = combined / "evaluations"
    threading.Thread(target=_checkpoint_loop,
                     args=(evals, hf_repo, f"combined/{combined.name}/evaluations", stop),
                     daemon=True).start()
    try:
        from src.eval.misalignment.odcv import odcv_judge
        odcv_judge.main(rollout_dir=str(combined), config=config,
                        max_workers=max_workers)
    finally:
        stop.set()
        # Final push regardless of how judging ended: a partial verdict cache is worth
        # keeping, since the judge resumes from exactly this file.
        try:
            _push_dir(combined, hf_repo, f"combined/{combined.name}")
        except Exception as e:  # noqa: BLE001
            print(f"    final push failed: {type(e).__name__}: {e}", flush=True)

    results = combined / "results.json"
    if results.is_file():
        print(">>> RESULTS")
        print(json.dumps(json.loads(results.read_text(encoding="utf-8")), indent=2)[:1500])
    (STATE / "ARM_DONE").write_text(str(combined), encoding="utf-8")
    print(">>> ARM COMPLETE", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
