# ABOUTME: Trains the two verbose-CoT arms on RunPod 2xH200 pods, by re-pointing the t10
# ABOUTME: driver at this experiment's configs rather than forking its 277 lines a third time.

"""Train a verbose-CoT arm on a throwaway 2xH200 RunPod pod.

    uv run python scratch/verbose_cot/train_pod.py --arm rows   bundle
    uv run python scratch/verbose_cot/train_pod.py --arm rows   up
    uv run python scratch/verbose_cot/train_pod.py --arm rows   status --pod <id>
    uv run python scratch/verbose_cot/train_pod.py --arm rows   pull   --pod <id>
    uv run python scratch/verbose_cot/train_pod.py --arm rows   push
    uv run python scratch/less/teardown.py --pod <id>

`--arm tokens` runs the token-matched arm instead. The two arms are independent: separate
pods, separate adapters, and either may be run without the other.

WHY A SHIM AND NOT A FORK. scratch/trait10_curiosity/train_pod.py is already a fork of
scratch/less/train_arms.py, and everything this experiment needs it to do -- credential-free
pod, public code bundle, `nika-` name prefix so a shared-account teardown can tell our pod
from a teammate's, torchrun DDP over 2 ranks, adapter pulled over :8080 and pushed from the
machine that holds the token -- it already does, identically. The only per-experiment facts
are seven module constants. A third copy would be a third place for the protocol to drift,
and the protocol being identical is precisely what makes these arms comparable. So this
loads that module and re-points those constants; no logic is duplicated and the t10 file is
not touched.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_T10 = ROOT / "scratch" / "trait10_curiosity" / "train_pod.py"
_spec = importlib.util.spec_from_file_location("t10_train_pod", _T10)
driver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(driver)

# One code bundle serves both arms: the pod runs whichever config `TRAIN_CONFIG` names, and
# the bundle is the repo, not the arm.
BUNDLE = "LASR-Callum/2026-08-25-verbose-cot-arm-code-bundle"

ARMS = {
    "rows": {
        "TRAIN_CONFIG":
            "configs/train/lora_qwen36_t2_9284_da716_verbose_dynbatch_2xh200.yaml",
        "ADAPTER_REPO":
            "LASR-Callum/qwen3.6-27b-lora-t2-9284-da716-verbose-r64-dynbatch",
        "OUT_NAME": "train_t2_9284_da716_verbose_dynbatch",
        "DEST": "output/adapters/verbose_rows",
        "POD_NAME": "nika-verbose-rows-arm",
    },
    "tokens": {
        "TRAIN_CONFIG":
            "configs/train/lora_qwen36_t2_9284_da_verbose_tokenmatched_dynbatch_2xh200.yaml",
        "ADAPTER_REPO":
            "LASR-Callum/qwen3.6-27b-lora-t2-9284-da-verbose-tokenmatched-r64-dynbatch",
        "OUT_NAME": "train_t2_9284_da_verbose_tokenmatched_dynbatch",
        "DEST": "output/adapters/verbose_tokenmatched",
        "POD_NAME": "nika-verbose-tokens-arm",
    },
}


def _select(arm: str) -> dict:
    """Point the driver's module globals at one arm; they are read at call time."""
    if arm not in ARMS:
        raise SystemExit(f"unknown --arm {arm!r}; expected one of {sorted(ARMS)}")
    spec = ARMS[arm]
    driver.BUNDLE = BUNDLE
    for key in ("TRAIN_CONFIG", "ADAPTER_REPO", "OUT_NAME", "DEST"):
        setattr(driver, key, spec[key])
    # CODE is a list LITERAL that embedded TRAIN_CONFIG when the module was imported, so
    # re-pointing the constant alone leaves the bundle shipping the t10 config and the pod
    # training the wrong arm -- silently, since the run would succeed on real data.
    #
    # Both arms' configs go in EVERY bundle, not just the selected one, because the two
    # arms share one bundle repo: bundling the second arm would otherwise overwrite the
    # first, and a pod that had not yet downloaded would come up on the wrong config. The
    # pod runs whichever config its bootstrap names, so carrying the other one is inert.
    driver.CODE = ([c for c in driver.CODE if not c.endswith(".yaml")]
                   + sorted(a["TRAIN_CONFIG"] for a in ARMS.values()))
    cfg = ROOT / spec["TRAIN_CONFIG"]
    assert cfg.is_file(), f"config not found: {cfg}"
    print(f">>> arm={arm}  config={spec['TRAIN_CONFIG']}")
    print(f">>> adapter={spec['ADAPTER_REPO']}")
    return spec


class Arm:
    """Fire entry point. `--arm` is chosen before the command runs."""

    def __init__(self, arm: str = "rows") -> None:
        """Bind one arm's constants onto the shared driver."""
        self.spec = _select(arm)

    def bundle(self) -> str:
        """Publish the code bundle the pod downloads (no credentials on the box)."""
        return driver.bundle(repo=BUNDLE)

    def up(self, gpus: int = 2, disk_gb: int = 200, gpu: str | None = None) -> str:
        """Provision the pod and start training. Name is fixed to this arm's `nika-` name."""
        return driver.up(gpus=gpus, name=self.spec["POD_NAME"], disk_gb=disk_gb, gpu=gpu)

    def status(self, pod: str, lines: int = 8) -> str:
        """Tail the pod's training log."""
        return driver.status(pod=pod, lines=lines)

    def pull(self, pod: str) -> str:
        """Fetch the adapter over the pod's :8080 proxy."""
        return driver.pull(pod=pod, dest=self.spec["DEST"])

    def push(self, private: bool = False) -> str:
        """Push the pulled adapter to HF from this machine, which holds the token."""
        return driver.push(dest=self.spec["DEST"], private=private)


if __name__ == "__main__":
    sys.exit(fire.Fire(Arm))
