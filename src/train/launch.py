# ABOUTME: The launch contract of `uv run train`: what a recipe may carry, what the command
# ABOUTME: line must supply, how `model=` resolves, and the W&B preflight that runs before any GPU spend.

"""One recipe, many arms — the contract that makes that safe.

`configs/train/sft.yaml` — the one train config — holds the training recipe (LoRA shape,
optimizer, schedule, sequence length) and NOTHING about a particular arm. Everything that changes from arm to
arm is a launch argument:

    uv run train --config configs/train/sft.yaml model=qwen36 data_repo=<org>/<mix> \\
        thinking=true [seed=0] [wandb=true] [data_revision=<sha>] [data_file=<legacy name>]

The trainer writes what it actually resolved BACK into the config it saves with the
adapter (`train_config.yaml`): the HF id `model=` stood for and the base revision it
loaded, the data file and revision it read, the token budget it batched under, and the
model profile it rendered and masked with (as a `profile:` block). That file is complete:
`uv run train --config train_config.yaml` re-runs the arm with no other argument, up to
GPU nondeterminism. This module is importable without torch/trl (linux-only in the lock),
which is why the contract lives here rather than in train_lora.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from src.infra.huggingface import hf_repo_id, push_run_dir
from src.model_profile import ModelProfile, model_key, model_profile
from src.utils import origin_url

# The project W&B runs land in unless the environment says otherwise (transformers'
# own default is the anonymous "huggingface" project).
DEFAULT_WANDB_PROJECT = "lasr"

# What every launch must supply — none has a default, because each is the arm's identity
# and a default would let two arms differ in something nobody typed.
LAUNCH_ARGS = ("model", "data_repo", "thinking")

# Keys a recipe no longer carries. Each says where the fact went, so an old per-arm config
# (configs/train/archive/) or a pre-2026-09-05 train_config.yaml pulled from an adapter
# fails with the fix in hand rather than running something silently different. An arm
# trained under the old shape re-runs from the code at its training_meta.json `git_sha`.
RETIRED_KEYS = {
    "train.dynamic_batching": "dynamic batching is always on; `train.token_budget: N` "
                              "overrides the resolved budget (cite a probe run)",
    "train.packing": "always off — dynamic batching pads, it never packs",
    "train.assistant_only_loss": "always on — the in-repo mask (CLAUDE.md gotcha 3)",
    "train.loss_type": "the loss is seq-mean-token-mean, computed by DynamicBatchTrainer",
    "train.mask_empty_think": "the generation-boundary rule masks a whole empty marker, "
                              "unconditionally",
    "train.load_in_4bit": "moved to configs/models/<key>.yaml `train.load_in_4bit`",
    "train.attn_implementation": "moved to configs/models/<key>.yaml "
                                 "`train.attn_implementation`",
    "train.push_to_hub": "the run pushes its own adapter; `push=false` opts out",
    "train.hub_model_id": "the adapter's name is built, never declared (src/naming.py)",
    "train.hub_strategy": "checkpoints stay local; the final adapter is the artifact",
    "train.report_to": "W&B is the only reporter: `wandb=true` on the command line "
                       "(a boolean, recorded in the artifact), never transformers' list",
    "model_class": "moved to configs/models/<key>.yaml `train.model_class`",
    "lora.target_modules": "moved to configs/models/<key>.yaml `train.lora_target_modules`",
    "data_path": "training data comes from the Hub: `data_repo=<org>/<name>`",
}


def check_retired_keys(cfg) -> None:
    """Refuse a config that carries a key the trainer no longer reads, naming the fix."""
    hits = [k for k in RETIRED_KEYS if OmegaConf.select(cfg, k) is not None]
    if hits:
        lines = "\n".join(f"  {k}: {RETIRED_KEYS[k]}" for k in hits)
        raise ValueError(
            "train config carries keys the trainer no longer reads:\n" + lines +
            "\nThe recipe (configs/train/sft.yaml) is the LoRA shape, optimizer, schedule and "
            "sequence length only; the model half is configs/models/<key>.yaml.")


def require_launch_args(cfg, config_path: str) -> None:
    """Refuse to start without the arm's identity: model, data and thinking declaration."""
    missing = [k for k in LAUNCH_ARGS if k not in cfg or OmegaConf.is_missing(cfg, k)]
    if missing:
        raise ValueError(
            f"`uv run train` needs {missing} as launch arguments:\n"
            f"  uv run train --config {config_path} model=qwen36 "
            "data_repo=<org>/<mix> thinking=true [seed=0]\n"
            "A recipe names no model and no data, so one file trains every arm; a "
            "train_config.yaml pulled from an adapter already carries them.")


def resolve_model(cfg) -> tuple[ModelProfile, str]:
    """The profile and the checkpoint id a launch's `model=` stands for.

    `model=` is a profile key (`qwen36`), an HF id, or a local path; anything the
    profile's `match` identifies. A `train_config.yaml` written by a run also carries a
    `profile:` block — the profile that run rendered and masked with — and that block
    wins over configs/models/, so a rerun does not depend on the directory having moved
    on. The verified-family refusal applies either way.

    Returns:
        (profile, model_id): `model_id` is the HF id or path to load — the profile's
        `model` when a bare key was given, else the value as typed.
    """
    value = str(cfg.model)
    stamp = cfg.get("profile")
    if stamp:
        profile = ModelProfile.from_dict(OmegaConf.to_container(stamp, resolve=True))
        assert profile.verified, (
            f"the config's `profile:` block ({profile.key}) has no verified template; "
            "a run cannot have produced it — remove the block to read configs/models/")
        assert model_key(value) == profile.key, (
            f"the config's `profile:` block is {profile.key!r} but `model` is {value!r}, "
            f"which is {model_key(value)!r}; they came from different runs")
    else:
        profile = model_profile(value)
    model_id = profile.model if value == profile.key else value
    return profile, model_id


def wandb_preflight(enabled: bool, env=None) -> str:
    """Refuse a W&B run that would die after the model loads; return the reporter line.

    transformers treats an unavailable reporter as a hard error, and wandb without a key
    in a non-interactive shell fails at `init` — which happens AFTER the 55GB base model
    is on the GPU. So: if the run reports to wandb (`wandb=true` on the command line, or a
    stamped train_config.yaml that ran with it), the environment must carry
    `WANDB_API_KEY` (from .env, or the pod's .env that `runpod up --push_env` wrote) or
    set `WANDB_MODE` to `disabled`/`offline`. The project defaults to `lasr` when the
    environment names none, so runs do not land in transformers' anonymous default.

    Args:
        enabled: The run's `wandb` flag.
        env: The environment mapping (os.environ by default; injectable for tests).
    """
    env = os.environ if env is None else env
    if not enabled:
        return "no reporter (wandb=false)"
    mode = str(env.get("WANDB_MODE") or "").lower()
    if mode in ("disabled", "offline"):
        return f"wandb {mode} (WANDB_MODE)"
    if not env.get("WANDB_API_KEY"):
        raise RuntimeError(
            "wandb=true, but WANDB_API_KEY is not set and WANDB_MODE is "
            "not disabled/offline. Put WANDB_API_KEY (and optionally WANDB_PROJECT, "
            "WANDB_ENTITY) in .env — `uv run runpod up --push_env` carries them to a pod, or "
            "scp the .env — or drop `wandb=true` for no reporter. Checked before "
            "anything downloads or loads, so a missing key costs no GPU time.")
    env.setdefault("WANDB_PROJECT", DEFAULT_WANDB_PROJECT)
    entity = env.get("WANDB_ENTITY") or "<the key's own account>"
    return f"wandb project={env['WANDB_PROJECT']} entity={entity}"


def write_back_pins(cfg, *, model_id: str, base_revision: str | None,
                    dataset_ref: dict, profile: ModelProfile) -> None:
    """Record into `cfg` what the run resolved, so the saved config is a complete rerun.

    Called once the pins exist and before the config is stamped into the adapter. The
    token budget is written by the trainer after it resolves it on the live GPU.
    """
    cfg.model = model_id
    if base_revision:
        cfg.base_model_revision = base_revision
    cfg.data_repo = dataset_ref["repo"]
    cfg.data_file = dataset_ref["file"]
    cfg.data_revision = dataset_ref["revision"]
    cfg.profile = profile.to_dict()


def recipe_name(config_path: str) -> str:
    """The recipe a run used, by the config's stem — recorded, never part of a name."""
    return Path(config_path).stem


def adapter_card_fields(meta: dict) -> dict:
    """The card an adapter ships with, built from its training_meta.json alone.

    Same card contract as every other artifact (CLAUDE.md: every upload carries a card),
    derived from the run's real metadata — the human-readable half beside the
    machine-readable stamp the eval framework consumes. Pure: everything it needs is in
    the stamp, which is what lets `push_adapter` publish a run whose own push died.
    """
    cfg, ds = meta["train_config"], meta["dataset"]
    return {
        "experiment": f"LoRA SFT adapter — recipe `{meta['recipe']}` on mixture "
                      f"`{meta['mix_subject']}` ({meta['model_profile']['key']}, "
                      f"seed {int(cfg['seed'])})",
        "date_generated": str(meta["timestamp"])[:8],
        "constitution": str(cfg.get("constitution") or
                            f"inherited from the training data ({ds['repo']}); "
                            "not declared at launch"),
        "source_repo": f"{origin_url()} @ {meta['git_sha']}",
        "models": f"base: {meta['base_model']}@"
                  f"{meta.get('base_model_revision') or 'unpinned local path'}",
        "generation_config": json.dumps({
            "recipe": meta["recipe"],
            "seed": int(cfg["seed"]), "thinking": bool(meta["thinking"]),
            "epochs": float(cfg["train"]["epochs"]), "lr": float(cfg["train"]["lr"]),
            "batch_size": int(cfg["train"]["batch_size"]),
            "grad_accum": int(cfg["train"]["grad_accum"]),
            "max_seq_len": int(cfg["train"]["max_seq_len"]),
            "lora": {"r": int(cfg["lora"]["r"]), "alpha": int(cfg["lora"]["alpha"]),
                     "dropout": float(cfg["lora"]["dropout"])},
            "dynamic_batching": {"token_budget": int(cfg["train"]["token_budget"]),
                                 "loss_agg": "seq-mean-token-mean"},
        }),
        "schema": "PEFT LoRA adapter (safetensors) + tokenizer + train_config.yaml "
                  "(the resolved config that ran, every launch argument and pin "
                  "written back: `uv run train --config train_config.yaml` re-runs "
                  "it) + training_meta.json {organism, thinking, recipe, mix_subject, "
                  "train_config, base_model, base_model_revision, model_profile, "
                  "dataset{repo,file,revision}, git_sha, timestamp}",
        "provenance": str(meta["command"]),
        "dataset": f"hf.co/datasets/{ds['repo']}@{ds['revision']} ({ds['file']})",
    }


def push_adapter(adapter_dir: Path, meta: dict | None = None) -> str:
    """Publish a saved adapter directory under the organism name its stamp records.

    `train_lora.main` calls this at the end of a run; `scratch/push_saved_adapter.py`
    calls it for a run whose push failed after the adapter was written. Either way the
    card comes from `training_meta.json` and the destination from
    `hf_repo_id(meta["organism"])` — the name the law minted at launch, qualified with
    the environment's org.
    """
    adapter_dir = Path(adapter_dir)
    if meta is None:
        meta = json.loads((adapter_dir / "training_meta.json").read_text())
    assert meta.get("organism"), (
        f"{adapter_dir}/training_meta.json has no `organism`; stamps from before "
        "2026-09-06 carry none — pass the name explicitly (scratch/push_saved_adapter.py)")
    return push_run_dir(adapter_dir, hf_repo_id(str(meta["organism"])),
                        adapter_card_fields(meta), private=True, repo_type="model")
