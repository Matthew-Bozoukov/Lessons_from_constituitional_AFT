# ABOUTME: LoRA SFT of a profiled base model on a Hub mixture under one recipe, via TRL's
# ABOUTME: SFTTrainer with token-budgeted dynamic batching. Runs on the GPU host: `uv run train`.

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from src.train.dynamic_batching import (  # noqa: E402
    plan_micro_batches,
    route_step,
    seq_mean_token_mean_loss,
)
from src.train.launch import (  # noqa: E402
    check_retired_keys,
    recipe_name,
    require_launch_args,
    resolve_model,
    wandb_preflight,
    write_back_pins,
)
from src.train.mask_gate import gate_generation_boundary  # noqa: E402
from src.model_profile import train_memory_entry  # noqa: E402
from src.naming import (  # noqa: E402
    check_hub_name, derive_artifact_name_from_legacy, legacy_subject, mix_subject_from,
    model_name, to_local)
from src.train.masking import (  # noqa: E402
    build_labels,
    check_thinking_declaration,
)


def _collate_padded(features: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Right-pad pre-tokenized examples, padding labels with -100 so they carry no loss.

    `attention_mask` is rebuilt here rather than read from the dataset: SFTTrainer fixes
    its signature columns to input_ids/labels/completion_mask/assistant_masks, so any
    attention_mask column is stripped before the collator runs.

    Args:
        features: Examples with `input_ids` and `labels`.
        pad_token_id: Token used to pad `input_ids`.

    Returns:
        A batch of stacked tensors: `input_ids`, `attention_mask`, `labels`.
    """
    width = max(len(f["input_ids"]) for f in features)
    pad_to = lambda seq, fill: seq + [fill] * (width - len(seq))  # noqa: E731
    batch = {
        "input_ids": torch.tensor([pad_to(f["input_ids"], pad_token_id) for f in features]),
        "attention_mask": torch.tensor(
            [pad_to([1] * len(f["input_ids"]), 0) for f in features]
        ),
        "labels": torch.tensor([pad_to(f["labels"], -100) for f in features]),
    }
    assert batch["input_ids"].shape == batch["labels"].shape == (len(features), width)
    assert (batch["labels"] != -100).any(), "batch has no supervised token; loss would be NaN"
    return batch


class DynamicBatchTrainer(SFTTrainer):
    """SFTTrainer whose step runs its examples as token-budgeted micro-batches.

    Single GPU: the step's examples are grouped by `plan_micro_batches` and run as
    fewer, fuller passes — verified gradient-equivalent to the legacy batch-1 path
    (docs/LOG.md 2026-08-10).

    DDP: every rank receives the SAME full step from the dataloader (see
    `get_train_dataloader`), computes the same deterministic `route_step` plan, and
    executes only its own share. Correctness rests on two mechanisms:

    - Loss scaling: each example's loss is its token-mean x 1/global_batch — equal
      weight regardless of length, same as single-GPU. DDP *averages* gradients
      over ranks, so each rank pre-multiplies by world_size; x N then /N restores
      the plain sum and no example's weight depends on which rank ran it. (The
      returned logged loss is the scaled local sum: Trainer gather-means it across
      ranks, which lands back on the single-GPU scale.)
    - Sync discipline: every backward runs under `no_sync` except each rank's LAST
      local pass, so ranks may run different pass counts and still all-reduce
      exactly once per step — no deadlock, no dummy passes.
    """

    def __init__(self, *args, token_budget: int, global_batch: int,
                 pad_token_id: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._token_budget = int(token_budget)
        self._global_batch = int(global_batch)
        self._pad_token_id = int(pad_token_id)
        self._stream_checked = False

    def get_train_dataloader(self):
        """Under DDP, every rank must see the SAME batches — do not shard.

        The stock Trainer loader is prepared by accelerate, which gives each rank a
        different slice of every batch. Routing needs the opposite: identical full
        steps everywhere, split by `route_step`, not by the sampler. A plain seeded
        DataLoader is identical across ranks by construction (same seed, same code,
        no rank-dependent state); device placement already happens per micro-batch
        inside training_step.
        """
        if self.args.world_size == 1:
            return super().get_train_dataloader()
        from torch.utils.data import DataLoader

        generator = torch.Generator()
        generator.manual_seed(int(self.args.seed))
        return DataLoader(
            self.train_dataset,
            batch_size=self._global_batch,
            shuffle=True,
            generator=generator,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

    def _check_stream_agreement(self, lengths: list[int]) -> None:
        """One-time tripwire: all ranks must hold the same step (lengths checksum)."""
        import torch.distributed as dist

        if self._stream_checked or self.args.world_size == 1 or not dist.is_initialized():
            self._stream_checked = True
            return
        checksum = torch.tensor([len(lengths), sum(lengths)], device=self.args.device)
        gathered = [torch.zeros_like(checksum) for _ in range(self.args.world_size)]
        dist.all_gather(gathered, checksum)
        assert all(torch.equal(g, gathered[0]) for g in gathered), (
            "ranks disagree on the step's rows — the dataloader is sharding; "
            "routing requires identical streams (get_train_dataloader override)")
        self._stream_checked = True

    def training_step(self, model, inputs, num_items_in_batch=None):  # noqa: ARG002
        model.train()
        features: list[dict] = inputs  # identity-collated: list of unpadded examples
        lengths = [len(f["input_ids"]) for f in features]
        self._check_stream_agreement(lengths)
        world_size = int(self.args.world_size)
        # Every rank computes the identical full plan and takes its own share; the
        # divisor stays global_batch even on a short final/smoke batch (legacy
        # behaviour, keeps loss curves comparable).
        local_plan = route_step(lengths, self._token_budget, world_size)[
            int(self.args.process_index)]
        scale = float(world_size)  # DDP mean -> sum; see class docstring
        total = None
        for j, part in enumerate(local_plan):
            is_last = j == len(local_plan) - 1
            sync_ctx = (contextlib.nullcontext() if (is_last or world_size == 1)
                        else self.accelerator.no_sync(model))
            with sync_ctx:
                batch = _collate_padded([features[i] for i in part], self._pad_token_id)
                batch = {k: v.to(self.args.device, non_blocking=True)
                         for k, v in batch.items()}
                labels = batch.pop("labels")
                with self.compute_loss_context_manager():
                    # No `labels` kwarg: the model must not compute its own
                    # (differently normalised) loss; we build it from the logits.
                    out = model(**batch, use_cache=False)
                    loss = seq_mean_token_mean_loss(
                        out.logits, labels, self._global_batch) * scale
                # Backward INSIDE the sync context: gradients accumulate locally,
                # and only the final pass's backward triggers the all-reduce.
                self.accelerator.backward(loss)
            total = loss.detach() if total is None else total + loss.detach()
        return total


def _repo_sha(model_id: str) -> str | None:
    """The commit sha an HF model id resolves to right now, or None for a local path."""
    if model_id.startswith(("/", ".")) or "/" not in model_id:
        return None
    from src.infra.huggingface import hf_api

    return hf_api().model_info(model_id).sha


def _git_sha() -> str:
    """Return the current git SHA if available, else 'nogit'."""
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001 - best-effort provenance only
        return "nogit"



def _warmup_kwargs(ratio: float, n_rows: int, global_batch: int, epochs: float) -> dict:
    """Express the warmup fraction in whichever field this trl's SFTConfig accepts.

    trl >= 1.0 removed `SFTConfig.warmup_ratio` in favour of `warmup_steps`; the training
    pod resolves `trl>=0.27` to the latest (1.10.0 on 2026-08-14) and so rejects the
    former, killing both DDP ranks before step 1. Convert rather than pin, so the config
    keeps stating the scientifically meaningful quantity (a fraction of the schedule)
    whichever trl is installed. The repo lock is trl 0.19.1, which DOES accept
    warmup_ratio, so pinning would break local runs instead.

    Args:
        ratio: Warmup as a fraction of total optimizer steps.
        n_rows: Dataset rows.
        global_batch: Examples per optimizer step.
        epochs: Training epochs.

    Returns:
        A kwargs dict carrying either `warmup_ratio` or the equivalent `warmup_steps`.
    """
    import dataclasses
    import math

    if "warmup_ratio" in {f.name for f in dataclasses.fields(SFTConfig)}:
        return {"warmup_ratio": ratio}
    total_steps = max(1, math.ceil(n_rows / max(global_batch, 1) * epochs))
    steps = max(1, round(ratio * total_steps)) if ratio > 0 else 0
    print(f">>> SFTConfig has no warmup_ratio (trl>=1.0): warmup_ratio={ratio} over "
          f"{total_steps} steps -> warmup_steps={steps}")
    return {"warmup_steps": steps}


def main(config: str, *overrides: str, smoke: bool = False) -> None:
    """Fine-tune a profiled base model with LoRA on a Hub mixture, under one recipe.

    Args:
        config: A RECIPE (`configs/train.yaml`), or a `train_config.yaml` pulled from
            an adapter — which already carries every launch argument and pin below.
        *overrides: OmegaConf dotlist overrides merged over the config, the same
            key=value convention as run_eval. The arm's identity arrives here, never in
            the recipe (src/train/launch.py): `model=qwen36 data_repo=<org>/<mix>
            thinking=true [seed=0] [wandb=true] [data_revision=<sha>] [data_file=<legacy name>]`.
            Positional so that a bare key=value token can never bind to `smoke`.
        smoke: If True, train 2 steps on 8 examples to validate wiring (no HF push).
            Keyword-only: pass `--smoke`.
    """
    # Fragmentation insurance for every training process (CLAUDE.md gotcha 2):
    # variable batch shapes strand the allocator's fixed blocks; expandable segments
    # let one region grow in place instead. The allocator reads this at first CUDA
    # use (model load, below), so setting it here covers every launch form — the
    # `train` alias, the scripts/ shim, torchrun — with no per-launcher export.
    # setdefault keeps an operator-provided value; deliberately NOT in .env or the
    # pod image, so it can never leak into vLLM serving (incompatible).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_initialized():
        print(">>> WARNING: CUDA already initialized before allocator config was set "
              "— PYTORCH_CUDA_ALLOC_CONF may not apply to this process")
    cfg = OmegaConf.load(config)
    if overrides:
        cfg.merge_with_dotlist([str(o) for o in overrides])
    # The launch contract (src/train/launch.py), all before anything downloads or loads:
    # an old per-arm config is refused with the fix, the arm's identity must have been
    # passed, and a W&B run without a key dies here rather than after the model is on
    # the GPU.
    check_retired_keys(cfg)
    require_launch_args(cfg, config)
    recipe = recipe_name(config)
    # W&B is a boolean launch argument (`wandb=true`), never under --smoke; it is the only
    # reporter this repo uses, so transformers' report_to list is built from it here.
    wandb_on = bool(cfg.get("wandb", False)) and not smoke
    report_to = ["wandb"] if wandb_on else []
    reporter = wandb_preflight(wandb_on)
    torch.manual_seed(int(cfg.seed))

    # Under `torchrun` every rank runs this file; these are 1/0 for a plain single-GPU run.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = local_rank == 0

    ts = time.strftime("%Y%m%d_%H%M%S")
    # Ranks start microseconds apart, so a per-rank timestamp would give each its own
    # smoke directory. Rank 0 broadcasts its choice through the environment instead.
    if world_size > 1:
        if is_main:
            os.environ["SYNTHDOC_RUN_TS"] = ts
        ts = os.environ.setdefault("SYNTHDOC_RUN_TS", ts)

    # `model=` is a profile key, an HF id or a path; a stamped train_config.yaml carries
    # the profile it ran with and that wins (src/train/launch.py resolve_model).
    profile, model_id = resolve_model(cfg)

    # --- data: from the HF dataset repo, pinned to the exact revision it resolves to ---
    from src.infra.huggingface import resolve_dataset

    data_path, dataset_ref = resolve_dataset(
        str(cfg.data_repo), cfg.get("data_file"), cfg.get("data_revision"))
    if is_main:
        print(f">>> dataset: {dataset_ref['repo']}@{dataset_ref['revision'][:12]} "
              f"({dataset_ref['file']})")
    ds = load_dataset("json", data_files=data_path, split="train")

    # THE ORGANISM'S NAME, built here and typed nowhere (src/naming.py): today's date,
    # the base model's registered key, the seed, and the SUBJECT of the mixture it is
    # about to train on. Two ways to get that subject, and the input's own name decides:
    #   * a mixture built under the law says it in its name (`...-da-7-cot-only-mix`);
    #   * a pre-law mixture keeps its old name on the Hub and says nothing — so the subject
    #     is derived from what its ROWS are (`source`, `supervise`), never from the name
    #     and never from the recipe's stem. The rows are in memory by now, which is why
    #     naming happens here rather than at config load; it is still before the
    #     tokenizer, the model, and the first GPU-hour.
    #
    # `hf_repo` in the config is the ONE override and it exists for one case: relaunching
    # a run that died, which would otherwise mint a second name under a second date. It
    # still has to pass the law.
    mix = mix_subject_from(str(cfg.data_repo))
    mix_from = "data_repo name"
    if not mix:
        # Pre-law input. The curated table (src/infra/legacy_names.yaml) speaks first — it is
        # how the Hub keeps its old names while everything built from them is named
        # well — and the rows speak only for a repo the table has never seen.
        mix = legacy_subject(str(cfg.data_repo))
        mix_from = "src/infra/legacy_names.yaml (pre-law name)"
    if not mix:
        mix = derive_artifact_name_from_legacy(ds)
        mix_from = "derived from data_repo rows (pre-law name)"
    override = (str(cfg.hf_repo)
                if "hf_repo" in cfg and not OmegaConf.is_missing(cfg, "hf_repo")
                and cfg.hf_repo else "")
    hf_repo = check_hub_name(override, what="model organism (hf_repo override)") \
        if override else model_name(model_id, int(cfg.seed), mix)
    # push=false is the deliberate opt-out for a pod without HF credentials, whose driver
    # pushes the pulled-back adapter instead.
    push = bool(cfg.get("push", True)) and not smoke

    # The run directory is the organism's own local name, so a relaunch of the same arm
    # on the same day lands where its checkpoints are (auto_resume) instead of beside them.
    out_dir = Path(cfg.output_dir) / (f"smoke_{ts}" if smoke else to_local(hf_repo))
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_main:
        print(f">>> output dir: {out_dir}")
        print(f">>> recipe: {recipe}  base model: {model_id} (profile {profile.key})")
        print(f">>> reporter: {reporter}")
        if world_size > 1:
            print(f">>> distributed: {world_size} ranks (DDP), this is rank {local_rank}")
        print(f">>> organism: {hf_repo}  (mix subject {mix!r}, {mix_from})")

    # The arm's eval-time thinking mode is a launch argument (the scientific record),
    # validated against the FULL dataset — the declaration is about the training data, and
    # a smoke subselect of a mostly-replay mixture can legitimately hold zero traces —
    # then stamped into the adapter as training_meta.json. No default.
    thinking = bool(cfg.thinking)
    # Rendered `text` rows need the family's empty-marker literal to classify; pure
    # interchange (`messages`) datasets don't.
    check_thinking_declaration(
        ds, thinking,
        empty_think=profile.empty_think if "text" in ds.column_names else None)
    print(f">>> thinking (declared, validated on all {len(ds)} rows): {thinking}")
    # An arm may unsupervise one property of its reasoning via per-row `mask_spans`.
    # Counted on the FULL dataset for the same reason as the thinking declaration: a
    # smoke subselect of a mostly-replay mixture legitimately holds zero masked rows,
    # but a dataset that declares the column and carries no span anywhere is the
    # ablation silently collapsing into its control.
    n_mask_rows = n_mask_spans = 0
    if "mask_spans" in ds.column_names:
        n_mask_rows = sum(1 for s_ in ds["mask_spans"] if s_)
        n_mask_spans = sum(len(s_) for s_ in ds["mask_spans"] if s_)
        assert n_mask_rows, (
            "dataset has a mask_spans column but no row carries any span; "
            "this arm would be identical to its control")
        print(f">>> mask_spans (validated on all {len(ds)} rows): {n_mask_rows} rows, "
              f"{n_mask_spans} spans")
        # The adapter must say which ablation produced it, or the arms cannot be told
        # apart after the fact.
        training_meta_mask = {
            "mask_spans_rows": n_mask_rows,
            "mask_spans_total": n_mask_spans,
            "mask_property": next((p_ for p_ in ds["mask_property"] if p_), None)
            if "mask_property" in ds.column_names else None,
        }
    else:
        training_meta_mask = {}

    # `supervise` selects WHICH assistant turns (and, for "cot", which part of one) are
    # training targets. Censused on the FULL dataset for the same reason as mask_spans:
    # a column that declares a non-default mode nowhere is an arm collapsed into its
    # control, and that must fail before the GPU bill starts.
    training_meta_supervise = {}
    if "supervise" in ds.column_names:
        supervise_counts = Counter(s_ or "all" for s_ in ds["supervise"])
        non_default = {m: n for m, n in supervise_counts.items() if m != "all"}
        assert non_default, (
            "dataset has a supervise column but every row is 'all'; "
            "this arm would be identical to its control")
        print(f">>> supervise (validated on all {len(ds)} rows): "
              f"{dict(supervise_counts.most_common())}")
        training_meta_supervise = {"supervise_counts": dict(supervise_counts)}

    # Pin the base model to the exact commit this run resolves, the way the dataset is
    # pinned above: an HF id names a moving head, and a rerun a month later should load
    # the weights that were trained on, not the ones that are there now. A stamped
    # train_config.yaml already carries the pin and it is honoured; a local path
    # (`/root/qwen36`) has no commit to pin and is recorded as given.
    base_revision = str(cfg.get("base_model_revision") or "") or _repo_sha(model_id)
    # Everything resolved so far goes back INTO the config, so the train_config.yaml this
    # run saves re-runs it with no other argument (the token budget follows below).
    write_back_pins(cfg, model_id=model_id, base_revision=base_revision,
                    dataset_ref=dataset_ref, profile=profile)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=base_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if smoke:
        ds = ds.select(range(min(8, len(ds))))
    print(f">>> dataset examples: {len(ds)}")
    if "messages" in ds.column_names:
        print(">>> FIRST EXAMPLE messages[0]:")
        print(json.dumps(ds[0]["messages"][0], indent=2)[:500])
    else:
        # Pre-rendered mixtures carry a plain `text` field so per-example chat-template
        # settings (e.g. thinking on/off) are fixed at build time, not training time.
        print(">>> FIRST EXAMPLE text (pre-rendered):")
        print(ds[0]["text"][:800])

    # The loss is ALWAYS assistant-only, via the in-repo mask — not a knob (the 20/80
    # ablation settled it: full-sequence training dilutes the signal with prompt
    # tokens, gotcha 3). TRL's own masking needs `{% generation %}` markers, which
    # Qwen3.6's chat template lacks, and its re-render from `messages` drops the
    # profile's preserve kwargs -- which would silently drop reasoning traces. So the
    # template is applied HERE for interchange datasets, and the masking is built off
    # the exact rendered strings, handing the trainer a ready-made batch.
    if "text" not in ds.column_names and "messages" in ds.column_names:
        # Model-agnostic interchange rows (see src/data/mixture/sources/): the stored
        # data carries semantics only; the model family's syntax -- think blocks
        # included -- is applied now, by the verified profile, where the mask gate
        # can see it. HF's json loader pads message dicts to a shared schema with
        # None for absent keys; those must not reach the template.
        ds = ds.map(
            lambda r: {"text": tokenizer.apply_chat_template(
                [{k: v for k, v in m.items() if v is not None} for m in r["messages"]],
                tokenize=False, add_generation_prompt=False, **profile.render_kwargs)},
            desc="rendering chat template (train-time, ModelProfile render_kwargs)",
        )
        print(f">>> interchange dataset rendered at train time with "
              f"{profile.family} render_kwargs={profile.render_kwargs}")
    if "text" not in ds.column_names:
        raise ValueError("training needs a `messages` (interchange) or "
                         "pre-rendered `text` column")
    max_len = int(cfg.train.max_seq_len)
    # Think supervision is NOT configurable either: the generation-boundary rule in
    # src/train/masking.py is the one way — mask what the model never generates (the
    # `<think>\n` prefill; a WHOLE empty marker, since a healthy model never closes
    # an empty block), supervise what it does (reasoning + `\n</think>` + answer).
    # Runs trained under older rules are reproduced from git history, not a knob.
    # (Orthogonal to the per-row `supervise` field below, which chooses which TURNS are
    # targets at all -- the rule then decides which of a target's tokens count.)
    # A row's optional `supervise` field must be consumed here -- remove_columns
    # discards it right after. Absent or null trains every turn. It is read BEFORE the
    # gate because the gate has to verify the mask this run will actually build: a
    # "cot" row checked as "all" would leave the arm's own code path unverified.
    modes = (list(ds["supervise"]) if "supervise" in ds.column_names
             else ["all"] * len(ds))
    gate_generation_boundary(ds["text"], tokenizer, max_len, profile, thinking,
                             supervise=modes)
    if "supervise" in ds.column_names:
        print(f">>> supervise in THIS selection: "
              f"{dict(Counter(m or 'all' for m in modes).most_common())}")
    # `mask_spans` (character spans of `text`) unsupervises one property of the reasoning
    # without altering the text, so an ablation arm and its control tokenize identically.
    # Validated on the full dataset above; consumed by the map below, which is the last
    # point it exists -- remove_columns discards it immediately after.
    if "mask_spans" in ds.column_names:
        print(f">>> mask_spans in THIS selection: "
              f"{sum(1 for s_ in ds['mask_spans'] if s_)}/{len(ds)} rows")
    ds = ds.map(
        lambda r: build_labels(r["text"], tokenizer, max_len, profile,
                               supervise=r.get("supervise") or "all",
                               mask_spans=r.get("mask_spans")),
        remove_columns=ds.column_names,
        desc="masking non-assistant and prefill tokens",
    )
    n_tok = sum(len(r) for r in ds["input_ids"])
    n_sup = sum(sum(1 for v in r if v != -100) for r in ds["labels"])
    print(f">>> assistant-only loss: {n_sup:,}/{n_tok:,} tokens supervised "
          f"({100 * n_sup / n_tok:.1f}%)")
    first = ds[0]
    kept = [v for v in first["labels"] if v != -100]
    print(">>> FIRST EXAMPLE masked (no loss):")
    print("   ", repr(tokenizer.decode(
        [i for i, v in zip(first["input_ids"], first["labels"]) if v == -100])[:300]))
    print(">>> FIRST EXAMPLE supervised (loss):")
    print("   ", repr(tokenizer.decode(kept)[:300]))

    # --- base model: the profile says how THIS checkpoint loads -----------------------
    # 4-bit QLoRA where the family supports it; bf16 LoRA where it does not (Qwen3.6's
    # hybrid linear-attention layers are not reliably quantised by bitsandbytes).
    bnb = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if profile.load_in_4bit
        else None
    )
    # Multimodal checkpoints (Qwen3.6) expose a conditional-generation class, not causal-LM.
    auto_cls = AutoModelForCausalLM
    if profile.model_class == "image_text_to_text":
        from transformers import AutoModelForImageTextToText

        auto_cls = AutoModelForImageTextToText
    # Under DDP each rank holds a COMPLETE copy of the model on its own GPU. "auto" would
    # instead shard one copy across every visible GPU, which collides with the replica the
    # other rank is building on the same device and deadlocks or OOMs.
    device_map = {"": local_rank} if world_size > 1 else "auto"
    model = auto_cls.from_pretrained(
        model_id,
        revision=base_revision,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation=profile.attn_implementation,
    )
    model.config.use_cache = False
    if smoke:
        names = [n for n, _ in model.named_modules()]
        print(f">>> model class: {type(model).__name__}, {len(names)} modules")
        print(">>> sample module paths (for the profile's lora_target_modules):")
        for n in names[:3] + [x for x in names if x.endswith("q_proj")][:2]:
            print(f"      {n}")

    # Continuing an existing adapter: load its trained weights and hand the trainer an
    # already-wrapped PeftModel, so a second epoch resumes from those weights instead of
    # re-initialising LoRA from scratch. `is_trainable=True` is essential -- without it
    # peft loads the adapter frozen and the run silently trains nothing.
    resume_adapter = cfg.train.get("resume_adapter")
    if resume_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(resume_adapter), is_trainable=True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable > 0, (
            f"adapter {resume_adapter} loaded with no trainable parameters; "
            f"is_trainable was not honoured")
        print(f">>> resumed adapter {resume_adapter} "
              f"({trainable:,} trainable parameters)")

    # peft treats a plain string as a regex over module paths, and a list as exact names;
    # the profile keeps the two types distinct (src/model_profile.py).
    targets = profile.lora_target_modules
    assert targets is not None, (
        f"configs/models/{profile.key}.yaml states no train.lora_target_modules; a "
        "verified family names the modules its LoRA targets")
    peft_cfg = LoraConfig(
        r=int(cfg.lora.r),
        lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )

    # --- dynamic batching, always: token-budgeted micro-batches within the step -------
    # The global batch (batch_size x grad_accum) and the shuffle are the scientific unit;
    # only the grouping into forward passes is decided here, and it is gradient-equivalent
    # to batch-1 accumulation (src/train/dynamic_batching.py). Under DDP every rank sees
    # the same step and route_step splits it, so the optimizer sees the same 16 examples
    # with one rank as with two.
    global_batch = int(cfg.train.batch_size) * int(cfg.train.grad_accum)
    lens = [len(r) for r in ds["input_ids"]]
    # Three tiers, strongest evidence wins: an explicit config override (must cite a
    # probe run in its comment) > this GPU's measured ceiling from the profile's
    # train.memory registry > the dataset's longest row. A registry hit only ever
    # UNLOCKS throughput (budget above the longest row lets long rows share passes);
    # the longest-row default introduces no failure mode batch-1 lacked on the same
    # data (both must run that row). NOT cfg.train.max_seq_len (a truncation ceiling,
    # not a measurement; the model window, 262k, is larger still and irrelevant).
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    measured = train_memory_entry(profile, gpu_name)
    if cfg.train.get("token_budget"):
        dyn_budget, budget_src = int(cfg.train.token_budget), "config (train.token_budget)"
    elif measured is not None:
        dyn_budget = int(measured["max_padded_tokens"])
        budget_src = f"measured ceiling for {measured['gpu']} ({measured['provenance']})"
    else:
        dyn_budget, budget_src = max(lens), (
            f"longest row (no train.memory entry for {gpu_name or 'this GPU'} in "
            f"configs/models/{profile.key}.yaml — add one from a "
            "scratch/probe_batch_memory.py run to unlock more)")
    # Written back so the saved config batches the same way on a rerun, whatever GPU it
    # lands on (grouping is gradient-equivalent either way; this pins the throughput).
    cfg.train.token_budget = dyn_budget
    n_passes = sum(
        len(plan_micro_batches(lens[s:s + global_batch], dyn_budget))
        for s in range(0, len(lens) - global_batch + 1, global_batch))
    print(f">>> dynamic batching: token_budget={dyn_budget} [{budget_src}], "
          f"global_batch={global_batch}, loss_agg=seq-mean-token-mean"
          + (f", DDP routing over {world_size} ranks (route_step)"
             if world_size > 1 else ""))
    print(f">>> ~{n_passes} forward passes/epoch vs {len(lens)} at batch 1 "
          f"({len(lens) / max(n_passes, 1):.1f}x fewer; dataloader-order estimate)")

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=float(cfg.train.epochs),
        # The dataloader delivers one whole optimizer step per batch and training_step
        # does its own accumulation, so HF's accumulation is 1.
        per_device_train_batch_size=global_batch,
        gradient_accumulation_steps=1,
        # Only the LoRA adapters carry gradients; the frozen base never does. Leaving this
        # True makes DDP scan the whole graph for unused parameters every step for nothing.
        ddp_find_unused_parameters=False,
        learning_rate=float(cfg.train.lr),
        lr_scheduler_type=cfg.train.lr_scheduler,
        **_warmup_kwargs(float(cfg.train.warmup_ratio), len(ds), global_batch,
                        float(cfg.train.epochs)),
        weight_decay=float(cfg.train.get("weight_decay", 0.0)),
        logging_steps=int(cfg.train.logging_steps),
        # Periodic checkpoints so a dead pod costs minutes, not the whole run; the run
        # directory is the organism's name, so a relaunch finds them (auto_resume).
        save_strategy=str(cfg.train.get("save_strategy", "epoch")),
        save_steps=int(cfg.train.get("save_steps", 500)),
        save_total_limit=int(cfg.train.get("save_total_limit", 2)),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=max_len,
        # Never packed: without the fla kernels Qwen3.6's gated-delta layers leak
        # recurrent state across packed examples. Dynamic batching pads instead.
        packing=False,
        max_steps=2 if smoke else -1,
        # `wandb=true` and not --smoke, preflighted above; nothing else ever reports.
        report_to=report_to,
        # The W&B run IS the adapter: same name, so the curve and the artifact line up.
        run_name=hf_repo,
        seed=int(cfg.seed),
        # The dataset arrives pre-tokenized with the assistant-only mask already in
        # `labels` (built above); TRL must not re-prepare or re-mask it.
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    trainer = DynamicBatchTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=None if resume_adapter else peft_cfg,
        # Identity collator: the step's examples reach training_step unpadded;
        # padding happens per micro-batch inside the step.
        data_collator=lambda f: f,
        token_budget=dyn_budget,
        global_batch=global_batch,
        pad_token_id=tokenizer.pad_token_id,
    )

    # One provenance stamp for every artifact this run publishes. Assembled AFTER every
    # write-back, so `train_config` is the complete rerun: the recipe plus every launch
    # argument and pin. The eval framework infers serve-time thinking mode from it
    # (CLAUDE.md, "The eval framework").
    training_meta = {
        "thinking": thinking,
        # The recipe locates the hyperparameters in this repo; `train_config` is what
        # actually ran, resolved (overrides merged, pins written back). Both travel with
        # the adapter because a recipe is edited in place: re-running this arm means
        # `uv run train --config train_config.yaml`, read from HERE.
        "recipe": recipe,
        "train_config_name": recipe,
        "mix_subject": mix,
        "mix_subject_from": mix_from,
        "train_config": OmegaConf.to_container(cfg, resolve=True),
        "base_model": model_id,
        "base_model_revision": base_revision,
        "model_profile": profile.to_dict(),
        "dataset": dataset_ref,
        # The exact invocation, overrides included. The resolved config above already
        # carries their EFFECT; this carries the fact that they were overrides, which is
        # what a reader needs to rerun the arm as it was run rather than as it is filed.
        "command": " ".join(sys.argv),
        "git_sha": _git_sha(),
        "timestamp": ts,
        **training_meta_mask,
        **training_meta_supervise,
    }

    # Resume from the newest checkpoint in the run directory when one is there, so a
    # restart continues rather than silently retraining from scratch at full cost.
    resume_ckpt = None
    if bool(cfg.train.get("auto_resume", False)):
        ckpts = sorted(out_dir.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[-1]))
        if ckpts:
            resume_ckpt = str(ckpts[-1])
            if is_main:
                print(f">>> resuming from {resume_ckpt}")

    print(">>> starting training")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))

    # Every rank holds identical adapter weights after the all-reduce, so only rank 0
    # writes the side artifacts; two ranks writing the same paths can interleave and
    # leave a truncated run_meta.json.
    if not is_main:
        return
    tokenizer.save_pretrained(str(adapter_dir))

    # The stamp built above; an adapter without it is a hard error at eval time
    # (CLAUDE.md, "The eval framework").
    (adapter_dir / "training_meta.json").write_text(json.dumps(training_meta, indent=2))
    # The resolved config as YAML beside the stamp: the same content training_meta
    # carries, in the form you hand back to `uv run train --config` — complete.
    (adapter_dir / "train_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    if push:
        from src.infra.huggingface import push_run_dir
        from src.utils import origin_url

        # Same card contract as every other artifact (CLAUDE.md: every upload carries a
        # card), derived from the run's real metadata — the human-readable half beside
        # the machine-readable training_meta.json the eval framework consumes.
        url = push_run_dir(adapter_dir, hf_repo, {
            "experiment": f"LoRA SFT adapter — recipe `{recipe}` on mixture `{mix}` "
                          f"({profile.key}, seed {int(cfg.seed)})",
            "date_generated": ts[:8],
            "constitution": str(cfg.get("constitution") or
                                f"inherited from the training data "
                                f"({dataset_ref['repo']}); "
                                "not declared at launch"),
            "source_repo": f"{origin_url()} @ {_git_sha()}",
            "models": f"base: {model_id}@{base_revision or 'unpinned local path'}",
            "generation_config": json.dumps({
                "recipe": recipe,
                "seed": int(cfg.seed), "thinking": thinking,
                "epochs": float(cfg.train.epochs), "lr": float(cfg.train.lr),
                "batch_size": int(cfg.train.batch_size),
                "grad_accum": int(cfg.train.grad_accum),
                "max_seq_len": max_len,
                "lora": {"r": int(cfg.lora.r), "alpha": int(cfg.lora.alpha),
                         "dropout": float(cfg.lora.dropout)},
                "dynamic_batching": {"token_budget": dyn_budget,
                                     "loss_agg": "seq-mean-token-mean"},
            }),
            "schema": "PEFT LoRA adapter (safetensors) + tokenizer + train_config.yaml "
                      "(the resolved config that ran, every launch argument and pin "
                      "written back: `uv run train --config train_config.yaml` re-runs "
                      "it) + training_meta.json {thinking, recipe, mix_subject, "
                      "train_config, base_model, base_model_revision, model_profile, "
                      "dataset{repo,file,revision}, git_sha, timestamp}",
            "provenance": " ".join(sys.argv),
            "dataset": f"hf.co/datasets/{dataset_ref['repo']}@{dataset_ref['revision']} "
                       f"({dataset_ref['file']})",
        }, private=True, repo_type="model")
        print(f">>> pushed adapter (with training_meta.json + card) to {url}")

    meta = {
        "git_sha": _git_sha(),
        "recipe": recipe,
        "base_model": model_id,
        "base_model_revision": base_revision,
        "dataset": dataset_ref,
        "n_examples": len(ds),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "smoke": smoke,
        "timestamp": ts,
        "world_size": world_size,
        "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        # transformers 5.x does not print the loss to stdout, so without this the loss
        # curve exists only in a reporter we may not be running.
        "log_history": trainer.state.log_history,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    for row in trainer.state.log_history:
        if "loss" in row:
            print(f">>> step {row.get('step')}  loss {row['loss']:.4f}")
    print(f">>> saved adapter to {adapter_dir}")
    print(f">>> run_meta: {out_dir / 'run_meta.json'}")


def cli() -> None:
    """Console entry (`uv run train --config ...`, [project.scripts])."""
    import fire

    fire.Fire(main)
