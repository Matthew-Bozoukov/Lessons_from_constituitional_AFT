# ABOUTME: Token-budget micro-batching within a fixed-size optimizer step, plus the
# ABOUTME: explicit per-example (seq-mean-token-mean) loss that makes the grouping loss-invariant.

"""Dynamic batching for LoRA SFT: fewer, fuller forward passes, identical gradient.

The trainer's optimizer step is a fixed set of `global_batch` examples chosen by the
dataloader shuffle (the scientific unit — identical to a `batch_size: 1,
grad_accum: 16` run with the same seed). This module changes only how those examples
are grouped into forward passes: short rows share a padded pass, long rows ride
alone, so 97% of the data stops paying the 8k-row worst case that forces batch 1.

Two pieces, and the second is what makes the first legal:

- ``plan_micro_batches``: greedy next-fit over descending lengths under a PADDED-token
  budget (``count x max_len``), the quantity that actually bounds activation and fp32
  logits memory. The trainer defaults the budget to the dataset's longest row — which
  makes dynamic batching introduce NO failure mode the legacy batch-1 path didn't
  already have on the same data (both must run that row), and nothing stronger: a
  dataset whose longest row exceeds anything previously run is unproven for BOTH
  paths, and fails the same way for both (an OOM when that row is reached). A
  measured `ModelProfile.train_memory` entry for the live GPU overrides the default.
  Neither ``max_seq_len`` (a truncation ceiling, not a measurement) nor the model's
  context window (262k for Qwen3.6) says anything about training memory.
- ``seq_mean_token_mean_loss``: each example's token-mean over its own supervised
  tokens, summed, divided by the constant ``global_batch``. Every example weighs
  1/global_batch regardless of length or grouping, so ANY partition of the step
  yields the same loss and gradient (linearity of the gradient of a sum). This is
  the weighting the legacy path produces implicitly (transformers divides the
  per-micro-batch mean by grad_accum when the model opts out of loss kwargs, as
  Qwen3_5ForConditionalGeneration does); here it is explicit and partition-proof.

The design follows verl's dynamic batch size (`verl/utils/seqlen_balancing.py`,
`rearrange_micro_batches`, Apache-2.0) and NeMo-RL's sequence-level loss under
dynamic batching; the cost model is adapted from their sum-of-real-tokens to padded
tokens because these micro-batches are padded, not packed. verl names this loss
aggregation mode "seq-mean-token-mean". Packing is deliberately NOT used: without
the fla/causal_conv1d kernels (absent from our lock), Qwen3.6's gated-delta layers
silently ignore `cu_seqlens` and leak recurrent state across packed examples.
"""

from __future__ import annotations


def plan_micro_batches(lengths: list[int], token_budget: int) -> list[list[int]]:
    """Partition one optimizer step's examples into token-budgeted micro-batches.

    Greedy next-fit over lengths sorted descending: a micro-batch accepts the next
    (shorter or equal) example while ``(count + 1) * max_len`` stays within budget.
    Deterministic for a given input. An example longer than the budget gets a
    singleton micro-batch — that is exactly the legacy batch-1 memory case, never
    worse.

    Args:
        lengths: Token count of each example in the step, in dataloader order.
        token_budget: Max padded tokens (batch x padded length) per forward pass.

    Returns:
        List of micro-batches, each a list of indices into `lengths`. Every index
        appears exactly once.
    """
    if not lengths:
        return []
    if token_budget < 1:
        raise ValueError(f"token_budget must be >= 1, got {token_budget}")
    if min(lengths) < 1:
        raise ValueError("every example must have at least one token")

    # Stable sort: ties keep dataloader order, so the plan is deterministic.
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    plan: list[list[int]] = []
    current: list[int] = []
    current_max = 0  # length of the first (longest) example in `current`
    for i in order:
        if current and (len(current) + 1) * current_max <= token_budget:
            current.append(i)
        else:
            if current:
                plan.append(current)
            current, current_max = [i], lengths[i]
    plan.append(current)
    return plan


def seq_mean_token_mean_loss(logits, labels, global_batch: int):
    """Per-example weighted causal-LM loss, invariant to micro-batch grouping.

    Each example contributes the mean cross-entropy over its OWN supervised tokens
    (labels != -100), scaled by 1/global_batch. Summing this over the micro-batches
    of a step reproduces the plain average of per-example losses — the same
    weighting a batch_size-1 x grad_accum-16 legacy run applies — no matter how the
    step was partitioned (verl calls this aggregation "seq-mean-token-mean").

    Args:
        logits: Float tensor [batch, seq_len, vocab] from a forward WITHOUT labels.
        labels: Long tensor [batch, seq_len]; -100 marks unsupervised positions
            (prompt, padding, think-prefill — already baked by build_labels).
        global_batch: The step's total example count across ALL micro-batches; the
            constant divisor that makes the loss partition-independent.

    Returns:
        Scalar loss tensor: sum over rows of (row token-mean) / global_batch.
    """
    import torch.nn.functional as F

    # Causal shift: position i predicts token i+1; position 0 is never a target.
    # The .float() matches the upcast transformers applies in its own loss path.
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]

    counts = shift_labels.ne(-100).sum(dim=1)
    if (counts == 0).any():
        bad = counts.eq(0).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"micro-batch rows {bad} have no supervised tokens after the causal "
            "shift; build_labels guarantees supervision, so this indicates a "
            "truncation or masking bug upstream"
        )

    per_token = F.cross_entropy(
        shift_logits.flatten(0, 1),
        shift_labels.flatten(),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    per_example = per_token.sum(dim=1) / counts
    return per_example.sum() / global_batch


# Fixed cost of one forward+backward pass, in padded-cell equivalents: measured
# 2026-08-10 on the H200 A/B (docs/LOG.md) — ~1.2s/pass fixed against ~1.3ms/cell.
# Re-measure if the model, GPU, or recipe changes (belongs beside train_memory).
ALPHA_CELLS = 1000


def route_step(lengths: list[int], token_budget: int, world_size: int,
               alpha: int = ALPHA_CELLS) -> list[list[list[int]]]:
    """Partition one optimizer step's examples into per-rank micro-batch plans.

    Option-2 routing (2026-08-11 design discussion): pack all examples into passes
    first (tight rectangles — similar lengths share, minimal padding), deal the
    passes to ranks biggest-first onto the least-loaded rank (Graham's LPT), then
    repair: split a multi-row pass on an overloaded rank and re-deal, keeping the
    change only if the load profile strictly improves. A rank's clock is modelled
    as ``sum(rows x longest) + alpha x passes``; splitting a pass never adds cells
    (``n1*max1 + n2*max2 <= n*max``), so repair can only trade ~1 pass of overhead
    for balance.

    ``world_size == 1`` returns ``[plan_micro_batches(...)]`` verbatim — the
    single-GPU path is byte-identical to the verified 2026-08-10 behaviour.

    The caller owns the DDP arithmetic this plan assumes (see DynamicBatchTrainer):
    every rank computes this same deterministic plan from the same lengths, runs
    only ``plans[rank]``, scales its loss by ``world_size`` (DDP averages gradients
    over ranks; x N then /N restores the plain 1/global_batch sum, so every example
    keeps exactly equal weight regardless of which rank ran it), and syncs only on
    its last local backward (unequal pass counts per rank are legal under no_sync).

    Returns:
        One plan per rank; each plan is a list of micro-batches of indices into
        `lengths`. Every index appears exactly once across all ranks.

    Raises:
        ValueError: a rank would be left with no pass (more ranks than splittable
            work — never the case for 16-example steps on <=8 GPUs).
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    passes = plan_micro_batches(lengths, token_budget)
    if world_size == 1:
        return [passes]
    if not passes:
        return [[] for _ in range(world_size)]

    def cost(part: list[int]) -> int:
        return len(part) * max(lengths[i] for i in part) + alpha

    def deal(parts: list[list[int]]) -> tuple[list[list[list[int]]], list[int]]:
        plans: list[list[list[int]]] = [[] for _ in range(world_size)]
        loads = [0] * world_size
        for i in sorted(range(len(parts)), key=lambda i: (-cost(parts[i]), i)):
            r = min(range(world_size), key=lambda k: (loads[k], k))
            plans[r].append(parts[i])
            loads[r] += cost(parts[i])
        return plans, loads

    def best_split(part: list[int]) -> tuple[list[int], list[int]]:
        # Rows arrive sorted descending (plan_micro_batches builds them that way),
        # so a single cut point suffices; pick the cut with the fewest total cells,
        # ties to the most even halves.
        n = len(part)
        cut = min(range(1, n), key=lambda c: (
            c * lengths[part[0]] + (n - c) * lengths[part[c]], abs(n - 2 * c)))
        return part[:cut], part[cut:]

    def profile(loads: list[int]) -> tuple[int, ...]:
        return tuple(sorted(loads, reverse=True))

    # Repair: strictly improve the sorted load profile (makespan first, then the
    # rest — which also pulls work onto empty ranks). Each accepted split grows
    # the pass count, so the loop is bounded by the example count.
    for _ in range(len(lengths)):
        plans, loads = deal(passes)
        current = profile(loads)
        accepted = False
        for r in sorted(range(world_size), key=lambda k: (-loads[k], k)):
            splittable = [p for p in plans[r] if len(p) > 1]
            if not splittable:
                continue
            target = max(splittable, key=lambda p: (cost(p), p[0]))
            trial = [p for p in passes if p is not target] + list(best_split(target))
            _, trial_loads = deal(trial)
            if profile(trial_loads) < current:
                passes = trial
                accepted = True
                break
        if not accepted:
            break

    plans, _ = deal(passes)
    if any(not plan for plan in plans):
        raise ValueError(
            f"route_step cannot give all {world_size} ranks work from "
            f"{len(lengths)} examples in {len(passes)} passes; use fewer GPUs "
            "for this step size")
    return plans
