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
  The loss only ever READS the logits at supervised positions, so the trainer asks the
  model for those alone (``supervised_positions`` -> the forward's ``logits_to_keep``):
  prompt, user and padding positions never reach `lm_head`, and a ~250k-vocab fp32
  logits row is only built for a token that carries loss. Dropped positions had zero
  weight, so loss and gradient are unchanged.

The design follows verl's dynamic batch size (`verl/utils/seqlen_balancing.py`,
`rearrange_micro_batches`, Apache-2.0) and NeMo-RL's sequence-level loss under
dynamic batching; the cost model is adapted from their sum-of-real-tokens to padded
tokens because these micro-batches are padded, not packed. verl names this loss
aggregation mode "seq-mean-token-mean". Packing is deliberately NOT used: on the
torch fallback Qwen3.6's gated-delta layers silently ignore `cu_seqlens` and leak
recurrent state across packed examples. The lock now carries the fla kernels (they
speed up the padded passes here), but packing on top of them is UNVERIFIED for this
family — it needs its own no-leak test before anything packs.
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


def plan_packs(lengths: list[int], token_budget: int) -> list[list[int]]:
    """Partition one optimizer step's examples into PACKS: sequences concatenated end to end.

    First-fit decreasing under a REAL-token budget (the sum of the members' lengths): the
    pack is one row with no padding, so the budget bounds exactly the tokens the forward pass
    touches. An example longer than the budget is a pack of its own — the same legacy
    batch-1 case as `plan_micro_batches`, never worse. Members are kept longest-first so
    `route_step`'s single-cut split applies unchanged.

    The maths is unchanged by construction: with boundaries respected (varlen attention,
    `cu_seqlens` in the gated-delta kernel, `seq_idx` in the conv), each member's forward is
    the computation it would get alone, and `seq_mean_token_mean_loss` weighs it as its own
    row via `segments`. Only the batching differs. scratch/pack_equality_check.py is the
    test that this holds on the live model; do not pack on a stack it has not passed on.

    Returns:
        List of packs, each a list of indices into `lengths`; every index exactly once.
    """
    if token_budget < 1:
        raise ValueError(f"token_budget must be >= 1, got {token_budget}")
    if any(n < 1 for n in lengths):
        raise ValueError("every example must have at least one token")
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    packs: list[list[int]] = []
    sums: list[int] = []
    for i in order:
        for k, total in enumerate(sums):
            if total + lengths[i] <= token_budget:
                packs[k].append(i)
                sums[k] = total + lengths[i]
                break
        else:
            packs.append([i])
            sums.append(lengths[i])
    return packs


def supervised_positions(labels):
    """Sequence positions whose logits the loss reads, for the forward's `logits_to_keep`.

    Position i predicts token i+1, so i is needed when ANY row of the micro-batch
    supervises token i+1. The union over rows is what one index tensor can express
    (transformers slices `hidden_states[:, idx, :]` before `lm_head`); rows that do not
    supervise a kept position are dropped again inside the loss.

    Args:
        labels: Long tensor [batch, seq_len]; -100 marks unsupervised positions.

    Returns:
        1-D long tensor of ascending positions in [0, seq_len - 1).
    """
    return labels[:, 1:].ne(-100).any(dim=0).nonzero(as_tuple=True)[0]


def seq_mean_token_mean_loss(logits, labels, global_batch: int, positions=None, segments=None):
    """Per-example weighted causal-LM loss, invariant to micro-batch grouping.

    Each example contributes the mean cross-entropy over its OWN supervised tokens
    (labels != -100), scaled by 1/global_batch. Summing this over the micro-batches
    of a step reproduces the plain average of per-example losses — the same
    weighting a batch_size-1 x grad_accum-16 legacy run applies — no matter how the
    step was partitioned (verl calls this aggregation "seq-mean-token-mean").

    Args:
        logits: Float tensor from a forward WITHOUT labels: [batch, seq_len, vocab], or
            [batch, len(positions), vocab] when the forward was given
            `logits_to_keep=positions`.
        labels: Long tensor [batch, seq_len]; -100 marks unsupervised positions
            (prompt, padding, think-prefill — already baked by build_labels).
        global_batch: The step's total example count across ALL micro-batches; the
            constant divisor that makes the loss partition-independent.
        positions: The `supervised_positions(labels)` the logits were restricted to, or
            None for full-sequence logits.
        segments: For a PACKED row (batch 1, several examples end to end): long tensor
            [1, seq_len] giving each position's example index in 0..n-1. Each example is then
            its own row of the weighting — the per-example token-mean over its own supervised
            tokens — exactly as it would be unpacked. None means one example per batch row.

    Returns:
        Scalar loss tensor: sum over rows of (row token-mean) / global_batch.
    """
    import torch
    import torch.nn.functional as F

    # Causal shift: position i predicts token i+1; position 0 is never a target.
    if segments is None:
        n_examples = labels.shape[0]
        owner_full = torch.arange(n_examples, device=labels.device)[:, None].expand(-1, labels.shape[1] - 1)
    else:
        assert labels.shape[0] == 1 and segments.shape == labels.shape, "a packed row is batch 1"
        n_examples = int(segments.max().item()) + 1
        # The token at position i+1 belongs to the example that owns position i+1: the label
        # side of the shift, so a pack boundary never lends a token across examples.
        owner_full = segments[:, 1:]
    counts = torch.bincount(owner_full[labels[:, 1:].ne(-100)], minlength=n_examples)
    if (counts == 0).any():
        bad = counts.eq(0).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"micro-batch rows {bad} have no supervised tokens after the causal "
            "shift; build_labels guarantees supervision, so this indicates a "
            "truncation or masking bug upstream"
        )

    if positions is None:
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        owner = owner_full
    else:
        assert logits.shape[1] == positions.numel(), (
            f"logits cover {logits.shape[1]} positions but {positions.numel()} were "
            "requested; the forward did not honour logits_to_keep")
        shift_logits = logits
        shift_labels = labels[:, positions + 1]
        owner = owner_full[:, positions]

    # Only supervised cells are upcast and scored: an unsupervised cell's cross-entropy
    # was always multiplied by zero, and at a ~250k vocab its fp32 row is what bounded
    # memory. The .float() matches the upcast transformers applies in its own loss path.
    supervised = shift_labels.ne(-100)
    per_token = F.cross_entropy(
        shift_logits[supervised].float(), shift_labels[supervised], reduction="none")
    rows = owner[supervised]
    per_example = torch.zeros(
        n_examples, dtype=per_token.dtype, device=per_token.device
    ).index_add(0, rows, per_token) / counts
    return per_example.sum() / global_batch


# Fixed cost of one forward+backward pass, in padded-cell equivalents: measured
# 2026-08-10 on the H200 A/B (docs/LOG.md) — ~1.2s/pass fixed against ~1.3ms/cell.
# Re-measure if the model, GPU, or recipe changes (belongs beside train_memory).
ALPHA_CELLS = 1000


def route_step(lengths: list[int], token_budget: int, world_size: int,
               alpha: int = ALPHA_CELLS, *, packed: bool = False) -> list[list[list[int]]]:
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

    ``packed=True`` routes PACKS (`plan_packs`) instead of padded passes: a pass then costs
    its real tokens plus alpha, and a split cuts a pack's member list, which is sorted
    longest-first like a pass. Everything else — the deal, the repair, the guarantees — is
    the same code.

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
    passes = plan_packs(lengths, token_budget) if packed else plan_micro_batches(lengths, token_budget)
    if world_size == 1:
        return [passes]
    if not passes:
        return [[] for _ in range(world_size)]

    def cost(part: list[int]) -> int:
        if packed:
            return sum(lengths[i] for i in part) + alpha
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
        if packed:  # cells are real tokens either way; cut for the most even halves
            cut = min(range(1, n), key=lambda c: (
                abs(sum(lengths[i] for i in part[:c]) - sum(lengths[i] for i in part[c:])), c))
        else:
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
