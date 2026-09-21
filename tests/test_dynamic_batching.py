# ABOUTME: Unit tests for token-budget micro-batch planning and the partition-invariant
# ABOUTME: seq-mean-token-mean loss (src/train/dynamic_batching.py). No network, no GPU.

import pytest

from src.train.dynamic_batching import (
    plan_micro_batches, plan_packs, route_step, seq_mean_token_mean_loss, supervised_positions)

# ---------------------------------------------------------------------------- planner


def test_every_index_exactly_once():
    lengths = [200, 200, 200, 500, 500, 200, 8000, 200, 200, 500, 200, 200, 8000, 200, 500, 200]
    plan = plan_micro_batches(lengths, 4096)
    flat = sorted(i for part in plan for i in part)
    assert flat == list(range(len(lengths)))


def test_budget_respected_except_singletons():
    lengths = [200, 200, 200, 500, 500, 200, 8000, 200, 200, 500, 200, 200, 8000, 200, 500, 200]
    for part in plan_micro_batches(lengths, 4096):
        padded = len(part) * max(lengths[i] for i in part)
        assert padded <= 4096 or len(part) == 1


def test_long_rows_ride_alone():
    lengths = [8000, 100, 8000, 100]
    plan = plan_micro_batches(lengths, 4096)
    singletons = [part for part in plan if len(part) == 1]
    assert sorted(part[0] for part in singletons) == [0, 2]


def test_toy_step_shrinks_to_three_passes():
    # The worked example from the design discussion: 2 x 8000 + 14 x 100 under 8192.
    lengths = [8000] * 2 + [100] * 14
    plan = plan_micro_batches(lengths, 8192)
    assert len(plan) == 3
    assert sorted(len(part) for part in plan) == [1, 1, 14]


def test_uniform_short_rows_fill_one_pass():
    assert len(plan_micro_batches([100] * 16, 8192)) == 1


def test_batch1_reproduced_by_tiny_budget():
    # budget 1 forces every example into its own pass: the legacy grouping.
    plan = plan_micro_batches([300, 200, 100], 1)
    assert sorted(map(tuple, plan)) == [(0,), (1,), (2,)]


def test_deterministic_including_ties():
    lengths = [500, 500, 500, 100, 100, 100]
    assert plan_micro_batches(lengths, 1500) == plan_micro_batches(lengths, 1500)


def test_empty_and_bad_inputs():
    assert plan_micro_batches([], 4096) == []
    with pytest.raises(ValueError):
        plan_micro_batches([100], 0)
    with pytest.raises(ValueError):
        plan_micro_batches([0, 100], 4096)


# ------------------------------------------------------------------------------- loss
# GPU-stack tests: torch is linux-only in this repo's lock, so these run on the pod
# (and any linux dev box) and skip cleanly on macOS. The skip guard is PER TEST —
# a module-level importorskip would skip the torch-free planner tests above too.


def _random_case(seed=0, batch=16, vocab=64):
    """Variable-length rows with -100 prompt masks, padded into one wide batch."""
    import torch
    g = torch.Generator().manual_seed(seed)
    rows = []
    for r in range(batch):
        n = int(torch.randint(4, 40, (1,), generator=g))
        ids = torch.randint(0, vocab, (n,), generator=g)
        labels = ids.clone()
        labels[: max(1, n // 3)] = -100  # unsupervised prompt prefix
        rows.append({"len": n, "labels": labels})
    width = max(r["len"] for r in rows)
    logits = torch.randn(batch, width, vocab, generator=g)
    labels = torch.full((batch, width), -100, dtype=torch.long)
    for r, row in enumerate(rows):
        labels[r, : row["len"]] = row["labels"]
    return logits, labels


def _reference_batch1(logits, labels, global_batch):
    """The legacy weighting: each row alone, per-row token-mean, / global_batch."""
    import torch

    total = torch.zeros(())
    for r in range(logits.shape[0]):
        total = total + seq_mean_token_mean_loss(
            logits[r : r + 1], labels[r : r + 1], global_batch)
    return total


def test_partition_invariance_matches_batch1_reference():
    torch = pytest.importorskip("torch")
    logits, labels, gb = *_random_case(), 16
    whole = seq_mean_token_mean_loss(logits, labels, gb)
    ref = _reference_batch1(logits, labels, gb)
    assert torch.allclose(whole, ref, atol=1e-5)
    # An arbitrary uneven partition — sums to the same loss.
    parts = [[0], [1, 2, 3, 4, 5, 6], [7, 8], list(range(9, 16))]
    split = sum(seq_mean_token_mean_loss(logits[p], labels[p], gb) for p in parts)
    assert torch.allclose(split, ref, atol=1e-5)


def test_token_mean_negative_control():
    """The test suite must be able to DETECT the wrong normaliser.

    Token-mean (pool all supervised tokens, one mean) is the aggregation the
    partition DOES change; if it agreed with the per-example reference, these
    tests would prove nothing.
    """
    torch = pytest.importorskip("torch")
    logits, labels, gb = *_random_case(), 16

    def token_mean(lg, lb):
        import torch.nn.functional as F

        sl = lg[:, :-1, :].float().flatten(0, 1)
        tl = lb[:, 1:].flatten()
        per = F.cross_entropy(sl, tl, ignore_index=-100, reduction="none")
        return per.sum() / tl.ne(-100).sum()

    parts = [[0], list(range(1, 16))]
    split_tm = sum(token_mean(logits[p], labels[p]) / len(parts) for p in parts)
    ref = _reference_batch1(logits, labels, gb)
    assert not torch.allclose(split_tm, ref, atol=1e-3)


def test_zero_supervised_row_raises():
    torch = pytest.importorskip("torch")
    logits, labels = _random_case()
    labels[3, :] = -100
    with pytest.raises(ValueError, match="no supervised tokens"):
        seq_mean_token_mean_loss(logits, labels, 16)


def test_padding_positions_carry_no_loss():
    """Widening a row with pad positions must not move its loss."""
    torch = pytest.importorskip("torch")
    logits, labels, gb = *_random_case(), 16
    ref = seq_mean_token_mean_loss(logits, labels, gb)
    wider = torch.cat([logits, torch.randn(*logits.shape[:2], logits.shape[2])[:, :7]], dim=1)
    wider_labels = torch.cat(
        [labels, torch.full((labels.shape[0], 7), -100, dtype=torch.long)], dim=1)
    assert torch.allclose(seq_mean_token_mean_loss(wider, wider_labels, gb), ref, atol=1e-5)


def _full_logits_reference(logits, labels, global_batch):
    """The pre-2026-09-20 loss, kept verbatim as the oracle: every position upcast and
    scored, unsupervised ones zeroed by ignore_index."""
    import torch.nn.functional as F

    shift_logits, shift_labels = logits[:, :-1, :].float(), labels[:, 1:]
    per_token = F.cross_entropy(
        shift_logits.flatten(0, 1), shift_labels.flatten(),
        ignore_index=-100, reduction="none").view(shift_labels.shape)
    return (per_token.sum(dim=1) / shift_labels.ne(-100).sum(dim=1)).sum() / global_batch


def test_supervised_positions_is_the_union_of_rows_targets():
    torch = pytest.importorskip("torch")
    labels = torch.tensor([[-100, -100, 5, 6, -100, -100],
                           [-100, -100, -100, -100, 7, -100]])
    # targets sit at 2,3 (row 0) and 4 (row 1); the logits that predict them are one back
    assert supervised_positions(labels).tolist() == [1, 2, 3]
    # a supervised token at position 0 has no predictor and asks for nothing
    assert supervised_positions(torch.tensor([[9, -100, -100]])).tolist() == []


def test_restricted_logits_give_the_full_logits_loss_and_gradient():
    """Scoring only the kept positions — as the trainer does via `logits_to_keep` — is
    the same loss AND the same gradient into the hidden states as scoring everything."""
    torch = pytest.importorskip("torch")
    _, labels = _random_case(seed=3)
    gb, hidden, vocab = 16, 12, 64
    g = torch.Generator().manual_seed(4)
    head = torch.randn(vocab, hidden, generator=g)
    states = torch.randn(*labels.shape, hidden, generator=g)

    full_in = states.clone().requires_grad_(True)
    ref = _full_logits_reference(full_in @ head.T, labels, gb)
    ref.backward()

    keep = supervised_positions(labels)
    assert keep.numel() < labels.shape[1] - 1, "case must actually drop positions"
    kept_in = states.clone().requires_grad_(True)
    loss = seq_mean_token_mean_loss(kept_in[:, keep, :] @ head.T, labels, gb, keep)
    loss.backward()

    assert torch.allclose(loss, ref, atol=1e-6)
    assert torch.allclose(kept_in.grad, full_in.grad, atol=1e-6)
    # and the full-logits call path still agrees with the oracle
    assert torch.allclose(seq_mean_token_mean_loss(states @ head.T, labels, gb), ref, atol=1e-6)


def test_restricted_logits_must_match_the_requested_positions():
    torch = pytest.importorskip("torch")
    logits, labels = _random_case(seed=5)
    keep = supervised_positions(labels)
    with pytest.raises(AssertionError, match="logits_to_keep"):
        seq_mean_token_mean_loss(logits, labels, 16, keep)  # full logits, keep passed



# ------------------------------------------------------------- train_memory registry


def test_train_memory_lookup_matches_substring_and_misses_honestly():
    from src.model_profile import model_profile, train_memory_entry
    QWEN36_PROFILE = model_profile("qwen36")

    hit = train_memory_entry(QWEN36_PROFILE, "NVIDIA H200")
    assert hit and hit["gpu"] == "H200" and hit["max_padded_tokens"] == 8000
    assert "provenance" in hit and "probe" in hit["provenance"]
    # H100 has been served on, never probed for training: MUST miss, never guess.
    assert train_memory_entry(QWEN36_PROFILE, "NVIDIA H100 80GB HBM3") is None
    assert train_memory_entry(QWEN36_PROFILE, "") is None


def test_every_train_memory_entry_carries_provenance():
    from src.model_profile import profiles
    MODEL_PROFILES = profiles()

    for profile in MODEL_PROFILES:
        for gpu, entry in profile.train_memory.items():
            assert entry.get("max_padded_tokens", 0) > 0, (profile.family, gpu)
            assert entry.get("provenance"), (
                f"{profile.family}/{gpu}: a train_memory entry without provenance "
                "is folklore, not a measurement")


# ------------------------------------------------------------------ DDP routing


def _flat(plans):
    return sorted(i for plan in plans for part in plan for i in part)


def test_route_ws1_is_exactly_the_single_gpu_planner():
    from src.train.dynamic_batching import route_step

    lengths = [8000, 100, 3000, 100, 500, 500, 200, 100] * 2
    assert route_step(lengths, 8000, 1) == [plan_micro_batches(lengths, 8000)]


def test_route_canonical_case_longs_separate_shorts_split():
    # Jamie's canonical example: two 8000s must land on different GPUs and the
    # fourteen 100s split 7/7 so no rank is left idle.
    from src.train.dynamic_batching import route_step

    lengths = [8000] * 2 + [100] * 14
    plans = route_step(lengths, 8000, 4)
    assert _flat(plans) == list(range(16))
    long_ranks = [r for r, plan in enumerate(plans)
                  for part in plan if any(lengths[i] == 8000 for i in part)]
    assert len(set(long_ranks)) == 2          # the 8000s never share a rank
    short_counts = sorted(sum(len(p) for p in plan) for plan in plans)[-2:]
    assert short_counts == [7, 7]             # the 100s split evenly
    assert all(plan for plan in plans)        # every rank has work


def test_route_every_index_once_and_budget_respected():
    from src.train.dynamic_batching import route_step

    lengths = [4000, 3000, 2000, 2000, 1000, 1000, 1000, 500, 500, 500, 500,
               200, 200, 200, 200, 200]
    for ws in (1, 2, 3, 4, 8):
        plans = route_step(lengths, 8000, ws)
        assert _flat(plans) == list(range(16)), ws
        for plan in plans:
            for part in plan:
                padded = len(part) * max(lengths[i] for i in part)
                assert padded <= 8000 or len(part) == 1


def test_route_split_never_worsens_load_profile():
    # Routed makespan must beat (or tie) dealing the raw passes without repair.
    from src.train.dynamic_batching import ALPHA_CELLS, route_step

    lengths = [4000] + [2000] * 3 + [500] * 12
    ws = 2

    def clock(plan):
        return sum(len(p) * max(lengths[i] for i in p) + ALPHA_CELLS for p in plan)

    routed = max(clock(plan) for plan in route_step(lengths, 8000, ws))
    # naive: LPT over unsplit passes only (reimplemented inline)
    passes = plan_micro_batches(lengths, 8000)
    loads = [0] * ws
    for part in sorted(passes, key=lambda p: -(len(p) * max(lengths[i] for i in p))):
        r = loads.index(min(loads))
        loads[r] += len(part) * max(lengths[i] for i in part) + ALPHA_CELLS
    assert routed <= max(loads)


def test_route_deterministic():
    from src.train.dynamic_batching import route_step

    lengths = [3000, 2000, 2000, 1000, 700, 700, 500, 500, 400, 300, 300,
               200, 200, 100, 100, 100]
    assert route_step(lengths, 4096, 4) == route_step(lengths, 4096, 4)


def test_route_refuses_unfillable_ranks():
    from src.train.dynamic_batching import route_step

    with pytest.raises(ValueError, match="ranks work"):
        route_step([100, 100], 8000, 4)  # 2 examples cannot occupy 4 ranks


def test_ddp_scaling_restores_the_exact_sum():
    """x world_size then DDP-mean must equal the single-GPU gradient exactly.

    Emulated on plain fp32 logits (no model, no GPU): per-rank losses are scaled
    by N, gradients averaged over ranks — the result must be bit-comparable to
    the unscaled whole-batch gradient.
    """
    torch = pytest.importorskip("torch")
    from src.train.dynamic_batching import route_step

    logits, labels, gb = *_random_case(), 16
    lengths = [int(labels[r].ne(-100).sum() + 3) for r in range(gb)]  # any lengths

    ref_logits = logits.clone().requires_grad_(True)
    seq_mean_token_mean_loss(ref_logits, labels, gb).backward()

    ws = 4
    ddp_logits = logits.clone().requires_grad_(True)
    for plan in route_step(lengths, max(lengths) * 3, ws):
        for part in plan:
            (seq_mean_token_mean_loss(ddp_logits[part], labels[part], gb) * ws).backward(
                gradient=torch.ones(()))  # accumulate per-rank scaled grads
    ddp_grad = ddp_logits.grad / ws  # DDP averages over ranks
    assert torch.allclose(ddp_grad, ref_logits.grad, atol=1e-6)


# ---------------------------------------------------------------------------- packing


def test_packs_respect_the_real_token_budget_and_cover_every_index():
    lengths = [200, 200, 200, 500, 500, 200, 8000, 200, 200, 500, 200, 200, 8000, 200, 500, 200]
    packs = plan_packs(lengths, 4096)
    assert sorted(i for p in packs for i in p) == list(range(len(lengths)))
    for p in packs:
        assert sum(lengths[i] for i in p) <= 4096 or len(p) == 1
        assert [lengths[i] for i in p] == sorted((lengths[i] for i in p), reverse=True)
    assert [p for p in packs if len(p) == 1 and lengths[p[0]] == 8000]  # the long rows ride alone
    # 14 short rows of 100 fit one 8192 pack with room to spare; padded passes would pay 14 x 8000
    assert len(plan_packs([8000] * 2 + [100] * 14, 8192)) == 3


def test_packed_routing_gives_every_rank_work_and_balances_tokens():
    lengths = [3000, 2900, 2800, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    plans = route_step(lengths, 8000, 2, packed=True)
    flat = sorted(i for plan in plans for pack in plan for i in pack)
    assert flat == list(range(len(lengths))) and all(plans)
    loads = [sum(lengths[i] for pack in plan for i in pack) for plan in plans]
    assert max(loads) - min(loads) <= 3000
    assert route_step(lengths, 8000, 1, packed=True) == [plan_packs(lengths, 8000)]


def test_packed_loss_equals_the_unpacked_loss_and_gradient():
    """A pack is several examples end to end with `segments` naming the owner of every position.
    Its loss must equal the batch of the same examples run separately — same per-example
    weighting, no token lent across a boundary."""
    torch = pytest.importorskip("torch")
    logits, labels = _random_case(seed=7, batch=4)
    lens = [int((labels[r] != -100).nonzero().max()) + 1 for r in range(4)]  # true lengths
    gb, vocab = 16, logits.shape[2]
    # unpacked reference: each row alone, padded, full logits
    ref_in = logits.clone().requires_grad_(True)
    ref = seq_mean_token_mean_loss(ref_in, labels, gb)
    ref.backward()
    # packed: concatenate the real parts; every example's first label is -100 already (prompt)
    ids = torch.cat([logits[r, :lens[r]] for r in range(4)])[None]
    lab = torch.cat([labels[r, :lens[r]] for r in range(4)])[None]
    seg = torch.cat([torch.full((lens[r],), r) for r in range(4)])[None]
    assert all(labels[r, 0] == -100 for r in range(4))
    packed_in = ids.clone().requires_grad_(True)
    keep = supervised_positions(lab)
    # the logits at a pack boundary's last position predict the NEXT example's first token; that
    # label is -100, so `supervised_positions` never keeps it and nothing crosses the boundary
    loss = seq_mean_token_mean_loss(packed_in[:, keep, :], lab, gb, keep, seg)
    loss.backward()
    assert torch.allclose(loss, ref, atol=1e-6)
    ref_grad = torch.cat([ref_in.grad[r, :lens[r]] for r in range(4)])[None]
    assert torch.allclose(packed_in.grad, ref_grad, atol=1e-6)


# ------------------------------------------------------------------------ token weighting


def test_token_mean_is_the_plain_mean_over_the_step_and_partition_invariant():
    """token_mean_loss with the STEP total as divisor: the micro-batch pieces add up to the
    one number a single full-batch token mean would give, whichever way the step is cut."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from src.train.dynamic_batching import token_mean_loss

    logits, labels = _random_case(seed=11)
    sl, tl = logits[:, :-1, :].float().flatten(0, 1), labels[:, 1:].flatten()
    step_tokens = int(tl.ne(-100).sum())
    plain = F.cross_entropy(sl, tl, ignore_index=-100, reduction="sum") / step_tokens
    whole = token_mean_loss(logits, labels, step_tokens)
    parts = [[0], [1, 2, 3, 4, 5, 6], [7, 8], list(range(9, 16))]
    split = sum(token_mean_loss(logits[p], labels[p], step_tokens) for p in parts)
    assert torch.allclose(whole, plain, atol=1e-5) and torch.allclose(split, plain, atol=1e-5)
    # and it is NOT the per-example weighting: a short row counts for less, not the same
    assert not torch.allclose(whole, seq_mean_token_mean_loss(logits, labels, 16), atol=1e-3)


def test_token_mean_restricted_and_packed_paths_agree_with_full_logits():
    torch = pytest.importorskip("torch")
    from src.train.dynamic_batching import token_mean_loss

    logits, labels = _random_case(seed=12, batch=4)
    step_tokens = int(labels[:, 1:].ne(-100).sum())
    ref = token_mean_loss(logits, labels, step_tokens)
    keep = supervised_positions(labels)
    assert torch.allclose(token_mean_loss(logits[:, keep, :], labels, step_tokens, keep), ref, atol=1e-6)
    lens = [int((labels[r] != -100).nonzero().max()) + 1 for r in range(4)]
    ids = torch.cat([logits[r, :lens[r]] for r in range(4)])[None]
    lab = torch.cat([labels[r, :lens[r]] for r in range(4)])[None]
    seg = torch.cat([torch.full((lens[r],), r) for r in range(4)])[None]
    keep = supervised_positions(lab)
    assert torch.allclose(token_mean_loss(ids[:, keep, :], lab, step_tokens, keep, seg), ref, atol=1e-6)
