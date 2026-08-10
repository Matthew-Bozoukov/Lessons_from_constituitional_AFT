# ABOUTME: Unit tests for token-budget micro-batch planning and the partition-invariant
# ABOUTME: seq-mean-token-mean loss (src/train/dynamic_batching.py). No network, no GPU.

import pytest

from src.train.dynamic_batching import plan_micro_batches, seq_mean_token_mean_loss

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
