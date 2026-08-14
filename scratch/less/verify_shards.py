# ABOUTME: Proves that sharding the gradient datastore across workers changes nothing that
# ABOUTME: matters — compares a sharded run against an unsharded one row by row.

"""Check that a sharded datastore agrees with an unsharded one.

    uv run python scratch/less/verify_shards.py --a output/less_grads/whole \\
        --b output/less_grads/split --split train

This is the test that licenses running the real datastore across 4 GPUs. Sharding is only
safe because each example's feature depends on nothing but that example, the checkpoint's
frozen Adam moments and the seeded projection basis -- no accumulation, no shared state.
That is an argument, and this turns it into a measurement.

It needs no second GPU: running shard 0 of 2 and shard 1 of 2 sequentially on one device
and comparing against the whole exercises exactly the code path that a 4-GPU run uses. The
extra GPUs add throughput, not new logic, so the correctness question is settled here.

What a failure would mean, in rough order of likelihood: the projection seed differs
between workers (the features live in unrelated bases); the LoRA parameter ordering is not
stable across processes (the flat vectors are permuted relative to the Adam moments); or
dropout is live, making the gradient a function of the RNG as well as the row.

The comparison is by cosine, not equality, and rows will NOT be bit-identical. Measured on
this model, two backward passes over the same row differ by 3.9e-03 relative -- CUDA
atomics and gradient-checkpoint recomputation -- for an end-to-end 1-cos of ~5e-05. That is
two orders of magnitude below the count-sketch's own approximation error, so it cannot move
a ranking; the default tolerance is set to separate it from real defects, which perturb
features far more. The bit-identical count is still reported, as information rather than as
a pass condition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from influence import load_split  # noqa: E402  (same-dir module)


def compare(a: dict, b: dict, tol: float) -> list[str]:
    """Compare two reassembled splits, returning human-readable problems."""
    problems: list[str] = []
    if set(a) != set(b):
        return [f"different checkpoints: {sorted(a)} vs {sorted(b)}"]

    for ckpt in sorted(a):
        x, y = a[ckpt], b[ckpt]
        if x["proj_seed"] != y["proj_seed"] or x["dim"] != y["dim"]:
            problems.append(f"{ckpt}: projection basis differs "
                            f"(seed {x['proj_seed']}/{y['proj_seed']}, dim {x['dim']}/{y['dim']})")
            continue
        if set(x["less_ids"]) != set(y["less_ids"]):
            only_a = set(x["less_ids"]) - set(y["less_ids"])
            only_b = set(y["less_ids"]) - set(x["less_ids"])
            problems.append(f"{ckpt}: row sets differ (+{len(only_a)} / -{len(only_b)})")
            continue

        # Shards emit rows strided, so align by id before comparing.
        pos = {i: k for k, i in enumerate(y["less_ids"])}
        yf = y["features"][[pos[i] for i in x["less_ids"]]]
        xf = x["features"]

        exact = int((xf == yf).all(dim=1).sum())
        delta = (xf - yf).abs()
        scale = xf.abs().amax().clamp(min=1e-12)
        cos = torch.nn.functional.cosine_similarity(xf, yf, dim=1)
        worst = float(cos.min())
        print(f"  {ckpt}: {len(xf)} rows | bit-identical {exact}/{len(xf)} | "
              f"max|Δ| {float(delta.max()):.3e} (rel {float(delta.max() / scale):.3e}) | "
              f"min cosine {worst:.8f}")
        if worst < 1 - tol:
            problems.append(f"{ckpt}: min row cosine {worst:.8f} below 1-{tol}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, type=Path, help="reference (unsharded) grads dir")
    ap.add_argument("--b", required=True, type=Path, help="sharded grads dir")
    ap.add_argument("--split", default="train", choices=("train", "val"))
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="allowed 1-cosine per row; sized to admit float nondeterminism "
                         "(~5e-05 measured) while catching real defects (>=1e-02)")
    args = ap.parse_args()

    print(f">>> reference {args.a}")
    a = load_split(args.a, args.split)
    print(f">>> sharded   {args.b}")
    b = load_split(args.b, args.split)

    problems = compare(a, b, args.tol)
    if problems:
        print("\nSHARDING IS NOT EQUIVALENT:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("\n>>> PASS: sharded and unsharded datastores agree; safe to fan out across GPUs")


if __name__ == "__main__":
    main()
