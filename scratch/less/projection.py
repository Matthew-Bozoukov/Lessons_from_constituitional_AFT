# ABOUTME: Random projection of LoRA-sized gradients down to d dims for the LESS
# ABOUTME: gradient datastore, plus the cosine-preservation check that validates it.

"""Project a 319M-dim LoRA gradient to d dims, preserving cosine similarity.

LESS (arXiv:2402.04333) projects with a dense Rademacher matrix via TRAK's CudaProjector.
That is not an option here: this LoRA has P = 318,767,104 parameters (r=64; gate/up/down on
all 64 Qwen3.6-27B layers, q/k/v/o on the 16 that use full attention rather than the hybrid
linear-attention path), so a dense P x 8192 matrix is ~9.8 TB, and
`fast-jl` ships sdist-only -- it compiles a CUDA extension, which this repo has repeatedly
lost hours to (docs/LOG.md, the CUDA-13 postmortem).

Count-sketch is used instead: every coordinate p hashes to one bucket h(p) with a random
sign s(p), and the sketch is a single scatter-add. It is an unbiased sketch of the inner
product -- E[<Sx, Sy>] = <x, y> -- with variance ~ (||x||^2 ||y||^2 + <x,y>^2) / d, so
error falls as 1/sqrt(d) exactly like a dense JL projection, just with a worse constant.

The constant is affordable because of an asymmetry worth stating plainly: count-sketch
costs O(P) per example REGARDLESS of d (the scatter touches each coordinate once; only the
output buffer grows), whereas a dense projection costs O(P*d). So where LESS must keep
d=8192 to stay tractable, we can raise d until the variance is gone and pay almost nothing.
Hence DEFAULT_DIM below is 32768, and `verify_rank_preservation` is the check that it is
in fact enough -- run it, do not assume it.

Determinism is a correctness requirement, not a nicety: train and validation features are
compared by cosine, so both sides -- and every GPU shard -- must sketch into the SAME
basis. The hash is drawn once from a CPU torch.Generator (CPU RNG is reproducible across
machines and accelerators in a way CUDA RNG is not) and then moved to the device.
"""

from __future__ import annotations

import torch

# Cost is O(P) whichever d we pick, so this is chosen for accuracy, not speed. See module
# docstring; validate with verify_rank_preservation rather than trusting the default.
DEFAULT_DIM = 32768


class CountSketchProjector:
    """A seeded count-sketch from P dims to d dims.

    The hash (bucket index + sign) is materialised once and reused for every example, so
    per-example work is one fused scatter-add over P coordinates.

    Attributes:
        p_total: Number of source coordinates (the flattened LoRA gradient length).
        dim: Sketch width d.
        seed: The value that fixes the basis. Train features, validation features and
            every shard MUST share it or their cosines are meaningless.
    """

    def __init__(self, p_total: int, dim: int = DEFAULT_DIM, seed: int = 0,
                 device: str | torch.device = "cuda") -> None:
        assert p_total > 0 and dim > 0, f"bad shape: p_total={p_total} dim={dim}"
        self.p_total, self.dim, self.seed = int(p_total), int(dim), int(seed)
        self.device = torch.device(device)

        # CPU generator on purpose: identical draws on every machine and every shard,
        # which a CUDA generator does not guarantee.
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        # index_add_ requires int64 indices; at P=319M that is ~2.6 GB resident, which is
        # the price of not regenerating the hash per example.
        self._bucket = torch.randint(0, self.dim, (self.p_total,), generator=gen,
                                     dtype=torch.int64).to(self.device)
        signs = torch.randint(0, 2, (self.p_total,), generator=gen,
                              dtype=torch.int8).to(self.device)
        self._sign = signs.mul_(2).sub_(1).to(torch.float32)  # {0,1} -> {-1,+1}

    def project(self, flat: torch.Tensor, chunk: int = 1 << 26) -> torch.Tensor:
        """Sketch one flattened gradient to `dim` dims.

        The scatter runs in chunks so the signed-gradient temporary stays bounded: at
        P=319M a whole-vector multiply would spike an extra 1.3 GiB per call, on top of
        the model, the moments and the hash. Chunking costs nothing (the work is identical
        and memory-bound) and keeps peak usage flat.

        Args:
            flat: A 1-D tensor of length `p_total`. Accumulated in float32 -- bf16 over
                319M scatter-adds loses far too much precision.
            chunk: Coordinates per scatter step.

        Returns:
            A 1-D float32 tensor of length `dim`, on this projector's device.
        """
        assert flat.ndim == 1 and flat.numel() == self.p_total, (
            f"expected a flat gradient of {self.p_total} elements, got {tuple(flat.shape)}")
        out = torch.zeros(self.dim, dtype=torch.float32, device=self.device)
        for a in range(0, self.p_total, chunk):
            b = min(a + chunk, self.p_total)
            out.index_add_(0, self._bucket[a:b],
                           flat[a:b].to(torch.float32) * self._sign[a:b])
        return out

    def memory_bytes(self) -> int:
        """Resident size of the hash, for the pod-sizing arithmetic."""
        return self._bucket.numel() * 8 + self._sign.numel() * 4


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().to(torch.float64)
    rb = b.argsort().argsort().to(torch.float64)
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


def verify_rank_preservation(p_total: int, dim: int, seed: int = 0, n_rows: int = 120,
                             device: str = "cpu", sparsity: float = 0.5) -> dict:
    """Measure whether the sketch preserves the RANKING of cosine similarities.

    This is THE check that licenses every downstream number, and it has to be a ranking
    check rather than an error check: LESS's output is an ordering of D, so a sketch with
    visible cosine error is still fine if it orders rows correctly, and a sketch with
    small error is still useless if it does not.

    Getting this measurement right requires a realistic SPREAD of true cosines. Comparing
    pairs that all sit at the same true similarity makes the rank correlation a comparison
    of noise against noise -- it reads near zero however good the projection is, which
    says nothing about d. So the rows here are built to span a range of similarity to one
    target, the way rows of D spread in usefulness to one validation subtask.

    Args:
        p_total: Source dimensionality to test at. Count-sketch variance depends on d and
            the vectors rather than P, so a scaled-down P still predicts behaviour at 319M.
        dim: Sketch width to test.
        seed: Projector seed.
        n_rows: How many rows to rank against the target.
        device: Where to run.
        sparsity: Fraction of coordinates zeroed, imitating a real LoRA gradient.

    Returns:
        rmse and bias of the cosine error, plus `spearman` and top-10/top-20 retention --
        the numbers that decide whether `dim` is wide enough.
    """
    proj = CountSketchProjector(p_total, dim=dim, seed=seed, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)

    def _draw() -> torch.Tensor:
        v = torch.randn(p_total, generator=gen)
        return v * (torch.rand(p_total, generator=gen) >= sparsity) if sparsity else v

    target = _draw().to(device)
    rows = [(a * target + _draw().to(device))
            for a in torch.linspace(0.0, 0.6, n_rows).tolist()]

    cs = torch.nn.functional.cosine_similarity
    true = torch.tensor([cs(target, r, dim=0).item() for r in rows])
    st = proj.project(target)
    got = torch.tensor([cs(st, proj.project(r), dim=0).item() for r in rows])

    def _kept(k: int) -> float:
        a = set(true.argsort(descending=True)[:k].tolist())
        b = set(got.argsort(descending=True)[:k].tolist())
        return len(a & b) / k

    err = got - true
    return {
        "dim": dim, "p_total": p_total, "n_rows": n_rows,
        "true_cos_sd": float(true.std()),
        "rmse": float((err ** 2).mean().sqrt()), "bias": float(err.mean()),
        "spearman": _spearman(true, got), "top10_kept": _kept(10), "top20_kept": _kept(20),
    }


if __name__ == "__main__":
    # Offline validation of DEFAULT_DIM. Measured at P=2M, 50% sparse, 120 rows spanning
    # true cosine 0.00-0.51 (numpy equivalent, 2026-08-14):
    #     d=2048   rmse 0.0214  spearman 0.9902  top10 90%
    #     d=8192   rmse 0.0100  spearman 0.9978  top10 80%   <- LESS's setting
    #     d=32768  rmse 0.0048  spearman 0.9995  top10 100%  <- DEFAULT_DIM
    #     d=131072 rmse 0.0023  spearman 1.0000  top10 100%
    # Ranking is what LESS consumes, and it is intact well before the error vanishes.
    print(f"{'d':>8}{'rmse':>11}{'spearman':>11}{'top10':>8}{'top20':>8}")
    for d in (2048, 8192, DEFAULT_DIM, 131072):
        s = verify_rank_preservation(p_total=2_000_000, dim=d)
        print(f"{d:>8}{s['rmse']:>11.5f}{s['spearman']:>11.4f}"
              f"{s['top10_kept']:>8.0%}{s['top20_kept']:>8.0%}")
