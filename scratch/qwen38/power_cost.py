# ABOUTME: oai_hf Step 3 sample-size and cost table: CI half-width, the smallest arm difference detectable at 80% power,
# ABOUTME: and GPU/judge cost and wall-clock per arm, from the measured 2026-11-18/19 runs. Run: uv run python scratch/qwen38/power_cost.py
import math

from scipy.stats import fisher_exact, norm

P = (10 + 8 + 8) / 192  # pooled share rate of the three Qwen3.8 arms
GPU_PER_H = 5.29  # 1x H200 SECURE, as billed for the eval pods
JUDGE_PER_64 = 0.28  # OpenRouter drop over the 64-rollout DA run
WALL_MIN_PER_64 = (
    48.6,
    36.2,
    17.9,
)  # Inspect wall-clock of the three n=64 runs at 24 concurrent
BOOT = 25 / 60 * GPU_PER_H  # one eval-pod boot (vLLM + weights), paid once per pod

z_a, z_b = norm.ppf(0.975), norm.ppf(0.80)
lo, hi = min(WALL_MIN_PER_64), max(WALL_MIN_PER_64)
mean = sum(WALL_MIN_PER_64) / 3
print(
    f"pooled share rate p = {P:.3f}; per 64 rollouts: {lo:.0f}-{hi:.0f} min (mean {mean:.0f}), "
    f"GPU ${lo / 60 * GPU_PER_H:.2f}-{hi / 60 * GPU_PER_H:.2f}, judge ${JUDGE_PER_64:.2f}; pod boot ${BOOT:.2f} once\n"
)
print(
    f"{'n per arm':>9} | {'95% CI half-width':>17} | {'detectable gap (80% power)':>26} | {'hours per arm':>13} | {'cost per arm':>12}"
)
for n in (64, 128, 256, 512, 1024, 2048):
    half = z_a * math.sqrt(P * (1 - P) / n)
    mde = (z_a + z_b) * math.sqrt(2 * P * (1 - P) / n)
    hours = n / 64 * mean / 60
    cost = n / 64 * (mean / 60 * GPU_PER_H + JUDGE_PER_64)
    print(
        f"{n:>9} | {100 * half:>15.1f}pp | {100 * mde:>24.1f}pp | {hours:>13.1f} | ${cost:>11.0f}"
    )
need = lambda d: math.ceil(((z_a + z_b) ** 2) * 2 * P * (1 - P) / d**2)
print(
    f"\nrollouts per arm to detect the observed nosynth-vs-DA gap (15.6% vs 12.5%, 3.1pp) at 80% power: {need(0.03125):,}"
)
for a, b, k1, k2 in (("nosynth", "da", 10, 8), ("nosynth", "datools", 10, 8)):
    print(
        f"Fisher exact {a} {k1}/64 vs {b} {k2}/64: p = {fisher_exact([[k1, 64 - k1], [k2, 64 - k2]])[1]:.2f}"
    )
