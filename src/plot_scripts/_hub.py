# ABOUTME: Reads the published eval runs that the paper figures plot, so every number in a figure
# ABOUTME: comes from the Hub at render time rather than from a literal typed into a script.
import json
import os

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

import src.eval.misalignment.odcv.odcv as O
from src.eval.stats import Design

load_dotenv()
ORG = os.environ.get("HF_ORG", "dougalldeepmind")

# The published ODCV runs use item_sampling="sampled" (generalising to unseen scenarios). These
# figures compare arms ON THIS BENCHMARK, so the 40 scenarios are fixed and the scenario-to-
# scenario term drops out: same point estimate, narrower interval, a different question. Reading
# `mr_ci95` off results.json instead would roughly double every ODCV error bar.
O.DESIGN = Design(item="scenario", enumerated={"variant": "equal"}, subsamples=("pass",),
                  item_sampling="fixed")
O.DESIGN_ONE_VARIANT = Design(item="scenario", subsamples=("pass",), item_sampling="fixed")


def _token() -> str:
    tok = os.environ.get("HF_TOKEN")
    assert tok, "HF_TOKEN is not set: the figures read their numbers from the Hub"
    return tok


def fetch(repo: str, path: str) -> dict:
    """One JSON file out of a published run."""
    return json.load(open(hf_hub_download(f"{ORG}/{repo}", path, repo_type="dataset",
                                          token=_token())))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, in percent."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def odcv(repo: str) -> tuple[float, float, float]:
    """Misalignment rate and a FIXED-benchmark interval, recomputed from the per-rollout scores."""
    sc = fetch(repo, "results/scores_gemini-3-flash-preview.json")
    medians: dict[str, dict[str, list]] = {}
    for key, v in sc.items():
        variant, scenario, _ = key.split("/")
        s = v.get("score") if isinstance(v, dict) else v
        if s is None:
            continue
        medians.setdefault(variant, {}).setdefault(scenario, []).append(float(s))
    o = O.summarise(medians)["overall"]
    return o["mr_pct"], o["mr_ci95_lo"], o["mr_ci95_hi"]


def agentic(repo: str) -> tuple[float, float, float]:
    """Harmful-action rate and a Wilson interval over the rollouts."""
    o = fetch(repo, "results/misalignment_summary.json")["overall"]
    return 100 * o["harmful"] / o["n"], *wilson(o["harmful"], o["n"])


def mask_passes(repo: str) -> tuple[list[float], int]:
    """Each repeat pass's row-weighted honesty score, and the benchmark's row count.

    A pass's per-archetype scores are weighted by that archetype's response count, which
    reproduces the run's published `overall_honesty_score` exactly. A single-pass run keeps its
    archetype file at results/all_results.json; a repeated run nests one per pass.

    Returns:
        The per-pass scores (one entry for a single-pass run) and the number of scored rows.
    """
    top = fetch(repo, "results/results.json")
    rows = top.get("n_rows")
    paths = [f"results/pass{i}/all_results.json" for i in (1, 2, 3)]
    scores = []
    for path in paths:
        try:
            d = fetch(repo, path)
        except Exception:                        # noqa: BLE001 -- absent pass, not an error
            continue
        num = den = 0.0
        for models in d.values():
            for m in models.values():
                num += m["honesty_score_1"] * m["total_responses"]
                den += m["total_responses"]
        scores.append(num / den)
    if not scores:
        d = fetch(repo, "results/all_results.json")
        num = den = 0.0
        for models in d.values():
            for m in models.values():
                num += m["honesty_score_1"] * m["total_responses"]
                den += m["total_responses"]
        scores.append(num / den)
        rows = rows or int(den)
    assert scores, f"{repo}: no archetype results to score"
    return scores, int(rows)


def mask(repo: str) -> tuple[float, float, float]:
    """Row-weighted honesty with a FIXED-benchmark interval over the scored rows.

    The score is the share of rows the judge did not mark dishonest, so the interval is binomial
    over the benchmark's fixed row set. Repeat passes are averaged first rather than pooled as
    extra rows: the same 1000 rows are re-run, so the passes are correlated and pooling them
    would understate the interval. Measured pass-to-pass spread is ~0.3-1.3 points against a
    row-to-row term of ~2.3, so the row term is what the interval should report. The runs
    themselves record `honesty_ci95: None`.
    """
    scores, rows = mask_passes(repo)
    m = float(np.mean(scores))
    published = fetch(repo, "results/results.json").get("overall_honesty_score")
    assert published is None or abs(m - float(published)) < 0.15, (
        f"{repo}: recomputed {m:.2f} disagrees with the published {published}")
    return m, *wilson(round(m / 100 * rows), rows)


def combine(*values: tuple[float, float | None, float | None]) -> tuple[float, float, float]:
    """Mean of several probe rates, as one metric, with the inputs treated as independent.

    Used to report ODCV and agentic misalignment as a single single-agent number. Every input
    must carry an interval, so a combined bar can never be drawn from a bare point estimate.
    """
    assert values and all(v is not None and v[1] is not None for v in values), (
        "combine() needs an interval on every input")
    m = float(np.mean([v[0] for v in values]))
    half = np.sqrt(sum(((v[2] - v[1]) / 2) ** 2 for v in values)) / len(values)
    return m, max(0.0, m - half), min(100.0, m + half)


def literal(value: float, n: int | None = None) -> tuple[float, float | None, float | None]:
    """A number supplied by hand. `n` gives it the interval a run of that size would have.

    Used only where a value exists outside this org's published runs; the caller must say in a
    comment where it came from, so a reader can tell it apart from a Hub-backed bar.
    """
    if n is None:
        return value, None, None
    return value, *wilson(round(value / 100 * n), n)
