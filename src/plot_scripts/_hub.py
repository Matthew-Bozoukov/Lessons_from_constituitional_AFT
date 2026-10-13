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


def mask(repo: str) -> tuple[float, None, None]:
    """Row-weighted honesty. The runs record `honesty_ci95: None`, so there is no interval."""
    v = fetch(repo, "results/results.json").get("overall_honesty_score")
    assert isinstance(v, (int, float)), f"{repo}: no overall_honesty_score"
    return float(v), None, None


def literal(value: float, n: int | None = None) -> tuple[float, float | None, float | None]:
    """A number supplied by hand. `n` gives it the interval a run of that size would have.

    Used only where a value exists outside this org's published runs; the caller must say in a
    comment where it came from, so a reader can tell it apart from a Hub-backed bar.
    """
    if n is None:
        return value, None, None
    return value, *wilson(round(value / 100 * n), n)
