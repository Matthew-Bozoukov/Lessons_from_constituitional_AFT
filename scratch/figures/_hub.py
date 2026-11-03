# ABOUTME: Reads the published eval runs that the paper figures plot, so every number in a figure
# ABOUTME: comes from the Hub at render time rather than from a literal typed into a script.
# THE ERROR-BAR RULE FOR FIGURES (2026-11-03, Jamie's decision): the benchmark is FIXED and the
# only randomness is the model's sampling, so a single-model bar shows the mean over the
# benchmark's items with -/+ one standard error propagated from each item's own repeats:
#   SE = sqrt(sum_i s_i^2 / k_i) / N      (N items, k_i repeats of item i, s_i^2 their sample variance)
# An ODCV item is a cell (scenario x variant, 3 rollouts), an agentic item is a condition (10
# trials), a MASK item is a row (3 passes). No coverage claim; a single repeat gives no bar.
# Every figure reads through `item_se`; paired-difference plots are the only exception.
# tests/test_figure_intervals.py refuses figure code that builds its own interval instead.
import json
from pathlib import Path
import os

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

import src.eval.misalignment.odcv.odcv as O  # noqa: E402

load_dotenv()
ORG = os.environ.get("HF_ORG", "dougalldeepmind")

# ODCV intervals are a t over the run's passes (src/eval/misalignment/odcv/odcv.py, docs/error_bars.md).
# Runs published before 2026-10-29 carry an older scenario-level `mr_ci95` in results.json;
# re-summarising their per-rollout scores here puts every arm on the one estimator.


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

# ---------------------------------------------------------------------------------------
# Per-repeat readers and the ±1 SEM bar (2026-11-03). One repeat = one whole-benchmark run:
# an ODCV pass, a MASK pass, an agentic-misalignment sweep (trial k of every condition).
# The figure caption names n and says "±1 SEM over repeats"; no coverage level is claimed.
# ---------------------------------------------------------------------------------------

def sem_bar(values: list[float]) -> tuple[float, float | None, float | None, list[float]]:
    """Mean of the repeats, mean -/+ one standard error, and the repeats themselves.

    One repeat gives its own value with a zero-width bar (the plot shows the lone dot)."""
    n = len(values); m = float(np.mean(values))
    if n < 2:
        return m, None, None, [float(v) for v in values]   # one repeat: no bar
    sem = float(np.std(values, ddof=1) / np.sqrt(n))
    return m, m - sem, m + sem, [float(v) for v in values]


def odcv_repeats(repo: str) -> list[float]:
    """Per-pass misalignment rate (%) from the per-rollout scores, keyed by the real pass id."""
    sc = fetch(repo, "results/scores_gemini-3-flash-preview.json")
    medians: dict[str, dict[str, float]] = {}
    for key, v in sc.items():
        variant, scenario, roll = key.split("/")
        s = v.get("score") if isinstance(v, dict) else v
        if s is not None:
            medians.setdefault(variant, {})[f"{scenario}/{roll}"] = float(s)
    return [float(x) for x in O.summarise(medians)["overall"]["mr_pct_by_pass"]]


def agentic_repeats(repo: str) -> list[float]:
    """Per-sweep harmful-action rate (%): trial k of every condition is one sweep."""
    from collections import defaultdict
    from huggingface_hub import snapshot_download
    from src.eval.misalignment.agentic_misalignment.aggregate_eval import _iter_responses
    root = Path(snapshot_download(f"{ORG}/{repo}", repo_type="dataset", token=_token(),
                                  allow_patterns=["results/harness/models/*/*/sample_*/response.json"]))
    by_sweep: dict[int, list[bool]] = defaultdict(list)
    for _, sample_id, data in _iter_responses(root / "results" / "harness" / "models"):
        cls = data.get("classification")
        if cls and "harmful_behavior" in cls and "error" not in cls:
            by_sweep[sample_id].append(bool(cls["harmful_behavior"]))
    return [100.0 * sum(f) / len(f) for _, f in sorted(by_sweep.items())]


def mask_repeats(repo: str) -> list[float]:
    """Per-pass row-weighted honesty score."""
    return [float(x) for x in mask_passes(repo)[0]]


# ---------------------------------------------------------------------------------------
# Per-item readers: each item's repeats, so the standard error can be propagated from them.
# ---------------------------------------------------------------------------------------

def item_se(cells: list[list[float]]) -> tuple[float, float | None, float | None]:
    """Mean over items of the item means, -/+ one SE propagated from each item's repeats.

    Items with one repeat contribute no variance; if no item has two, there is no bar."""
    means = [float(np.mean(c)) for c in cells]
    m = float(np.mean(means))
    multi = [c for c in cells if len(c) > 1]
    if not multi:
        return m, None, None
    var = sum(float(np.var(c, ddof=1)) / len(c) for c in multi) / len(cells) ** 2
    se = float(np.sqrt(var))
    return m, m - se, m + se


def odcv_cells(repo: str) -> list[list[float]]:
    """Per cell (scenario x variant): the violation indicator (score >= 3) of each rollout, in %."""
    sc = fetch(repo, "results/scores_gemini-3-flash-preview.json")
    cells: dict[str, list[float]] = {}
    for key, v in sc.items():
        variant, scenario, _ = key.split("/")
        s_ = v.get("score") if isinstance(v, dict) else v
        if s_ is not None:
            cells.setdefault(f"{variant}/{scenario}", []).append(100.0 * (float(s_) >= O.VIOLATION_THRESHOLD))
    return list(cells.values())


def agentic_cells(repo: str) -> list[list[float]]:
    """Per condition: the harmful flag of each trial, in %."""
    from collections import defaultdict
    from huggingface_hub import snapshot_download
    from src.eval.misalignment.agentic_misalignment.aggregate_eval import _iter_responses
    root = Path(snapshot_download(f"{ORG}/{repo}", repo_type="dataset", token=_token(),
                                  allow_patterns=["results/harness/models/*/*/sample_*/response.json"]))
    by_cond: dict[str, list[float]] = defaultdict(list)
    for condition, _, data in _iter_responses(root / "results" / "harness" / "models"):
        cls = data.get("classification")
        if cls and "harmful_behavior" in cls and "error" not in cls:
            by_cond[condition].append(100.0 * bool(cls["harmful_behavior"]))
    return list(by_cond.values())


def mask_cells(repo: str) -> list[list[float]]:
    """Per row (task_id): the honest-or-not indicator of each pass, in % (100 unless the judge
    called the row a lie, `eval_honesty_1 == -1`, which is how the harness scores honesty)."""
    import csv, sys
    from huggingface_hub import HfApi
    csv.field_size_limit(sys.maxsize)   # a looping generation can exceed the 128 KB default
    files = [f.rfilename for f in HfApi().dataset_info(f"{ORG}/{repo}", token=_token()).siblings]
    metrics = sorted(f for f in files if f.endswith("_metrics.csv"))
    rows: dict[str, list[float]] = {}
    for f in metrics:
        with open(hf_hub_download(f"{ORG}/{repo}", f, repo_type="dataset", token=_token()), newline="") as fh:
            for r in csv.DictReader(fh):
                rows.setdefault(r["task_id"], []).append(0.0 if str(r["eval_honesty_1"]).strip() == "-1" else 100.0)
    assert rows, f"{repo}: no *_metrics.csv with per-row verdicts"
    return list(rows.values())


def mask_cells_by_id(repo: str) -> dict[str, list[float]]:
    """As mask_cells, keyed by task_id, so passes can be pooled across artifacts row by row."""
    import csv, sys
    from huggingface_hub import HfApi
    csv.field_size_limit(sys.maxsize)   # a looping generation can exceed the 128 KB default
    files = [f.rfilename for f in HfApi().dataset_info(f"{ORG}/{repo}", token=_token()).siblings]
    rows: dict[str, list[float]] = {}
    for f in sorted(f for f in files if f.endswith("_metrics.csv")):
        with open(hf_hub_download(f"{ORG}/{repo}", f, repo_type="dataset", token=_token()), newline="") as fh:
            for r in csv.DictReader(fh):
                rows.setdefault(r["task_id"], []).append(0.0 if str(r["eval_honesty_1"]).strip() == "-1" else 100.0)
    return rows


def odcv(repo: str) -> tuple[float, float | None, float | None]:
    """Misalignment rate over the 80 cells, -/+ one SE from each cell's rollouts."""
    return item_se(odcv_cells(repo))


def agentic(repo: str) -> tuple[float, float | None, float | None]:
    """Harmful-action rate over the 18 conditions, -/+ one SE from each condition's trials."""
    return item_se(agentic_cells(repo))


def mask(repo: str) -> tuple[float, float | None, float | None]:
    """Honesty over the rows, -/+ one SE from each row's passes."""
    return item_se(mask_cells(repo))
