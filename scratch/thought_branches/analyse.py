# ABOUTME: Driver for the offline good-vs-bad ODCV contrast — fetch runs, compute features,
# ABOUTME: contrast, cluster forks, write figures + results.md into a dated run directory.

"""Run the descriptive half of Thought Branches, end to end.

    uv run python -m scratch.thought_branches.analyse fetch      # pull runs from HF
    uv run python -m scratch.thought_branches.analyse run        # analyse + plot

Nothing here needs a GPU, a container or an API key: judge severities are already
published beside the rollouts, embeddings are local, and the whole corpus fits in
memory. That is the point — the expensive causal half only gets pointed at branch
points this half says are interesting.
"""

from __future__ import annotations

import json
from pathlib import Path

import fire

from src.naming import run_dir
from src.utils import write_run_meta

from scratch.thought_branches.descriptive import (
    all_contrasts,
    all_contrasts_within,
    as_rows,
    cluster_forks,
    feature_table,
)
from scratch.thought_branches.report import (
    fig_by_arm,
    fig_contrasts,
    fig_fork,
    fig_shape,
    write_results,
)
from scratch.thought_branches.trajectory import load_corpus

DATA = Path("output/thought_branches/data")
OUT_BASE = Path("output/thought_branches")

# ODCV runs published in the contract layout that carry BOTH transcripts and judge
# scores. Two control arms, one synthetic-SFT arm, two paired responder arms — enough
# spread that a marker surviving inside every arm is not an artefact of one recipe.
RUNS = (
    "LASR-Callum/2026-08-19-odcv-numina-control-716-seed0",
    "LASR-Callum/2026-08-24-odcv-matched-control-716-5x",
    "LASR-Callum/2026-08-31-odcv-qwen3-6-27b-lora-t2-9284-synthdoc-716-dynbatch-r64",
    "LASR-Callum/2026-08-28-odcv-gpt-responder-685-seed42-paired-eval",
    "LASR-Callum/2026-08-24-odcv-grok-responder-703-paired-eval",
)


def fetch(dest: str = str(DATA), runs: tuple[str, ...] = RUNS) -> None:
    """Download rollout transcripts and judge scores for the analysed runs.

    Args:
        dest: Where the run directories land.
        runs: HF dataset repo ids in the published ODCV layout.
    """
    from huggingface_hub import snapshot_download

    root = Path(dest)
    root.mkdir(parents=True, exist_ok=True)
    for rid in runs:
        local = root / rid.split("/")[-1]
        snapshot_download(
            rid,
            repo_type="dataset",
            local_dir=local,
            allow_patterns=[
                "rollouts/**/messages_record.txt",
                "results/*.json",
                "metadata/run_meta.json",
                "README.md",
            ],
        )
        n = len(list(local.rglob("messages_record.txt")))
        print(f"{rid}: {n} transcripts -> {local}")


def _caveat(within, name: str) -> str:
    """The within-scenario estimate for one feature, phrased for a figure subtitle."""
    s = next((x for x in within if x.name == name), None)
    if s is None or not s.n_cells:
        return ""
    return (
        f"Holding scenario fixed too, the gap disappears: Δ = {s.delta:+.3f} "
        f"[{s.lo:+.3f}, {s.hi:+.3f}] over {s.n_cells} cells. The marker is largely "
        "tracking which scenario ran, not which outcome followed."
    )


def run(
    data: str = str(DATA),
    subject: str = "odcv_thought_branches_descriptive",
    k_clusters: int = 12,
    seed: int = 0,
) -> None:
    """Compute contrasts and fork clusters, then write figures and results.

    Args:
        data: Directory of downloaded ODCV run directories.
        subject: Naming-law subject; the run directory and every figure take their name
            from it, with today's date prefixed.
        k_clusters: Fork clusters to fit.
        seed: Seed for clustering and bootstraps, so a figure regenerates identically.
    """
    out = run_dir(OUT_BASE, subject)
    out.mkdir(parents=True, exist_ok=True)

    trajs = load_corpus(Path(data))
    rows = feature_table(trajs)
    scored = [r for r in rows if r.violation is not None]
    base = sum(1 for r in scored if r.violation) / max(1, len(scored))
    print(
        f"{len(trajs)} rollouts, {len(scored)} judged, base violation rate {base:.1%}"
    )

    pooled = all_contrasts(rows, seed=seed)
    within = all_contrasts_within(rows, seed=seed)
    clusters, assignments = cluster_forks(trajs, k=k_clusters, seed=seed)

    figures = [
        fig_contrasts(within, out, subject, pooled=pooled),
        fig_fork(rows, clusters, base, out, subject),
        fig_shape(rows, out, subject),
        fig_by_arm(
            rows,
            "commit_before_write",
            out,
            subject,
            caveat=_caveat(within, "commit_before_write"),
        ),
    ]
    md, js = write_results(out, subject, rows, pooled, within, clusters, figures)

    (out / f"{subject}_features.jsonl").write_text(
        "\n".join(json.dumps(r) for r in as_rows(rows)), encoding="utf-8"
    )
    (out / f"{subject}_fork_assignments.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "key": t.key,
                    "arm": t.arm,
                    "severity": t.severity,
                    "violation": t.is_violation,
                    "cluster": c,
                    "obstacle": f.obstacle_text,
                    "fork_thought": f.thought,
                    "fork_action": f.action,
                    "fork_rel_pos": f.rel_pos,
                }
            )
            for t, f, c in assignments
        ),
        encoding="utf-8",
    )
    write_run_meta(
        out,
        {
            "data": data,
            "subject": subject,
            "k_clusters": k_clusters,
            "seed": seed,
            "runs": list(RUNS),
        },
        extra={"n_rollouts": len(rows), "base_violation_rate": base},
    )
    print(f"\nwrote {md}")
    for p in figures:
        print(f"  {p}")


if __name__ == "__main__":
    fire.Fire({"fetch": fetch, "run": run})
