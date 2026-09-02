# ABOUTME: sae_diff — a pretrained SAE labels every document with ~65k interpretable
# ABOUTME: concepts, corpora are diffed on latent frequency, and each hypothesis lands here.

"""SAE dataset diffing, as Property rows.

The method is arXiv:2512.10092 (Jiang, Sun, Dunlap, Smith, Nanda). One reader LLM encodes
every document, a pretrained SAE decomposes each token's activations into ~65,536 latents
that carry human-readable labels, and max-pooling over tokens turns a document into a
boolean vector over concepts. Diffing two corpora is then arithmetic:

    per-corpus latent frequency -> target minus max-of-others -> top-K latents
    -> relabel each on OUR excerpts -> summarise into <=10 hypotheses
    -> an LLM judge reads every document and scores every hypothesis

What this producer adds to the four already here is a hypothesis space nobody had to
write down. `clusters` pays an autorater per record to invent vocabulary; the SAE arrives
with 65k concepts pre-labelled, and the embeddings are REUSABLE — every later comparison
over the same corpora is arithmetic on cached matrices, no GPU.

**Port status: the run itself lives in `scratch/sae_properties/`** — `run_embed.py` (the
only GPU stage), `run_diff.py`, and the vendored, patched upstream pipeline under
`third_party/interp_embed/`. That split is deliberate and mirrors LESS: the pod-side job
stays a pod-side job. What is here is the boundary — `produce()` reads the run directory's
artifacts, which are the interface, so porting the drivers changes nothing downstream.

Two things this producer does NOT do, both on purpose:

* It does not re-measure prevalence. The diff run already had a judge read every document
  for every hypothesis, and `Property.prevalence` is that reading. Re-measuring with a
  freshly written detector would put a DIFFERENT instrument's number in the row.
* It does not invent a detector. The detector is the question the verifier actually
  asked, so the instrument in `detector` is the one whose readings are in `prevalence`.
  `interpreter_model` records which model answered it.

One caveat worth carrying into any reading of these rows (LOG 2026-08-20): SAE latent
labels come from the SAE's own training distribution, and on our corpora some are
syntactic detectors wearing semantic labels — a latent labelled "Offensive request from
the user" fires on 84% of the courtroom corpus. The relabel-and-verify steps upstream are
what make a hypothesis quotable; an unverified latent label is not evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.properties.registry import Property
from src.utils import git_sha

SOURCE = "sae_diff"
SCRATCH_MODULE = "scratch/sae_properties"


def _verified_rates(
    diff_dir: Path, corpus: str, n_hypotheses: int
) -> list[dict] | None:
    """The judge's per-hypothesis rates for one corpus, or None if it was not verified.

    A killed run can leave verification directories behind that scored a DIFFERENT
    hypotheses file, and their directory names sort after the good ones — so reports are
    matched on hypothesis count and picked by the report's own timestamp, never by path
    order (this bit us on 2026-08-20).

    Args:
        diff_dir: The `diff_<target>` directory.
        corpus: Corpus name, matching the `verify_<corpus>` subdirectory.
        n_hypotheses: How many hypotheses the current hypotheses file holds.

    Returns:
        `summary_by_hypothesis`, newest first by report timestamp, or None.
    """
    reports = []
    for path in (diff_dir / f"verify_{corpus}").rglob("verification_report.json"):
        report = json.loads(path.read_text())
        if report.get("metadata", {}).get("num_hypotheses") == n_hypotheses:
            reports.append(report)
    if not reports:
        return None
    reports.sort(key=lambda r: r["metadata"]["timestamp"])
    return reports[-1]["summary_by_hypothesis"]


def _rate(summary: list[dict], index: int) -> float | None:
    """One hypothesis's verified share, as a fraction.

    Args:
        summary: A report's `summary_by_hypothesis`.
        index: Position in the hypotheses file; the verifier preserves that order.

    Returns:
        The share in [0, 1], or None when the hypothesis is absent from the report.
    """
    for row in summary:
        if row["hypothesis_idx"] == index:
            rates = row.get("verification_rates") or {}
            first = next(iter(rates.values()), None)
            if isinstance(first, dict) and first.get("percentage") is not None:
                return float(first["percentage"]) / 100.0
    return None


def produce(records, cfg, out_dir: str | Path, target=None) -> list[Property]:
    """Turn one SAE diff run's verified hypotheses into Property rows.

    Args:
        records: Unused. Prevalence comes from the run's own verification pass, which
            already read every document — see the module docstring.
        cfg: The producer's config block. Keys: `run_dir` (required) — a
            `diff_<target>` directory written by `scratch/sae_properties/run_diff.py`;
            `target_corpus` (required) — whose prevalence goes in `prevalence`;
            `channel` (default "response") — which channel was embedded;
            `min_prevalence` (default 0.0) — drop hypotheses below this in the target;
            `min_gap` (default 0.0) — drop hypotheses whose target share does not exceed
            the best other corpus by at least this much.
        out_dir: Where to write this producer's preview artifacts.
        target: Unused; diffing describes corpora rather than tracing an outcome back.

    Returns:
        Property rows, largest target-minus-others gap first.

    Raises:
        KeyError: If the config names no `run_dir` or no `target_corpus`.
        FileNotFoundError: If that run directory holds no `hypotheses.json`.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(cfg)
    run_dir = Path(str(cfg["run_dir"]))
    target_corpus = str(cfg["target_corpus"])
    channel = str(cfg.get("channel", "response"))
    min_prevalence = float(cfg.get("min_prevalence", 0.0))
    min_gap = float(cfg.get("min_gap", 0.0))

    hyp_path = run_dir / "hypotheses.json"
    if not hyp_path.exists():
        raise FileNotFoundError(
            f"no hypotheses.json in {run_dir} — run the diff first:\n"
            f"  uv run --project {SCRATCH_MODULE} python {SCRATCH_MODULE}/run_diff.py "
            f"--config configs/properties/sae_diff.yaml run=<embed-run>"
        )
    payload = json.loads(hyp_path.read_text())
    hypotheses = payload.get("differences") or []

    corpora = sorted({p.name[len("verify_") :] for p in run_dir.glob("verify_*")})
    rates = {c: _verified_rates(run_dir, c, len(hypotheses)) for c in corpora}
    rates = {c: s for c, s in rates.items() if s is not None}
    if target_corpus not in rates:
        raise KeyError(
            f"{target_corpus!r} was not verified in {run_dir}; "
            f"verified: {sorted(rates)}"
        )

    run = run_dir.parent.name
    rows: list[tuple[float, Property]] = []
    for i, hypothesis in enumerate(hypotheses):
        description = str(hypothesis.get("description") or "").strip()
        if not description:
            continue
        by_corpus = {c: _rate(s, i) for c, s in rates.items()}
        prevalence = by_corpus.get(target_corpus)
        if prevalence is None or prevalence < min_prevalence:
            continue
        others = [
            v for c, v in by_corpus.items() if c != target_corpus and v is not None
        ]
        gap = prevalence - max(others) if others else prevalence
        if gap < min_gap:
            continue

        rows.append(
            (
                gap,
                Property.make(
                    source=SOURCE,
                    run=run,
                    key=f"h{i:02d}",
                    # The hypothesis is already a statement about one document; the label is its
                    # first clause, the description the whole of it.
                    label=description.split(". ")[0][:120],
                    description=description,
                    # The verifier's own question, so the instrument matches the reading.
                    detector=f"Does this {channel} exhibit the following property? {description}",
                    channel=channel,
                    confidence=str(hypothesis.get("confidence", "")) or "medium",
                    caveat="SAE latent labels come from the SAE's training distribution; this "
                    "hypothesis is trustworthy because a judge verified it per document, "
                    "not because the latents were labelled convincingly.",
                    prevalence=prevalence,
                    corpus={
                        "name": target_corpus,
                        "channel": channel,
                        "run_dir": str(run_dir),
                    },
                    support={
                        "verified_prevalence_by_corpus": by_corpus,
                        "gap_vs_best_other": gap,
                        "sae_estimated_difference": hypothesis.get(
                            "percentage_difference"
                        ),
                        "latents": hypothesis.get("feature_ids") or [],
                    },
                    evidence={"examples": (hypothesis.get("examples") or [])[:3]},
                    provenance={
                        "producer": SOURCE,
                        "run_dir": str(run_dir),
                        "scratch_module": SCRATCH_MODULE,
                        "git_sha": git_sha(),
                        "method": "arXiv:2512.10092 dataset diffing",
                    },
                ),
            )
        )

    rows.sort(key=lambda pair: -pair[0])
    properties = [p for _, p in rows]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sae_diff_preview.md").write_text(_preview(properties, corpora))
    return properties


def _preview(properties: list[Property], corpora: list[str]) -> str:
    """A markdown mirror, so the numbers are greppable without opening the registry.

    Args:
        properties: The rows, in export order.
        corpora: Every corpus the run verified.

    Returns:
        The markdown.
    """
    lines = [
        "# sae_diff — verified hypotheses",
        "",
        "| gap | " + " | ".join(corpora) + " | property |",
        "|---|" + "---|" * len(corpora) + "---|",
    ]
    for prop in properties:
        by = prop.support["verified_prevalence_by_corpus"]
        cells = " | ".join(
            "—" if by.get(c) is None else f"{by[c]:.2f}" for c in corpora
        )
        lines.append(
            f"| +{prop.support['gap_vs_best_other']:.2f} | {cells} | {prop.label} |"
        )
    return "\n".join(lines) + "\n"
