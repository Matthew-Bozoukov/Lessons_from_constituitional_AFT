# ABOUTME: Coverage reporting. Emitted automatically at the end of every run so
# ABOUTME: under-represented parts of the spec are visible rather than assumed absent.

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from .core.recipe import Recipe
from .core.specs import load_spec
from .core.types import Document
from .plugins.chunkers import build_chunker


def _matrix(
    docs: Sequence[Document], row_of, col_of
) -> tuple[dict[str, dict[str, int]], list[str], list[str]]:
    """Build a count matrix from two document accessors.

    Args:
        docs: Documents to count.
        row_of: Callable returning a list of row keys per document.
        col_of: Callable returning the column key per document.

    Returns:
        Tuple of (matrix, sorted row keys, sorted column keys).
    """
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    cols: set[str] = set()
    for doc in docs:
        col = col_of(doc)
        cols.add(col)
        for row in row_of(doc):
            matrix[row][col] += 1
    return (
        {r: dict(c) for r, c in matrix.items()},
        sorted(matrix),
        sorted(cols),
    )


def _md_table(matrix: dict[str, dict[str, int]], rows: list[str], cols: list[str],
              row_label: str, max_rows: int = 400) -> str:
    """Render a count matrix as a markdown table with a totals column."""
    head = f"| {row_label} | " + " | ".join(cols) + " | total |"
    sep = "|" + "---|" * (len(cols) + 2)
    lines = [head, sep]
    for r in rows[:max_rows]:
        cells = [str(matrix[r].get(c, 0)) for c in cols]
        total = sum(matrix[r].get(c, 0) for c in cols)
        lines.append(f"| `{r}` | " + " | ".join(cells) + f" | {total} |")
    if len(rows) > max_rows:
        lines.append(f"| _...{len(rows) - max_rows} more rows_ | " + " | ".join([""] * len(cols)) + " | |")
    return "\n".join(lines)


def coverage_report(
    corpus: Sequence[Document],
    cfg: dict[str, Any],
    run_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, str]:
    """Write the coverage report, the slicing index, and the coverage heatmap.

    The report joins the corpus back to chunk_id so that spec material the sampler
    never reached - or reached only under one doc type - is stated explicitly. A
    corpus that looks large can still have holes, and holes are the failure mode
    that silently limits what the finetuned model learns.

    Args:
        corpus: The full final-stage corpus, including dropped documents.
        cfg: Resolved config.
        run_dir: Run directory.
        manifest: The run manifest (for counts and filter summaries).

    Returns:
        Mapping of artifact name to path.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    kept = [d for d in corpus if d.filter_verdict == "keep"]
    recipe = Recipe.from_config(cfg["recipe"])

    spec_cfg = cfg.get("spec") or {}
    spec = load_spec(spec_cfg["id"], spec_cfg.get("path"))
    all_chunks = build_chunker(spec_cfg.get("chunker") or {}).chunk(spec)
    all_chunk_ids = [c.chunk_id for c in all_chunks]

    seen = Counter(cid for d in kept for cid in d.scenario.chunk_ids)
    holes = [cid for cid in all_chunk_ids if seen[cid] == 0]
    counts_sorted = sorted(seen.values())
    p10 = counts_sorted[len(counts_sorted) // 10] if counts_sorted else 0
    thin = [cid for cid in all_chunk_ids if 0 < seen[cid] <= p10]

    chunk_by_type, chunk_rows, type_cols = _matrix(
        kept, lambda d: d.scenario.chunk_ids, lambda d: d.scenario.doc_type
    )
    empty_cells = sum(
        1 for cid in all_chunk_ids for t in type_cols if chunk_by_type.get(cid, {}).get(t, 0) == 0
    )
    total_cells = max(1, len(all_chunk_ids) * len(type_cols))

    pressure_axis = "stakes_holder" if "stakes_holder" in recipe.axes else (
        recipe.axis_names[0] if recipe.axis_names else ""
    )
    type_by_pressure, type_rows, pressure_cols = (
        _matrix(kept, lambda d: [d.scenario.doc_type], lambda d: str(d.scenario.axes.get(pressure_axis, "")))
        if pressure_axis
        else ({}, [], [])
    )
    type_by_grouping, tg_rows, tg_cols = _matrix(
        kept, lambda d: [d.scenario.doc_type], lambda d: d.scenario.grouping_strategy
    )

    index_path = _write_index(corpus, run_dir, recipe.axis_names)
    plot_path = ""
    if (cfg.get("report") or {}).get("plot", True):
        from .plots import coverage_heatmap

        plot_path = coverage_heatmap(
            chunk_by_type, all_chunk_ids, type_cols, run_dir / "coverage_heatmap.png",
            title=f"Chunk coverage by document type - {manifest.get('run_id', '')}",
        )

    lines: list[str] = [
        f"# Coverage report - `{manifest.get('run_id', '')}`",
        "",
        f"- spec: `{spec.spec_id}` ({spec.path}, sha `{spec.sha}`)",
        f"- chunker: `{(spec_cfg.get('chunker') or {}).get('granularity', 'bullet')}`, "
        f"chunks: **{len(all_chunk_ids)}**",
        f"- documents: **{len(corpus)}** generated, **{len(kept)}** kept "
        f"({100 * len(kept) / max(1, len(corpus)):.1f}%)",
        f"- generator: `{(cfg.get('generation') or {}).get('model', '')}`, "
        f"revision passes: **{len(cfg.get('revision') or [])}**",
        f"- total cost: **${manifest.get('cost_usd_total', 0):.2f}** "
        f"(cache hits: {manifest.get('cache', {}).get('hits', 0)})",
        "",
        "## Spec coverage",
        "",
        f"- chunks never used: **{len(holes)}** of {len(all_chunk_ids)}",
        f"- chunks at or below the 10th percentile ({p10} docs): **{len(thin)}**",
        f"- empty (chunk x doc_type) cells: **{empty_cells}** of {total_cells} "
        f"({100 * empty_cells / total_cells:.1f}%)",
        "",
    ]
    if holes:
        lines += ["Chunks with zero coverage:", ""]
        lines += [f"- `{cid}`" for cid in holes[:60]]
        if len(holes) > 60:
            lines.append(f"- _...{len(holes) - 60} more_")
        lines.append("")
    else:
        lines += ["No chunk is completely uncovered.", ""]

    lines += [
        "## chunk_id x doc_type",
        "",
        _md_table(chunk_by_type, chunk_rows, type_cols, "chunk_id"),
        "",
    ]
    if pressure_axis:
        lines += [
            f"## doc_type x {pressure_axis}",
            "",
            _md_table(type_by_pressure, type_rows, pressure_cols, "doc_type"),
            "",
        ]
    lines += [
        "## doc_type x grouping_strategy",
        "",
        _md_table(type_by_grouping, tg_rows, tg_cols, "doc_type"),
        "",
        "## Axis marginals (kept documents)",
        "",
    ]
    for axis in recipe.axis_names:
        tally = Counter(str(d.scenario.axes.get(axis, "")) for d in kept)
        target = recipe.axes[axis]
        lines.append(f"**{axis}**")
        lines.append("")
        lines.append("| value | n | share | target |")
        lines.append("|---|---|---|---|")
        for value in sorted(tally):
            share = tally[value] / max(1, len(kept))
            lines.append(
                f"| {value} | {tally[value]} | {share:.3f} | {target.get(value, 0):.3f} |"
            )
        lines.append("")

    lines += ["## Stage-over-stage", "", "| stage | n | ok | errors | mean words | cost usd |", "|---|---|---|---|---|---|"]
    for stage, c in (manifest.get("counts") or {}).items():
        lines.append(
            f"| `{stage}` | {c.get('n', 0)} | {c.get('n_ok', 0)} | {c.get('n_error', 0)} "
            f"| {c.get('mean_words', 0)} | {c.get('cost_usd', 0):.4f} |"
        )
    lines.append("")

    dropped = Counter(d.dropped_by for d in corpus if d.dropped_by)
    if dropped:
        lines += ["## Filter effect", "", "| filter | dropped |", "|---|---|"]
        lines += [f"| `{k}` | {v} |" for k, v in dropped.most_common()]
        lines.append("")
    agreement = manifest.get("agreement") or {}
    if agreement:
        lines += ["## Inter-rater agreement", "", "```json", json.dumps(agreement, indent=2), "```", ""]

    lines += [
        "## Artifacts",
        "",
        f"- slicing index: `{index_path.name}`",
    ]
    if plot_path:
        lines.append(f"- coverage heatmap: `{Path(plot_path).name}`")
    lines.append("")

    report_path = run_dir / "coverage_report.md"
    report_path.write_text("\n".join(lines))

    return {
        "coverage_report": str(report_path),
        "index": str(index_path),
        "heatmap": plot_path,
    }


def _write_index(corpus: Sequence[Document], run_dir: Path, axis_names: list[str]) -> Path:
    """Write the parquet slicing index over (doc_id, chunk_id, doc_type, axes...).

    One row per (document, chunk) pair, so a groupby on chunk_id is a direct join
    back to the spec.

    Args:
        corpus: All documents.
        run_dir: Run directory.
        axis_names: Axis columns to include.

    Returns:
        Path to the written parquet.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for doc in corpus:
        for cid in doc.scenario.chunk_ids:
            row = {
                "doc_id": doc.doc_id,
                "scenario_hash": doc.scenario.scenario_hash,
                "chunk_id": cid,
                "spec_id": doc.scenario.spec_id,
                "doc_type": doc.scenario.doc_type,
                "grouping_strategy": doc.scenario.grouping_strategy,
                "n_chunks": len(doc.scenario.chunks),
                "filter_verdict": doc.filter_verdict or "",
                "dropped_by": doc.dropped_by,
                "n_words": len(doc.text().split()),
            }
            row.update({f"axis_{a}": str(doc.scenario.axes.get(a, "")) for a in axis_names})
            rows.append(row)

    path = run_dir / "coverage_index.parquet"
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return path
