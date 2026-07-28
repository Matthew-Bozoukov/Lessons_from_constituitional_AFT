# ABOUTME: The corpus catalogue: register every finished corpus, list what exists, and
# ABOUTME: compare two corpora as a paired difference on the shared scenarios.

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
INDEX_NAME = "corpora.json"


def index_path(output_dir: Path | str) -> Path:
    """Return the catalogue path for an output directory."""
    return Path(output_dir) / INDEX_NAME


def _top(mixture: dict[str, Any] | None, k: int = 3) -> str:
    """Summarize a mixture as its heaviest components."""
    if not mixture:
        return ""
    items = sorted(mixture.items(), key=lambda kv: -float(kv[1]))
    shown = ", ".join(f"{name}={float(w):.2f}" for name, w in items[:k])
    return shown + (f", +{len(items) - k} more" if len(items) > k else "")


def summarize(result: Any) -> dict[str, Any]:
    """Build the catalogue entry for a finished run.

    Records the fields that distinguish one corpus from another, so a researcher can
    tell from the listing alone which corpus is "the all-multiturn one" without
    opening any config.

    Args:
        result: A RunResult.

    Returns:
        The catalogue entry.
    """
    cfg = result.config
    recipe = cfg.get("recipe") or {}
    final = result.stages[-1]
    counts = result.counts.get(final, {})
    return {
        "name": cfg.get("name") or result.run_id,
        "run_id": result.run_id,
        "path": str(result.run_dir),
        "timestamp_utc": result.manifest.get("timestamp_utc", ""),
        "git_sha": result.manifest.get("git_sha", ""),
        "seed": cfg.get("seed", 0),
        "spec_id": (cfg.get("spec") or {}).get("id", ""),
        "spec_sha": result.manifest.get("spec", {}).get("spec_sha", ""),
        "chunker": ((cfg.get("spec") or {}).get("chunker") or {}).get("granularity", ""),
        "n_chunks": result.manifest.get("spec", {}).get("n_chunks", 0),
        "n_documents": counts.get("n", 0),
        "n_kept": counts.get("n_keep", 0),
        "mean_words": counts.get("mean_words", 0),
        "doc_type": _top(recipe.get("doc_type")),
        "grouping": _top(recipe.get("grouping")),
        "chunks_per_example": _top(recipe.get("chunks_per_example")),
        "generator_model": (cfg.get("generation") or {}).get("model", ""),
        "template": (cfg.get("generation") or {}).get("template", ""),
        "revision_dose": len(cfg.get("revision") or []),
        "revision_kinds": [e.get("kind") for e in (cfg.get("revision") or [])],
        "filters": [e.get("kind") for e in (cfg.get("filters") or [])],
        "cost_usd": result.manifest.get("cost_usd_total", 0.0),
        "hf_repo": result.manifest.get("hf_repo"),
        "exports": result.exports,
        "stages": result.stages,
    }


def register(result: Any) -> Path:
    """Add or update a corpus in the catalogue.

    Args:
        result: A RunResult.

    Returns:
        Path to the catalogue file.
    """
    output_dir = Path(result.config.get("output_dir", "output/synthdoc"))
    path = index_path(output_dir)
    entry = summarize(result)
    with _LOCK:
        entries = load_index(output_dir)
        entries = [e for e in entries if e.get("run_id") != entry["run_id"]]
        entries.append(entry)
        entries.sort(key=lambda e: e.get("timestamp_utc", ""), reverse=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries, indent=2, default=str))
    return path


def load_index(output_dir: Path | str) -> list[dict[str, Any]]:
    """Load the catalogue, returning [] when it does not exist yet."""
    path = index_path(output_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def format_index(entries: list[dict[str, Any]]) -> str:
    """Render the catalogue as an aligned table."""
    if not entries:
        return "No corpora yet. Run one with: uv run python -m synthdoc.cli run --config base.yaml"
    cols = [
        ("name", 26),
        ("spec_id", 20),
        ("doc_type", 34),
        ("n_kept", 8),
        ("rev", 4),
        ("generator_model", 28),
        ("cost_usd", 9),
    ]
    header = "  ".join(f"{c:<{w}}" for c, w in cols)
    lines = [header, "-" * len(header)]
    for e in entries:
        row = {
            "name": str(e.get("name", ""))[:26],
            "spec_id": str(e.get("spec_id", ""))[:20],
            "doc_type": str(e.get("doc_type", ""))[:34],
            "n_kept": f"{e.get('n_kept', 0)}/{e.get('n_documents', 0)}",
            "rev": str(e.get("revision_dose", 0)),
            "generator_model": str(e.get("generator_model", ""))[:28],
            "cost_usd": f"${float(e.get('cost_usd', 0) or 0):.2f}",
        }
        lines.append("  ".join(f"{row[c]:<{w}}" for c, w in cols))
    lines.append("")
    lines.append("Paths:")
    lines += [f"  {e.get('name')}: {e.get('path')}" for e in entries]
    return "\n".join(lines)


def list_hf(org: str, prefix: str = "synthdoc-", limit: int = 200) -> list[dict[str, Any]]:
    """List saved corpora from a HuggingFace namespace.

    HuggingFace is the durable home for corpora, so this - not the local index - is
    the authoritative listing. Details come from each repo's manifest.json; a repo
    whose manifest cannot be read still appears, with only the fields the Hub knows.

    Args:
        org: HF namespace, e.g. "LASR-Callum".
        prefix: Only list dataset repos whose name starts with this.
        limit: Maximum repos to inspect.

    Returns:
        Catalogue entries, newest first.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    entries: list[dict[str, Any]] = []
    for info in list(api.list_datasets(author=org, limit=limit)):
        name = info.id.split("/")[-1]
        if prefix and not name.startswith(prefix):
            continue
        entry: dict[str, Any] = {
            "name": name.removeprefix(prefix),
            "hf_repo": info.id,
            "path": f"hf://{info.id}",
            "timestamp_utc": str(getattr(info, "last_modified", "") or ""),
            "private": bool(getattr(info, "private", False)),
        }
        try:
            manifest_path = hf_hub_download(
                repo_id=info.id, filename="manifest.json", repo_type="dataset"
            )
            manifest = json.loads(Path(manifest_path).read_text())
            cfg = manifest.get("config") or {}
            recipe = cfg.get("recipe") or {}
            final = (manifest.get("stages") or [""])[-1]
            counts = (manifest.get("counts") or {}).get(final, {})
            entry.update(
                {
                    "run_id": manifest.get("run_id", ""),
                    "git_sha": manifest.get("git_sha", ""),
                    "spec_id": (cfg.get("spec") or {}).get("id", ""),
                    "spec_sha": (manifest.get("spec") or {}).get("spec_sha", ""),
                    "chunker": ((cfg.get("spec") or {}).get("chunker") or {}).get("granularity", ""),
                    "doc_type": _top(recipe.get("doc_type")),
                    "grouping": _top(recipe.get("grouping")),
                    "n_documents": counts.get("n", 0),
                    "n_kept": counts.get("n_keep", 0),
                    "generator_model": (cfg.get("generation") or {}).get("model", ""),
                    "template": (cfg.get("generation") or {}).get("template", ""),
                    "revision_dose": len(cfg.get("revision") or []),
                    "cost_usd": manifest.get("cost_usd_total", 0.0),
                    "stages": manifest.get("stages", []),
                }
            )
        except Exception as e:  # manifest missing, private, or unreadable
            entry["note"] = f"manifest unavailable: {type(e).__name__}"
        entries.append(entry)

    entries.sort(key=lambda e: str(e.get("timestamp_utc", "")), reverse=True)
    return entries


def fetch_hf(repo_id: str, cache_dir: Path | str = "output/synthdoc_cache/hf") -> Path:
    """Download a corpus's stage snapshots from HuggingFace into a local scratch dir.

    Used so that `compare` works against corpora that exist only on the Hub.

    Args:
        repo_id: Dataset repo id, with or without an `hf://` prefix.
        cache_dir: Where to place the downloaded parquet files.

    Returns:
        A directory laid out like a local run directory.
    """
    from huggingface_hub import HfApi, hf_hub_download

    repo_id = repo_id.removeprefix("hf://").strip("/")
    out = Path(cache_dir) / repo_id.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    wanted = [
        f
        for f in api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        if f.startswith("data/stage_") and f.endswith(".parquet")
    ] + ["manifest.json"]

    for remote in wanted:
        try:
            local = hf_hub_download(repo_id=repo_id, filename=remote, repo_type="dataset")
        except Exception:
            continue
        target = out / Path(remote).name
        target.write_bytes(Path(local).read_bytes())
    return out


def resolve_corpus(ref: str, cache_dir: Path | str = "output/synthdoc_cache/hf") -> Path:
    """Resolve a corpus reference to a local directory.

    Accepts a local run directory, an `hf://org/repo` URL, or a bare `org/repo`
    dataset id, so the same commands work whether or not the corpus is still on disk.

    Args:
        ref: The corpus reference.
        cache_dir: Scratch directory for Hub downloads.

    Returns:
        A directory containing stage snapshots.
    """
    path = Path(ref)
    if path.exists():
        return path
    if ref.startswith("hf://") or (ref.count("/") == 1 and not ref.startswith(".")):
        return fetch_hf(ref, cache_dir)
    raise FileNotFoundError(
        f"Corpus {ref!r} is neither a local directory nor an hf://org/repo reference."
    )


def final_snapshot(run_dir: Path | str) -> Path:
    """Return the final-stage parquet in a run directory.

    Args:
        run_dir: A run directory.

    Returns:
        Path to the filtered-stage parquet.

    Raises:
        FileNotFoundError: If no filtered snapshot is present.
    """
    matches = sorted(Path(run_dir).glob("stage_*_filtered.parquet"))
    if not matches:
        raise FileNotFoundError(f"No stage_*_filtered.parquet in {run_dir}")
    return matches[-1]


def _stats(frame) -> dict[str, Any]:
    """Compute corpus-level summary statistics from a snapshot frame."""
    kept = frame[frame.filter_verdict == "keep"]
    out: dict[str, Any] = {
        "n_documents": int(len(frame)),
        "n_kept": int(len(kept)),
        "keep_rate": round(len(kept) / max(1, len(frame)), 4),
        "mean_words": round(float(kept.n_words.mean()), 1) if len(kept) else 0.0,
        "mean_turns": round(float(kept.n_turns.mean()), 2) if len(kept) else 0.0,
        "cost_usd": round(float(frame.cost_usd.sum()), 4),
    }
    scores = frame["filter_scores"].apply(lambda s: dict(s) if s is not None else {})
    for field in sorted({k for s in scores for k in s}):
        values = [s[field] for s in scores if s.get(field) is not None]
        if values:
            out[f"mean_{field}"] = round(sum(values) / len(values), 3)
    return out


def compare(a_dir: Path | str, b_dir: Path | str, label_a: str = "A",
            label_b: str = "B") -> dict[str, Any]:
    """Compare two corpora, both overall and as a paired difference.

    The paired comparison is the one worth reading. Corpora that share scenarios join
    on `scenario_hash`, and the per-scenario difference removes the variance from
    which scenarios happened to be sampled - which is usually larger than the effect
    being measured. Where the arms share no scenarios (a recipe-axis ablation), only
    the marginal comparison is meaningful and the result says so.

    Args:
        a_dir: First run directory.
        b_dir: Second run directory.
        label_a: Display label for the first corpus.
        label_b: Display label for the second corpus.

    Returns:
        Overall stats for each side, paired deltas, and a per-doc_type breakdown.
    """
    import pandas as pd

    a = pd.read_parquet(final_snapshot(resolve_corpus(str(a_dir))))
    b = pd.read_parquet(final_snapshot(resolve_corpus(str(b_dir))))

    result: dict[str, Any] = {
        "a": {"label": label_a, "path": str(a_dir), **_stats(a)},
        "b": {"label": label_b, "path": str(b_dir), **_stats(b)},
    }
    result["delta"] = {
        key: round(result["b"][key] - result["a"][key], 4)
        for key in result["a"]
        if isinstance(result["a"].get(key), (int, float))
        and isinstance(result["b"].get(key), (int, float))
    }

    shared = set(a.scenario_hash) & set(b.scenario_hash)
    result["n_shared_scenarios"] = len(shared)
    result["paired"] = bool(shared) and len(shared) == len(set(a.scenario_hash)) == len(
        set(b.scenario_hash)
    )

    # Prefer scenario_hash: identical conditions on both sides. When the ablation
    # changed the recipe itself, no scenario_hash can match, but example i is still
    # example i - it differs from its counterpart only in the swept axis - so
    # sample_index recovers a genuine paired comparison instead of falling back to
    # marginals and losing most of the signal.
    join_key = "scenario_hash"
    if not shared and "sample_index" in a.columns and "sample_index" in b.columns:
        join_key = "sample_index"
        result["note"] = (
            "No shared scenario_hash: this ablation changed the recipe, so the "
            "conditions themselves differ. Paired on sample_index instead - example i "
            "in each arm differs only in the swept axis."
        )
    elif not shared:
        result["note"] = (
            "No shared scenarios and no sample_index: only the marginal comparison "
            "above is meaningful."
        )
        return result
    result["join_key"] = join_key

    def flat(frame):
        """Flatten the filter_scores struct into columns for joining."""
        scores = frame["filter_scores"].apply(lambda s: dict(s) if s is not None else {})
        wide = pd.DataFrame(list(scores), index=frame.index)
        keep = [join_key, "doc_type", "n_words", "n_turns", "filter_verdict"]
        return pd.concat([frame[keep], wide], axis=1)

    joined = flat(a).merge(flat(b), on=join_key, suffixes=("_a", "_b"))
    if joined.empty:
        result["note"] = "Corpora share no rows on either join key."
        return result
    metrics = ["n_words", "n_turns"] + [
        c[:-2] for c in joined.columns if c.endswith("_a") and c.startswith("autorater_")
    ] + (["dedup_max_sim"] if "dedup_max_sim_a" in joined.columns else [])

    paired: dict[str, Any] = {}
    for metric in metrics:
        ca, cb = f"{metric}_a", f"{metric}_b"
        if ca not in joined or cb not in joined:
            continue
        diff = (joined[cb] - joined[ca]).dropna()
        if diff.empty:
            continue
        paired[metric] = {
            "mean_delta": round(float(diff.mean()), 4),
            "std_delta": round(float(diff.std(ddof=1)), 4) if len(diff) > 1 else 0.0,
            "n": int(len(diff)),
            "b_higher_pct": round(float((diff > 0).mean()), 4),
        }
    result["paired_deltas"] = paired

    result["keep_flip"] = {
        "a_keep_b_drop": int(
            ((joined.filter_verdict_a == "keep") & (joined.filter_verdict_b == "drop")).sum()
        ),
        "a_drop_b_keep": int(
            ((joined.filter_verdict_a == "drop") & (joined.filter_verdict_b == "keep")).sum()
        ),
    }

    by_type: dict[str, Any] = {}
    for doc_type, group in joined.groupby("doc_type_a"):
        entry: dict[str, Any] = {"n": int(len(group))}
        for metric in ("n_words", "autorater_overall"):
            ca, cb = f"{metric}_a", f"{metric}_b"
            if ca in group and cb in group:
                diff = (group[cb] - group[ca]).dropna()
                if not diff.empty:
                    entry[f"delta_{metric}"] = round(float(diff.mean()), 3)
        by_type[str(doc_type)] = entry
    result["by_doc_type"] = by_type
    return result


def format_comparison(result: dict[str, Any]) -> str:
    """Render a comparison as markdown."""
    a, b = result["a"], result["b"]
    lines = [
        f"# Corpus comparison: `{a['label']}` vs `{b['label']}`",
        "",
        f"- shared scenarios: **{result['n_shared_scenarios']}** "
        f"(fully paired: **{result['paired']}**)",
        "",
        "## Marginals",
        "",
        f"| metric | {a['label']} | {b['label']} | delta |",
        "|---|---|---|---|",
    ]
    for key in a:
        if key in ("label", "path") or not isinstance(a[key], (int, float)):
            continue
        lines.append(f"| {key} | {a[key]} | {b.get(key, '')} | {result['delta'].get(key, '')} |")

    if result.get("note"):
        lines += ["", f"> {result['note']}", ""]

    if result.get("paired_deltas"):
        lines += [
            "",
            "## Paired deltas (per-scenario, b minus a)",
            "",
            "Lower variance than the marginals: the scenario is held fixed.",
            "",
            "| metric | mean delta | std | n | % where b higher |",
            "|---|---|---|---|---|",
        ]
        for metric, stats in result["paired_deltas"].items():
            lines.append(
                f"| {metric} | {stats['mean_delta']:+} | {stats['std_delta']} | "
                f"{stats['n']} | {stats['b_higher_pct']:.1%} |"
            )
        flip = result.get("keep_flip", {})
        lines += [
            "",
            f"Filter verdict flips: **{flip.get('a_drop_b_keep', 0)}** documents that "
            f"`{a['label']}` dropped were kept by `{b['label']}`; "
            f"**{flip.get('a_keep_b_drop', 0)}** went the other way.",
            "",
        ]

    if result.get("by_doc_type"):
        lines += ["## Paired delta by doc_type", "", "| doc_type | n | delta words | delta rater |", "|---|---|---|---|"]
        for doc_type, stats in sorted(result["by_doc_type"].items()):
            lines.append(
                f"| {doc_type} | {stats.get('n', 0)} | "
                f"{stats.get('delta_n_words', '-')} | {stats.get('delta_autorater_overall', '-')} |"
            )
        lines.append("")
    return "\n".join(lines)
