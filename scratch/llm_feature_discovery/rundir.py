# ABOUTME: RunDir — the one place that knows what a feature-discovery run directory
# ABOUTME: contains and how each artifact is read and written.

"""The run-directory contract.

Every stage of this pipeline reads and writes the same directory, so the directory *is*
the interface between stages — there is no orchestrator holding state. That makes its
layout the thing most worth having in exactly one place: a stage that spells
`unique_features.txt` itself is a stage that can drift from the one that wrote it.

    features.jsonl          one {scenario_id, trait_id, features} per labelled trace
    unique_features.txt     the vocabulary, one string per line, in embedding-row order
    feature_counts.json     [[feature, occurrences], ...], most common first
    embeddings.npy          (n x d) fp16, L2-normalised, rows aligned to unique_features
    probe_embeddings.npy    (3 x d) the embedding sanity probes, if the run saved them
    embed_meta.json         embedding model, dimensions, sanity cosines
    umap_coords.npy         (n x m) the reduction the clusterer actually clustered
    feature_cluster_map.json   feature -> cluster id, NOISE FEATURES OMITTED
    clusters.json           {meta, clusters}: the canonical result
    report.md               human-readable mirror of clusters.json + the audit
    report_audit.json       redundancy pairs and keyword-probe counts
    dashboard.html          browsable version of the same
    clustering_comparison.* one clustering gated against another

`unique_features.txt` and `embeddings.npy` are row-aligned, and nothing downstream works
if they are not, so `read_embeddings` checks rather than trusting it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RunDir:
    """A feature-discovery run directory.

    Attributes:
        path: The directory itself. It need not exist yet; `ensure` creates it.
    """

    path: Path

    @classmethod
    def at(cls, path: str | Path) -> RunDir:
        """Build a RunDir from a string or Path.

        Args:
            path: The run directory.

        Returns:
            The RunDir.
        """
        return cls(Path(path))

    def ensure(self) -> RunDir:
        """Create the directory if it does not exist.

        Returns:
            Self, so this chains onto construction.
        """
        self.path.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def name(self) -> str:
        """The directory's basename, used as the run's title in reports.

        Returns:
            The basename.
        """
        return self.path.name

    def file(self, filename: str) -> Path:
        """Path to one artifact inside the run.

        Args:
            filename: Artifact filename.

        Returns:
            The absolute path, whether or not it exists.
        """
        return self.path / filename

    # ---------------------------------------------------------------- trace features ----

    def has_trace_features(self) -> bool:
        """Whether any traces have been labelled yet.

        Returns:
            True if features.jsonl exists.
        """
        return self.file("features.jsonl").exists()

    def read_trace_features(self) -> list[dict]:
        """Read the per-trace feature records.

        Returns:
            One {scenario_id, trait_id, features} dict per labelled trace.
        """
        return _read_jsonl(self.file("features.jsonl"))

    def labelled_scenario_ids(self) -> set[str]:
        """Scenario ids already labelled, so a resumed run can skip them.

        Returns:
            The ids, empty if nothing has been labelled.
        """
        if not self.has_trace_features():
            return set()
        return {record["scenario_id"] for record in self.read_trace_features()}

    def open_trace_features_for_append(self):
        """Open features.jsonl for appending.

        Extraction appends each trace's features the moment they land, so an interrupted
        run resumes instead of restarting.

        Returns:
            A writable text file handle the caller must close.
        """
        self.ensure()
        return self.file("features.jsonl").open("a")

    # ---------------------------------------------------------------- vocabulary --------

    def read_unique_features(self) -> list[str]:
        """Read the unique feature vocabulary, in embedding-row order.

        Returns:
            The feature strings.
        """
        return [line for line in self.file("unique_features.txt").read_text().splitlines()
                if line.strip()]

    def write_unique_features(self, features: list[str]) -> None:
        """Write the unique feature vocabulary.

        Args:
            features: Feature strings, in the order their embeddings will be in.
        """
        self.ensure()
        self.file("unique_features.txt").write_text("\n".join(features) + "\n")

    def write_feature_counts(self, counts: list[tuple[str, int]]) -> None:
        """Write the occurrence count of every unique feature.

        Args:
            counts: (feature, occurrences) pairs, most common first.
        """
        self.ensure()
        self.file("feature_counts.json").write_text(json.dumps(counts, indent=1))

    # ---------------------------------------------------------------- embeddings --------

    def read_embed_meta(self) -> dict:
        """Read the embedding metadata written by the embedding stage.

        Returns:
            Parsed embed_meta.json.
        """
        return json.loads(self.file("embed_meta.json").read_text())

    def read_embeddings(self, dtype=np.float32) -> np.ndarray:
        """Read the embedding matrix, checking it lines up with the vocabulary.

        Args:
            dtype: dtype to cast to. float32 is what every consumer wants; the file is fp16.

        Returns:
            (n x d) matrix whose rows correspond to read_unique_features().

        Raises:
            RuntimeError: If the matrix and the vocabulary disagree on length.
        """
        embeddings = np.asarray(np.load(self.file("embeddings.npy")), dtype=dtype)
        n_features = len(self.read_unique_features())
        if embeddings.shape[0] != n_features:
            raise RuntimeError(f"{self.path}: embeddings {embeddings.shape} vs "
                               f"{n_features} unique features")
        return embeddings

    def read_probe_embeddings(self) -> np.ndarray | None:
        """Read the embedding sanity probe vectors, if this run kept them.

        Runs embedded before 2026-08-18 stored only the probes' cosines, not the vectors.

        Returns:
            (3 x d) float32 matrix, or None.
        """
        probe_path = self.file("probe_embeddings.npy")
        if not probe_path.exists():
            return None
        return np.asarray(np.load(probe_path), dtype=np.float32)

    def write_umap_coords(self, coords: np.ndarray) -> None:
        """Save the reduction the clusterer clustered, so it can be re-examined.

        Args:
            coords: (n x m) UMAP coordinates.
        """
        self.ensure()
        np.save(self.file("umap_coords.npy"), np.asarray(coords, dtype=np.float32))

    # ---------------------------------------------------------------- clustering --------

    def read_feature_cluster_map(self) -> dict[str, int]:
        """Read feature -> cluster id.

        Noise features are absent from this map by construction; look features up with
        `.get` and treat a miss as "belongs to no cluster".

        Returns:
            The mapping.
        """
        return json.loads(self.file("feature_cluster_map.json").read_text())

    def write_feature_cluster_map(self, feature_to_cluster: dict[str, int]) -> None:
        """Write feature -> cluster id, which must already exclude noise.

        Args:
            feature_to_cluster: The mapping.

        Raises:
            ValueError: If a negative (noise) cluster id is present.
        """
        negative = [f for f, c in feature_to_cluster.items() if c < 0]
        if negative:
            raise ValueError(f"{len(negative)} noise features in the cluster map "
                             f"(e.g. {negative[:3]}); noise must be omitted, not stored")
        self.ensure()
        self.file("feature_cluster_map.json").write_text(json.dumps(feature_to_cluster))

    def read_clusters(self) -> dict:
        """Read the canonical clustering result.

        Returns:
            Parsed clusters.json: {"meta": {...}, "clusters": [...]}.
        """
        return json.loads(self.file("clusters.json").read_text())

    def write_clusters(self, meta: dict, clusters: list[dict]) -> None:
        """Write the canonical clustering result.

        Args:
            meta: Run metadata (params, counts, provenance).
            clusters: Per-cluster records, most prevalent first.
        """
        self.ensure()
        self.file("clusters.json").write_text(
            json.dumps({"meta": meta, "clusters": clusters}, indent=1))

    def cluster_count(self) -> int:
        """How many clusters this run actually produced.

        `meta["k"]` is the resolution knob of the k-means clusterer that was removed on
        2026-08-18, not a count; runs from before then carry only that, hence the fallback.

        Returns:
            The number of clusters.
        """
        meta = self.read_clusters()["meta"]
        return meta.get("n_clusters", meta.get("k"))

    # ---------------------------------------------------------------- reports ----------

    def write_text(self, filename: str, text: str) -> Path:
        """Write one text artifact.

        Args:
            filename: Artifact filename.
            text: Its contents.

        Returns:
            The path written.
        """
        self.ensure()
        path = self.file(filename)
        path.write_text(text)
        return path

    def append_text(self, filename: str, text: str) -> Path:
        """Append to one text artifact, creating it if absent.

        Args:
            filename: Artifact filename.
            text: Text to append.

        Returns:
            The path written.
        """
        self.ensure()
        path = self.file(filename)
        path.write_text((path.read_text() if path.exists() else "") + text)
        return path

    def write_json(self, filename: str, payload) -> Path:
        """Write one JSON artifact.

        Args:
            filename: Artifact filename.
            payload: Anything json-serialisable.

        Returns:
            The path written.
        """
        return self.write_text(filename, json.dumps(payload, indent=1))


def _read_jsonl(path: Path) -> list[dict]:
    """Read a jsonl file, skipping blank lines.

    Args:
        path: The file.

    Returns:
        One dict per line.
    """
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream a jsonl file that is not part of a run directory (e.g. the input SFT file).

    Args:
        path: The file.

    Yields:
        One dict per non-blank line.
    """
    for line in Path(path).read_text().splitlines():
        if line.strip():
            yield json.loads(line)
