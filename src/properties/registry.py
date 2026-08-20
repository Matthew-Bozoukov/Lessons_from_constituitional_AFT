# ABOUTME: The Property row schema and properties.jsonl — the shared List of Properties.
# ABOUTME: THE ONLY WRITER: producers return Property objects, this file persists them.

"""The List of Properties.

Four producers feed one list. That list is what the ablation stage consumes, so a row has
to be legible to code that knows nothing about how it was made. Hence one schema, one
writer, and one field whose meaning is fixed across producers:

    prevalence   the share of records in the SAME corpus exhibiting this property.

Everything else on a row is advisory detail a reader uses to judge the label. `support` is
explicitly producer-specific and nothing merges on it.

Why one writer. Each producer already has its own run directory and its own artifacts, and
if each also appended to the shared list in its own way, the list would carry four
dialects of the same row and every consumer would have to know all four. Producers return
`Property` objects; `PropertyRegistry` is the only thing that turns them into lines.

The id is `<source>:<run>:<key>`, stable across reruns of the same run directory, so a
property referenced in an ablation config, a train config name, a LOG entry and a dataset
card all point at the same thing months later.

    property_id   "clusters:20260812_092119:g030"
    source        which producer emitted it
    label         the property, as a short phrase
    description   what the move is
    detector      the yes/no test a judge applies to ONE record  <- makes it actionable
    channel       query | reasoning | response
    confidence    the interpreter's confidence in the label; read before ablating
    caveat        what would make the label wrong
    prevalence    THE comparable number
    n_records     records exhibiting it
    corpus        which corpus that prevalence was measured on (repo + revision)
    support       producer-specific detail — never merged on
    evidence      examples, so a reader can judge the label
    provenance    run dir, git sha, models, params
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

DEFAULT_PATH = Path("output/properties/properties.jsonl")
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Property:
    """One named, detectable property of a corpus.

    Attributes:
        property_id: `<source>:<run>:<key>`. Stable across reruns of the same run dir.
        source: The producer that emitted it.
        label: The property, as a short phrase.
        description: What the move is, in a couple of sentences.
        detector: The yes/no test a judge applies to one record. Required: a property
            with no detector cannot be ablated, and cannot be shown to have been ablated.
        channel: Which channel it lives in — query, reasoning or response.
        confidence: The interpreter's own confidence in the label — high, medium or low.
            On the row rather than buried in provenance because it is what a reader
            checks before spending a training run on the property.
        caveat: What would make the label wrong, in one sentence, or "".
        interpreter_model: The model that wrote the label and the detector. A detector is
            an instrument; which model wrote it belongs beside its readings.
        prevalence: Share of the corpus's records exhibiting it, in [0, 1].
        n_records: Records exhibiting it.
        n_instances: Observations, repeats included (a record can exhibit it twice).
        corpus: What prevalence was measured on: {"repo", "file", "revision"} or {"path"}.
        target_id: The behaviour this was traced back from, for producers that need one.
        support: Producer-specific detail (cluster ids, influence scores, trait mix).
        evidence: Examples, so a reader can judge the label without rerunning anything.
        provenance: Run dir, git sha, models, embedding meta, grouping params.
        schema_version: Bumped when the row shape changes.
    """

    property_id: str
    source: str
    label: str
    detector: str
    channel: str = "reasoning"
    description: str = ""
    confidence: str = "medium"
    caveat: str = ""
    interpreter_model: str = ""
    prevalence: float | None = None
    n_records: int | None = None
    n_instances: int | None = None
    corpus: dict = field(default_factory=dict)
    target_id: str | None = None
    support: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Reject a row that cannot do the job a property row exists to do.

        Raises:
            ValueError: On a missing label or detector, or a prevalence outside [0, 1].
        """
        if not self.label.strip():
            raise ValueError(f"{self.property_id}: a property needs a label")
        if not self.detector.strip():
            raise ValueError(
                f"{self.property_id} ({self.label!r}): a property needs a detector. "
                "Without one it cannot select the rows an ablation edits, and cannot "
                "show afterwards that its prevalence moved — see shared/interpret.py.")
        if self.prevalence is not None and not 0.0 <= self.prevalence <= 1.0:
            raise ValueError(f"{self.property_id}: prevalence {self.prevalence} is not a "
                             "share of the corpus")

    @classmethod
    def make(cls, source: str, run: str, key: str, **kwargs) -> Property:
        """Build a Property with the id convention applied.

        Args:
            source: Producer name.
            run: Run identifier (a run directory's basename).
            key: Producer-local key, e.g. `c030` for cluster 30.
            **kwargs: The rest of the row.

        Returns:
            The Property.
        """
        return cls(property_id=f"{source}:{run}:{key}", source=source, **kwargs)

    @classmethod
    def from_dict(cls, row: dict) -> Property:
        """Rebuild a Property from a jsonl line.

        Args:
            row: The parsed line.

        Returns:
            The Property, ignoring any field a newer writer added.
        """
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in row.items() if k in known})

    def to_dict(self) -> dict:
        """This property as a jsonl-safe dict.

        Returns:
            The row.
        """
        return {"property_id": self.property_id, "source": self.source,
                "label": self.label, "description": self.description,
                "detector": self.detector, "channel": self.channel,
                "confidence": self.confidence, "caveat": self.caveat,
                "interpreter_model": self.interpreter_model,
                "prevalence": self.prevalence, "n_records": self.n_records,
                "n_instances": self.n_instances, "corpus": self.corpus,
                "target_id": self.target_id, "support": self.support,
                "evidence": self.evidence, "provenance": self.provenance,
                "schema_version": self.schema_version}

    def with_prevalence(self, measured: dict, corpus: dict | None = None) -> Property:
        """Attach a measured prevalence from `shared/interpret.prevalence`.

        A producer's own prevalence is derived from how it grouped its evidence, which is
        a different quantity from "what share of records does the detector say yes to".
        Re-measuring with the detector is what makes two producers' numbers comparable,
        and this is where that measurement lands.

        Args:
            measured: The dict `interpret.prevalence` returns.
            corpus: The corpus stamp it was measured on; keeps the existing one if None.

        Returns:
            A new Property carrying the measured numbers.
        """
        return replace(self, prevalence=measured["prevalence"],
                       n_records=measured["hits"], corpus=corpus or self.corpus,
                       support={**self.support,
                                "detector_measurement": {k: measured[k] for k in
                                                         ("n", "hits", "ci_low",
                                                          "ci_high", "n_errors")}})


class PropertyRegistry:
    """properties.jsonl, and the only thing that writes it.

    Attributes:
        path: The jsonl file.
    """

    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        """Open (without creating) a registry file.

        Args:
            path: Where the list lives.
        """
        self.path = Path(path)

    def read(self) -> list[Property]:
        """Read every property in the list.

        Returns:
            The properties, in file order; empty when the file does not exist yet.
        """
        if not self.path.exists():
            return []
        return [Property.from_dict(json.loads(line))
                for line in self.path.read_text(encoding="utf-8").split("\n")
                if line.strip()]

    def get(self, property_id: str) -> Property:
        """Look one property up by id.

        Args:
            property_id: The id.

        Returns:
            The property.

        Raises:
            KeyError: If no property has that id.
        """
        for prop in self.read():
            if prop.property_id == property_id:
                return prop
        raise KeyError(f"no property {property_id!r} in {self.path}. Known ids: "
                       f"{[p.property_id for p in self.read()][:10]}")

    def write(self, properties: list[Property]) -> Path:
        """Replace the list with these properties.

        Args:
            properties: The rows, most prevalent first by convention.

        Returns:
            The path written.

        Raises:
            ValueError: On a duplicate property_id — two rows with one id makes every
                downstream reference ambiguous.
        """
        duplicates = [pid for pid, n in Counter(p.property_id for p in properties).items()
                      if n > 1]
        if duplicates:
            raise ValueError(f"duplicate property_ids: {duplicates}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "".join(json.dumps(p.to_dict(), ensure_ascii=False) + "\n"
                    for p in properties), encoding="utf-8")
        return self.path

    def add(self, properties: list[Property], replace_source: bool = True) -> Path:
        """Merge a producer's rows into the list.

        Args:
            properties: The new rows. They must all come from one source.
            replace_source: Drop existing rows from the same source+run first, so
                re-running a producer updates its rows rather than duplicating them.

        Returns:
            The path written.

        Raises:
            ValueError: If the rows come from more than one source (each producer's
                output is merged separately, so a mixed batch is a caller bug).
        """
        if not properties:
            return self.path
        sources = {p.source for p in properties}
        if len(sources) != 1:
            raise ValueError(f"add() takes one source's rows at a time, got {sources}")
        existing = self.read()
        if replace_source:
            runs = {p.property_id.rsplit(":", 1)[0] for p in properties}
            existing = [p for p in existing
                        if p.property_id.rsplit(":", 1)[0] not in runs]
        return self.write(existing + list(properties))

    def report(self) -> str:
        """A markdown mirror of the list, so numbers are greppable without a json reader.

        Returns:
            The markdown.
        """
        properties = sorted(self.read(), key=lambda p: -(p.prevalence or 0))
        lines = [f"# List of Properties — {len(properties)} rows", "",
                 f"Source: `{self.path}`", ""]
        by_source = Counter(p.source for p in properties)
        lines += ["| producer | properties |", "|---|--:|"]
        lines += [f"| {s} | {n} |" for s, n in sorted(by_source.items())]
        lines += ["", "| property_id | label | channel | prevalence | records |",
                  "|---|---|---|--:|--:|"]
        for prop in properties:
            share = "—" if prop.prevalence is None else f"{prop.prevalence:.1%}"
            records = "—" if prop.n_records is None else prop.n_records
            lines.append(f"| `{prop.property_id}` | {prop.label} | {prop.channel} | "
                         f"{share} | {records} |")
        return "\n".join(lines) + "\n"


def label_collisions(properties: list[Property], threshold: float = 0.85,
                     backend: str = "openrouter") -> list[dict]:
    """Find near-duplicate labels ACROSS producers, so a merged list can be read honestly.

    Two producers describing the same behaviour in different words is the expected outcome,
    not an error — it is corroboration, and it is exactly what makes a property worth
    spending a training run on. But an unlabelled duplicate looks like two independent
    findings, and ablating both wastes a pod. This flags them; a human decides.

    Args:
        properties: The merged list.
        threshold: Cosine above which two labels are called near-duplicates.
        backend: Embedding backend for the labels.

    Returns:
        {"a", "b", "cosine"} triples, most similar first, across DIFFERENT RUNS only.
    """
    from src.properties.shared.embed import embed

    if len(properties) < 2:
        return []
    vectors, _ = embed([p.label for p in properties], backend=backend, probe=False)
    similarity = vectors @ vectors.T
    # `<source>:<run>`, not `source`. One config can run one producer twice — the same
    # clusterer over the reasoning channel and over the response channel — and those are
    # two independent fits whose labels agreeing IS corroboration. Comparing on `source`
    # alone would skip exactly those pairs as if they were within-fit duplicates.
    fit = [p.property_id.rsplit(":", 1)[0] for p in properties]
    out = []
    for i in range(len(properties)):
        for j in range(i + 1, len(properties)):
            if fit[i] == fit[j]:
                continue
            if similarity[i, j] >= threshold:
                out.append({"a": properties[i].property_id, "b": properties[j].property_id,
                            "a_label": properties[i].label, "b_label": properties[j].label,
                            "cosine": round(float(similarity[i, j]), 4)})
    return sorted(out, key=lambda r: -r["cosine"])
