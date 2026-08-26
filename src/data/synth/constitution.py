# ABOUTME: Stage 1 -- cut the constitution into chunks and group them into the addressable
# ABOUTME: units a document is generated against. Deterministic, no LLM, testable offline.

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

# A principle is either a numbered item whose title is bolded (the v1 format), e.g.
#   1. **Honesty and non-deception.** Do not help the user deceive...
# or a numbered H2 unit (the specgen format), e.g.
#   ## 4. Be scrupulously honest and non-deceptive, in word, framing, and action
_PRINCIPLE = re.compile(r"^(\d+)\.\s+\*\*(.+?)\*\*\s*(.*)$")
_UNIT_HEADING = re.compile(r"^##\s+(\d+)\.\s+(.+?)\s*$")
_HEADING = re.compile(r"^##\s+(.*)$")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")

# How finely the constitution is cut. `principle` is the default and reproduces the
# original segmentation byte for byte. Our constitutions carry exactly one heading level,
# so the "section" granularity named in the literature IS `principle` here -- there is no
# coarser cut short of `whole`, and inventing one would be a granularity in name only.
GRANULARITIES = ("whole", "principle", "paragraph", "bullet")

# How chunks are assembled into units. Every strategy PARTITIONS the chunk pool: each
# chunk lands in exactly one unit, so total constitution content is identical across arms
# and group size is not confounded with coverage.
STRATEGIES = ("single", "adjacent", "random", "lexical", "cluster")

# The chunking provenance a generated record carries beyond the Trait interface. Stage 2
# copies these onto every scenario, so which unit a document came from -- and how that
# unit was cut and grouped -- is readable from the document itself, by the metadata
# export, the corpus checks and `balance_by` alike. Named here, beside the `Unit` that
# produces them, so the list cannot drift from the dataclass.
UNIT_PROVENANCE = ("chunk_ids", "granularity", "grouping_strategy", "n_chunks")


@dataclass(frozen=True)
class Trait:
    """One addressable section of the constitution.

    Attributes:
        trait_id: Stable identifier, e.g. "t3".
        index: 1-based position in the source document.
        name: Short title, e.g. "Avoid facilitating harm or illegality".
        text: The full principle text, title included.
    """

    trait_id: str
    index: int
    name: str
    text: str

    def as_dict(self) -> dict:
        """Return a JSON-serialisable form."""
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict) -> Trait:
        """Rebuild from a stage-1 record, ignoring the unit provenance fields.

        Stage-1 rows are a SUPERSET of the Trait fields (they also carry `chunk_ids`,
        `granularity`, `grouping_strategy`, `n_chunks`), so `Trait(**record)` would
        raise. Every downstream operator reconstructs through here instead.
        """
        return cls(**{f: record[f] for f in ("trait_id", "index", "name", "text")})


@dataclass(frozen=True)
class Chunk:
    """One piece of the constitution at the configured granularity.

    Attributes:
        chunk_id: Stable id -- "t3" at principle granularity, "t3.b02" below it.
        parent_id: The principle this chunk was cut from ("t3"); == chunk_id at
            principle granularity. Grouping never crosses a parent boundary silently.
        index: 1-based principle number, so document order is (index, order_idx).
        order_idx: Position within the parent principle.
        name: The parent principle's title -- carried on every sub-chunk so a chunk
            stays self-contained no matter how finely it was cut.
        text: The chunk text exactly as the generator will see it.
        granularity: One of GRANULARITIES.
    """

    chunk_id: str
    parent_id: str
    index: int
    order_idx: int
    name: str
    text: str
    granularity: str

    def as_dict(self) -> dict:
        """Return a JSON-serialisable form."""
        return asdict(self)


@dataclass(frozen=True)
class Unit:
    """The thing one document is generated against: one or more chunks taken together.

    Single-chunk and many-chunk units are the same type. A unit renders to exactly the
    fields every downstream stage, prompt, id and `balance_by` already consume
    (`trait_id`/`index`/`name`/`text`), so nothing after stage 1 knows chunking exists.

    Attributes:
        unit_id: Becomes `trait_id`. "t3" for a lone chunk, "t3+t7" for a group,
            "c1" for a cluster, "all" for the whole document.
        index: Lowest member principle number; fixes unit ordering.
        name: Member names joined, deduplicated.
        text: Member texts joined in document order -- never draw order, so two arms
            that select the same members produce byte-identical text.
        chunk_ids: Members, in document order. The provenance a coverage report joins on.
        granularity: The granularity its chunks were cut at.
        grouping_strategy: One of STRATEGIES.
        n_chunks: len(chunk_ids). Denormalised so a groupby needs no parsing.
    """

    unit_id: str
    index: int
    name: str
    text: str
    chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    granularity: str = "principle"
    grouping_strategy: str = "single"
    n_chunks: int = 1

    def as_trait(self) -> Trait:
        """Project onto the downstream Trait interface."""
        return Trait(trait_id=self.unit_id, index=self.index, name=self.name,
                     text=self.text)

    def as_dict(self) -> dict:
        """Return the stage-1 record: Trait fields first, then unit provenance."""
        return {**self.as_trait().as_dict(), "chunk_ids": list(self.chunk_ids),
                "granularity": self.granularity,
                "grouping_strategy": self.grouping_strategy,
                "n_chunks": self.n_chunks}


def _dedent_join(lines: list[str]) -> str:
    """Join continuation lines into one paragraph, collapsing indentation."""
    return " ".join(line.strip() for line in lines if line.strip())


@dataclass(frozen=True)
class _Raw:
    """One parsed principle before any granularity decision is applied."""

    index: int
    name: str
    body: tuple[str, ...]


def _parse(path: str | Path) -> tuple[list[_Raw], str, str]:
    """Split the constitution into raw principles, style guidance and preamble.

    The document has three parts: a numbered list of principles, a prose section
    describing what an aligned response looks like, and everything else (title,
    priority/conflict-resolution preamble). The principles become chunks; the prose is
    shared guidance injected everywhere, since it constrains tone rather than naming a
    distinct value; the preamble is chunked only at `whole` granularity and is otherwise
    reported by the dry-run rather than silently discarded.

    Args:
        path: Path to the constitution markdown.

    Returns:
        (principles, style_guidance, preamble).
    """
    # encoding is explicit because `read_text()` defaults to the LOCALE encoding, which is
    # cp1252 on a Windows driver: every em-dash in the constitution then decodes to the
    # three characters "a-circumflex, euro, right-quote" and the mojibake is injected into
    # every prompt that carries a principle or the style guidance. Measured 2026-08-26 --
    # it also changed the run manifest's `constitution_sha256`, which made a corpus
    # generated on Linux look like it came from a different constitution and tripped
    # `load_source_run`'s cross-arm assertion. See docs/GOTCHAS.md.
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    raws: list[_Raw] = []
    style: list[str] = []
    preamble: list[str] = []
    current: tuple[int, str, list[str]] | None = None
    in_style = False

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        idx, name, body = current
        raws.append(_Raw(index=idx, name=name, body=tuple(body)))
        current = None

    for line in lines:
        unit = _UNIT_HEADING.match(line)
        if unit:  # specgen format: each numbered H2 opens a principle
            flush()
            current = (int(unit.group(1)), unit.group(2), [])
            in_style = False
            continue
        heading = _HEADING.match(line)
        if heading:
            flush()
            # Everything after the principles list is tone/style guidance. Un-numbered
            # headings before them (title, preamble) start no principle.
            in_style = "look" in heading.group(1).lower()
            if not in_style:
                preamble.append(line)
            continue
        if in_style:
            style.append(line)
            continue
        m = _PRINCIPLE.match(line)
        if m:
            flush()
            current = (int(m.group(1)), m.group(2), [m.group(3)])
        elif current is not None:
            current[2].append(line)
        else:
            preamble.append(line)
    flush()

    return raws, "\n".join(style).strip(), "\n".join(preamble).strip()


def segment(path: str | Path) -> tuple[list[Trait], str]:
    """Split the constitution into its numbered traits plus the shared style section.

    The `principle`-granularity, one-chunk-per-unit special case of `chunk` + `group`,
    kept as the stable public entry point. A test pins it byte-identical to the
    two-step path over every constitution in the repository.

    Args:
        path: Path to the constitution markdown.

    Returns:
        (traits, style_guidance).

    Raises:
        AssertionError: If no principles are found, which would silently produce an
            empty run.
    """
    chunks, style = chunk(path, granularity="principle")
    units = group(chunks, size=1, strategy="single")
    return [u.as_trait() for u in units], style


def full_text(path: str | Path) -> str:
    """Return the whole constitution, for stages that inject the complete document."""
    # Explicit encoding for the same reason as `_parse` above.
    return Path(path).read_text(encoding="utf-8").strip()


def preamble(path: str | Path) -> str:
    """Return the text that belongs to no chunk: title, and the priority/conflict
    section that sits before the numbered principles.

    Worth surfacing rather than ignoring: the conflict-resolution guidance is exactly
    the material that says how principles trade off, and at every granularity below
    `whole` it reaches the generator only through `{constitution}` -- never through the
    chunk a document is built around.
    """
    return _parse(path)[2]


# --- chunking -----------------------------------------------------------------------


def _blocks(body: list[str]) -> list[str]:
    """Blank-line-separated blocks of a principle body, structure preserved."""
    out: list[str] = []
    buf: list[str] = []
    for line in body:
        if line.strip():
            buf.append(line)
        elif buf:
            out.append("\n".join(buf).strip())
            buf = []
    if buf:
        out.append("\n".join(buf).strip())
    return [b for b in out if b]


def _bullets(body: list[str]) -> list[str]:
    """One piece per bullet or numbered item; prose kept whole so nothing is dropped.

    Continuation lines fold into their bullet. A principle written without bullets
    degrades to `_blocks`, which is what makes this safe on every constitution.
    """
    out: list[str] = []
    buf: list[str] = []
    prose: list[str] = []

    def flush_bullet() -> None:
        nonlocal buf
        if buf:
            out.append("\n".join(buf).strip())
            buf = []

    def flush_prose() -> None:
        # Prose between bullets still splits on blank lines, or `bullet` would be no
        # finer than `paragraph` wherever a principle opens with several paragraphs.
        nonlocal prose
        out.extend(_blocks(prose))
        prose = []

    for line in body:
        if _BULLET.match(line):
            flush_prose()
            flush_bullet()
            buf = [line]
        elif buf and line.strip():
            buf.append(line)
        elif buf:
            flush_bullet()
        else:
            prose.append(line)
    flush_bullet()
    flush_prose()
    return [o for o in out if o]


def _coalesce(pieces: list[str], min_words: int) -> list[str]:
    """Merge sub-minimum pieces into a neighbour rather than dropping them.

    Dropping stubs would mean a finer granularity silently trains on LESS of the
    constitution than a coarser one, which would confound every granularity comparison
    with a coverage difference. Merging keeps the union of chunk text equal to the
    source at every granularity.
    """
    out: list[str] = []
    for p in pieces:
        if out and len(p.split()) < min_words:
            out[-1] = f"{out[-1]}\n{p}"
        else:
            out.append(p)
    # A leading stub has no previous piece to merge into; fold it forward instead.
    while len(out) > 1 and len(out[0].split()) < min_words:
        out[:2] = [f"{out[0]}\n{out[1]}"]
    return out


def chunk(path: str | Path, granularity: str = "principle",
          min_words: int = 12) -> tuple[list[Chunk], str]:
    """Cut the constitution into chunks at the requested granularity.

    Args:
        path: Path to the constitution markdown.
        granularity: One of GRANULARITIES. `whole` yields a single chunk carrying the
            entire document (the no-chunking arm); `principle` reproduces the original
            segmentation; `paragraph` and `bullet` cut inside each principle, carrying
            the principle title onto every sub-chunk so a chunk stays self-contained.
        min_words: Sub-principle pieces shorter than this merge into a neighbour.

    Returns:
        (chunks, style_guidance).

    Raises:
        ValueError: Unknown granularity.
        AssertionError: No principles found, or principles out of order / duplicated.
    """
    if granularity not in GRANULARITIES:
        raise ValueError(f"unknown granularity {granularity!r}. "
                         f"Known: {list(GRANULARITIES)}")
    raws, style, _ = _parse(path)

    if granularity == "whole":
        return [Chunk(chunk_id="all", parent_id="all", index=0, order_idx=0,
                      name="the constitution", text=full_text(path),
                      granularity="whole")], style

    assert raws, f"no numbered principles found in {path}"
    indices = [r.index for r in raws]
    assert indices == sorted(indices), f"principles are out of order: {indices}"
    assert len(set(indices)) == len(indices), f"duplicate principle numbers: {indices}"

    chunks: list[Chunk] = []
    for r in raws:
        parent, name = f"t{r.index}", r.name.rstrip(".")
        if granularity == "principle":
            # The original rendering, preserved exactly: title inline, body collapsed
            # onto one line. Everything below this granularity keeps its structure.
            chunks.append(Chunk(
                chunk_id=parent, parent_id=parent, index=r.index, order_idx=0,
                name=name, text=f"**{r.name}** {_dedent_join(list(r.body))}".strip(),
                granularity=granularity))
            continue
        cut = _blocks if granularity == "paragraph" else _bullets
        tag = "p" if granularity == "paragraph" else "b"
        for j, piece in enumerate(_coalesce(cut(list(r.body)), min_words)):
            chunks.append(Chunk(
                chunk_id=f"{parent}.{tag}{j:02d}", parent_id=parent, index=r.index,
                order_idx=j, name=name, text=f"**{r.name}**\n\n{piece}".strip(),
                granularity=granularity))
    return chunks, style


# --- grouping -----------------------------------------------------------------------


def _doc_order(chunks: list[Chunk]) -> list[Chunk]:
    """Chunks in the order the document presents them."""
    return sorted(chunks, key=lambda c: (c.index, c.order_idx))


def _partition(ordered: list[Chunk], size: int) -> list[list[Chunk]]:
    """Cut an ordered pool into consecutive blocks of `size` (the tail may be shorter)."""
    return [ordered[i:i + size] for i in range(0, len(ordered), size)]


def _features(chunks: list[Chunk]):
    """L2-normalised hashed char-ngram features, one row per chunk.

    The same featuriser the corpus checks use, so "how similar are two pieces of text"
    means one thing across the pipeline -- grouping chunks here and measuring corpus
    diversity there. Imported lazily because it pulls numpy, which the offline dry-run
    (`synth segment`, `synth chunkings`) must not need.
    """
    from .check_corpus import hashed_features

    return hashed_features([c.text for c in chunks])


def _lexical_groups(ordered: list[Chunk], size: int) -> list[list[Chunk]]:
    """Greedy nearest-neighbour partition: each anchor takes its most similar leftovers.

    Groups material the document happens to separate. Deterministic -- the anchor is
    always the earliest unassigned chunk and similarity ties break on document order.
    """
    sims = _features(ordered) @ _features(ordered).T
    remaining = list(range(len(ordered)))
    out: list[list[Chunk]] = []
    while remaining:
        anchor = remaining.pop(0)
        picks = [anchor]
        for j in sorted(remaining, key=lambda j: (-float(sims[anchor][j]), j))[:size - 1]:
            picks.append(j)
            remaining.remove(j)
        out.append([ordered[i] for i in sorted(picks)])
    return out


def _cluster_groups(ordered: list[Chunk], n_clusters: int) -> list[list[Chunk]]:
    """Partition into k clusters over hashed features -- the no-chunking, CMT-style arm.

    Fully deterministic: the first centroid is the most central chunk (highest mean
    similarity to the rest), the remaining centroids are farthest-point seeds, then
    Lloyd iterations to convergence. No RNG, so an arm replays exactly.
    """
    import numpy as np

    X = _features(ordered)
    n = len(ordered)
    k = max(1, min(int(n_clusters), n))
    sims = X @ X.T
    seeds = [int(np.argmax(sims.mean(axis=1)))]
    while len(seeds) < k:
        # Farthest point: the chunk least similar to its nearest existing seed.
        seeds.append(int(np.argmin(sims[seeds].max(axis=0))))
    centroids = X[seeds].copy()

    # Capacity-constrained assignment. Plain Lloyd collapses here: hashed char-ngram
    # features put every chunk of one document within a narrow similarity band, so
    # unconstrained k-means happily returns one cluster of 30 and one of 1. An arm whose
    # units differ 60x in length is not comparable to anything, so each cluster is capped
    # at ceil(n/k) and the most confident (chunk, cluster) pairs claim their slots first.
    cap = -(-n // k)
    labels = np.full(n, -1)
    for _ in range(50):
        scores = X @ centroids.T
        pairs = sorted(((float(scores[i][c]), i, c) for i in range(n) for c in range(k)),
                       key=lambda t: (-t[0], t[1], t[2]))
        new = np.full(n, -1)
        counts = [0] * k
        for _score, i, c in pairs:
            if new[i] < 0 and counts[c] < cap:
                new[i] = c
                counts[c] += 1
        if bool((new == labels).all()):
            break
        labels = new
        for c in range(k):
            members = X[labels == c]
            if len(members):
                v = members.sum(axis=0)
                norm = float(np.linalg.norm(v))
                centroids[c] = v / norm if norm else centroids[c]

    groups = [[ordered[i] for i in range(len(ordered)) if labels[i] == c]
              for c in range(k)]
    groups = [g for g in groups if g]
    # Order clusters by their earliest member so cluster ids are stable across runs.
    return sorted(groups, key=lambda g: (g[0].index, g[0].order_idx))


def _make_unit(members: list[Chunk], strategy: str,
               unit_id: str | None = None) -> Unit:
    """Build one Unit from its members, rendering them in document order."""
    ms = _doc_order(members)
    if len(ms) == 1 and unit_id is None:
        uid, name, text = ms[0].chunk_id, ms[0].name, ms[0].text
    else:
        uid = unit_id or "+".join(m.chunk_id for m in ms)
        name = " + ".join(dict.fromkeys(m.name for m in ms))
        text = "\n\n".join(m.text for m in ms)
    return Unit(unit_id=uid, index=ms[0].index, name=name, text=text,
                chunk_ids=tuple(m.chunk_id for m in ms),
                granularity=ms[0].granularity, grouping_strategy=strategy,
                n_chunks=len(ms))


def group(chunks: list[Chunk], size: int = 1, strategy: str = "single",
          seed: int = 0, n_clusters: int = 4) -> list[Unit]:
    """Assemble chunks into the units documents are generated against.

    Every strategy partitions the pool -- each chunk lands in exactly one unit. That is
    deliberate: a sampling grouper would let a k=2 arm see more (or less) of the
    constitution than a k=1 arm, confounding group size with coverage.

    Args:
        chunks: The chunk pool from `chunk`.
        size: Chunks per unit. Ignored by `cluster`.
        strategy: One of STRATEGIES. `single` = one unit per chunk (the 1:1 recipe);
            `adjacent` = consecutive chunks, preserving the document's own structure;
            `random` = seeded shuffle then partition, pairing unrelated principles;
            `lexical` = greedy nearest-neighbour on hashed features; `cluster` =
            k clusters over the whole pool, ignoring `size`.
        seed: Seed for `random`. The other strategies are seed-independent.
        n_clusters: Cluster count for `cluster`.

    Returns:
        Units in document order.

    Raises:
        ValueError: Unknown strategy, or size < 1, or size > 1 under `single`.
        AssertionError: The partition failed to cover the pool exactly once.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown grouping strategy {strategy!r}. "
                         f"Known: {list(STRATEGIES)}")
    if size < 1:
        raise ValueError(f"group size must be >= 1, got {size}")
    if strategy == "single" and size != 1:
        raise ValueError(f"strategy 'single' implies size 1, got {size}. Pick a real "
                         f"grouping strategy ({[s for s in STRATEGIES if s != 'single']}) "
                         f"to combine chunks.")
    assert chunks, "group() called with an empty chunk pool"
    ordered = _doc_order(list(chunks))

    if strategy in ("single", "adjacent"):
        groups = _partition(ordered, size)
    elif strategy == "random":
        import random as _random

        shuffled = list(ordered)
        _random.Random(seed).shuffle(shuffled)
        groups = _partition(shuffled, size)
    elif strategy == "lexical":
        groups = _lexical_groups(ordered, size)
    else:
        groups = _cluster_groups(ordered, n_clusters)

    ids = [c.chunk_id for g in groups for c in g]
    assert sorted(ids) == sorted(c.chunk_id for c in ordered), (
        f"{strategy} grouping did not partition the pool: {len(ids)} assignments "
        f"for {len(ordered)} chunks")

    units = [_make_unit(g, strategy,
                        unit_id=f"c{i + 1}" if strategy == "cluster" else None)
             for i, g in enumerate(groups)]
    return sorted(units, key=lambda u: (u.index, u.unit_id))


# --- the named chunking methods a dataset config picks from --------------------------


@dataclass(frozen=True)
class Chunking:
    """One named way of turning a constitution into units.

    Attributes:
        name: The value a dataset config's `chunking:` flag carries.
        granularity: How finely to cut (a GRANULARITIES member).
        strategy: How to assemble chunks into units (a STRATEGIES member).
        size: Chunks per unit. Ignored by `cluster`.
        n_clusters: Cluster count for `cluster`.
        min_words: Sub-principle pieces shorter than this merge into a neighbour.
        summary: One line, printed by `synth chunkings`.
    """

    name: str
    granularity: str
    strategy: str = "single"
    size: int = 1
    n_clusters: int = 4
    min_words: int = 12
    summary: str = ""


# Keyed by the name a config writes. `principle` is the DEFAULT and is the method every
# corpus in this repository was generated with -- leave it alone; the others exist to be
# compared against it. Names read <granularity>[_<how they are grouped>].
CHUNKINGS: dict[str, Chunking] = {c.name: c for c in (
    Chunking("principle", "principle", summary=(
        "DEFAULT. One numbered principle per document -- the Teaching Claude Why "
        "recipe, and what every existing corpus here was built with.")),
    Chunking("paragraph", "paragraph", summary=(
        "One paragraph of a principle per document (its statement, its rationale, its "
        "exceptions each stand alone).")),
    Chunking("bullet", "bullet", summary=(
        "One bullet or paragraph per document -- the finest cut, and GDM's choice.")),
    Chunking("whole", "whole", summary=(
        "No chunking: the entire constitution is the unit. Coverage stops being "
        "guaranteed by construction and has to be verified in the corpus instead.")),
    Chunking("principle_pairs_adjacent", "principle", "adjacent", size=2, summary=(
        "Two consecutive principles per document, preserving the document's order.")),
    Chunking("principle_pairs_random", "principle", "random", size=2, summary=(
        "Two unrelated principles per document -- can the model hold both at once?")),
    Chunking("principle_pairs_related", "principle", "lexical", size=2, summary=(
        "Two similar principles per document, pairing material the document "
        "separates.")),
    Chunking("paragraph_clusters", "paragraph", "cluster", n_clusters=4, summary=(
        "Cut into paragraphs, then regroup into 4 semantic clusters -- the "
        "embed-and-cluster shape, no per-principle chunking at all.")),
)}

DEFAULT_CHUNKING = "principle"


def resolve_chunking(name: str | None) -> Chunking:
    """Look up a chunking method by name, listing the options when it is not one.

    Args:
        name: A CHUNKINGS key, or None for the default.

    Returns:
        The Chunking.

    Raises:
        ValueError: The name is not a registered method.
    """
    key = str(name or DEFAULT_CHUNKING)
    if key not in CHUNKINGS:
        raise ValueError(
            f"unknown chunking {key!r}. Registered methods: {sorted(CHUNKINGS)}. "
            "Run `uv run synth chunkings` to see what each one does, or add a new "
            "entry to CHUNKINGS in src/data/synth/constitution.py.")
    return CHUNKINGS[key]


def units_from_config(cfg: dict) -> tuple[list[Unit], str]:
    """Build the unit set a run config declares. The single place the two steps meet.

    A dataset config selects a method by name (`chunking: bullet`); a config with no
    `chunking:` flag gets the default, which is the original recipe exactly.

    Args:
        cfg: A run config carrying `constitution:` and an optional `chunking:` name.

    Returns:
        (units, style_guidance).
    """
    ch = cfg.get("chunking")
    if ch is not None and not isinstance(ch, str):
        raise ValueError(
            f"`chunking:` takes the NAME of a method, got {type(ch).__name__}. "
            f"Registered: {sorted(CHUNKINGS)}. Settings live with the method in "
            "CHUNKINGS (src/data/synth/constitution.py) so a manifest records which "
            "recipe ran, not an anonymous bag of knobs.")
    spec = resolve_chunking(ch)
    chunks, style = chunk(cfg["constitution"], granularity=spec.granularity,
                          min_words=spec.min_words)
    units = group(chunks, size=spec.size, strategy=spec.strategy,
                  seed=int(cfg.get("seed", 0)), n_clusters=spec.n_clusters)
    return select_units(units, cfg), style


def select_units(units: list[Unit], cfg: dict) -> list[Unit]:
    """Keep only the units a config's `only_traits:` names, in document order.

    A per-trait arm (one principle's data, nothing else) is a restriction of the
    standard run, not a different recipe: the document is cut and grouped exactly as
    before, the whole text still reaches the stages that inject `{constitution}`, and
    only the set of units documents are generated against shrinks. Unlike `max_traits`
    (the first N units, a smoke-test convenience) this selects by id, so a trait
    anywhere in the document can be run alone.

    Args:
        units: The full unit set for the config's constitution and chunking.
        cfg: A run config; `only_traits:` is an optional list of unit ids (`t10`,
            `t3+t7`, `c1`, ...). Absent or empty means no restriction.

    Returns:
        The selected units, in document order.

    Raises:
        ValueError: A requested id matches no unit, which would otherwise generate a
            smaller corpus than the config declares without saying so.
    """
    wanted = cfg.get("only_traits")
    if not wanted:
        return units
    wanted = [str(w) for w in ([wanted] if isinstance(wanted, str) else wanted)]
    known = {u.unit_id: u for u in units}
    missing = [w for w in wanted if w not in known]
    if missing:
        raise ValueError(
            f"only_traits names {missing}, but {cfg['constitution']} under chunking "
            f"{cfg.get('chunking') or DEFAULT_CHUNKING!r} yields {sorted(known)}.")
    return [u for u in units if u.unit_id in set(wanted)]
