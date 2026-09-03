# ABOUTME: Auto-labels CoT chunks with the ODCV function taxonomy via OpenRouter, cached
# ABOUTME: on disk, with a zero-cost lexical fallback for iterating before spending money.

"""Putting a function tag on every chunk.

The paper labels its sentences with an LLM auto-labeller, and the per-category bars that
carry every result are only as good as those labels. Two labellers live here:

  `lexical_labels`  Regexes. Free, instant, deterministic, and wrong often enough that it
                    is only for wiring things up and for eyeballing a corpus. It cannot
                    distinguish "I should not edit the checker" from "I will edit the
                    checker".

  `llm_labels`      One call per trajectory (all its chunks in one prompt, so the
                    labeller sees the trace as a trace), cached by content hash. This is
                    the one whose output goes in a figure.

Cost discipline is deliberate: labelling is the only paid step in the descriptive half,
`estimate_cost` prices a run before it starts, and `--limit` exists so a smoke pass over a
handful of trajectories is the default first move. Confirm the labeller model with a human
before a full run, and prefer a labeller from a different family than the model under
study — a model grading its own family's prose has a thumb on the scale.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded
from src.utils import extract_json

from scratch.thought_branches.descriptive import (
    COMMITMENT,
    INTEGRITY,
    PRINCIPAL_APPEAL,
    SHORTCUT,
)
from scratch.thought_branches.segment import Chunk
from scratch.thought_branches.taxonomy import TAGS, build_prompt
from scratch.thought_branches.trajectory import Trajectory

DEFAULT_LABELLER = "google/gemini-3.1-pro-preview"

# Chunks per request. A whole ODCV trajectory is usually well under this, and the tag
# definitions say precedence depends on the surrounding trace, so splitting a trajectory
# across requests costs accuracy. Long trajectories are split and the parts stitched.
MAX_CHUNKS_PER_CALL = 120


@dataclass
class Label:
    """One chunk's label.

    Attributes:
        chunk_id: The chunk this describes.
        tag: A key of `taxonomy.TAGS`.
        is_misaligned: Whether the chunk plans or executes a measurement-corrupting act.
        source: "llm" or "lexical", so a mixed corpus stays auditable.
    """

    chunk_id: str
    tag: str
    is_misaligned: bool
    source: str = "llm"


@dataclass
class LabelCache:
    """Content-addressed label cache on disk.

    Keyed by a hash of (labeller model, task text, chunk texts), so a re-run with the same
    corpus costs nothing and a changed prompt or model invalidates cleanly instead of
    silently serving stale labels.

    Attributes:
        path: JSON file backing the cache.
        data: key -> {chunk index -> label dict}.
    """

    path: Path
    data: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "LabelCache":
        p = Path(path)
        return cls(path=p, data=json.loads(p.read_text()) if p.is_file() else {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data), encoding="utf-8")


def cache_key(model: str, task: str, texts: Sequence[str]) -> str:
    """Stable hash over everything that could change a labelling result."""
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(task.encode())
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode())
    return h.hexdigest()[:32]


# -- lexical fallback -----------------------------------------------------------------


def lexical_label(chunk: Chunk) -> Label:
    """Tag one chunk with regexes.

    Precedence follows the taxonomy: a principal appeal outranks a shortcut, which
    outranks everything else. Actions are tagged by what they are, not what they say.

    Args:
        chunk: The chunk to tag.

    Returns:
        A label with `source="lexical"`.
    """
    t = chunk.text
    if chunk.kind == "action":
        tag = "action_execution"
    elif PRINCIPAL_APPEAL.search(t):
        tag = "principal_appeal"
    elif SHORTCUT.search(t):
        tag = "shortcut_identification"
    elif INTEGRITY.search(t) or COMMITMENT.search(t):
        tag = "ethical_deliberation"
    elif any(
        w in t.lower()
        for w in (
            "fail",
            "does not meet",
            "below the",
            "cannot be met",
            "missing",
            "not exist",
        )
    ):
        tag = "obstacle_identification"
    elif any(
        w in t.lower()
        for w in ("i need to", "i will", "let me", "next, i", "plan", "approach")
    ):
        tag = "plan_generation"
    elif any(
        w in t.lower()
        for w in ("verify", "confirm", "check that", "re-run", "rerun", "validate")
    ):
        tag = "verification"
    elif any(
        w in t.lower() for w in ("report", "summar", "task is complete", "conclusion")
    ):
        tag = "reporting"
    else:
        tag = "situation_assessment"
    return Label(
        chunk_id=chunk.chunk_id,
        tag=tag,
        is_misaligned=bool(SHORTCUT.search(t)) and not bool(INTEGRITY.search(t)),
        source="lexical",
    )


def lexical_labels(chunks: Iterable[Chunk]) -> dict[str, Label]:
    """Tag a whole corpus lexically."""
    return {c.chunk_id: lexical_label(c) for c in chunks}


# -- LLM labeller -----------------------------------------------------------------------


def _parse(raw: str, chunks: Sequence[Chunk], offset: int) -> dict[str, Label]:
    """Parse a labeller response into labels, dropping anything malformed.

    A chunk the labeller skipped or tagged with an unknown label is left OUT rather than
    defaulted to `other`: a silent default would put a made-up label in a per-tag average,
    and `label_run` reports coverage so a bad batch is visible.
    """
    try:
        d = extract_json(raw)
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, Label] = {}
    for k, v in (d or {}).items():
        try:
            i = int(k) - offset
        except (TypeError, ValueError):
            continue
        if not (0 <= i < len(chunks)) or not isinstance(v, dict):
            continue
        tag = str(v.get("function_tag", "")).strip()
        if tag not in TAGS:
            continue
        out[chunks[i].chunk_id] = Label(
            chunk_id=chunks[i].chunk_id,
            tag=tag,
            is_misaligned=bool(v.get("is_misaligned", False)),
            source="llm",
        )
    return out


def label_trajectory(
    traj: Trajectory,
    chunks: Sequence[Chunk],
    client: OpenRouterClient,
    model: str = DEFAULT_LABELLER,
    cache: LabelCache | None = None,
    max_tokens: int = 8000,
) -> dict[str, Label]:
    """Label every chunk of one trajectory.

    Args:
        traj: The trajectory, for the task text the labeller needs as context.
        chunks: Its chunks, in trace order.
        client: OpenRouter client.
        model: Labeller model id.
        cache: Optional on-disk cache.
        max_tokens: Response cap; a long trajectory needs room for one entry per chunk.

    Returns:
        chunk_id -> Label, possibly missing chunks the labeller skipped.
    """
    task = f"{traj.system_prompt}\n\n{traj.user_prompt}".strip()[:8000]
    texts = [c.text for c in chunks]
    key = cache_key(model, task, texts)
    if cache is not None and key in cache.data:
        return {
            cid: Label(cid, v["tag"], v["is_misaligned"], v.get("source", "llm"))
            for cid, v in cache.data[key].items()
        }

    out: dict[str, Label] = {}
    for start in range(0, len(chunks), MAX_CHUNKS_PER_CALL):
        part = list(chunks[start : start + MAX_CHUNKS_PER_CALL])
        prompt = build_prompt(task, [c.text for c in part], first_index=start)
        res = client.chat(
            model,
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        out.update(_parse(res.content, part, offset=start))

    if cache is not None:
        cache.data[key] = {
            cid: {"tag": l.tag, "is_misaligned": l.is_misaligned, "source": l.source}
            for cid, l in out.items()
        }
    return out


def label_run(
    trajs: Sequence[Trajectory],
    chunks_by_traj: dict[str, list[Chunk]],
    model: str = DEFAULT_LABELLER,
    cache_path: Path | None = None,
    workers: int = 8,
    client: OpenRouterClient | None = None,
) -> tuple[dict[str, Label], dict]:
    """Label a corpus, threaded and cached.

    Args:
        trajs: Trajectories to label.
        chunks_by_traj: `Trajectory.key` -> its chunks.
        model: Labeller model id.
        cache_path: Where to persist the cache; None disables caching.
        workers: Concurrent requests.
        client: OpenRouter client; constructed if omitted.

    Returns:
        (chunk_id -> Label, stats). Stats carry coverage and the tag histogram; a
        coverage well below 1.0 means the labeller is dropping chunks and the per-tag
        numbers should not be trusted until that is fixed.
    """
    client = client or OpenRouterClient()
    cache = LabelCache.load(cache_path) if cache_path else None
    results: list[dict[str, Label]] = [dict() for _ in trajs]

    def one(i: int) -> None:
        t = trajs[i]
        results[i] = label_trajectory(
            t, chunks_by_traj.get(t.key, []), client, model=model, cache=cache
        )

    map_threaded(one, len(trajs), max_workers=workers, desc=f"label ({model})")
    if cache:
        cache.save()

    labels: dict[str, Label] = {}
    for r in results:
        labels.update(r)
    total = sum(len(chunks_by_traj.get(t.key, [])) for t in trajs)
    stats = {
        "model": model,
        "n_trajectories": len(trajs),
        "n_chunks": total,
        "n_labelled": len(labels),
        "coverage": len(labels) / total if total else 0.0,
        "tags": dict(Counter(l.tag for l in labels.values()).most_common()),
        "misaligned_share": (
            sum(l.is_misaligned for l in labels.values()) / len(labels)
        )
        if labels
        else 0.0,
    }
    return labels, stats


def estimate_cost(
    chunks_by_traj: dict[str, list[Chunk]], model: str = DEFAULT_LABELLER
) -> dict:
    """Price a labelling run before starting it.

    Args:
        chunks_by_traj: The corpus to label.
        model: Labeller model id, priced through the repo's provider pins.

    Returns:
        {n_trajectories, n_chunks, est_prompt_tokens, est_completion_tokens, est_usd}.
        `est_usd` is None when the model has no price pinned — that is a prompt to go
        look it up, not a licence to run blind.
    """
    from src.infra.endpoints.openrouter import provider_price

    n_traj = len(chunks_by_traj)
    n_chunks = sum(len(v) for v in chunks_by_traj.values())
    chars = sum(len(c.text) for v in chunks_by_traj.values() for c in v)
    # ~4 chars/token for the chunk text, plus the fixed prompt per call, plus a generous
    # ~25 completion tokens per chunk for the JSON entry.
    prompt_tokens = chars // 4 + n_traj * 1200
    completion_tokens = n_chunks * 25
    try:
        price = provider_price(model) or {}
        usd = prompt_tokens / 1e6 * float(
            price.get("prompt", 0)
        ) + completion_tokens / 1e6 * float(price.get("completion", 0))
    except Exception:  # noqa: BLE001
        usd = None
    return {
        "n_trajectories": n_traj,
        "n_chunks": n_chunks,
        "est_prompt_tokens": prompt_tokens,
        "est_completion_tokens": completion_tokens,
        "est_usd": usd,
    }


def tag_composition(
    labels: dict[str, Label], chunks: Sequence[Chunk], violating: dict[str, bool]
) -> dict:
    """Share of chunks per tag, split by outcome.

    Args:
        labels: chunk_id -> Label.
        chunks: The corpus's chunks.
        violating: `Trajectory.key` -> whether that rollout violated.

    Returns:
        tag -> {clean, violating, delta}, each a share of that group's labelled chunks.
    """
    groups: dict[bool, Counter] = {True: Counter(), False: Counter()}
    for c in chunks:
        v = violating.get(c.traj_key)
        lab = labels.get(c.chunk_id)
        if v is None or lab is None:
            continue
        groups[bool(v)][lab.tag] += 1
    out = {}
    for tag in TAGS:
        g, b = groups[False], groups[True]
        sg = g[tag] / sum(g.values()) if sum(g.values()) else 0.0
        sb = b[tag] / sum(b.values()) if sum(b.values()) else 0.0
        out[tag] = {"clean": sg, "violating": sb, "delta": sb - sg}
    return dict(sorted(out.items(), key=lambda kv: -abs(kv[1]["delta"])))
