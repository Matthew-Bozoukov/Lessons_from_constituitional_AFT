# ABOUTME: Registry of property-extraction sources: each module turns one kind of input —
# ABOUTME: training corpus, ODCV rollouts, agentic rollouts — into the shared Record type.

"""One adapter per input type, keyed by name.

A config names a source and passes it kwargs; the adapter does the rest:

    source:
      name: mixture_rows
      repo: LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train
      only_source: synthdoc_difficult_advice

Two properties of an adapter are declared rather than discovered, because both change what
a caller is allowed to do with it:

    has_outcomes  whether every Record carries the source's own judged outcome. Producers
                  that split good traces from bad refuse a source without one.
    ablatable     whether this is a TRAINING CORPUS an ablation may rewrite. Rollouts are
                  evidence about a trained model, not training data — ablating them would
                  produce a dataset nothing was ever trained on, so `ablation/base.py`
                  checks this before it edits anything.

`targets.py` is not in the registry: a Target is not an input type but the behaviour a
producer traces back from, and it is built out of the records a source already yielded.
"""

from __future__ import annotations

from src.properties.sources.base import (  # noqa: F401  (re-exported contract)
    CHANNELS,
    Record,
    SourceAdapter,
    Target,
    first_turns,
)
from src.properties.sources import (
    agentic_rollouts,
    mixture_rows,
    odcv_rollouts,
)

SOURCES: dict[str, SourceAdapter] = {
    adapter.name: adapter
    for adapter in (
        mixture_rows.ADAPTER,
        odcv_rollouts.ADAPTER,
        agentic_rollouts.ADAPTER,
    )
}


def load_source(spec: dict) -> tuple[list[Record], SourceAdapter]:
    """Load the source a config's `source:` block names.

    Args:
        spec: The block, e.g. {"name": "mixture_rows", "repo": ..., "limit": 100}.
            Every key but `name` is passed to the adapter's `load`.

    Returns:
        (records, the adapter), so a caller can check `ablatable`/`has_outcomes` without
        looking the name up again.

    Raises:
        KeyError: If `name` is missing or names no registered source.
    """
    kwargs = dict(spec)
    name = kwargs.pop("name")
    if name not in SOURCES:
        raise KeyError(f"unknown source {name!r}; registered: {sorted(SOURCES)}")
    adapter = SOURCES[name]
    return adapter.load(**kwargs), adapter
