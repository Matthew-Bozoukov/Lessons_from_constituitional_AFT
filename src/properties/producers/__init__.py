# ABOUTME: Registry of property producers: name -> ProducerSpec with a lazily imported
# ABOUTME: produce(), resolved only when selected, mirroring src/eval/__init__.py.

"""One entry per way of earning a property.

Same shape as the eval registry, for the same reason: a producer's dependencies (torch for
LESS's gradients, umap for the clusterers) must not be dragged in by `import
src.properties`. A producer module is imported only when its producer is selected.

Every producer is ONE module — its package `__init__.py` — exposing ONE function, always
with the same signature:

    produce(records, cfg, out_dir, target=None) -> list[Property]

so `scripts/properties/discover.py` runs any of them without knowing which. What differs
is declared in the spec rather than discovered by calling:

| producer          | evidence it reads              | needs a Target | ported? |
|-------------------|--------------------------------|----------------|---------|
| feature_discovery | free-text features per record  | no             | reads its run dir |
| trace_clusters    | whole records, embedded        | no             | yes, in full |
| turf              | attributes, both channels      | YES            | reads its run dir |
| less              | gradient influence ranking     | YES            | reads its run dir |

Three of the four are mid-port from `scratch/`, and their modules here are NOT stubs: each
reads the artifacts its scratch module already writes (`clusters.json`,
`trace_result.json`, `scores.jsonl`) and turns them into Property rows. The ARTIFACTS are
the interface, so when the producer code moves in it lands beside its `produce()` and
stops reading from a foreign run directory — nothing downstream of `registry.py` changes.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ProducerSpec:
    """How one producer is invoked and what it needs.

    Attributes:
        package: Package under src.properties.producers; its `__init__` defines produce().
        needs_target: True when the producer explains an OUTCOME and so needs a Target —
            TURF traces a case back, LESS scores influence on a validation set. False for
            producers that describe a corpus.
        needs_gpu: True when the producer cannot run on the driving laptop (LESS computes
            per-example gradients). Embedding does not count: `shared/embed.py` rents its
            own pod.
        ported: False while the producer's code still lives under `scratch/`; `produce()`
            then reads that module's run directory rather than running it.
        scratch_path: Where the un-ported code lives, so the error message when its
            artifacts are missing says what to run.
    """

    package: str
    needs_target: bool = False
    needs_gpu: bool = False
    ported: bool = True
    scratch_path: str = ""


PRODUCERS: dict[str, ProducerSpec] = {
    "trace_clusters": ProducerSpec("trace_clusters"),
    "feature_discovery": ProducerSpec(
        "feature_discovery", ported=False,
        scratch_path="scratch/llm_feature_discovery"),
    "turf": ProducerSpec("turf", needs_target=True, ported=False,
                         scratch_path="scratch/turf"),
    "less": ProducerSpec("less", needs_target=True, needs_gpu=True, ported=False,
                         scratch_path="scratch/less"),
}


def resolve(name: str):
    """Import and return one producer's `produce` (the only place they are imported).

    Args:
        name: A key of PRODUCERS.

    Returns:
        That producer's `produce` function.

    Raises:
        KeyError: If the name is not registered.
    """
    if name not in PRODUCERS:
        raise KeyError(f"unknown producer {name!r}; registered: {sorted(PRODUCERS)}")
    module = f"src.properties.producers.{PRODUCERS[name].package}"
    return importlib.import_module(module).produce
