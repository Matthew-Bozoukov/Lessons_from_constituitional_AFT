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

| producer          | evidence it reads              | needs a Target | implemented |
|-------------------|--------------------------------|----------------|-------------|
| trace_clusters    | whole records, embedded        | no             | yes         |
| feature_discovery | free-text features per record  | no             | not yet     |
| turf              | attributes, both channels      | YES            | not yet     |
| less              | gradient influence ranking     | YES            | not yet     |

Three of the four are empty packages: their code still lives under `scratch/` and lands in
their `__init__.py` when it moves. They stay in the registry because the registry is the
list of producers this module is FOR, and a name that is planned but missing should fail
by saying so — `resolve()` raises with the path to the code — rather than by being absent
and reading as a typo.
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
        scratch_path: Where the code lives while the package here is still a placeholder,
            so the error `resolve()` raises names what to port.
    """

    package: str
    needs_target: bool = False
    needs_gpu: bool = False
    scratch_path: str = ""

    @property
    def implemented(self) -> bool:
        """Whether this producer's code is in `src/` yet.

        Returns:
            True when the package holds the producer, False while it is a placeholder.
        """
        return not self.scratch_path


PRODUCERS: dict[str, ProducerSpec] = {
    "trace_clusters": ProducerSpec("trace_clusters"),
    "feature_discovery": ProducerSpec(
        "feature_discovery", scratch_path="scratch/feature_discovery"),
    "turf": ProducerSpec("turf", needs_target=True, scratch_path="scratch/turf"),
    "less": ProducerSpec("less", needs_target=True, needs_gpu=True,
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
        NotImplementedError: If the producer is still a placeholder package.
    """
    if name not in PRODUCERS:
        raise KeyError(f"unknown producer {name!r}; registered: {sorted(PRODUCERS)}")
    spec = PRODUCERS[name]
    if not spec.implemented:
        raise NotImplementedError(
            f"producer {name!r} is a placeholder: its code is still in "
            f"{spec.scratch_path}. Port it into "
            f"src/properties/producers/{spec.package}/__init__.py as a "
            "produce(records, cfg, out_dir, target=None) -> list[Property], or drop it "
            "from this config's `producers:` block.")
    module = f"src.properties.producers.{spec.package}"
    return importlib.import_module(module).produce
