# ABOUTME: Importing this package registers every built-in plugin.
# ABOUTME: A new plugin module must be imported here to be resolvable by name.

from . import (  # noqa: F401
    chunkers,
    exporters,
    filters,
    generators,
    groupers,
    planners,
    revisers,
    strategies,
)

__all__ = [
    "chunkers",
    "exporters",
    "filters",
    "generators",
    "groupers",
    "planners",
    "revisers",
    "strategies",
]
