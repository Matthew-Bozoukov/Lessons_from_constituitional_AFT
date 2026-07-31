# ABOUTME: Loads and renders the prompt packs in control/prompts/. Every string a
# ABOUTME: generation, revision, or rating model ever sees is loaded through here.

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

# StrictUndefined so a template referencing a variable the caller forgot to pass
# fails loudly at render time instead of silently emitting an empty prompt section.
_ENV = Environment(undefined=StrictUndefined, trim_blocks=False, lstrip_blocks=False)


class PromptError(KeyError):
    """Raised when a prompt pack entry is missing or malformed."""


@functools.lru_cache(maxsize=None)
def load_pack(name: str) -> dict[str, Any]:
    """Load one prompt pack YAML from control/prompts/.

    Args:
        name: Pack name without extension, e.g. "generation", "doc_types".

    Returns:
        The parsed mapping.

    Raises:
        PromptError: If the pack file does not exist.
    """
    path = PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise PromptError(
            f"No prompt pack {name!r} at {path}. Available: "
            f"{sorted(p.stem for p in PROMPTS_DIR.glob('*.yaml'))}"
        )
    return yaml.safe_load(path.read_text()) or {}


def entry(pack: str, key: str) -> dict[str, Any]:
    """Fetch one entry from a pack.

    Args:
        pack: Pack name.
        key: Entry key, e.g. a template version or doc_type name.

    Returns:
        The entry mapping.

    Raises:
        PromptError: If the key is absent from the pack.
    """
    data = load_pack(pack)
    if key not in data:
        raise PromptError(
            f"{key!r} not found in control/prompts/{pack}.yaml. "
            f"Available: {sorted(data)}"
        )
    return data[key]


def render(template_text: str, **variables: Any) -> str:
    """Render a Jinja2 template string with strict undefined checking.

    Args:
        template_text: The raw template.
        **variables: Template variables.

    Returns:
        The rendered text, with trailing whitespace stripped.
    """
    return _ENV.from_string(template_text).render(**variables).strip()


def axis_fragment(axis: str, value: str) -> dict[str, str]:
    """Return the prompt fragment for one axis value.

    Args:
        axis: Axis name as it appears in the recipe, e.g. "tools".
        value: The sampled value, e.g. "agentic".

    Returns:
        Dict with keys axis (the human label), value, and text.

    Raises:
        PromptError: If the axis or value is not declared in axes.yaml.
    """
    pack = load_pack("axes")
    if axis not in pack:
        raise PromptError(
            f"Axis {axis!r} is used in the recipe but not declared in "
            f"control/prompts/axes.yaml. Declared axes: {sorted(pack)}"
        )
    spec = pack[axis]
    values = spec.get("values") or {}
    if value not in values:
        raise PromptError(
            f"Axis {axis!r} has no value {value!r} in control/prompts/axes.yaml. "
            f"Declared values: {sorted(values)}"
        )
    return {
        "axis": spec.get("label", axis),
        "value": value,
        "text": str(values[value]).strip(),
    }


def declared_axes() -> list[str]:
    """Return the axis names declared in axes.yaml."""
    return sorted(load_pack("axes"))


def declared_doc_types() -> list[str]:
    """Return the doc_type names declared in doc_types.yaml."""
    return sorted(load_pack("doc_types"))


def doc_type_instructions(doc_type: str) -> str:
    """Return the instruction block for a doc type.

    Args:
        doc_type: Doc type name.

    Returns:
        The instructions text.

    Raises:
        PromptError: If the doc type is not declared.
    """
    return str(entry("doc_types", doc_type)["instructions"]).strip()
