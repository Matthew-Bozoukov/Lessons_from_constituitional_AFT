# ABOUTME: Plot registry and the one-call renderer. Every figure is regenerable from the
# ABOUTME: results store alone, for any subset of recipes, with no run state involved.

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..core.registry import names, resolve
from . import tier_a  # noqa: F401  - import side effect is the registration
from .theme import CATEGORICAL, PaletteError, apply_style, recipe_colors

# Render order. Also the order figures appear in the report.
TIER_A_PLOTS: tuple[str, ...] = (
    "clause_heatmap",
    "retrieval_vs_application",
    "compliance_vs_tension",
    "ood_decay",
    "robustness_delta",
    "side_effect_panel",
    "checkpoint_trajectory",
)


def render(name: str, df, out_dir: Path | str, **kwargs: Any) -> str:
    """Render one registered plot.

    Args:
        name: Registered plot name.
        df: A results frame.
        out_dir: Directory to write into.
        **kwargs: Passed to the plot function (recipes, axis, ...).

    Returns:
        The written path, or "" if the plot had nothing to draw.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return resolve("plot", name)(df, out / f"{name}.png", **kwargs)


def render_all(
    df,
    out_dir: Path | str,
    plots: Sequence[str] = TIER_A_PLOTS,
    recipes: Sequence[str] | None = None,
    **kwargs: Any,
) -> dict[str, str]:
    """Render every Tier A figure.

    A plot that raises is recorded against its name rather than aborting the rest: a
    report with six figures and one stated failure is more useful than no report.

    Args:
        df: A results frame.
        out_dir: Directory to write into.
        plots: Plot names to render.
        recipes: Restrict every figure to these recipes.
        **kwargs: Passed through to each plot.

    Returns:
        plot name -> written path. Plots with nothing to draw map to "", and plots that
        failed map to a string beginning with "ERROR:".
    """
    written: dict[str, str] = {}
    for name in plots:
        try:
            written[name] = render(name, df, out_dir, recipes=recipes, **kwargs)
        except PaletteError as e:
            written[name] = f"ERROR: {e}"
        except Exception as e:  # noqa: BLE001 - one bad figure must not sink the report
            written[name] = f"ERROR: {type(e).__name__}: {e}"
    return written


def registered() -> list[str]:
    """Return every registered plot name."""
    return names("plot")


__all__ = [
    "CATEGORICAL",
    "PaletteError",
    "TIER_A_PLOTS",
    "apply_style",
    "recipe_colors",
    "registered",
    "render",
    "render_all",
]
