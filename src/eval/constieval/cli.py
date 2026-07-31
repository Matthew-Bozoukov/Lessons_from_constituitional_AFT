# ABOUTME: Command line entry point. Run with `uv run python -m src.eval.constieval.cli <command>`.
# ABOUTME: Commands: items, run, report, plot, validate, clauses, registry.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fire

from . import plots, report as report_mod
from .config import ConfigError, load_config
from .estimate import estimate as estimate_cost
from .judge_check import check_judge
from .study import run_study
from .control import loader
from .core import registry
from .core.store import ResultsStore
from .items.itemset import ItemSet
from .judges.base import JudgeConfig
from .pipeline.judging import build_judge_client
from .pipeline.run import prepare_itemset, run_eval


def _overrides(pairs: str | None) -> dict[str, Any]:
    """Parse `a.b=1,c=x` into a dotted-override dict.

    Args:
        pairs: Comma-separated key=value string, or None.

    Returns:
        Mapping of dotted key to parsed value (JSON when parseable, else string).

    Raises:
        ValueError: If an entry is not key=value.
    """
    out: dict[str, Any] = {}
    for item in _split_top_level(pairs or ""):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Override {item!r} is not key=value")
        key, value = item.split("=", 1)
        try:
            out[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            out[key.strip()] = value
    return out


def _split_top_level(text: str) -> list[str]:
    """Split on commas that are not inside brackets, braces, or quotes.

    A naive split breaks every JSON-list override - `axes=["a","b"]` becomes two
    fragments, the second of which is not a key=value pair. Since list-valued overrides
    are the common case here (axes, wrappers, difficulties), the split has to track
    nesting.

    Args:
        text: Comma-separated key=value string.

    Returns:
        The top-level fragments.
    """
    parts: list[str] = []
    depth = 0
    in_str = False
    escape = False
    current: list[str] = []
    for ch in text:
        if in_str:
            current.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    parts.append("".join(current))
    return parts


def _load(config: str, set: str | None, smoke: bool, **extra: Any) -> dict[str, Any]:
    """Load a config with CLI overrides applied.

    Args:
        config: Config path or filename in control/configs/.
        set: Comma-separated dotted overrides.
        smoke: Use the offline smoke config when no config was named.
        **extra: Additional overrides applied last, skipping None values.

    Returns:
        The resolved config.
    """
    overrides = _overrides(set)
    overrides.update({k: v for k, v in extra.items() if v is not None})
    return load_config("smoke.yaml" if smoke and config == "base.yaml" else config, overrides)


def _results_frame(results: str):
    """Load results from a file, a directory, or a glob into a frame.

    Args:
        results: Path to a .jsonl/.parquet file, or a directory of run directories.

    Returns:
        A results DataFrame.
    """
    path = Path(results)
    store = ResultsStore.load_dir(path) if path.is_dir() else ResultsStore.load(path)
    return store.to_frame()


class ItemsCLI:
    """Build and inspect the frozen item set."""

    def build(
        self, config: str = "base.yaml", set: str | None = None, smoke: bool = False
    ) -> str:
        """Build and freeze the item set.

        Args:
            config: Config path or control/configs/ filename.
            set: Comma-separated dotted overrides.
            smoke: Build offline with the echo generator.

        Returns:
            A human-readable summary.
        """
        cfg = _load(config, set, smoke)
        itemset = prepare_itemset(cfg, rebuild=True)
        root = Path(cfg["itemset"]["dir"]) / itemset.itemset_id
        counts = "\n".join(f"  {k:28s} {v}" for k, v in itemset.counts().items())
        return (
            f"itemset: {itemset.itemset_id}\n"
            f"dir:     {root}\n"
            f"clauses: {itemset.meta.get('n_clauses', 0)} "
            f"({itemset.meta.get('n_held_out', 0)} held out)\n"
            f"counts:\n{counts}\n\n"
            f"Pin this id into your run config as `itemset.id` so every checkpoint is "
            f"measured against the same items."
        )

    def show(
        self, config: str = "base.yaml", itemset: str | None = None, sample: int = 3
    ) -> str:
        """Show an item set's counts and a few sample prompts.

        Args:
            config: Config path, used only for the item set directory.
            itemset: Item set id; the newest is used when omitted.
            sample: Number of sample items to print.

        Returns:
            A human-readable summary.
        """
        cfg = load_config(config)
        loaded = ItemSet.find(cfg["itemset"]["dir"], itemset)
        lines = [f"itemset: {loaded.itemset_id}", f"clause set: {loaded.clause_set_id}", ""]
        lines += [f"  {k:28s} {v}" for k, v in loaded.counts().items()]
        for item in list(loaded)[:sample]:
            lines += [
                "",
                f"--- {item.family} / {item.clause_id} / {item.difficulty} / {item.condition}",
                item.prompt[:600],
            ]
        return "\n".join(lines)


class CLI:
    """Constitution-internalization eval suite (Tier A)."""

    def __init__(self) -> None:
        """Attach the sub-command groups."""
        self.items = ItemsCLI()

    def run(
        self,
        config: str = "base.yaml",
        recipe: str | None = None,
        step: int | None = None,
        model: str | None = None,
        adapter: str | None = None,
        base_url: str | None = None,
        max_items: int = 0,
        set: str | None = None,
        smoke: bool = False,
        rebuild_items: bool = False,
        report: bool = True,
    ) -> str:
        """Evaluate one checkpoint end to end.

        Args:
            config: Config path or control/configs/ filename.
            recipe: Override run.recipe - the comparison key for every plot.
            step: Override run.checkpoint_step.
            model: Override target.model. A HuggingFace repo id with the `hf` provider.
            adapter: Override target.client.adapter - a LoRA repo id or local path.
            base_url: Override target.base_url.
            max_items: Cap base items for a quick pass; pairing is preserved.
            set: Comma-separated dotted overrides.
            smoke: Offline run with echo providers and a tiny item set.
            rebuild_items: Force an item-set rebuild instead of reusing the frozen one.
            report: Render figures and the markdown mirror for this run.

        Returns:
            A human-readable summary.
        """
        cfg = _load(
            config,
            set,
            smoke,
            **{
                "run.recipe": recipe,
                "run.checkpoint_step": step,
                "target.model": model,
                "target.client.adapter": adapter,
                "target.base_url": base_url,
            },
        )
        result = run_eval(cfg, rebuild_items=rebuild_items, max_items=max_items)
        lines = [result.summary()]
        if report and cfg.get("report", {}).get("enabled", True):
            written = report_mod.build_report(
                result.store,
                result.run_dir / "report",
                make_plots=bool(cfg["report"].get("plots", True)),
                title=f"Constitution internalization — {result.manifest['recipe']} "
                f"(step {result.manifest['checkpoint_step']})",
            )
            lines.append(f"report:  {written['markdown']}")
            for warning in written["warnings"]:
                lines.append(f"WARNING: {warning}")
        return "\n".join(lines)

    def study(
        self,
        arms: str = "base=qwen36_base.yaml,lora=qwen36_lora.yaml",
        name: str = "study",
        out: str = "output/constieval/studies",
        max_items: int = 0,
        cross_check: str = "",
        cross_check_n: int = 120,
        set: str | None = None,
        stop_on_error: bool = False,
    ) -> str:
        """Run every arm end to end and bundle all artifacts into one folder.

        The item set is resolved ONCE and handed to every arm, so the comparison is valid
        by construction rather than by remembering to pin an id in two configs.

        Args:
            arms: Comma-separated `name=config.yaml` specs. The name becomes the recipe on
                every row that arm produces, and its colour in every figure.
            name: Study name; the bundle directory is `<name>_<timestamp>`.
            out: Where bundles are written.
            max_items: Cap base items for a quick pass; parent/child pairing is preserved.
            cross_check: Reference judge model to validate the run's judge against, e.g.
                anthropic/claude-sonnet-4.5. Empty skips it.
            cross_check_n: Rows to re-grade in the cross-check.
            set: Comma-separated dotted overrides applied to every arm.
            stop_on_error: Abort on the first failed arm instead of carrying on. Off by
                default so a served checkpoint that is down cannot discard an arm that
                already succeeded.

        Returns:
            The study summary.
        """
        result = run_study(
            arms,
            name=name,
            out_root=out,
            overrides=_overrides(set),
            max_items=max_items,
            cross_check=cross_check,
            cross_check_n=cross_check_n,
            keep_going=not stop_on_error,
        )
        return result.summary()

    def report(
        self,
        results: str,
        out: str = "output/src/eval/constieval/report",
        recipes: str | None = None,
        plots: bool = True,
    ) -> str:
        """Build the Tier A report from one or more result files.

        Args:
            results: A results file, or a directory of run directories.
            out: Output directory.
            recipes: Comma-separated recipes to restrict to, for a pairwise comparison.
            plots: Render figures as well as tables.

        Returns:
            A human-readable summary.
        """
        df = _results_frame(results)
        selected = [r.strip() for r in recipes.split(",")] if recipes else None
        written = report_mod.build_report(df, out, recipes=selected, make_plots=plots)
        lines = [f"markdown: {written['markdown']}", f"summary:  {written['summary']}"]
        for name, path in written["figures"].items():
            lines.append(f"  {name:28s} {path or '(no data)'}")
        for warning in written["warnings"]:
            lines.append(f"WARNING: {warning}")
        return "\n".join(lines)

    def plot(
        self,
        name: str,
        results: str,
        out: str = "output/src/eval/constieval/report/figures",
        recipes: str | None = None,
        axis: str = "compliance",
    ) -> str:
        """Render one figure standalone, for a write-up.

        Args:
            name: Registered plot name; see `registry`.
            results: A results file or directory.
            out: Output directory.
            recipes: Comma-separated recipes to restrict to.
            axis: Eval axis, for the plots that take one.

        Returns:
            The written path, or a note that there was nothing to draw.
        """
        df = _results_frame(results)
        selected = [r.strip() for r in recipes.split(",")] if recipes else None
        path = plots.render(name, df, out, recipes=selected, axis=axis)
        return path or f"{name}: no data to plot"

    def estimate(
        self, config: str = "base.yaml", arms: int = 2, set: str | None = None
    ) -> str:
        """Project what a run will cost, without spending anything.

        Args:
            config: Config path or control/configs/ filename.
            arms: How many models will be compared against this item set. Item
                generation is charged once; generation and judging are charged per arm.
            set: Comma-separated dotted overrides, so a cheaper variant can be priced
                without editing the config.

        Returns:
            A cost breakdown.
        """
        return estimate_cost(_load(config, set, smoke=False), arms=arms).render()

    def judge_agreement(
        self,
        run_dir: str,
        config: str = "cheap.yaml",
        reference: str = "anthropic/claude-sonnet-4.5",
        n: int = 120,
        set: str | None = None,
    ) -> str:
        """Cross-check a run's judge against a stronger reference judge on a sample.

        The suite ships no gold set, so this is how a cheap judge earns trust: re-grade a
        stratified sample with a strong model and measure how often the pass/fail decision
        changes. Only the reference calls are paid for - the completions are reused.

        Args:
            run_dir: A completed run directory.
            config: Config the run used, for the item set and clause set.
            reference: Reference judge model id.
            n: Approximate number of rows to re-grade.
            set: Comma-separated dotted overrides.

        Returns:
            The agreement report.
        """
        cfg = _load(config, set, smoke=False)
        meta = json.loads((Path(run_dir) / "run_meta.json").read_text())
        itemset = ItemSet.find(cfg["itemset"]["dir"], meta.get("itemset_id"))
        reference_cfg = JudgeConfig.from_config(cfg)
        reference_cfg.model = reference
        report = check_judge(
            run_dir,
            itemset,
            loader.clause_set(str(cfg["clause_set"])),
            build_judge_client(cfg),
            reference_cfg,
            n=n,
            seed=int(cfg.get("seed", 0)),
            max_workers=int(cfg.get("max_workers", 8)),
        )
        out = Path(run_dir) / "judge_agreement.json"
        out.write_text(json.dumps(report.to_dict(), indent=2))
        return f"{report.render()}\n\nwrote {out}"

    def validate(self, config: str = "base.yaml", set: str | None = None) -> str:
        """Validate a config without running anything.

        Args:
            config: Config path or control/configs/ filename.
            set: Comma-separated dotted overrides.

        Returns:
            Confirmation, or the first problem found.
        """
        try:
            cfg = _load(config, set, smoke=False)
        except ConfigError as e:
            return f"INVALID: {e}"
        clauses = loader.clause_set(str(cfg["clause_set"]))
        return (
            f"OK\n"
            f"  clause set: {clauses.spec_id} ({len(clauses)} clauses, "
            f"{len(clauses.held_out)} held out, {len(clauses.fakes)} distractors)\n"
            f"  target:     {cfg['target']['provider']}/{cfg['target']['model']}\n"
            f"  judge:      {cfg['judge']['provider']}/{cfg['judge']['model']}\n"
            f"  axes:       {', '.join(loader.declared_axes())}"
        )

    def clauses(self, config: str = "base.yaml") -> str:
        """List the clause set a config names.

        Args:
            config: Config path or control/configs/ filename.

        Returns:
            One line per clause.
        """
        clauses = loader.clause_set(str(load_config(config)["clause_set"]))
        lines = [f"{clauses.spec_id}  fingerprint={clauses.fingerprint}", ""]
        for clause in clauses:
            flag = " [HELD OUT]" if clause.held_out else ""
            lines.append(
                f"  tier {clause.priority_tier}  {clause.clause_id:38s} {clause.title}{flag}"
            )
        lines += ["", f"distractors: {len(clauses.fakes)}"]
        return "\n".join(lines)

    def registry(self) -> str:
        """List every registered plugin, by extension point."""
        import src.eval.constieval.items  # noqa: F401
        import src.eval.constieval.judges  # noqa: F401
        import src.eval.constieval.plots  # noqa: F401

        return "\n".join(
            f"{kind}:\n" + "\n".join(f"  {n}" for n in registry.names(kind))
            for kind in registry.kinds()
        )

    def axes(self) -> str:
        """List every declared eval axis with its rubric settings."""
        lines = []
        for axis in loader.declared_axes():
            spec = loader.rubric(axis)
            lines.append(
                f"  {axis:24s} scale 0-{spec['scale_max']}  pass@{spec['pass_at']}  "
                f"{spec['direction']:14s} families={','.join(spec['applies_to'])}"
            )
        return "eval axes:\n" + "\n".join(lines)


def main() -> None:
    """Entry point for `python -m src.eval.constieval.cli`."""
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
