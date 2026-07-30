# ABOUTME: Command line entry point. Run with `uv run python -m synthdoc.cli <command>`.
# ABOUTME: Commands: run, sweep, validate, chunks, scenarios, inspect, publish, registry.

from __future__ import annotations

import json
from typing import Any

import fire

from . import config as config_mod
from .core.specs import available_specs, load_spec
from .pipeline import build_scenarios, run_pipeline
from .plugins.chunkers import build_chunker
from .snapshots import load_snapshot
from .sweep import load_sweep, run_sweep


def _overrides(pairs: str | None) -> dict[str, Any]:
    """Parse `a.b=1,c=x` into a dotted-override dict.

    Args:
        pairs: Comma-separated key=value string, or None.

    Returns:
        Mapping of dotted key to parsed value (JSON when parseable, else string).
    """
    out: dict[str, Any] = {}
    for item in (pairs or "").split(","):
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


class CLI:
    """Synthetic document generation pipeline."""

    def run(
        self,
        config: str = "base.yaml",
        n: int | None = None,
        run_id: str | None = None,
        set: str | None = None,
        smoke: bool = False,
    ) -> str:
        """Run the pipeline end to end.

        Args:
            config: Config path, or a filename in synthdoc/control/configs/.
            n: Override recipe.n.
            run_id: Override the generated run id.
            set: Comma-separated dotted overrides, e.g. "generation.model=gen-b".
            smoke: Offline 8-document run using the echo provider and no pushes.

        Returns:
            A human-readable summary.
        """
        overrides = _overrides(set)
        if smoke:
            overrides.setdefault("llm.provider", "echo")
            overrides.setdefault("snapshots.backend", "local")
            # A smoke run stays on disk so it can be inspected.
            overrides.setdefault("snapshots.cleanup_local", False)
            n = n or 8
        cfg = config_mod.load_config(config, overrides)
        result = run_pipeline(cfg, n=n, run_id=run_id)
        final = result.stages[-1]
        counts = result.counts.get(final, {})
        return (
            f"run_id: {result.run_id}\n"
            f"dir:    {result.run_dir}\n"
            f"stages: {' -> '.join(result.stages)}\n"
            f"kept:   {counts.get('n_keep', 0)}/{counts.get('n', 0)}\n"
            f"cost:   ${result.manifest.get('cost_usd_total', 0):.2f}\n"
            f"export: {result.exports.get('main', '')}\n"
            f"report: {result.manifest.get('report', {}).get('coverage_report', '')}"
        )

    def sweep(
        self,
        config: str,
        n: int | None = None,
        dry_run: bool = False,
    ) -> str:
        """Run a single-axis sweep.

        Args:
            config: Sweep config path, or a filename in control/configs/sweeps/.
            n: Override recipe.n for every arm.
            dry_run: Check pairing and write the report without generating.

        Returns:
            A human-readable summary.
        """
        result = run_sweep(load_sweep(config), n=n, dry_run=dry_run)
        return (
            f"sweep:  {result.sweep_id}\n"
            f"axis:   {result.axis}\n"
            f"arms:   {', '.join(a.name for a in result.arms)}\n"
            f"paired: {result.pairing['paired']} "
            f"({result.pairing['shared_fraction']:.1%} shared scenarios)\n"
            f"report: {result.report_path}"
        )

    def validate(self, config: str = "base.yaml") -> str:
        """Validate a config without running anything.

        Args:
            config: Config path or control/configs/ filename.

        Returns:
            Confirmation with the resolved stage sequence.
        """
        cfg = config_mod.load_config(config)
        n_rev = len(cfg.get("revision") or [])
        stages = ["stage_00_generated"] + [
            f"stage_{i:02d}_revised" for i in range(1, n_rev + 1)
        ] + [f"stage_{n_rev + 1:02d}_filtered"]
        return f"OK. {config}\nstages: {' -> '.join(stages)}"

    def chunks(self, config: str = "base.yaml", limit: int = 10, full: bool = False) -> str:
        """Show how a spec chunks, without running generation.

        Args:
            config: Config path or control/configs/ filename.
            limit: Chunks to print.
            full: Print full chunk text rather than the first line.

        Returns:
            A listing of chunks.
        """
        cfg = config_mod.load_config(config)
        spec_cfg = cfg["spec"]
        spec = load_spec(spec_cfg["id"], spec_cfg.get("path"))
        chunks = build_chunker(spec_cfg.get("chunker") or {}).chunk(spec)
        lines = [f"{spec.spec_id}: {len(chunks)} chunks from {spec.path}", ""]
        for c in chunks[:limit]:
            body = c.text if full else c.text.splitlines()[0][:110]
            lines.append(f"[{c.chunk_id}] ({c.meta.get('register')}) {body}")
        if len(chunks) > limit:
            lines.append(f"... {len(chunks) - limit} more")
        return "\n".join(lines)

    def scenarios(self, config: str = "base.yaml", n: int = 10) -> str:
        """Sample scenarios without generating documents.

        Args:
            config: Config path or control/configs/ filename.
            n: Scenarios to sample.

        Returns:
            One line per sampled experimental condition.
        """
        cfg = config_mod.load_config(config)
        scenarios, diagnostics = build_scenarios(cfg, n)
        lines = [json.dumps(diagnostics, indent=2), ""]
        for s in scenarios:
            axes = " ".join(f"{k}={v}" for k, v in sorted(s.axes.items()))
            lines.append(
                f"{s.scenario_hash}  {s.doc_type:<22} {s.grouping_strategy:<9} "
                f"k={len(s.chunks)}  {axes}"
            )
        return "\n".join(lines)

    def inspect(self, snapshot: str, index: int = 0, doc_id: str | None = None) -> str:
        """Print one document from a stage snapshot.

        Args:
            snapshot: Path to a stage .jsonl or .parquet in a run directory.
            index: Row index to show when doc_id is not given.
            doc_id: Show this document instead of index.

        Returns:
            The rendered document with its lineage.
        """
        from .core.parsing import render_document

        docs = load_snapshot(snapshot)
        doc = next((d for d in docs if d.doc_id == doc_id), None) if doc_id else docs[index]
        if doc is None:
            return f"doc_id {doc_id} not in {snapshot}"
        head = (
            f"doc_id: {doc.doc_id}  scenario: {doc.scenario.scenario_hash}\n"
            f"stage:  {doc.stage_name}  verdict: {doc.filter_verdict} "
            f"{('(' + doc.dropped_by + ')') if doc.dropped_by else ''}\n"
            f"type:   {doc.scenario.doc_type}  grouping: {doc.scenario.grouping_strategy}\n"
            f"chunks: {', '.join(doc.scenario.chunk_ids)}\n"
            f"axes:   {json.dumps(doc.scenario.axes)}\n"
            f"scores: {json.dumps(doc.filter_scores)}\n"
            f"lineage: {' -> '.join(f'{r.kind}@{r.model}' for r in doc.lineage)}\n"
            + ("error:  " + doc.error + "\n" if doc.error else "")
            + "-" * 72
        )
        return head + "\n" + render_document(doc.turns)

    def corpora(
        self,
        org: str | None = None,
        output_dir: str = "output/synthdoc",
        local: bool = False,
    ) -> str:
        """List saved corpora, from HuggingFace by default.

        HuggingFace is the durable home for corpora; the local listing only shows
        runs that still have a working directory on this machine.

        Args:
            org: HF namespace to list. Defaults to the org in control/configs/base.yaml.
            output_dir: Local run directory, used with --local.
            local: List the local catalogue instead of the Hub.

        Returns:
            An aligned table of saved corpora.
        """
        from .corpora import format_index, list_hf, load_index

        if local:
            return format_index(load_index(output_dir))
        if org is None:
            try:
                org = (config_mod.load_config("base.yaml").get("snapshots") or {}).get("org")
            except Exception:
                org = None
        if not org:
            return (
                "No HF org configured. Pass --org <namespace>, set snapshots.org in "
                "base.yaml, or use --local to list this machine's run directories."
            )
        return format_index(list_hf(org))

    def compare(self, a: str, b: str, out: str | None = None) -> str:
        """Compare two saved corpora and report the effect size.

        Reports paired per-scenario deltas where the corpora share scenarios, which
        removes the variance from which scenarios were sampled.

        Args:
            a: Baseline corpus: a local run directory, `hf://org/repo`, or `org/repo`.
            b: Corpus to compare against it, in the same forms.
            out: Optional path to write the markdown report to.

        Returns:
            The comparison as markdown.
        """
        from pathlib import Path as _Path

        from .corpora import compare as compare_corpora
        from .corpora import format_comparison

        result = compare_corpora(a, b, _Path(a).name, _Path(b).name)
        text = format_comparison(result)
        if out:
            _Path(out).write_text(text)
        return text

    def publish(
        self,
        export: str,
        repo: str,
        card: str,
        kind: str = "petri",
        private: bool = False,
        dry_run: bool = True,
        chunk_size: int = 50,
    ) -> str:
        """Publish a dataset to HuggingFace in the shape the visualizer reads.

        Writes the AGENTS.md dataset card, the small `manifest.json` the static
        site bakes in, and the per-item shards the browser fetches lazily.

        Defaults to a dry run: uploading is a side effect on a shared namespace,
        so it must be asked for explicitly with `--dry_run=False`.

        Args:
            export: Petri export bundle directory, or a dialogue `.jsonl` file.
            repo: Target dataset repo, `org/<YYYY-MM-DD>-<short-description>`.
            card: JSON or YAML file holding the required dataset-card fields.
            kind: `petri` for an audit export, `dialogues` for a dialogue JSONL.
            private: Create the repo private. Public repos need no reader token.
            dry_run: Stage and report without contacting the Hub.
            chunk_size: Records per lazily-fetched chunk (`dialogues` only).

        Returns:
            A human-readable result line.
        """
        from .publish import card_from_file, publish_dialogue_dataset, publish_petri_run

        fields = card_from_file(card)
        if kind == "petri":
            return publish_petri_run(
                export, repo, fields, private=private, dry_run=dry_run
            )
        if kind == "dialogues":
            return publish_dialogue_dataset(
                export,
                repo,
                fields,
                chunk_size=chunk_size,
                private=private,
                dry_run=dry_run,
            )
        raise ValueError(f"Unknown kind {kind!r}; expected 'petri' or 'dialogues'")

    def axes(self) -> str:
        """List every ablatable axis with the exact sweep key to vary it."""
        from .ablations import catalog_text

        return catalog_text()

    def registry(self) -> str:
        """List every registered plugin and every declared prompt entry."""
        from .control import loader
        from .core import registry as reg

        import synthdoc.plugins  # noqa: F401

        lines = ["# Registered plugins", ""]
        for kind in reg.kinds():
            lines.append(f"- {kind}: {', '.join(reg.names(kind))}")
        lines += [
            "",
            "# Declared in control/prompts/",
            "",
            f"- doc_types: {', '.join(loader.declared_doc_types())}",
            f"- axes: {', '.join(loader.declared_axes())}",
            f"- generation templates: {', '.join(sorted(loader.load_pack('generation')))}",
            f"- revision kinds: {', '.join(sorted(loader.load_pack('revision')))}",
            f"- rubrics: {', '.join(sorted(loader.load_pack('rubrics')))}",
            "",
            f"# Specs in control/specs/: {', '.join(available_specs()) or '(none)'}",
        ]
        return "\n".join(lines)

    def configs(self) -> str:
        """List the run and sweep configs in the control area."""
        base = sorted(p.name for p in config_mod.CONTROL_CONFIGS.glob("*.yaml"))
        sweeps = sorted(p.name for p in (config_mod.CONTROL_CONFIGS / "sweeps").glob("*.yaml"))
        return (
            f"control/configs/:        {', '.join(base)}\n"
            f"control/configs/sweeps/: {', '.join(sweeps)}"
        )


def main() -> None:
    """Fire entry point."""
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
