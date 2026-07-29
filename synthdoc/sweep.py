# ABOUTME: Sweep runner: base config + ONE varied axis + a list of arms.
# ABOUTME: Multi-axis sweeps are rejected at validation; arms share seeds so comparisons are paired.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from . import config as config_mod
from .config import ConfigError
from .pipeline import RunResult, build_scenarios, run_pipeline


@dataclass
class Arm:
    """One arm of a sweep.

    Attributes:
        name: Short arm name, used in the run id and the report.
        value: The value assigned to the swept axis.
    """

    name: str
    value: Any


@dataclass
class SweepResult:
    """Outcome of a sweep.

    Attributes:
        sweep_id: Sweep identifier.
        sweep_dir: Directory holding the sweep report and comparison table.
        axis: The dotted config key that was varied.
        arms: The arms that were run.
        runs: Arm name -> RunResult.
        pairing: Scenario-pairing diagnostics across arms.
        report_path: Path to sweep_report.md.
    """

    sweep_id: str
    sweep_dir: Path
    axis: str
    arms: list[Arm]
    runs: dict[str, RunResult] = field(default_factory=dict)
    pairing: dict[str, Any] = field(default_factory=dict)
    report_path: str = ""


def load_sweep(path: str | Path) -> dict[str, Any]:
    """Load and validate a sweep config.

    Args:
        path: Sweep YAML path, or a bare filename in control/configs/sweeps/.

    Returns:
        The sweep config as a plain dict.

    Raises:
        ConfigError: If the sweep is malformed or varies more than one axis.
    """
    p = Path(path)
    if not p.exists():
        candidate = config_mod.CONTROL_CONFIGS / "sweeps" / p.name
        if not candidate.exists():
            available = sorted(
                x.name for x in (config_mod.CONTROL_CONFIGS / "sweeps").glob("*.yaml")
            )
            raise ConfigError(f"Sweep config not found: {path}. Available: {available}")
        p = candidate

    raw = OmegaConf.to_container(OmegaConf.load(p), resolve=True)
    if not isinstance(raw, dict):
        raise ConfigError(f"Sweep config {p} is not a mapping")

    for required in ("base", "axis", "arms"):
        if required not in raw:
            raise ConfigError(f"sweep.{required} is required in {p}")

    axis = raw["axis"]
    if isinstance(axis, (list, tuple)):
        raise ConfigError(
            f"sweep.axis must be a single dotted key, got {list(axis)}. "
            "Multi-axis sweeps are rejected: with more than one axis moving, an arm "
            "difference cannot be attributed, which is the entire point of a sweep. "
            "Run one sweep per axis."
        )
    if not isinstance(axis, str) or not axis:
        raise ConfigError(f"sweep.axis must be a non-empty dotted key, got {axis!r}")

    base_overrides = raw.get("base_overrides") or {}
    if not isinstance(base_overrides, dict):
        raise ConfigError("sweep.base_overrides must be a mapping of dotted key -> value")
    if axis in base_overrides:
        raise ConfigError(
            f"sweep.base_overrides sets {axis!r}, which is the swept axis. "
            "base_overrides is applied identically to every arm and must not touch it."
        )

    arms = raw["arms"]
    if not isinstance(arms, list) or len(arms) < 2:
        raise ConfigError("sweep.arms must be a list of at least two arms")

    seen: set[str] = set()
    for i, arm in enumerate(arms):
        if not isinstance(arm, dict) or "name" not in arm or "value" not in arm:
            raise ConfigError(f"sweep.arms[{i}] needs both `name` and `value`")
        extra = set(arm) - {"name", "value"}
        if extra:
            raise ConfigError(
                f"sweep.arms[{i}] has extra keys {sorted(extra)}. An arm may set only "
                "the swept axis; anything else would make the sweep multi-axis."
            )
        if arm["name"] in seen:
            raise ConfigError(f"Duplicate arm name {arm['name']!r}")
        seen.add(arm["name"])
    return raw


def check_pairing(base_cfg: dict[str, Any], axis: str, arms: list[Arm],
                  n: int | None = None) -> dict[str, Any]:
    """Sample each arm's scenarios and measure how well the arms are paired.

    Run before any generation so an arm that silently reshuffles the scenario set is
    caught for free rather than after paying for it. Axes downstream of sampling
    (generator model, revision, filters) pair perfectly; axes inside the recipe
    change the sample by design, and the shared fraction quantifies by how much.

    Args:
        base_cfg: The resolved base config.
        axis: Dotted key being varied.
        arms: The arms.
        n: Optional recipe.n override.

    Returns:
        Diagnostics including the per-arm scenario counts and the shared fraction.
    """
    hashes: dict[str, set[str]] = {}
    for arm in arms:
        cfg = config_mod.load_config_dict(base_cfg, {axis: arm.value})
        scenarios, _ = build_scenarios(cfg, n)
        hashes[arm.name] = {s.scenario_hash for s in scenarios}

    sets = list(hashes.values())
    shared = set.intersection(*sets) if sets else set()
    union = set.union(*sets) if sets else set()
    identical = bool(union) and len(shared) == len(union)

    largest = max(sets, key=len) if sets else set()
    nested = bool(sets) and not identical and all(s <= largest for s in sets)
    counts = {name: len(h) for name, h in hashes.items()}
    index_paired = len(set(counts.values())) == 1

    if identical:
        note = (
            "paired: every arm sampled an identical scenario set, so arm differences "
            "are attributable to the axis alone. Join arms on scenario_hash."
        )
    elif nested:
        note = (
            "nested: smaller arms are subsets of the largest, which is what a scaling "
            "sweep should look like. The shared subset is exactly paired - join on "
            "scenario_hash and compare within it."
        )
    elif index_paired:
        note = (
            "index-paired: this axis feeds the sampler, so no scenario_hash can match. "
            "Example i still differs from its counterpart only in the swept axis, so "
            "join arms on sample_index for a genuine paired comparison. `cli compare` "
            "does this automatically."
        )
    else:
        note = (
            "unpaired: the arms differ in both the scenarios sampled and their count. "
            "Only marginal comparisons are meaningful. Consider equalising recipe.n."
        )

    return {
        "axis": axis,
        "per_arm": counts,
        "n_shared_scenarios": len(shared),
        "shared_fraction": round(len(shared) / max(1, len(union)), 4),
        "paired": identical,
        "nested": nested,
        "index_paired": index_paired,
        "join_key": "scenario_hash" if (identical or nested) else (
            "sample_index" if index_paired else None
        ),
        "note": note,
    }


def run_sweep(
    sweep_cfg: dict[str, Any],
    n: int | None = None,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
) -> SweepResult:
    """Run every arm of a sweep and write the comparison report.

    Args:
        sweep_cfg: Loaded sweep config.
        n: Override for recipe.n across all arms.
        output_dir: Sweep output directory.
        dry_run: Only check pairing and write the report; run no generation.

    Returns:
        A SweepResult.
    """
    axis = sweep_cfg["axis"]
    arms = [Arm(name=str(a["name"]), value=a["value"]) for a in sweep_cfg["arms"]]
    # base_overrides is applied identically to every arm, so it holds a confound
    # fixed rather than adding a second varied axis.
    base_cfg = config_mod.load_config(sweep_cfg["base"], sweep_cfg.get("base_overrides"))
    n = n if n is not None else sweep_cfg.get("n")

    sweep_id = sweep_cfg.get("id") or f"{axis.replace('.', '_')}_{config_mod.timestamp()}"
    sweep_dir = Path(output_dir or "output/synthdoc_sweeps") / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    pairing = check_pairing(base_cfg, axis, arms, n)
    result = SweepResult(
        sweep_id=sweep_id, sweep_dir=sweep_dir, axis=axis, arms=arms, pairing=pairing
    )

    if not dry_run:
        for arm in arms:
            cfg = config_mod.load_config_dict(base_cfg, {axis: arm.value})
            result.runs[arm.name] = run_pipeline(
                cfg, n=n, run_id=f"{sweep_id}__{arm.name}"
            )

    result.report_path = write_sweep_report(result, sweep_cfg)
    (sweep_dir / "sweep_manifest.json").write_text(
        json.dumps(
            {
                "sweep_id": sweep_id,
                "axis": axis,
                "arms": [{"name": a.name, "value": a.value} for a in arms],
                "base": sweep_cfg["base"],
                "n": n,
                "git_sha": config_mod.git_sha(),
                "timestamp_utc": config_mod.timestamp(),
                "pairing": pairing,
                "runs": {k: str(v.run_dir) for k, v in result.runs.items()},
            },
            indent=2,
            default=str,
        )
    )
    return result


def write_sweep_report(result: SweepResult, sweep_cfg: dict[str, Any]) -> str:
    """Write sweep_report.md comparing the arms.

    Args:
        result: The sweep result.
        sweep_cfg: The sweep config.

    Returns:
        Path to the written report.
    """
    lines = [
        f"# Sweep `{result.sweep_id}`",
        "",
        f"- varied axis: `{result.axis}`",
        f"- base config: `{sweep_cfg['base']}`"
        + (
            f" (held fixed across arms: `{json.dumps(sweep_cfg['base_overrides'])}`)"
            if sweep_cfg.get("base_overrides")
            else ""
        ),
        "- arms: " + ", ".join(f"`{a.name}` = `{a.value}`" for a in result.arms),
        "",
        "## Pairing",
        "",
        f"- shared scenarios: **{result.pairing['n_shared_scenarios']}** "
        f"({result.pairing['shared_fraction']:.1%} of the union)",
        f"- fully paired: **{result.pairing['paired']}**"
        + (", nested" if result.pairing.get("nested") else "")
        + (", index-paired" if result.pairing.get("index_paired") else ""),
        f"- join arms on: **`{result.pairing.get('join_key') or 'n/a (marginals only)'}`**",
        "",
        f"> {result.pairing['note']}",
        "",
    ]

    if result.runs:
        lines += [
            "## Arm comparison",
            "",
            "| arm | value | generated | kept | keep rate | mean words | rater mean | cost usd |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for arm in result.arms:
            run = result.runs.get(arm.name)
            if run is None:
                continue
            final = run.stages[-1]
            counts = run.counts.get(final, {})
            n_total = counts.get("n", 0)
            n_keep = counts.get("n_keep", 0)
            rater = [
                d.filter_scores.get("autorater_overall")
                for d in run.corpus
                if d.filter_scores.get("autorater_overall") is not None
            ]
            rater_mean = f"{sum(rater) / len(rater):.2f}" if rater else "-"
            lines.append(
                f"| `{arm.name}` | `{arm.value}` | {n_total} | {n_keep} | "
                f"{n_keep / max(1, n_total):.1%} | {counts.get('mean_words', 0)} | "
                f"{rater_mean} | {run.manifest.get('cost_usd_total', 0):.2f} |"
            )
        baseline = result.arms[0]
        base_run = result.runs.get(baseline.name)
        if base_run is not None and len(result.arms) > 1:
            lines += [
                "",
                f"## Effect size vs `{baseline.name}`",
                "",
                "Paired per-scenario deltas where the arms share scenarios; marginal",
                "differences otherwise. This is the number the ablation exists to produce.",
                "",
            ]
            for arm in result.arms[1:]:
                run = result.runs.get(arm.name)
                if run is None:
                    continue
                try:
                    from .corpora import compare

                    cmp = compare(base_run.run_dir, run.run_dir, baseline.name, arm.name)
                except (FileNotFoundError, ImportError) as e:
                    lines += [f"- `{arm.name}`: comparison unavailable ({e})", ""]
                    continue
                deltas = cmp.get("paired_deltas") or {}
                if deltas:
                    parts = ", ".join(
                        f"{m} {s['mean_delta']:+g}" for m, s in sorted(deltas.items())
                    )
                    lines.append(
                        f"- `{arm.name}` (paired, n={cmp['n_shared_scenarios']}): {parts}"
                    )
                else:
                    d = cmp.get("delta", {})
                    parts = ", ".join(
                        f"{m} {d[m]:+g}" for m in sorted(d) if m.startswith(("mean_", "keep_"))
                    )
                    lines.append(f"- `{arm.name}` (marginal, no shared scenarios): {parts}")
            lines.append("")
            lines.append(
                "Full breakdown: `uv run python -m synthdoc.cli compare "
                f"--a <{baseline.name}_dir> --b <arm_dir>`"
            )

        lines += [
            "",
            "## Joining arms",
            "",
            "- Across arms: `scenario_hash` when the arms sample identical conditions,",
            "  otherwise `sample_index` (example i differs only in the swept axis).",
            "- Across stages within an arm: `doc_id` (constant across stages, but",
            "  arm-specific because it hashes the run id).",
            "",
            "```python",
            "import pandas as pd",
            "a = pd.read_parquet('<arm_a_dir>/stage_NN_filtered.parquet')",
            "b = pd.read_parquet('<arm_b_dir>/stage_NN_filtered.parquet')",
            f"paired = a.merge(b, on='{result.pairing.get('join_key') or 'scenario_hash'}',"
            " suffixes=('_a', '_b'))",
            "```",
            "",
            "## Run directories",
            "",
        ]
        lines += [f"- `{name}`: `{run.run_dir}`" for name, run in result.runs.items()]
    else:
        lines += ["## Arm comparison", "", "_Dry run: pairing checked, no generation performed._", ""]

    path = result.sweep_dir / "sweep_report.md"
    path.write_text("\n".join(lines) + "\n")
    return str(path)
