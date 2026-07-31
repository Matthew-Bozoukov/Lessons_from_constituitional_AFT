# ABOUTME: Runs a whole multi-arm study end to end and bundles every artifact into one directory.
# ABOUTME: One command, one self-contained output folder you can archive, share, or hand to a write-up.

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from . import report as report_mod
from .config import ConfigError, git_sha, load_config, timestamp
from .control import loader
from .core.store import ResultsStore
from .estimate import estimate
from .judges.base import JudgeConfig
from .pipeline.judging import build_judge_client
from .pipeline.run import prepare_itemset, run_eval


class StudyError(ValueError):
    """Raised when a study is misconfigured in a way that would waste a run."""


@dataclass
class ArmResult:
    """The outcome of one arm.

    Attributes:
        name: Arm name; becomes the recipe on every row it produces.
        config: Config file the arm used.
        status: "ok", "failed", or "skipped".
        run_id: Run id, when it ran.
        run_dir: Where the run wrote its artifacts.
        spend_usd: What the arm cost.
        warnings: Health warnings (truncation, error rate) from the run.
        error: Failure message, when status is "failed".
        seconds: Wall-clock time.
    """

    name: str
    config: str
    status: str = "ok"
    run_id: str = ""
    run_dir: str = ""
    spend_usd: float = 0.0
    warnings: list[str] = field(default_factory=list)
    error: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "name": self.name,
            "config": self.config,
            "status": self.status,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "spend_usd": round(self.spend_usd, 4),
            "warnings": self.warnings,
            "error": self.error,
            "seconds": round(self.seconds, 1),
        }


@dataclass
class StudyResult:
    """Everything one study produced.

    Attributes:
        study_id: Study name plus timestamp; also the bundle directory name.
        bundle: The self-contained output directory.
        itemset_id: The frozen item set every arm shared.
        arms: One entry per arm, successful or not.
        report: Paths written by the report step.
        judge_agreement: The cross-check result, when it was run.
        manifest: Everything above, as written to study.json.
    """

    study_id: str
    bundle: Path
    itemset_id: str
    arms: list[ArmResult]
    report: dict[str, Any] = field(default_factory=dict)
    judge_agreement: dict[str, Any] | None = None
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def ok_arms(self) -> list[ArmResult]:
        """Arms that produced results."""
        return [a for a in self.arms if a.status == "ok"]

    @property
    def total_spend(self) -> float:
        """Total USD across every arm."""
        return round(sum(a.spend_usd for a in self.arms), 2)

    def summary(self) -> str:
        """Return the human-readable end-of-study summary."""
        lines = [
            "",
            "=" * 74,
            f"  STUDY  {self.study_id}",
            "=" * 74,
            f"  item set   {self.itemset_id}",
            f"  bundle     {self.bundle}",
            f"  spend      ${self.total_spend:.2f}",
            "",
            "  arms:",
        ]
        for arm in self.arms:
            mark = {"ok": "ok  ", "failed": "FAIL", "skipped": "skip"}.get(arm.status, "?   ")
            lines.append(f"    [{mark}] {arm.name:22s} ${arm.spend_usd:6.2f}  {arm.seconds:6.1f}s")
            if arm.error:
                lines.append(f"           {arm.error[:150]}")
        if self.judge_agreement:
            agree = self.judge_agreement.get("overall", {})
            lines += [
                "",
                f"  judge cross-check: raw {agree.get('raw', 0):.3f} · "
                f"kappa {agree.get('kappa', 0):+.3f} -> {self.judge_agreement.get('verdict', '?')}",
            ]

        problems = [f"{a.name}: {w}" for a in self.arms for w in a.warnings]
        if problems:
            lines += ["", "  !! HEALTH WARNINGS - read before trusting any number:"]
            lines += [f"     - {p}" for p in problems]
        for warning in self.report.get("warnings") or []:
            lines.append(f"     - {warning}")

        lines += [
            "",
            "  outputs:",
            f"    figures    {self.bundle / 'report' / 'figures'}",
            f"    tables     {self.bundle / 'report' / 'tier_a_results.md'}",
            f"    raw rows   {self.bundle / 'runs'}/<arm>/results.jsonl",
            f"    manifest   {self.bundle / 'study.json'}",
            "=" * 74,
            "",
        ]
        return "\n".join(lines)


def _arm_configs(arms: Sequence[str] | str) -> list[tuple[str, str]]:
    """Parse `name=config.yaml` arm specs.

    Args:
        arms: A list, or a comma-separated string, of `name=config` pairs.

    Returns:
        (name, config) pairs in order.

    Raises:
        StudyError: If a spec is malformed or a name repeats.
    """
    items = arms.split(",") if isinstance(arms, str) else list(arms)
    out: list[tuple[str, str]] = []
    for spec in items:
        spec = str(spec).strip()
        if not spec:
            continue
        if "=" not in spec:
            raise StudyError(
                f"Arm {spec!r} is not name=config. Example: base=qwen36_base.yaml"
            )
        name, config = spec.split("=", 1)
        name, config = name.strip(), config.strip()
        if any(n == name for n, _ in out):
            raise StudyError(f"Duplicate arm name {name!r}; every arm needs a distinct recipe name")
        out.append((name, config))
    if not out:
        raise StudyError("A study needs at least one arm")
    return out


def run_study(
    arms: Sequence[str] | str,
    name: str = "study",
    out_root: Path | str = "output/constieval/studies",
    overrides: dict[str, Any] | None = None,
    max_items: int = 0,
    cross_check: str = "",
    cross_check_n: int = 120,
    keep_going: bool = True,
) -> StudyResult:
    """Run every arm against one frozen item set and bundle the results.

    The item set is resolved ONCE, from the first arm's config, and handed to every
    subsequent arm. Letting each arm resolve its own would be the easiest way to end up
    comparing two models measured on different items - which the report would refuse, but
    only after the money was spent.

    Args:
        arms: `name=config` specs. The name becomes the recipe on every row.
        name: Study name; the bundle directory is `<name>_<timestamp>`.
        out_root: Where bundles are written.
        overrides: Dotted config overrides applied to every arm.
        max_items: Cap base items for a quick pass; pairing is preserved.
        cross_check: Reference judge model for the agreement check; "" skips it.
        cross_check_n: Rows to re-grade in the cross-check.
        keep_going: Continue after an arm fails. A served checkpoint that is not up
            should not throw away an arm that already succeeded.

    Returns:
        The StudyResult.

    Raises:
        StudyError: If the arms are malformed, or every arm failed.
    """
    specs = _arm_configs(arms)
    study_id = f"{name}_{timestamp()}"
    bundle = Path(out_root) / study_id
    (bundle / "runs").mkdir(parents=True, exist_ok=True)
    started = time.time()

    base_cfg = load_config(specs[0][1], dict(overrides or {}))
    itemset = prepare_itemset(base_cfg)
    if max_items:
        itemset = itemset.subsample(max_items, seed=int(base_cfg.get("seed", 0)))

    projected = estimate(base_cfg, arms=len(specs))
    print(f"study {study_id}: {len(specs)} arm(s), item set {itemset.itemset_id} "
          f"({len(itemset)} items), projected ${projected.total:.2f}")

    results: list[ArmResult] = []
    for arm_name, config_path in specs:
        arm = ArmResult(name=arm_name, config=config_path)
        started_arm = time.time()
        print(f"\n--- arm {arm_name} ({config_path})")
        try:
            cfg = load_config(config_path, {**(overrides or {}), "run.recipe": arm_name})
            run = run_eval(cfg, itemset=itemset, run_id=f"{study_id}__{arm_name}")
            arm.run_id = run.run_id
            arm.run_dir = str(run.run_dir)
            arm.spend_usd = float(run.manifest.get("spend_usd_total", 0.0))
            arm.warnings = list(run.warnings)
            # Copied, not referenced: the bundle has to stay meaningful after output/ is
            # cleaned or the study is moved to another machine.
            shutil.copytree(run.run_dir, bundle / "runs" / arm_name, dirs_exist_ok=True)
            print(run.summary())
        except (ConfigError, Exception) as e:  # noqa: BLE001 - one arm must not sink the study
            arm.status = "failed"
            arm.error = f"{type(e).__name__}: {e}"
            print(f"    FAILED: {arm.error[:200]}")
            if not keep_going:
                arm.seconds = time.time() - started_arm
                results.append(arm)
                raise
        arm.seconds = time.time() - started_arm
        results.append(arm)

    ok = [a for a in results if a.status == "ok"]
    if not ok:
        raise StudyError(
            "Every arm failed; nothing to report. First error: "
            + (results[0].error if results else "unknown")
        )

    store = ResultsStore.load(*[Path(a.run_dir) / "results.jsonl" for a in ok])
    written = report_mod.build_report(
        store,
        bundle / "report",
        title=f"Constitution internalization — {name}",
    )

    agreement: dict[str, Any] | None = None
    if cross_check:
        print(f"\n--- judge cross-check against {cross_check}")
        try:
            from .judge_check import check_judge

            reference = JudgeConfig.from_config(base_cfg)
            reference.model = cross_check
            report = check_judge(
                Path(ok[0].run_dir),
                itemset,
                loader.clause_set(str(base_cfg["clause_set"])),
                build_judge_client(base_cfg),
                reference,
                n=cross_check_n,
                seed=int(base_cfg.get("seed", 0)),
                max_workers=int(base_cfg.get("max_workers", 8)),
            )
            agreement = report.to_dict()
            (bundle / "judge_agreement.json").write_text(json.dumps(agreement, indent=2))
            print(report.render())
        except Exception as e:  # noqa: BLE001 - a failed cross-check must not lose the study
            agreement = {"error": f"{type(e).__name__}: {e}"}
            print(f"    cross-check failed: {e}")

    shutil.copytree(
        Path(base_cfg["itemset"]["dir"]) / itemset.itemset_id.split("_n")[0],
        bundle / "itemset",
        dirs_exist_ok=True,
    )

    result = StudyResult(
        study_id=study_id,
        bundle=bundle,
        itemset_id=itemset.itemset_id,
        arms=results,
        report=written,
        judge_agreement=agreement,
    )
    result.manifest = {
        "study_id": study_id,
        "created_utc": timestamp(),
        "git_sha": git_sha(),
        "itemset_id": itemset.itemset_id,
        "n_items": len(itemset),
        "clause_set": base_cfg["clause_set"],
        "judge_model": (base_cfg.get("judge") or {}).get("model"),
        "arms": [a.to_dict() for a in results],
        "total_spend_usd": result.total_spend,
        "projected_spend_usd": projected.total,
        "seconds": round(time.time() - started, 1),
        "report": written,
        "judge_agreement": agreement,
        "warnings": [w for a in results for w in a.warnings] + list(written.get("warnings") or []),
    }
    (bundle / "study.json").write_text(json.dumps(result.manifest, indent=2, default=str))
    (bundle / "README.md").write_text(_bundle_readme(result, base_cfg))
    return result


def _bundle_readme(result: StudyResult, cfg: dict[str, Any]) -> str:
    """Write the orientation note that ships inside every bundle."""
    arms = "\n".join(
        f"- **{a.name}** — `{a.config}`, {a.status}, ${a.spend_usd:.2f}" for a in result.arms
    )
    warnings = result.manifest.get("warnings") or []
    warning_block = (
        "## Health warnings\n\n"
        + "\n".join(f"- **{w}**" for w in warnings)
        + "\n\nRead these before quoting any number.\n\n"
        if warnings
        else ""
    )
    return f"""<!-- ABOUTME: Self-contained results bundle for one constieval study. -->
<!-- ABOUTME: Generated by src.eval.constieval.study; every path below is relative to this folder. -->

# {result.study_id}

Constitution-internalization study. Everything needed to read, re-plot, or audit this
result is in this folder.

- item set: `{result.itemset_id}` ({result.manifest.get('n_items')} items)
- clause set: `{cfg.get('clause_set')}` — cut from the trait document the training data encodes
- judge: `{(cfg.get('judge') or {}).get('model')}`
- spend: ${result.total_spend:.2f}

## Arms

{arms}

{warning_block}## What is here

| Path | What it is |
|---|---|
| `report/figures/` | The seven Tier A figures |
| `report/tier_a_results.md` | Every figure's numbers in greppable form |
| `report/summary.json` | Headline table, machine-readable |
| `runs/<arm>/results.jsonl` | One row per (item, axis, score) — the source of every plot |
| `runs/<arm>/completions.jsonl` | Raw model outputs, for spot-checking a surprising score |
| `runs/<arm>/run_meta.json` | Config, git sha, health, spend |
| `itemset/` | The frozen items every arm was measured on |
| `judge_agreement.json` | Cheap-vs-strong judge cross-check, if it was run |
| `study.json` | This study's full manifest |

## How to read it

There is **no gold set**, so absolute levels on a single axis are only as good as the
rubric behind them. What is sound is the **comparison**: every arm hit the same frozen item
set with the same judge and identical sampling, so the only difference is the weights. Read
the difference panel on the clause heatmap and the paired robustness/OOD deltas; treat a
lone axis score as descriptive.

## Regenerate the figures

```bash
uv run python -m src.eval.constieval.cli report --results runs --out report
```
"""
