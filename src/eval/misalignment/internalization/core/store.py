# ABOUTME: The results store: one row per (run, recipe, clause, item, axis, score).
# ABOUTME: Every plot and table in the suite derives from this one table and nothing else.

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Sequence

from .types import Completion, Item, Verdict

# The store's columns, declared explicitly rather than inferred. Inference would give
# a run with no OOD items a null ood_axis column and a run with them a string column,
# and the two would stop concatenating - which is the one thing this table exists to do.
COLUMNS: tuple[str, ...] = (
    # run identity
    "run_id",
    "recipe",
    "checkpoint_step",
    "model_id",
    "itemset_id",
    # item identity
    "item_id",
    "parent_item_id",
    "clause_id",
    "clause_title",
    "principle",
    "priority_tier",
    "held_out",
    # item coordinates
    "family",
    "difficulty",
    "condition",
    "pressure",
    "ood_axis",
    "ood_value",
    "variant",
    # measurement
    "axis",
    "score",
    "raw_score",
    "passed",
    # provenance
    "judge_model",
    "judge_rationale",
    "error",
)


@dataclass
class ScoreRow:
    """One measurement: one axis scored on one item for one run.

    A single completion produces several rows - an application item is scored on
    compliance, tension recognition, and justification quality from the same
    generation. That is why the axis is a column rather than a separate table.
    """

    run_id: str
    recipe: str
    item_id: str
    clause_id: str
    axis: str
    score: float
    checkpoint_step: int = 0
    model_id: str = ""
    itemset_id: str = ""
    parent_item_id: str = ""
    clause_title: str = ""
    principle: str = ""
    priority_tier: int = 3
    held_out: bool = False
    family: str = ""
    difficulty: str = "na"
    condition: str = "clean"
    pressure: str = ""
    ood_axis: str = ""
    ood_value: str = ""
    variant: int = 0
    raw_score: float = 0.0
    passed: bool = False
    judge_model: str = ""
    judge_rationale: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the row as a dict in declared column order."""
        d = asdict(self)
        return {c: d[c] for c in COLUMNS}


@dataclass
class RunContext:
    """Identity shared by every row a single evaluation run writes.

    Attributes:
        run_id: Unique id for this evaluation run.
        recipe: The training data recipe under test. The comparison key for every
            plot, and the thing that gets a consistent colour across all of them.
        checkpoint_step: Training step of the checkpoint, for trajectory plots.
        model_id: Model actually queried.
        itemset_id: Fingerprint of the frozen item set. Rows carrying different
            itemset_ids must not be compared, and the report says so loudly.
    """

    run_id: str
    recipe: str
    checkpoint_step: int = 0
    model_id: str = ""
    itemset_id: str = ""


class ResultsStore:
    """An append-only collection of ScoreRows with JSONL and parquet persistence."""

    def __init__(self, rows: Iterable[ScoreRow | dict[str, Any]] | None = None) -> None:
        """Initialize, optionally seeding with existing rows."""
        self.rows: list[dict[str, Any]] = []
        for row in rows or ():
            self.append(row)

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self.rows)

    def append(self, row: ScoreRow | dict[str, Any]) -> None:
        """Append one row, normalising to the declared column order.

        Args:
            row: A ScoreRow or an equivalent mapping.

        Raises:
            ValueError: If a mapping carries a column the schema does not declare,
                which is nearly always a typo that would silently vanish downstream.
        """
        if isinstance(row, ScoreRow):
            self.rows.append(row.to_dict())
            return
        unknown = sorted(set(row) - set(COLUMNS))
        if unknown:
            raise ValueError(f"Row has undeclared columns {unknown}; declared: {list(COLUMNS)}")
        defaults = {f.name: f.default for f in fields(ScoreRow) if f.default is not None}
        self.rows.append({c: row.get(c, defaults.get(c)) for c in COLUMNS})

    def extend(self, rows: Iterable[ScoreRow | dict[str, Any]]) -> None:
        """Append many rows."""
        for row in rows:
            self.append(row)

    def to_frame(self):
        """Return the store as a pandas DataFrame with stable dtypes.

        Returns:
            A DataFrame with exactly COLUMNS, even when empty, so downstream
            groupbys never fail on a missing column for a run that happened to
            contain no items of some family.
        """
        import pandas as pd

        if not self.rows:
            return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS})
        df = pd.DataFrame(self.rows, columns=list(COLUMNS))
        for col, dtype in (
            ("score", "float64"),
            ("raw_score", "float64"),
            ("checkpoint_step", "int64"),
            ("variant", "int64"),
            ("priority_tier", "int64"),
        ):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(dtype)
        for col in ("passed", "held_out"):
            df[col] = df[col].fillna(False).astype(bool)
        return df

    def write(self, path: Path | str) -> str:
        """Write the store to JSONL, and to parquet alongside it when available.

        Args:
            path: Output .jsonl path.

        Returns:
            The written JSONL path as a string.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for row in self.rows:
                fh.write(json.dumps(row) + "\n")
        try:
            self.to_frame().to_parquet(out.with_suffix(".parquet"), index=False)
        except (ImportError, ValueError):
            # Parquet is a convenience for analysis; JSONL is the source of truth.
            pass
        return str(out)

    @classmethod
    def load(cls, *paths: Path | str) -> ResultsStore:
        """Load one or more result files into a single store.

        Concatenating runs is how a recipe comparison is assembled: each run writes
        its own file, and the report reads all of them.

        Args:
            *paths: .jsonl or .parquet paths.

        Returns:
            A ResultsStore holding every row.

        Raises:
            FileNotFoundError: If a path does not exist.
        """
        store = cls()
        for p in paths:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"No results at {path}")
            if path.suffix == ".parquet":
                import pandas as pd

                store.extend(pd.read_parquet(path).to_dict("records"))
                continue
            for line in path.read_text().splitlines():
                if line.strip():
                    store.append(json.loads(line))
        return store

    @classmethod
    def load_dir(cls, directory: Path | str, pattern: str = "*/results.jsonl") -> ResultsStore:
        """Load every result file under a directory.

        Args:
            directory: Root to search.
            pattern: Glob relative to the root.

        Returns:
            A ResultsStore holding every matching row.

        Raises:
            FileNotFoundError: If nothing matches.
        """
        root = Path(directory)
        paths = sorted(root.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No results matching {pattern!r} under {root}")
        return cls.load(*paths)


def build_rows(
    ctx: RunContext,
    item: Item,
    verdicts: Sequence[Verdict],
    clause_title: str = "",
    principle: str = "",
    priority_tier: int = 3,
    held_out: bool = False,
    completion: Completion | None = None,
) -> list[ScoreRow]:
    """Fan one item's verdicts out into store rows.

    Args:
        ctx: Run identity applied to every row.
        item: The item judged.
        verdicts: One verdict per axis.
        clause_title: Denormalised clause title, so plots need no second table.
        principle: Denormalised parent principle.
        priority_tier: Denormalised conflict-ordering tier.
        held_out: Whether the clause was excluded from data generation.
        completion: The completion, used to carry a generation error onto the rows.

    Returns:
        One ScoreRow per verdict.
    """
    gen_error = completion.error if completion is not None else ""
    return [
        ScoreRow(
            run_id=ctx.run_id,
            recipe=ctx.recipe,
            checkpoint_step=ctx.checkpoint_step,
            model_id=ctx.model_id,
            itemset_id=ctx.itemset_id,
            item_id=item.item_id,
            parent_item_id=item.parent_item_id,
            clause_id=item.clause_id,
            clause_title=clause_title,
            principle=principle,
            priority_tier=priority_tier,
            held_out=held_out,
            family=item.family,
            difficulty=item.difficulty,
            condition=item.condition,
            pressure=item.pressure,
            ood_axis=item.ood_axis,
            ood_value=item.ood_value,
            variant=item.variant,
            axis=v.axis,
            score=v.score,
            raw_score=v.raw_score,
            passed=v.passed,
            judge_model=v.judge_model,
            judge_rationale=v.rationale,
            error=v.error or gen_error,
        )
        for v in verdicts
    ]
