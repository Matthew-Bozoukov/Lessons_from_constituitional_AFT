# ABOUTME: Projects the cost of a run from its config alone, with no API calls.
# ABOUTME: Counts are exact; token sizes are estimates, so treat the total as a budget not a quote.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .control import loader
from .core.llm import PriceTable

# Per-call token estimates, measured from rendered prompts on the shipped prompt packs.
# Only the token sizes are approximate - every call COUNT below is exact.
TOKENS = {
    "item_gen_in": 700,
    "item_gen_out": 300,
    "ood_rewrite_in": 600,
    "ood_rewrite_out": 300,
    "target_in": 500,
    # Measured on Qwen3.6-27B over a 238-item pass: ~1,500 tokens of trace plus ~450 of
    # answer. The first version of this table guessed 900 and under-quoted the target side
    # by ~2x, so it is now calibrated against a real run rather than intuition.
    "target_out_thinking": 2000,
    "target_out_plain": 400,
    "judge_in": 1200,
    "judge_out": 150,
}


@dataclass
class Estimate:
    """A projected cost breakdown.

    Attributes:
        counts: Exact item and call counts.
        costs: USD per stage.
        assumptions: The token estimates used, so a surprising total can be checked.
    """

    counts: dict[str, int] = field(default_factory=dict)
    costs: dict[str, float] = field(default_factory=dict)
    assumptions: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> float:
        """Total projected USD."""
        return round(sum(self.costs.values()), 2)

    def render(self) -> str:
        """Return a human-readable breakdown."""
        lines = ["counts:"]
        lines += [f"  {k:34s} {v:>7,}" for k, v in self.counts.items()]
        lines += ["", "projected cost (USD):"]
        lines += [f"  {k:34s} {v:>7.2f}" for k, v in self.costs.items()]
        lines += ["", f"  {'TOTAL':34s} {self.total:>7.2f}", ""]
        lines += [
            "Counts are exact; token sizes are estimates, so treat this as a budget, not a quote.",
            "Item generation is a ONE-TIME cost shared by every arm and cached across re-runs.",
        ]
        unpriced = self.assumptions.get("unpriced_models") or []
        if unpriced:
            lines += ["", f"WARNING: no price listed for {', '.join(unpriced)} - counted as $0."]
        return "\n".join(lines)


def _clause_stats(cfg: dict[str, Any], fakes_per_clause: int) -> tuple[int, int, int]:
    """Return (clauses in scope, clauses with a stated rationale, available distractors).

    The distractor count is taken from the clause set rather than assumed: not every
    clause has one, and assuming otherwise over-counts the fake_clause family - the exact
    kind of drift that makes an estimator worse than no estimator.
    """
    clauses = loader.clause_set(str(cfg["clause_set"]))
    in_scope = (
        list(clauses.clauses)
        if (cfg.get("itemset") or {}).get("include_held_out", True)
        else list(clauses.trained)
    )
    distractors = sum(
        min(fakes_per_clause, len(clauses.fakes_for(c.clause_id))) for c in in_scope
    )
    return len(in_scope), sum(1 for c in in_scope if c.rationale.strip()), distractors


def estimate(cfg: dict[str, Any], arms: int = 1) -> Estimate:
    """Project the cost of a run from its config.

    Every count here is derived the same way the builders derive theirs, so a config
    change shows up in the estimate before it shows up on the bill.

    Args:
        cfg: A resolved run config.
        arms: How many models will be evaluated against this item set. Item generation
            is charged once; generation and judging are charged per arm.

    Returns:
        The Estimate.
    """
    itemset = cfg.get("itemset") or {}
    families = itemset.get("families") or {}
    transforms = itemset.get("transforms") or {}
    fakes_cfg = families.get("fake_clause") or {}
    fakes_per_clause = int(fakes_cfg.get("per_clause", 0)) if fakes_cfg.get("enabled", True) else 0
    n_clause, n_rationale, n_distractors = _clause_stats(cfg, fakes_per_clause)

    def fam(name: str) -> dict[str, Any]:
        """Return a family's config if enabled, else an empty mapping."""
        spec = families.get(name) or {}
        return spec if spec.get("enabled", True) else {}

    app_cfg = fam("application")
    n_app = (
        n_clause * len(app_cfg.get("difficulties") or []) * int(app_cfg.get("variants", 0))
        if app_cfg
        else 0
    )
    n_retrieval = n_clause * int((fam("retrieval") or {}).get("variants", 0))
    # Each distractor is paired with a matched genuine probe, hence x2.
    n_fake = n_distractors * 2
    conflict_cfg = fam("conflict")
    n_conflict = int(conflict_cfg.get("pairs", 0)) * int(conflict_cfg.get("variants", 0)) if conflict_cfg else 0
    n_over = n_clause * int((fam("over_refusal") or {}).get("variants", 0))
    n_persona = int((fam("persona_drift") or {}).get("n", 0))

    pressure = transforms.get("pressure") or {}
    n_pressure = (
        n_clause * int(pressure.get("per_clause", 1)) * len(pressure.get("wrappers") or [])
        if pressure.get("enabled", True)
        else 0
    )

    ood = transforms.get("ood") or {}
    n_ood = 0
    if ood.get("enabled", True):
        max_distance = int(ood.get("max_distance", 0))
        for axis_name in ood.get("axes") or []:
            values = [
                v
                for v in loader.ood_axis(axis_name)["values"]
                if int(v["distance"]) > 0 and (not max_distance or int(v["distance"]) <= max_distance)
            ]
            n_ood += n_clause * int(ood.get("per_clause", 1)) * len(values)

    n_items = n_app + n_retrieval + n_fake + n_conflict + n_over + n_persona + n_pressure + n_ood
    n_app_family = n_app + n_pressure + n_ood

    # Judge calls, counted the way judge_all counts them: per (item, axis), after the
    # rubrics' `conditions` and `requires` gates.
    judge_calls = 0
    for axis in loader.declared_axes():
        spec = loader.rubric(axis)
        conditions = spec.get("conditions") or []
        clean_only = conditions == ["clean"]
        for family in spec.get("applies_to") or []:
            if family == "application":
                pool = n_app if clean_only else n_app_family
                if "rationale" in (spec.get("requires") or []):
                    # Only clauses that state a reason are gradeable on this axis.
                    pool = int(pool * (n_rationale / n_clause)) if n_clause else 0
                judge_calls += pool
            elif family == "conflict":
                judge_calls += n_conflict
            elif family == "retrieval":
                judge_calls += n_retrieval
            elif family == "fake_clause":
                judge_calls += n_fake
            elif family == "over_refusal":
                judge_calls += n_over
            elif family == "persona_drift":
                judge_calls += n_persona

    prices = PriceTable(cfg.get("pricing") or {})
    gen_model = str((itemset.get("generator") or {}).get("model", ""))
    judge_model = str((cfg.get("judge") or {}).get("model", ""))
    target = cfg.get("target") or {}
    target_model = str(target.get("model", ""))
    thinking = bool(target.get("enable_thinking", True))
    target_out = TOKENS["target_out_thinking"] if thinking else TOKENS["target_out_plain"]

    # Only families whose scenarios are written by a model cost a generation call.
    scenario_calls = n_app + n_conflict + n_over
    item_gen_cost = scenario_calls * prices.cost(
        gen_model, TOKENS["item_gen_in"], TOKENS["item_gen_out"]
    ) + n_ood * prices.cost(gen_model, TOKENS["ood_rewrite_in"], TOKENS["ood_rewrite_out"])

    target_cost = arms * n_items * prices.cost(target_model, TOKENS["target_in"], target_out)
    judge_cost = arms * judge_calls * prices.cost(
        judge_model, TOKENS["judge_in"], TOKENS["judge_out"]
    )

    return Estimate(
        counts={
            "clauses": n_clause,
            "clauses with stated rationale": n_rationale,
            "items (clean)": n_app + n_retrieval + n_fake + n_conflict + n_over + n_persona,
            "items (pressure)": n_pressure,
            "items (ood)": n_ood,
            "items TOTAL": n_items,
            "item-generation calls (once)": scenario_calls + n_ood,
            "target completions per arm": n_items,
            "judge calls per arm": judge_calls,
            "arms": arms,
        },
        costs={
            f"item generation ({gen_model})": round(item_gen_cost, 2),
            f"target x{arms} ({target_model})": round(target_cost, 2),
            f"judging x{arms} ({judge_model})": round(judge_cost, 2),
        },
        assumptions={"tokens": TOKENS, "unpriced_models": sorted(prices.unpriced)},
    )
