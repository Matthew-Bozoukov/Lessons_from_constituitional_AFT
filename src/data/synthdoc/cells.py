# ABOUTME: The model-eval-model cell structure: CELLS registry, deterministic planning,
# ABOUTME: minimal-pair perturbation, generation and assembly. All wording comes from the config.

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from src.endpoints.openrouter import OpenRouterClient

from .core import Checkpoint, Usage, call_tagged, run_items

# model-eval-model documents make the model reason about a response to a difficult-advice scenario
# and work out whether it was the right call. Cells run over a COMPLETED
# difficult-advice run (its scenarios and gold responses), so arm differences are
# attributable to format rather than content.


def _eval_response_text(p: dict) -> str:
    """Return the response a model-eval-model cell evaluates.

    The flawed cells read the perturbed response -- strictly, so a document whose
    perturbation failed can never silently degrade into evaluating the gold response.
    Routing every cell through one accessor is what keeps generation blind: good and
    flawed twins build byte-identical prompts except for this text.
    """
    if p["response_kind"] == "flawed":
        assert p.get("flawed_response"), \
            f"{p['record_id']}: flawed cell reached generation without a perturbed response"
        return p["flawed_response"]
    return p["gold_response"]


def _control_messages(p: dict, constitution: str, P: dict) -> list[dict]:
    """Build the generation call for the reasoning-only control."""
    return [
        {"role": "system", "content": P["control_system"]},
        {"role": "user", "content": P["control_user"].format(
            constitution=constitution, trait_name=p["trait_name"],
            trait_text=p["trait_text"],
            style_line=P["explicitness_styles"][p["explicitness"]],
            system=p["system"], user=p["user"], response=p["gold_response"])},
    ]


def _wrap_transcript(p: dict, P: dict) -> str:
    """Render the clean transcript-in-user-turn wrapper for a critique record."""
    return P["transcript_wrappers"][p["variant_ix"]].format(
        system=p["system"], user=p["user"], response=_eval_response_text(p))


def _critique_messages(p: dict, constitution: str, P: dict) -> list[dict]:
    """Build the generation call for the other-attribution critique cells (m3/m4).

    The user message is the exact wrapper the training record will carry plus the
    format scaffolding, which assembly strips. Nothing here may depend on whether the
    response is good or flawed beyond the response text itself.
    """
    return [
        {"role": "system", "content": P["critique_system"].format(
            constitution=constitution, trait_name=p["trait_name"],
            trait_text=p["trait_text"],
            style_line=P["explicitness_styles"][p["explicitness"]])},
        {"role": "user",
         "content": _wrap_transcript(p, P) + "\n\n---\n" + P["critique_format"]},
    ]


def _reflect_messages(p: dict, constitution: str, P: dict) -> list[dict]:
    """Build the generation call for the self-reflection cells (m1/m2).

    The response under evaluation sits in a genuine assistant turn -- attribution is
    structural, never verbal -- and the reflection prompt follows as real chat history.
    Identical template for the good and flawed twins.
    """
    return [
        {"role": "system", "content": P["reflect_system"].format(
            system=p["system"], constitution=constitution, trait_name=p["trait_name"],
            trait_text=p["trait_text"],
            style_line=P["explicitness_styles"][p["explicitness"]])},
        {"role": "user", "content": p["user"]},
        {"role": "assistant", "content": _eval_response_text(p)},
        {"role": "user", "content": P["reflect_variants"][p["reflect_ix"]]
         + "\n\n---\n" + P["reflect_format"]},
    ]


def _model_eval_model_metadata(r: dict, verdict: str | None = None) -> dict:
    """Metadata every model-eval-model training record carries.

    `supervise` declares which assistant turns are training targets: "final" for the
    self cells (the first response is context, not a target) and "all" otherwise. It is
    threaded from here through the render/mixture/masking chain.
    """
    flaw = r.get("flaw") or {}
    return {
        "record_id": r["record_id"],
        "cell": r["cell"],
        "attribution": r["attribution"],
        "response_kind": r["response_kind"],
        "flaw_type": flaw.get("type"),
        "flaw_severity": flaw.get("severity"),
        "explicitness": r["explicitness"],
        "verdict": verdict,
        "scenario_id": r["scenario_id"],
        "trait_id": r["trait_id"],
        "trait_name": r["trait_name"],
        "domain": r.get("domain", ""),
        "situation": r["situation"],
        "shortcut": r.get("shortcut", ""),
        "source_run": r.get("source_run", ""),
        "supervise": CELLS[r["cell"]].supervise,
    }


def _assemble_control(r: dict, P: dict) -> dict:
    """Control record: the original exchange with only the reasoning trace replaced."""
    return {
        "messages": [
            {"role": "system", "content": r["system"]},
            {"role": "user", "content": r["user"]},
            {"role": "assistant", "content": r["gold_response"],
             "reasoning_content": r["reasoning"]},
        ],
        "metadata": _model_eval_model_metadata(r),
    }


def _assemble_critique(r: dict, P: dict) -> dict:
    """Critique record (m3/m4): transcript in the user turn, evaluation as the reply."""
    return {
        "messages": [
            {"role": "system", "content": P["evaluator_system"]},
            {"role": "user", "content": _wrap_transcript(r, P)},
            {"role": "assistant", "content": r["response"],
             "reasoning_content": r["reasoning"]},
        ],
        "metadata": _model_eval_model_metadata(r, verdict=r["assessment"]),
    }


def _assemble_reflect(r: dict, P: dict) -> dict:
    """Self-reflection record (m1/m2): the evaluated response sits in the model's own
    prior turn -- carrying NO reasoning trace, matching how Qwen renders history at
    inference -- and only the final turn is a training target (`supervise: "final"`)."""
    return {
        "messages": [
            {"role": "system", "content": r["system"]},
            {"role": "user", "content": r["user"]},
            {"role": "assistant", "content": _eval_response_text(r)},
            {"role": "user", "content": P["reflect_variants"][r["reflect_ix"]]},
            {"role": "assistant", "content": r["response"],
             "reasoning_content": r["reasoning"]},
        ],
        "metadata": _model_eval_model_metadata(r, verdict=r["assessment"]),
    }


@dataclass(frozen=True)
class CellSpec:
    """One model-eval-model cell: who the response is attributed to, whether it is good or flawed,
    and how its documents are generated and assembled.

    Attributes:
        cell: Registry key, also the `cell` field on every record.
        attribution: "self" | "other" | None (the control evaluates nothing).
        response_kind: "good" | "flawed" | None.
        model_key: Which `models:` block in the config prices and runs this cell; also
            the per-stage usage key, so measured estimates line up per cell family.
        tags: Required tagged blocks in the generation output.
        verdicts: Verdicts the cell's <assessment> tag may carry (empty = no verdict).
        supervise: Which assistant turns of the assembled record are training targets.
        build_messages: (plan record, constitution, prompt library P) -> messages.
        assemble: (generated record, prompt library P) -> training record.
    """

    cell: str
    attribution: str | None
    response_kind: str | None
    model_key: str
    tags: tuple[str, ...]
    verdicts: tuple[str, ...]
    supervise: str
    build_messages: Callable[[dict, str, dict], list[dict]]
    assemble: Callable[[dict, dict], dict]


# M5 is a mixture of cells, not a cell.
CELLS: dict[str, CellSpec] = {
    "control": CellSpec(
        cell="control", attribution=None, response_kind=None, model_key="control",
        tags=("reasoning",), verdicts=(), supervise="all",
        build_messages=_control_messages, assemble=_assemble_control),
    "m4_other_good": CellSpec(
        cell="m4_other_good", attribution="other", response_kind="good",
        model_key="critique", tags=("reasoning", "response", "assessment"),
        verdicts=("sound", "issue_found"), supervise="all",
        build_messages=_critique_messages, assemble=_assemble_critique),
    "m3_other_flawed": CellSpec(
        cell="m3_other_flawed", attribution="other", response_kind="flawed",
        model_key="critique", tags=("reasoning", "response", "assessment"),
        verdicts=("sound", "issue_found"), supervise="all",
        build_messages=_critique_messages, assemble=_assemble_critique),
    "m2_self_good": CellSpec(
        cell="m2_self_good", attribution="self", response_kind="good",
        model_key="reflect", tags=("reasoning", "response", "assessment"),
        verdicts=("held", "revised"), supervise="final",
        build_messages=_reflect_messages, assemble=_assemble_reflect),
    "m1_self_flawed": CellSpec(
        cell="m1_self_flawed", attribution="self", response_kind="flawed",
        model_key="reflect", tags=("reasoning", "response", "assessment"),
        verdicts=("held", "revised"), supervise="final",
        build_messages=_reflect_messages, assemble=_assemble_reflect),
}

# Accepted <assessment> spellings -> canonical verdict.
_VERDICTS = {"sound": "sound", "issue_found": "issue_found", "issue found": "issue_found",
             "issue": "issue_found", "held": "held", "hold": "held", "revised": "revised"}


def _norm_verdict(raw: str, allowed: tuple[str, ...]) -> str:
    """Canonicalise an <assessment> verdict against the cell's allowed set."""
    v = _VERDICTS.get(raw.strip().lower().replace("-", "_"))
    if v is None or v not in allowed:
        raise ValueError(f"unrecognised <assessment> verdict for this cell: {raw!r}")
    return v


def _weighted_labels(n: int, weights: dict[str, float], rng: random.Random) -> list[str]:
    """Return n labels matching the weights as closely as rounding allows.

    Deterministic allocation rather than sampling, so coverage (explicitness styles,
    flaw type x severity) is by construction and a smoke run's tiny n still gets a
    sensible split.
    """
    assert weights, "weights must be non-empty"
    total = sum(weights.values())
    keys = sorted(weights)
    counts = {k: int(n * weights[k] / total) for k in keys}
    remainder = sorted(keys, key=lambda k: (counts[k] - n * weights[k] / total, k))
    for k in remainder[: n - sum(counts.values())]:
        counts[k] += 1
    out = [k for k in keys for _ in range(counts[k])]
    rng.shuffle(out)
    return out


def _weighted_flaws(n: int, flaws: dict, rng: random.Random, P: dict) -> list[dict]:
    """Allocate n (type, severity) flaw assignments from the config's weight tables."""
    types, sevs = flaws["types"], flaws["severities"]
    bad = sorted(set(types) - set(P["flaw_types"])) + \
        sorted(set(sevs) - set(P["flaw_severities"]))
    assert not bad, f"unknown flaw type/severity in config: {bad}"
    crossed = {f"{t}|{s}": tw * sw
               for t, tw in types.items() for s, sw in sevs.items()}
    return [{"type": k.split("|")[0], "severity": k.split("|")[1]}
            for k in _weighted_labels(n, crossed, rng)]


def plan_model_eval_model_records(source: list[dict], cells: dict[str, int],
                                  explicitness: dict[str, float], seed: int, P: dict,
                                  source_run: str = "",
                                  flaws: dict | None = None) -> list[dict]:
    """Allocate source scenarios to model-eval-model cells. Deterministic, no LLM calls.

    Each cell draws its own trait-stratified, seeded sample from the source run, so
    trait coverage is by construction and cross-cell scenario reuse is deliberate
    (cells are separate training arms). Flawed cells additionally get a (type, severity)
    flaw assignment from the config's weight tables -- coverage over the flaw grid is
    likewise by construction, not sampling luck.

    Args:
        source: A completed difficult-advice run's stage-6 final records.
        cells: Cell name -> number of documents to plan. Zero-count cells are skipped.
        explicitness: Style label -> weight (see P["explicitness_styles"]).
        seed: Base RNG seed; each cell derives its own stream from it.
        source_run: Provenance label (HF repo or run dir) carried into metadata.
        flaws: The config's `flaws` block ({types: {..}, severities: {..}}). Required
            when any flawed cell is enabled.

    Returns:
        One plan record per document, `record_id = "<scenario_id>::<cell>"`.

    Raises:
        ValueError: An enabled cell is not registered, asks for more documents than the
            source run holds, or is flawed while `flaws` is missing.
    """
    enabled = {c: int(n) for c, n in cells.items() if int(n) > 0}
    unknown = sorted(set(enabled) - set(CELLS))
    if unknown:
        raise ValueError(
            f"unregistered cell(s) enabled: {unknown}. Registered: {sorted(CELLS)}.")
    bad_style = sorted(set(explicitness) - set(P["explicitness_styles"]))
    assert not bad_style, f"unknown explicitness style(s): {bad_style}"
    if any(CELLS[c].response_kind == "flawed" for c in enabled) and not flaws:
        raise ValueError("a flawed cell is enabled but the config has no `flaws` block")

    by_trait: dict[str, list[dict]] = {}
    for r in sorted(source, key=lambda r: r["scenario_id"]):
        by_trait.setdefault(r["trait_id"], []).append(r)

    plans: list[dict] = []
    for cell in sorted(enabled):
        want = enabled[cell]
        if want > len(source):
            raise ValueError(f"{cell}: wants {want} documents but the source run has "
                             f"only {len(source)}")
        spec = CELLS[cell]
        rng = random.Random(f"{seed}:{cell}")
        pools = {t: rng.sample(rows, len(rows)) for t, rows in sorted(by_trait.items())}
        order = sorted(pools)
        picked: list[dict] = []
        i = 0
        while len(picked) < want:
            pool = pools[order[i % len(order)]]
            if pool:
                picked.append(pool.pop())
            i += 1
        styles = _weighted_labels(want, explicitness, rng)
        cell_flaws = _weighted_flaws(want, flaws, rng, P) \
            if spec.response_kind == "flawed" else [None] * want
        for r, style, flaw in zip(picked, styles, cell_flaws):
            plans.append({
                "record_id": f"{r['scenario_id']}::{cell}",
                "cell": cell,
                "attribution": spec.attribution,
                "response_kind": spec.response_kind,
                "scenario_id": r["scenario_id"],
                "trait_id": r["trait_id"],
                "trait_name": r["trait_name"],
                "trait_text": r["trait_text"],
                "domain": r.get("domain", ""),
                "situation": r["situation"],
                "shortcut": r.get("shortcut", ""),
                "system": r["system"],
                "user": r["user"],
                "gold_reasoning": r["reasoning"],
                "gold_response": r["response"],
                "flaw": flaw,
                "explicitness": style,
                "variant_ix": rng.randrange(len(P["transcript_wrappers"])),
                "reflect_ix": rng.randrange(len(P["reflect_variants"])),
                "source_run": source_run,
            })
    return plans


# Minimal-pair guard: a flawed response drifting far from the gold response's length is
# a surface tell a classifier (or the trained model) can exploit instead of the flaw.
_LENGTH_RATIO_BOUNDS = (0.8, 1.25)


def _length_matched(gold: str, flawed: str) -> tuple[bool, float]:
    """Return whether the pair's word-count ratio sits inside the allowed band."""
    ratio = max(len(flawed.split()), 1) / max(len(gold.split()), 1)
    lo, hi = _LENGTH_RATIO_BOUNDS
    return lo <= ratio <= hi, ratio


def perturb_responses(plans: list[dict], client: OpenRouterClient, usage: Usage,
                      model: str, temperature: float, max_tokens: int, workers: int,
                      templates: dict, P: dict,
                      ckpt: Checkpoint | None = None) -> list[dict]:
    """Create the minimal-pair flawed response for every flawed-cell plan record.

    One flaw per document, from the record's assigned (type, severity); length and
    register held to the gold response (out-of-band pairs are resampled once, then the
    item fails and is dropped by `resilient` -- never silently passed through).

    Returns:
        The flawed plan records extended with `flawed_response` and `change_summary`
        (metadata forever -- `checks.check_blindness` proves it never trains).
    """
    flawed = [p for p in plans if p["response_kind"] == "flawed"]

    def one(p: dict) -> dict:
        flaw = p["flaw"]
        ratio = 0.0
        for _ in range(2):
            parsed = call_tagged(
                client, usage, model,
                [{"role": "system", "content": templates["system"]},
                 {"role": "user", "content": templates["user"].format(
                     trait_name=p["trait_name"], trait_text=p["trait_text"],
                     user=p["user"], response=p["gold_response"],
                     flaw_type=flaw["type"],
                     flaw_definition=P["flaw_types"][flaw["type"]],
                     flaw_severity=flaw["severity"],
                     severity_guidance=P["flaw_severities"][flaw["severity"]])}],
                temperature, max_tokens, "perturb",
                ("flawed_response", "change_summary"))
            ok, ratio = _length_matched(p["gold_response"], parsed["flawed_response"])
            if ok:
                return {**p, "flawed_response": parsed["flawed_response"],
                        "change_summary": parsed["change_summary"],
                        "length_ratio": round(ratio, 3)}
        raise ValueError(f"{p['record_id']}: minimal pair length ratio {ratio:.2f} "
                         f"outside {_LENGTH_RATIO_BOUNDS} after retry")

    return run_items(flawed, one, workers, "model_eval_model:perturb", ckpt)


def generate_model_eval_model_documents(plans: list[dict], client: OpenRouterClient,
                                        usage: Usage, model_cfgs: dict[str, dict],
                                        constitution: str, P: dict, workers: int,
                                        ckpt: Checkpoint | None = None) -> list[dict]:
    """Generate each planned model-eval-model document via its cell's prompt builder.

    Args:
        plans: Plan records (perturbed where flawed).
        client: OpenRouter client.
        usage: Tally; calls are recorded under the cell's `model_key`, so a measured
            estimate can price each cell family separately.
        model_cfgs: model_key -> {model, temperature, max_tokens}.
        constitution: Full constitution text.
        workers: Thread pool size.
        ckpt: Optional checkpoint keyed by `record_id`.

    Returns:
        Plan records extended with the cell's generated fields, failures dropped.
    """
    def one(p: dict) -> dict:
        spec = CELLS[p["cell"]]
        m = model_cfgs[spec.model_key]
        parsed = call_tagged(client, usage, m["model"],
                             spec.build_messages(p, constitution, P),
                             m["temperature"], m["max_tokens"], spec.model_key,
                             spec.tags)
        out = {**p, **{k: parsed[k] for k in spec.tags}}
        if "assessment" in out:
            out["assessment"] = _norm_verdict(out["assessment"], spec.verdicts)
        return out

    return run_items(plans, one, workers, "model_eval_model:generate", ckpt)


def to_model_eval_model_sft(records: list[dict], P: dict) -> list[dict]:
    """Assemble generated model-eval-model records into training form, one assembler per cell."""
    return [CELLS[r["cell"]].assemble(r, P) for r in records]
