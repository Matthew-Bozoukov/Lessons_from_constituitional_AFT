# ABOUTME: Stage functions for every synthdoc document type: the difficult-advice
# ABOUTME: stages 2-6 and the MEM cells (planning, perturbation, generation, assembly).

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from src.endpoints.openrouter import OpenRouterClient

from . import prompts
from .constitution import Trait
from .core import Checkpoint, Usage, call_json, call_tagged, resilient, run_items

# --- stage 2 -----------------------------------------------------------------------


def generate_scenarios(traits: list[Trait], client: OpenRouterClient, usage: Usage,
                       model: str, per_trait: int, per_call: int, temperature: float,
                       max_tokens: int, workers: int) -> list[dict]:
    """Generate difficult situations for each trait.

    Scenarios are requested in batches rather than all at once: a single call asking for
    40 situations would exceed any sane `max_tokens` and truncate its JSON. Batching also
    improves diversity, since each call is told to vary domains only within its own batch.

    Args:
        traits: The segmented constitution.
        client: OpenRouter client.
        usage: Tally.
        model: Model id.
        per_trait: Scenarios requested per trait, in total.
        per_call: Scenarios requested per API call.
        temperature: Sampling temperature.
        max_tokens: Completion cap.
        workers: Thread pool size.

    Returns:
        One record per scenario.
    """
    # (trait index, batch index, how many this batch asks for)
    batches: list[tuple[int, int, int]] = []
    for ti in range(len(traits)):
        remaining = per_trait
        bi = 0
        while remaining > 0:
            n = min(per_call, remaining)
            batches.append((ti, bi, n))
            remaining -= n
            bi += 1

    def one(k: int) -> list[dict]:
        ti, bi, n = batches[k]
        t = traits[ti]
        parsed, _ = call_json(
            client, usage, model,
            prompts.SCENARIO_SYSTEM,
            prompts.SCENARIO_USER.format(trait_name=t.name, trait_text=t.text, n=n),
            temperature, max_tokens, stage="scenarios",
        )
        assert isinstance(parsed, list), f"{t.trait_id}: expected a JSON array, got {type(parsed)}"
        return [{
            "scenario_id": f"{t.trait_id}_b{bi:02d}_s{j:03d}",
            "trait_id": t.trait_id,
            "trait_name": t.name,
            "trait_text": t.text,
            "domain": s.get("domain", ""),
            "situation": s["situation"],
            "shortcut": s.get("shortcut", ""),
        } for j, s in enumerate(parsed)]

    nested = resilient(one, len(batches), workers, "stage2:scenarios")
    return [r for group in nested for r in group]


# --- stage 3 -----------------------------------------------------------------------


def draft_prompts(scenarios: list[dict], client: OpenRouterClient, usage: Usage, model: str,
                  temperature: float, max_tokens: int, workers: int) -> list[dict]:
    """Write a first-attempt system and user prompt for each scenario."""
    def one(i: int) -> dict:
        s = scenarios[i]
        parsed, _ = call_json(
            client, usage, model,
            prompts.DRAFT_SYSTEM,
            prompts.DRAFT_USER.format(situation=s["situation"], shortcut=s["shortcut"]),
            temperature, max_tokens, stage="draft",
        )
        return {**s, "draft_system": parsed["system"], "draft_user": parsed["user"]}

    return resilient(one, len(scenarios), workers, "stage3:draft")


# --- stage 4 -----------------------------------------------------------------------


def refine_prompts(drafts: list[dict], client: OpenRouterClient, usage: Usage, model: str,
                   constitution: str, temperature: float, max_tokens: int,
                   workers: int, ckpt: Checkpoint | None = None) -> list[dict]:
    """Rewrite each draft prompt into a sharper test of its target trait.

    The full constitution and the specific target trait are both injected, so the model
    can tell which principle the prompt is supposed to stress.
    """
    def one(d: dict) -> dict:
        parsed, _ = call_json(
            client, usage, model,
            prompts.REFINE_SYSTEM,
            prompts.REFINE_USER.format(
                constitution=constitution, trait_name=d["trait_name"],
                trait_text=d["trait_text"], draft_system=d["draft_system"],
                draft_user=d["draft_user"],
            ),
            temperature, max_tokens, stage="refine",
        )
        return {**d, "system": parsed["system"], "user": parsed["user"],
                "refine_changes": parsed.get("changes", "")}

    return run_items(drafts, one, workers, "stage4:refine", ckpt)


# --- stage 5 -----------------------------------------------------------------------


def generate_responses(refined: list[dict], client: OpenRouterClient, usage: Usage, model: str,
                       style_guidance: str, temperature: float, max_tokens: int,
                       workers: int, ckpt: Checkpoint | None = None) -> list[dict]:
    """Answer each refined prompt with explicit reasoning, steered by the target trait."""
    def one(r: dict) -> dict:
        parsed = call_tagged(
            client, usage, model,
            [{"role": "system", "content": prompts.RESPONSE_SYSTEM.format(
                system=r["system"], trait_name=r["trait_name"], trait_text=r["trait_text"],
                style_guidance=style_guidance)},
             {"role": "user", "content": prompts.RESPONSE_USER.format(user=r["user"])}],
            temperature, max_tokens, "respond", ("reasoning", "response"),
        )
        return {**r, "draft_reasoning": parsed["reasoning"], "draft_response": parsed["response"]}

    return run_items(refined, one, workers, "stage5:respond", ckpt)


# --- stage 6 -----------------------------------------------------------------------


def rewrite_responses(responses: list[dict], client: OpenRouterClient, usage: Usage, model: str,
                      constitution: str, temperature: float, max_tokens: int,
                      workers: int, ckpt: Checkpoint | None = None) -> list[dict]:
    """Rewrite each response to maximally exhibit its target trait.

    The blog calls this the critical step: the reviewer sees the whole transcript with the
    relevant constitution section in context, then rewrites rather than scores.
    """
    def one(r: dict) -> dict:
        parsed = call_tagged(
            client, usage, model,
            [{"role": "system", "content": prompts.REWRITE_SYSTEM},
             {"role": "user", "content": prompts.REWRITE_USER.format(
                 constitution=constitution, trait_name=r["trait_name"],
                 trait_text=r["trait_text"], system=r["system"], user=r["user"],
                 reasoning=r["draft_reasoning"], response=r["draft_response"])}],
            temperature, max_tokens, "rewrite", ("reasoning", "response", "changes"),
        )
        return {**r, "reasoning": parsed["reasoning"], "response": parsed["response"],
                "rewrite_changes": parsed.get("changes", "")}

    return run_items(responses, one, workers, "stage6:rewrite", ckpt)


def to_sft(records: list[dict]) -> list[dict]:
    """Convert final records into chat form with the trait carried in metadata.

    Args:
        records: Stage-6 output.

    Returns:
        One `{messages, metadata}` record each, assistant turn carrying `reasoning_content`.
    """
    out = []
    for r in records:
        out.append({
            "messages": [
                {"role": "system", "content": r["system"]},
                {"role": "user", "content": r["user"]},
                {"role": "assistant", "content": r["response"],
                 "reasoning_content": r["reasoning"]},
            ],
            "metadata": {
                "scenario_id": r["scenario_id"],
                "trait_id": r["trait_id"],
                "trait_name": r["trait_name"],
                "trait_text": r["trait_text"],
                "domain": r.get("domain", ""),
                "shortcut": r.get("shortcut", ""),
                "situation": r["situation"],
            },
        })
    return out


# =====================================================================
# MEM (model-evaluates-model): cells over a completed difficult-advice run
# =====================================================================

# MEM documents make the model reason about a response to a difficult-advice scenario
# and work out whether it was the right call. Cells run over a COMPLETED
# difficult-advice run (its scenarios and gold responses), so arm differences are
# attributable to format rather than content.


def _eval_response_text(p: dict) -> str:
    """Return the response a MEM cell evaluates.

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


def _control_messages(p: dict, constitution: str) -> list[dict]:
    """Build the generation call for the reasoning-only control."""
    return [
        {"role": "system", "content": prompts.CONTROL_REASONING_SYSTEM},
        {"role": "user", "content": prompts.CONTROL_REASONING_USER.format(
            constitution=constitution, trait_name=p["trait_name"],
            trait_text=p["trait_text"],
            style_line=prompts.EXPLICITNESS_STYLES[p["explicitness"]],
            system=p["system"], user=p["user"], response=p["gold_response"])},
    ]


def _wrap_transcript(p: dict) -> str:
    """Render the clean transcript-in-user-turn wrapper for a critique record."""
    return prompts.TRANSCRIPT_WRAP_VARIANTS[p["variant_ix"]].format(
        system=p["system"], user=p["user"], response=_eval_response_text(p))


def _critique_messages(p: dict, constitution: str) -> list[dict]:
    """Build the generation call for the other-attribution critique cells (m3/m4).

    The user message is the exact wrapper the training record will carry plus the
    format scaffolding, which assembly strips. Nothing here may depend on whether the
    response is good or flawed beyond the response text itself.
    """
    return [
        {"role": "system", "content": prompts.MEM_CRITIQUE_SYSTEM.format(
            constitution=constitution, trait_name=p["trait_name"],
            trait_text=p["trait_text"],
            style_line=prompts.EXPLICITNESS_STYLES[p["explicitness"]])},
        {"role": "user",
         "content": _wrap_transcript(p) + "\n\n---\n" + prompts.MEM_CRITIQUE_FORMAT},
    ]


def _reflect_messages(p: dict, constitution: str) -> list[dict]:
    """Build the generation call for the self-reflection cells (m1/m2).

    The response under evaluation sits in a genuine assistant turn -- attribution is
    structural, never verbal -- and the reflection prompt follows as real chat history.
    Identical template for the good and flawed twins.
    """
    return [
        {"role": "system", "content": prompts.MEM_REFLECT_SYSTEM.format(
            system=p["system"], constitution=constitution, trait_name=p["trait_name"],
            trait_text=p["trait_text"],
            style_line=prompts.EXPLICITNESS_STYLES[p["explicitness"]])},
        {"role": "user", "content": p["user"]},
        {"role": "assistant", "content": _eval_response_text(p)},
        {"role": "user", "content": prompts.REFLECT_VARIANTS[p["reflect_ix"]]
         + "\n\n---\n" + prompts.MEM_REFLECT_FORMAT},
    ]


def _mem_metadata(r: dict, verdict: str | None = None) -> dict:
    """Metadata every MEM training record carries.

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


def _assemble_control(r: dict) -> dict:
    """Control record: the original exchange with only the reasoning trace replaced."""
    return {
        "messages": [
            {"role": "system", "content": r["system"]},
            {"role": "user", "content": r["user"]},
            {"role": "assistant", "content": r["gold_response"],
             "reasoning_content": r["reasoning"]},
        ],
        "metadata": _mem_metadata(r),
    }


def _assemble_critique(r: dict) -> dict:
    """Critique record (m3/m4): transcript in the user turn, evaluation as the reply."""
    return {
        "messages": [
            {"role": "system", "content": prompts.MEM_EVAL_SYSTEM},
            {"role": "user", "content": _wrap_transcript(r)},
            {"role": "assistant", "content": r["response"],
             "reasoning_content": r["reasoning"]},
        ],
        "metadata": _mem_metadata(r, verdict=r["assessment"]),
    }


def _assemble_reflect(r: dict) -> dict:
    """Self-reflection record (m1/m2): the evaluated response sits in the model's own
    prior turn -- carrying NO reasoning trace, matching how Qwen renders history at
    inference -- and only the final turn is a training target (`supervise: "final"`)."""
    return {
        "messages": [
            {"role": "system", "content": r["system"]},
            {"role": "user", "content": r["user"]},
            {"role": "assistant", "content": _eval_response_text(r)},
            {"role": "user", "content": prompts.REFLECT_VARIANTS[r["reflect_ix"]]},
            {"role": "assistant", "content": r["response"],
             "reasoning_content": r["reasoning"]},
        ],
        "metadata": _mem_metadata(r, verdict=r["assessment"]),
    }


@dataclass(frozen=True)
class CellSpec:
    """One MEM cell: who the response is attributed to, whether it is good or flawed,
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
        build_messages: (plan record, constitution) -> generation message list.
        assemble: generated record -> `{messages, metadata}` training record.
    """

    cell: str
    attribution: str | None
    response_kind: str | None
    model_key: str
    tags: tuple[str, ...]
    verdicts: tuple[str, ...]
    supervise: str
    build_messages: Callable[[dict, str], list[dict]]
    assemble: Callable[[dict], dict]


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


def _weighted_flaws(n: int, flaws: dict, rng: random.Random) -> list[dict]:
    """Allocate n (type, severity) flaw assignments from the config's weight tables."""
    types, sevs = flaws["types"], flaws["severities"]
    bad = sorted(set(types) - set(prompts.FLAW_TYPES)) + \
        sorted(set(sevs) - set(prompts.FLAW_SEVERITIES))
    assert not bad, f"unknown flaw type/severity in config: {bad}"
    crossed = {f"{t}|{s}": tw * sw
               for t, tw in types.items() for s, sw in sevs.items()}
    return [{"type": k.split("|")[0], "severity": k.split("|")[1]}
            for k in _weighted_labels(n, crossed, rng)]


def plan_mem_records(source: list[dict], cells: dict[str, int],
                 explicitness: dict[str, float], seed: int,
                 source_run: str = "", flaws: dict | None = None) -> list[dict]:
    """Allocate source scenarios to MEM cells. Deterministic, no LLM calls.

    Each cell draws its own trait-stratified, seeded sample from the source run, so
    trait coverage is by construction and cross-cell scenario reuse is deliberate
    (cells are separate training arms). Flawed cells additionally get a (type, severity)
    flaw assignment from the config's weight tables -- coverage over the flaw grid is
    likewise by construction, not sampling luck.

    Args:
        source: A completed difficult-advice run's stage-6 final records.
        cells: Cell name -> number of documents to plan. Zero-count cells are skipped.
        explicitness: Style label -> weight (see prompts.EXPLICITNESS_STYLES).
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
    bad_style = sorted(set(explicitness) - set(prompts.EXPLICITNESS_STYLES))
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
        cell_flaws = _weighted_flaws(want, flaws, rng) \
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
                "variant_ix": rng.randrange(len(prompts.TRANSCRIPT_WRAP_VARIANTS)),
                "reflect_ix": rng.randrange(len(prompts.REFLECT_VARIANTS)),
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
                [{"role": "system", "content": prompts.PERTURB_SYSTEM},
                 {"role": "user", "content": prompts.PERTURB_USER.format(
                     trait_name=p["trait_name"], trait_text=p["trait_text"],
                     user=p["user"], response=p["gold_response"],
                     flaw_type=flaw["type"],
                     flaw_definition=prompts.FLAW_TYPES[flaw["type"]],
                     flaw_severity=flaw["severity"],
                     severity_guidance=prompts.FLAW_SEVERITIES[flaw["severity"]])}],
                temperature, max_tokens, "perturb",
                ("flawed_response", "change_summary"))
            ok, ratio = _length_matched(p["gold_response"], parsed["flawed_response"])
            if ok:
                return {**p, "flawed_response": parsed["flawed_response"],
                        "change_summary": parsed["change_summary"],
                        "length_ratio": round(ratio, 3)}
        raise ValueError(f"{p['record_id']}: minimal pair length ratio {ratio:.2f} "
                         f"outside {_LENGTH_RATIO_BOUNDS} after retry")

    return run_items(flawed, one, workers, "mem:perturb", ckpt)


def generate_mem_documents(plans: list[dict], client: OpenRouterClient, usage: Usage,
                       model_cfgs: dict[str, dict], constitution: str,
                       workers: int, ckpt: Checkpoint | None = None) -> list[dict]:
    """Generate each planned MEM document via its cell's prompt builder.

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
                             spec.build_messages(p, constitution),
                             m["temperature"], m["max_tokens"], spec.model_key,
                             spec.tags)
        out = {**p, **{k: parsed[k] for k in spec.tags}}
        if "assessment" in out:
            out["assessment"] = _norm_verdict(out["assessment"], spec.verdicts)
        return out

    return run_items(plans, one, workers, "mem:generate", ckpt)


def to_mem_sft(records: list[dict]) -> list[dict]:
    """Assemble generated MEM records into training form, one assembler per cell."""
    return [CELLS[r["cell"]].assemble(r) for r in records]
