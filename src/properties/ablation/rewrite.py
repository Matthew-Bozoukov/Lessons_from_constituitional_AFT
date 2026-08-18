# ABOUTME: The recommended ablation: one LLM call per affected row rewrites the property
# ABOUTME: out (or substitutes a named replacement), leaving everything else in place.

"""Rewrite the property out of the rows that have it.

Callum's recommendation, 2026-08-17, and the reason it is the default:

    "taking an existing dataset and then doing an ad hoc, specific rewrite to vary a
    targeted property of that data is a very good kind of experiment... it's fairly cheap,
    because you only have to run one LLM per example... and it gets you two datasets that
    will be very similar in most ways."

That last clause is the whole point. `filter` changes which rows exist; `regenerate`
changes the pipeline and therefore many things at once. A rewrite changes one thing in one
place and leaves the row count, the scenario, the user's question and the final decision
alone — so a difference between the arms is attributable to the property far more tightly.

Two modes:

    remove       rewrite the channel so the property is gone and nothing replaces it.
    substitute   replace it with a NAMED other property. This is the mode that answers
                 "is it this reasoning move, or is it having any reasoning move at all?" —
                 a remove-arm that regresses might only be showing that shorter reasoning
                 is worse. `substitute_with` is per-property prose and lives in the config.

Three guards, because a rewrite is an LLM editing training data and every failure mode of
that is quiet:

* **Preserve list.** The rewriter is told, explicitly, what must survive unchanged — the
  decision, the facts, the register. A rewrite that also changes the conclusion has
  ablated two things.
* **Re-detection.** Every rewritten row goes back through the SAME detector. A row the
  detector still flags is reported as a miss, not silently counted as ablated.
* **Length check.** The rewritten channel's length is compared against the original.
  The peer-critique corpus was separable at AUC 0.9973 on LENGTH ALONE (2026-08-17); an
  ablation that shortens every flagged row has introduced exactly that artefact, and a
  classifier — or the model — can learn the arm from it instead of from the property.
"""

from __future__ import annotations

import statistics
import threading

from src.properties import block
from src.properties.ablation.base import (
    AblationResult,
    candidates,
    check_channel,
    check_corpus,
)
from src.properties.registry import Property
from src.properties.shared import interpret as interpret_mod
from src.properties.sources.base import Record, SourceAdapter

KIND = "rewrite"
MODES = ("remove", "substitute")
# Above this relative length change the arm is separable on length alone; see the
# peer-critique defect of 2026-08-17.
LENGTH_DRIFT_WARN = 0.15

REWRITE_SYSTEM = """\
You are editing one piece of AI training data for a controlled experiment. Two datasets \
will be trained on and compared, and they must differ in EXACTLY ONE respect.

THE PROPERTY TO {verb}:
  {label}
  {description}

{instruction}

MUST SURVIVE UNCHANGED — changing any of these makes the experiment measure two things at \
once, which is worse than not running it:
{preserve}

Also hold the length roughly constant. A systematically shorter (or longer) rewrite lets a \
classifier — and the model being trained — identify which dataset a row came from without \
ever seeing the property, which is the exact failure this experiment must avoid.

Write natural text. It should read as though it was generated this way, not as though \
something was cut out of it: no stubs, no dangling connectives, no summary of what was \
removed.

Return only JSON:
{{"rewritten": "<the full rewritten text>", "changed": true | false, "note": "<one clause \
on what you changed, or why you changed nothing>"}}"""

REMOVE_INSTRUCTION = """\
Rewrite the text below so that the property is ABSENT. Everything the property's sentences \
were doing for the argument must either be dropped or carried by reasoning of a different \
kind — do not paraphrase the property into different words."""

SUBSTITUTE_INSTRUCTION = """\
Rewrite the text below so that the property is replaced by this one instead:

  {substitute_with}

The replacement should do comparable work in the argument, at comparable length, so the \
two datasets differ in WHICH move is made rather than in whether a move is made at all."""

DEFAULT_PRESERVE = [
    "the decision the assistant reaches, and whether it declines",
    "every fact, name, number and quotation",
    "the user's question and situation",
    "the register, voice and formatting of the text",
    "the overall length, to within about 10%",
]


def applicable(prop: Property, records: list[Record], adapter: SourceAdapter,
               cfg=None) -> tuple[bool, str]:
    """Whether rewriting can run against this property and corpus.

    Args:
        prop: The property.
        records: The corpus.
        adapter: The source adapter.
        cfg: The ablation config block.

    Returns:
        (ok, reason).
    """
    ok, reason = check_corpus(adapter)
    if not ok:
        return ok, reason
    # The query is the user's turn: rewriting it changes the scenario, which is a
    # different experiment (and one `uv run synth` already runs).
    ok, reason = check_channel(prop, ("reasoning", "response"))
    if not ok:
        return ok, reason
    if not any(r.channel(prop.channel).strip() for r in records):
        return False, f"no record carries text in the {prop.channel!r} channel"
    if cfg is not None and hasattr(cfg, "get"):
        mode = cfg.get("mode", "remove")
        if mode not in MODES:
            return False, f"mode must be one of {MODES}, got {mode!r}"
        if mode == "substitute" and not cfg.get("substitute_with"):
            return False, ("substitute mode needs `substitute_with:` — the replacement "
                           "property, in prose, in this config")
    return True, ""


def _write_back(record: Record, channel: str, text: str) -> dict:
    """Put rewritten text back into a row without disturbing anything else.

    Args:
        record: The record.
        channel: "reasoning" or "response".
        text: The rewritten text.

    Returns:
        The row, with only that channel replaced.

    Raises:
        ValueError: If the row is pre-rendered (a single `text` string), which this
            cannot edit safely — the channel boundaries are the family's syntax.
    """
    row = dict(record.raw)
    if "messages" not in row:
        raise ValueError(
            f"{record.record_id}: pre-rendered rows carry the family's chat syntax in one "
            "string, so a channel cannot be replaced without re-deriving that syntax. "
            "Rewrite the interchange corpus and let training render it.")
    messages = [dict(m) for m in row["messages"]]
    for message in messages:
        if message.get("role") == "assistant":
            message["reasoning_content" if channel == "reasoning" else "content"] = text
            break
    row["messages"] = messages
    return row


def apply(prop: Property, records: list[Record], cfg) -> AblationResult:
    """Detect the property, rewrite the rows that have it, and re-detect.

    Args:
        prop: The property to rewrite out.
        records: The corpus.
        cfg: The ablation config block. Keys: `mode` (remove | substitute),
            `substitute_with` (substitute mode; per-property prose), `rewriter` (model),
            `preserve` (list, defaults to DEFAULT_PRESERVE), `description_override`,
            `workers`, `max_tokens`, `verify_rewrites` (default True),
            `detector {model, workers}`.

    Returns:
        The AblationResult, with the re-detection miss rate and the length drift in
        `report`.
    """
    from omegaconf import OmegaConf

    from src.endpoints.openrouter import OpenRouterClient, map_threaded
    from src.utils import extract_json

    cfg = OmegaConf.create(cfg)
    mode = str(cfg.get("mode", "remove"))
    rewriter = str(cfg.get("rewriter", "anthropic/claude-sonnet-5"))
    workers = int(cfg.get("workers", 8))
    preserve = list(cfg.get("preserve") or DEFAULT_PRESERVE)
    instruction = (SUBSTITUTE_INSTRUCTION.format(
        substitute_with=str(cfg["substitute_with"])) if mode == "substitute"
        else REMOVE_INSTRUCTION)
    system = REWRITE_SYSTEM.format(
        verb="REMOVE" if mode == "remove" else "REPLACE",
        label=prop.label,
        description=str(cfg.get("description_override") or prop.description),
        instruction=instruction,
        preserve="\n".join(f"* {p}" for p in preserve))

    judged, _untouched = candidates(records, cfg)
    verdicts = interpret_mod.detect(
        judged, prop.label, prop.detector, channel=prop.channel,
        **block(cfg, "detector"))
    flagged = {v["record_id"] for v in verdicts if v.get("exhibits")}
    targets = [r for r in judged if r.record_id in flagged]
    print(f">>> detector flagged {len(targets)}/{len(judged)} rows for {prop.label!r}")

    client = OpenRouterClient()
    lock = threading.Lock()
    rewritten: dict[str, str] = {}
    failures: list[dict] = []

    def run(i: int) -> None:
        record = targets[i]
        original = record.channel(prop.channel)
        try:
            result = client.chat(
                model=rewriter, temperature=1.0,
                max_tokens=int(cfg.get("max_tokens", 8192)),
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content":
                           f"<text>\n{original}\n</text>"}])
            parsed = extract_json(result.content)
            text = str(parsed.get("rewritten", "")).strip()
            if not text:
                raise ValueError("rewriter returned no text")
            if not parsed.get("changed", True):
                with lock:
                    failures.append({"record_id": record.record_id,
                                     "error": "rewriter reported no change: "
                                              f"{parsed.get('note', '')[:160]}"})
                return
        except Exception as exc:  # noqa: BLE001 - one bad row must not kill the pass
            with lock:
                failures.append({"record_id": record.record_id,
                                 "error": f"{type(exc).__name__}: {exc}"[:300]})
            return
        with lock:
            rewritten[record.record_id] = text

    if targets:
        map_threaded(run, len(targets), max_workers=workers, desc="rewriting")

    rows, drifts = [], []
    for record in records:
        text = rewritten.get(record.record_id)
        if text is None:
            rows.append(dict(record.raw))
            continue
        original = record.channel(prop.channel)
        drifts.append((len(text) - len(original)) / max(1, len(original)))
        rows.append(_write_back(record, prop.channel, text))

    report = {"mode": mode, "rewriter": rewriter, "preserve": preserve,
              "n_rewritten": len(rewritten), "n_rewrite_failures": len(failures),
              "rewrite_failures": failures[:20],
              "detector_errors": sum(1 for v in verdicts if v.get("exhibits") is None)}
    if drifts:
        report["length_drift"] = {
            "mean": round(statistics.fmean(drifts), 4),
            "median": round(statistics.median(drifts), 4)}
        if abs(statistics.fmean(drifts)) > LENGTH_DRIFT_WARN:
            report["length_drift"]["warning"] = (
                f"mean length changed by {statistics.fmean(drifts):+.1%}; a bag-of-words "
                "classifier separated the peer-critique arms on length alone at AUC 0.85 "
                "(2026-08-17). Check the arms are not separable before training on this.")
            print(f"!!! {report['length_drift']['warning']}")

    if bool(cfg.get("verify_rewrites", True)) and rewritten:
        # The same detector, on the rewritten text. A row it still flags was not ablated,
        # however confident the rewriter's note was.
        redone = [r for r in records if r.record_id in rewritten]
        patched = [_patched_record(r, prop.channel, rewritten[r.record_id])
                   for r in redone]
        after = interpret_mod.detect(
            patched, prop.label, prop.detector, channel=prop.channel,
            **block(cfg, "detector"))
        misses = sorted(v["record_id"] for v in after if v.get("exhibits"))
        report["rewrite_misses"] = len(misses)
        report["rewrite_miss_ids"] = misses[:20]
        report["rewrite_success_rate"] = round(
            1 - len(misses) / max(1, len(rewritten)), 4)
        print(f">>> re-detection: {len(misses)}/{len(rewritten)} rewritten rows still "
              f"carry {prop.label!r}")

    return AblationResult(kind=KIND, property_id=prop.property_id, rows=rows,
                          changed_ids=sorted(rewritten), detected_ids=sorted(flagged),
                          report=report)


def _patched_record(record: Record, channel: str, text: str) -> Record:
    """A copy of a record with one channel replaced, for re-detection.

    Args:
        record: The original.
        channel: Which channel was rewritten.
        text: The rewritten text.

    Returns:
        The patched Record.
    """
    import dataclasses

    return dataclasses.replace(record, **{channel: text})
