# ABOUTME: The weakest ablation: a judge quotes the spans that do the property's work and
# ABOUTME: those tokens are unsupervised, leaving text — and tokenisation — untouched.

"""Mask the property's tokens instead of removing them.

The intervention: the property's text stays in the sequence, so the model still READS it,
but it carries no loss, so the model is never trained to PRODUCE it. That makes this the
cleanest arm to interpret — the masked corpus tokenises identically to its control, so
nothing is confounded by a tokenisation change — and the weakest, because a masked arm can
still learn the property from the surrounding text's dependence on it. Say so when
reporting a null from this ablation.

The mechanism is already in the training path: `train_lora.py` reads a per-row `mask_spans`
column of CHARACTER spans of the RENDERED text and passes them to `build_labels`, which
unsupervises every token touching one. So the work here is to produce those spans:

    1. the detector says which rows carry the property
    2. a judge quotes VERBATIM spans of the reasoning that do the property's work
    3. each quote is located by exact string search in the rendered row

Exact search, never fuzzy. A quote that is absent, or that appears more than once, is a
hard error for that row: a fuzzy match would mask the wrong tokens, and the entire value of
this ablation is that the masked set is inspectable.

Rendering matters and is checked. Spans are character offsets into the string
`train_lora.py` will build, so this file renders with the same tokenizer and the same
`ModelProfile.render_kwargs`, and stamps `mask_render_model` on every row. A config whose
model differs from the train config's is the one way these spans go silently wrong.
"""

from __future__ import annotations

import re
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

KIND = "mask"

SELECT_SYSTEM = """\
You are marking up an AI assistant's private reasoning for a training experiment.

We are studying ONE property of the reasoning:
  {label}
  {description}

A detector has already flagged this trace as containing that property{evidence_note}.

Your job: quote the exact spans of the reasoning that DO that property's work, and nothing \
else.

Rules:
* Each span must be a VERBATIM contiguous substring of the reasoning, copied character for \
character, including punctuation and capitalisation. It will be located by exact string \
search and anything that does not match is discarded as an error.
* Each span must be UNIQUE in the trace — if a short phrase appears twice, extend it until \
it is unique.
* Quote the whole clause or sentence that carries the move, not a bare keyword. A reader \
should be able to see the reasoning step in what you quote.
* Do NOT quote reasoning that does other work — identifying who is affected, the \
assistant's uncertainty about facts, the final decision, or what it will say to the user — \
even when it sits in the same sentence. If a sentence is half this property and half \
something else, quote only the half that is this property.
* If a span is genuinely inseparable from other reasoning, quote it and say so in `notes`.
* Return no spans at all if the trace does not really do this — an empty list is a valid \
and useful answer, and a better one than a stretch.

Return only JSON:
{{"spans": ["verbatim span", "..."], "notes": "one sentence on anything ambiguous"}}"""


def applicable(prop: Property, records: list[Record], adapter: SourceAdapter,
               cfg=None) -> tuple[bool, str]:
    """Whether masking can run against this property and corpus.

    Args:
        prop: The property.
        records: The corpus.
        adapter: The source adapter.
        cfg: The ablation config block; `model` is required (it fixes the rendering).

    Returns:
        (ok, reason).
    """
    ok, reason = check_corpus(adapter)
    if not ok:
        return ok, reason
    # Masking edits labels, not text. A property of the response could be masked in
    # principle, but unsupervising the answer trains the model to produce no answer, which
    # is a different experiment; keep this to the reasoning.
    ok, reason = check_channel(prop, ("reasoning",))
    if not ok:
        return ok, reason
    if not any(r.reasoning.strip() for r in records):
        return False, ("no record in this corpus carries a reasoning trace; there is "
                       "nothing to mask")
    if cfg is not None and not (cfg.get("model") if hasattr(cfg, "get") else None):
        return False, ("mask needs `model:` — spans are character offsets into the "
                       "rendered training text, so the rendering model must match the "
                       "train config's exactly")
    return True, ""


def _render(records: list[Record], model_id: str) -> tuple[dict[str, str], object, object]:
    """Render every record the way training will render it.

    Args:
        records: The corpus.
        model_id: The model whose tokenizer and ModelProfile define the rendering.

    Returns:
        (record_id -> rendered text, tokenizer, profile).

    Raises:
        ValueError: If a record has no `messages` to render (a pre-rendered corpus keeps
            its own `text`, which is used as-is).
    """
    from transformers import AutoTokenizer

    from src.model_profile import model_profile

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    profile = model_profile(model_id)
    rendered = {}
    for record in records:
        if "text" in record.raw:
            rendered[record.record_id] = record.raw["text"]
            continue
        if "messages" not in record.raw:
            raise ValueError(f"{record.record_id}: no `messages` or `text` to render")
        messages = [{k: v for k, v in m.items() if v is not None}
                    for m in record.raw["messages"]]
        rendered[record.record_id] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            **profile.render_kwargs)
    return rendered, tokenizer, profile


def _think_region(text: str) -> tuple[int, int]:
    """Locate the assistant's reasoning inside a rendered conversation.

    Args:
        text: A rendered row.

    Returns:
        (start, end) character offsets of the content between the think tags.

    Raises:
        ValueError: If the row does not hold exactly one reasoning block — masking a row
            with two would need to know which one the detector meant.
    """
    opens = [m.end() for m in re.finditer(r"<think>\n?", text)]
    closes = [m.start() for m in re.finditer(r"\n?</think>", text)]
    if len(opens) != 1 or len(closes) != 1:
        raise ValueError(f"expected exactly one think block, found "
                         f"{len(opens)} open / {len(closes)} close")
    return opens[0], closes[0]


def _locate(text: str, span: str, region: tuple[int, int]) -> tuple[int, int]:
    """Find one verbatim span inside the reasoning region.

    Args:
        text: The rendered row.
        span: The verbatim substring the judge returned.
        region: (start, end) of the reasoning block.

    Returns:
        (start, end) character offsets of the span.

    Raises:
        ValueError: If the span is absent, ambiguous, or falls outside the reasoning.
    """
    low, high = region
    hits = [m.start() for m in re.finditer(re.escape(span), text)]
    inside = [h for h in hits if h >= low and h + len(span) <= high]
    if not inside:
        where = "outside the reasoning block" if hits else "not present"
        raise ValueError(f"span {where}: {span[:90]!r}")
    if len(inside) > 1:
        raise ValueError(f"span occurs {len(inside)}x, not unique: {span[:90]!r}")
    return inside[0], inside[0] + len(span)


def apply(prop: Property, records: list[Record], cfg) -> AblationResult:
    """Detect the property, quote its spans, and emit a corpus carrying `mask_spans`.

    Args:
        prop: The property to mask.
        records: The corpus.
        cfg: The ablation config block. Keys: `model` (required — must equal the train
            config's `model`), `judge` (span selector), `detector {model, workers}`,
            `workers`, `description_override` (per-property prose for the span selector,
            which is where the property-specific wording belongs).

    Returns:
        The AblationResult. Rows carry `mask_spans`, `mask_property` and
        `mask_render_model`; rows the detector cleared are unchanged.

    Raises:
        ValueError: If `model` is missing.
    """
    from omegaconf import OmegaConf

    from src.endpoints.openrouter import OpenRouterClient, map_threaded
    from src.utils import extract_json

    cfg = OmegaConf.create(cfg)
    model_id = cfg.get("model")
    if not model_id:
        raise ValueError("mask needs `model:` so its spans match the training rendering")
    judge = str(cfg.get("judge", "anthropic/claude-sonnet-5"))
    workers = int(cfg.get("workers", 8))

    judged, _untouched = candidates(records, cfg)
    verdicts = interpret_mod.detect(
        judged, prop.label, prop.detector, channel=prop.channel,
        **block(cfg, "detector"))
    flagged = {v["record_id"]: v for v in verdicts if v.get("exhibits")}
    targets = [r for r in judged if r.record_id in flagged]
    print(f">>> detector flagged {len(targets)}/{len(judged)} rows for {prop.label!r}")

    # Only the candidates need rendering; rendering 10,000 replay rows to mask 716 is
    # minutes of tokenizer work for nothing.
    rendered, _tokenizer, _profile = _render(targets, str(model_id))
    # `description_override` is where a per-property wording lives: the interpreter's
    # description names the move for a reader, but a span selector often needs it spelled
    # out in the corpus's own vocabulary. That belongs in the config, not in this file.
    description = str(cfg.get("description_override") or prop.description)
    client = OpenRouterClient()
    lock = threading.Lock()
    spans_by_id: dict[str, list[tuple[int, int]]] = {}
    failures: list[dict] = []

    def run(i: int) -> None:
        record = targets[i]
        text = rendered[record.record_id]
        try:
            region = _think_region(text)
        except ValueError as exc:
            with lock:
                failures.append({"record_id": record.record_id, "error": str(exc)})
            return
        evidence = flagged[record.record_id].get("evidence")
        system = SELECT_SYSTEM.format(
            label=prop.label, description=description,
            evidence_note=f", quoting this: {evidence!r}" if evidence else "")
        try:
            result = client.chat(
                model=judge, temperature=0.0, max_tokens=2000,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content":
                           f"<reasoning>\n{text[region[0]:region[1]]}\n</reasoning>"}])
            quoted = extract_json(result.content).get("spans", [])
            located = [_locate(text, s, region) for s in quoted]
        except Exception as exc:  # noqa: BLE001 - one bad row must not kill the pass
            with lock:
                failures.append({"record_id": record.record_id,
                                 "error": f"{type(exc).__name__}: {exc}"[:300]})
            return
        with lock:
            if located:
                spans_by_id[record.record_id] = located

    if targets:
        map_threaded(run, len(targets), max_workers=workers, desc="selecting spans")

    rows = []
    for record in records:
        row = dict(record.raw)
        spans = spans_by_id.get(record.record_id)
        # Every row carries the column, empty where nothing was masked: an absent column on
        # some rows and not others makes the dataset's schema depend on the ablation's hit
        # rate, and `datasets` would pad it with None anyway.
        row["mask_spans"] = [list(s) for s in (spans or [])]
        row["mask_property"] = prop.property_id if spans else None
        row["mask_render_model"] = str(model_id) if spans else None
        rows.append(row)

    return AblationResult(
        kind=KIND, property_id=prop.property_id, rows=rows,
        changed_ids=sorted(spans_by_id),
        detected_ids=sorted(flagged),
        report={"judge": judge, "render_model": str(model_id),
                "n_spans": sum(len(s) for s in spans_by_id.values()),
                "n_span_failures": len(failures),
                "span_failures": failures[:20],
                "detector_errors": sum(1 for v in verdicts
                                       if v.get("exhibits") is None)})
