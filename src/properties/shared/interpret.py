# ABOUTME: Evidence -> a label AND a detector rubric. The detector is what turns a named
# ABOUTME: cluster into something an ablation can act on and a verification can measure.

"""Naming a group, and writing the test that decides whether a record has it.

Every producer ends with a pile of evidence that belongs together — features in a cluster,
attributes near a centroid, the top of an influence ranking — and needs two things out of
it. A LABEL, so humans can talk about it. And a DETECTOR, so code can act on it.

The detector is the part that is easy to skip and expensive to skip. Without one, a
property is a sentence in a report: you cannot find the rows that have it, you cannot
rewrite them, and after training an ablated arm you cannot show that the property's
prevalence actually dropped. With one, all three are the same LLM-judge call:

    ablation/filter.py     which rows have it -> drop or split them
    ablation/mask.py       where in the reasoning it lives -> unsupervise those spans
    ablation/rewrite.py    rewrite it out, and re-run the detector to confirm
    ablation/verify.py     prevalence before vs after — did the intervention land?

So `interpret()` returns both, from one call, and refuses a reply that gives only the
label. The detector is written as a strict yes/no test on ONE record, with an explicit
"does not count" clause: the failure mode of an LLM-written detector is that it drifts
into "is this record good?", which every well-behaved record passes and which therefore
measures nothing.

Interpretation runs at temperature 0. Naming is not a place for sampling diversity: the
label is a name that other artifacts key on, and two runs over the same cluster should not
disagree about what it is called.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

INTERPRET_TEMPERATURE = 0.0
DEFAULT_MODEL = "anthropic/claude-sonnet-5"
# How many pieces of evidence the interpreter sees. SURF summarises from its top-100
# closest members; 50 is that halved, which is what TURF settled on and enough that a
# label is not driven by three outliers.
EVIDENCE_SAMPLE = 50

# What the interpreter is looking at. The two producers' evidence is not the same kind of
# thing, and telling the model it is reading one when it is reading the other is not a
# cosmetic error: "say what these descriptions have in common" and "say what these
# transcripts have in common" are different tasks, and the second needs the warning about
# subject matter that the first does not.
EVIDENCE_FRAMING = {
    "features": "short descriptions of what individual records in a corpus do. They were "
                "written independently, one record at a time, by a model that saw no "
                "cluster structure",
    "records": "excerpts of the records themselves — raw reasoning transcripts, grouped "
               "by how similar their text is. Be careful: text similarity tracks SUBJECT "
               "MATTER as strongly as behaviour, so if the only thing these share is a "
               "topic, say so in the caveat and give the property a low confidence rather "
               "than inventing a behavioural label for a topical cluster",
}

INTERPRET_SYSTEM = """\
You are naming a behavioural property of AI training data so that it can be ABLATED and \
the ablation measured.

You will see evidence drawn from one cluster: EVIDENCE_IS. Your job is to say what they \
have in common, and to write a test another model can apply to a SINGLE record to decide \
whether that record has this property.

Two things must be true of the label:
- It names a BEHAVIOUR or a MOVE the record makes, not a topic it is about. "Weighs \
likelihood against severity before deciding" is a property; "About medical questions" is a \
topic.
- It is specific enough that a reasonable person could disagree about whether a given \
record has it. A label everything passes is not a property.

Three things must be true of the detector:
- It decides yes/no about ONE record, from the record alone, with no comparison to others.
- It states what does NOT count, including the nearest thing this is easily confused with. \
This clause is the whole value of the detector: without it, detectors drift into "is this \
record any good?", which everything passes.
- It never mentions this cluster, this corpus, or how the property was found. A detector \
that says "records in this cluster" cannot be run on a rollout from a different model.

Return ONLY this JSON:
{"label": "<the property, under 10 words, sentence case>",
 "description": "<2-3 sentences: what the move is and why a record would make it>",
 "detector": "<the yes/no test, addressed to a judge, including the does-not-count clause>",
 "channel": "query" | "reasoning" | "response",
 "confidence": "high" | "medium" | "low",
 "caveat": "<one sentence: what would make this label wrong, or \\"\\" if nothing>"}"""

INTERPRET_USER = """\
Evidence from one cluster ({n_shown} of {n_total} items){outcome_note}:

{evidence}

{extra}"""

# Added to the user message when a producer knows how the records carrying this evidence
# turned out. Outcome is context for the label, never permission to write the detector in
# terms of it: "the response was judged harmful" is an outcome, not a property.
OUTCOME_NOTE = ", drawn from records whose judged outcome was: {outcome}"


@dataclass(frozen=True)
class Interpretation:
    """A named property plus the test that finds it.

    Attributes:
        label: The property, as a short phrase.
        description: What the move is, in 2-3 sentences.
        detector: A yes/no test a judge applies to one record.
        channel: Which channel the property lives in.
        confidence: The interpreter's own confidence in the label.
        caveat: What would make the label wrong, or "".
        evidence: The evidence strings the interpreter actually saw.
        model: The interpreting model.
    """

    label: str
    description: str
    detector: str
    channel: str
    confidence: str = "medium"
    caveat: str = ""
    evidence: list[str] = field(default_factory=list)
    model: str = DEFAULT_MODEL

    def to_dict(self) -> dict:
        """This interpretation as a plain dict.

        Returns:
            The json-safe record that becomes a property row's core fields.
        """
        return {"label": self.label, "description": self.description,
                "detector": self.detector, "channel": self.channel,
                "confidence": self.confidence, "caveat": self.caveat,
                "interpreter_model": self.model}


def sample_evidence(evidence: list[str], n: int = EVIDENCE_SAMPLE,
                    seed: int = 0) -> list[str]:
    """Take a reproducible sample of a group's evidence.

    Random rather than head-of-list: producers hand evidence over sorted by distance to
    the centroid, and labelling only the closest members describes the group's core while
    hiding how far it stretches.

    Args:
        evidence: All the group's evidence strings.
        n: How many to show the interpreter.
        seed: Sampling seed, so a rerun names the group the same way.

    Returns:
        The sample, or everything when there is less than `n`.
    """
    if len(evidence) <= n:
        return list(evidence)
    return random.Random(seed).sample(list(evidence), n)


def interpret(evidence: list[str], channel: str = "reasoning",
              outcome: str | None = None, extra: str = "",
              model: str = DEFAULT_MODEL, n_shown: int = EVIDENCE_SAMPLE,
              seed: int = 0, evidence_kind: str = "features",
              client=None) -> Interpretation:
    """Name one group and write its detector.

    Args:
        evidence: The group's evidence strings (features, attributes, excerpts).
        channel: Which channel this group's evidence describes; the interpreter may
            override it in its reply, and its answer wins — it saw the evidence.
        evidence_kind: What those strings ARE — "features" (a model's descriptions of
            records) or "records" (excerpts of the records themselves). Chooses the
            framing the interpreter is given; see EVIDENCE_FRAMING.
        outcome: How the records carrying this evidence turned out ("judged harmful",
            "resisted"), or None. Context for the label, never grounds for a detector.
        extra: Anything else the producer wants the interpreter to know (e.g. which target
            behaviour this group was retrieved against).
        model: OpenRouter model.
        n_shown: Evidence items to show.
        seed: Sampling seed.
        client: An OpenRouterClient; one is built when omitted.

    Returns:
        The Interpretation.

    Raises:
        ValueError: If `evidence` is empty, the evidence kind is unknown, or the reply
            omits a label or a detector — a property without a detector cannot be ablated
            or verified, so it is not a property this module will export.
    """
    from src.endpoints.openrouter import OpenRouterClient
    from src.utils import extract_json

    if not evidence:
        raise ValueError("cannot interpret an empty group")
    if evidence_kind not in EVIDENCE_FRAMING:
        raise ValueError(f"evidence_kind must be one of {sorted(EVIDENCE_FRAMING)}, "
                         f"got {evidence_kind!r}")
    client = client or OpenRouterClient()
    shown = sample_evidence(evidence, n_shown, seed)
    user = INTERPRET_USER.format(
        n_shown=len(shown), n_total=len(evidence),
        outcome_note=OUTCOME_NOTE.format(outcome=outcome) if outcome else "",
        evidence="\n".join(f"* {e}" for e in shown), extra=extra).strip()
    # A literal replace, not .format(): the prompt ends in a JSON contract whose braces
    # would have to be doubled, and a doubled brace in a prompt is a bug waiting to ship.
    system = INTERPRET_SYSTEM.replace("EVIDENCE_IS", EVIDENCE_FRAMING[evidence_kind])
    result = client.chat(model=model, temperature=INTERPRET_TEMPERATURE, max_tokens=2000,
                         messages=[{"role": "system", "content": system},
                                   {"role": "user", "content": user}])
    parsed = extract_json(result.content)
    missing = [k for k in ("label", "detector") if not str(parsed.get(k, "")).strip()]
    if missing:
        raise ValueError(
            f"interpreter returned no {missing}: a property with no detector cannot be "
            f"ablated or verified.\n{result.content[:400]}")
    return Interpretation(
        label=str(parsed["label"]).strip(),
        description=str(parsed.get("description", "")).strip(),
        detector=str(parsed["detector"]).strip(),
        channel=str(parsed.get("channel") or channel).strip(),
        confidence=str(parsed.get("confidence", "medium")).strip(),
        caveat=str(parsed.get("caveat", "")).strip(),
        evidence=shown, model=model)


def interpret_many(groups: dict[int, list[str]], workers: int = 8,
                   client=None, **kwargs) -> dict[int, Interpretation]:
    """Interpret several groups concurrently.

    Args:
        groups: group id -> its evidence strings.
        workers: Concurrent requests.
        client: An OpenRouterClient shared across the calls.
        **kwargs: Passed to `interpret`.

    Returns:
        group id -> Interpretation, omitting any group whose interpretation failed. The
        failures are printed rather than raised: one unnameable cluster out of eighty
        should cost that cluster, not the run.
    """
    from src.endpoints.openrouter import OpenRouterClient, map_threaded

    client = client or OpenRouterClient()
    ids = sorted(groups)

    def run(i: int) -> tuple[int, Interpretation | str]:
        try:
            return ids[i], interpret(groups[ids[i]], client=client, **kwargs)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            return ids[i], f"{type(exc).__name__}: {exc}"

    out: dict[int, Interpretation] = {}
    for group_id, result in map_threaded(run, len(ids), max_workers=workers,
                                         desc="interpreting groups"):
        if isinstance(result, Interpretation):
            out[group_id] = result
        else:
            print(f"!!! group {group_id} not interpreted: {result}")
    return out


# --- running a detector ---------------------------------------------------------------

DETECT_SYSTEM = """\
You decide whether ONE record exhibits ONE specific property. You are not judging quality, \
helpfulness, or alignment — only presence.

PROPERTY: {label}

TEST:
{detector}

Apply the test as written, including anything it says does not count. When the record is \
genuinely borderline, answer no: a detector that says yes to borderline cases inflates \
every prevalence measured with it, and prevalence is the number this whole pipeline turns \
on.

Return ONLY this JSON:
{{"exhibits": true | false, "evidence": "<a short verbatim quote from the record, or \
\\"\\" if false>", "note": "<one clause on why, especially if borderline>"}}"""


def detect(records, label: str, detector: str, channel: str = "reasoning",
           model: str = DEFAULT_MODEL, workers: int = 16, client=None) -> list[dict]:
    """Run one property's detector over records.

    THE measurement primitive of this module. `prevalence` on a property row, the row set
    an ablation edits, and the before/after in `ablation/verify.py` are all this function.

    Args:
        records: Records to test.
        label: The property's label.
        detector: The property's detector rubric.
        channel: Which channel the judge reads.
        model: OpenRouter judge model.
        workers: Concurrent requests.
        client: An OpenRouterClient; one is built when omitted.

    Returns:
        One {"record_id", "exhibits", "evidence", "note"} per record, in order. A record
        whose judgement fails carries `exhibits: None` and an `error`, and callers must
        exclude those from the denominator rather than counting them as absent.
    """
    from src.endpoints.openrouter import OpenRouterClient, map_threaded
    from src.utils import extract_json

    client = client or OpenRouterClient()
    system = DETECT_SYSTEM.format(label=label, detector=detector)

    def run(i: int) -> dict:
        record = records[i]
        try:
            result = client.chat(
                model=model, temperature=0.0, max_tokens=800,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content":
                           f"<record>\n{record.channel(channel)}\n</record>"}])
            parsed = extract_json(result.content)
            return {"record_id": record.record_id,
                    "exhibits": bool(parsed.get("exhibits")),
                    "evidence": str(parsed.get("evidence", ""))[:600],
                    "note": str(parsed.get("note", ""))[:300]}
        except Exception as exc:  # noqa: BLE001 - recorded per record
            return {"record_id": record.record_id, "exhibits": None, "evidence": "",
                    "note": "", "error": f"{type(exc).__name__}: {exc}"[:300]}

    return map_threaded(run, len(records), max_workers=workers,
                        desc=f"detecting: {label[:40]}")


def prevalence(verdicts: list[dict]) -> dict:
    """Summarise detector verdicts into a prevalence with a confidence interval.

    Args:
        verdicts: Rows from `detect`.

    Returns:
        {"n", "hits", "prevalence", "ci_low", "ci_high", "n_errors"}. Errors are excluded
        from `n`, so a run where the judge failed on half the records reports a prevalence
        over the half it actually judged and says so in `n_errors`.
    """
    from src.utils import wilson

    judged = [v for v in verdicts if v.get("exhibits") is not None]
    hits = sum(1 for v in judged if v["exhibits"])
    low, high = wilson(hits, len(judged))
    return {"n": len(judged), "hits": hits,
            "prevalence": round(hits / len(judged), 4) if judged else None,
            "ci_low": low, "ci_high": high,
            "n_errors": len(verdicts) - len(judged)}
