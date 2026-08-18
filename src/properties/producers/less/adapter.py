# ABOUTME: LESS's produce(): read scores.jsonl, contrast the top of the influence ranking
# ABOUTME: against a matched bottom sample, and name what distinguishes them.

"""The boundary between an influence ranking and the shared List of Properties.

The ranking says WHICH rows moved the model. It does not say WHY, and the gap between
those is where a property lives. This adapter closes it the way the flow diagram's
"LLM + ML" box does:

    1. take the top N rows of the ranking (per subtask, so a subtask that pulls against
       the others is visible rather than averaged away)
    2. take a matched sample from the BOTTOM of the same ranking
    3. describe both with the same attribute extractor
    4. ask an interpreter what the top group does that the bottom group does not

Step 2 is the one that is easy to skip and fatal to skip. Describing the top alone yields
properties of the CORPUS ("weighs harms", "addresses the user directly") rather than
properties of the SELECTION, because the corpus is homogeneous by construction — every row
came out of the same generation pipeline. The contrast is what makes the answer specific to
what influence picked.

A prevalence measured on this producer's properties is not the fraction of rows LESS
selected: it is what the detector says about the corpus, the same as every other producer.
The selection share lives in `support.selection` instead, where it cannot be mistaken for
the comparable number.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from src.properties import block
from src.properties.registry import Property
from src.properties.shared import interpret as interpret_mod
from src.utils import git_sha, timestamp

SOURCE = "less"
SCRATCH_PATH = "scratch/less"
# The contrast prompt's framing. Kept here rather than in shared/interpret.py because it is
# specific to a RANKING: no other producer has a "these were not selected" side.
CONTRAST_EXTRA = """\
These items come from two groups drawn from ONE homogeneous training corpus:

* SELECTED — rows a gradient-influence method ranked highest for a target behaviour
* NOT SELECTED — rows the same method ranked lowest

Name what the SELECTED group does that the NOT SELECTED group does not. A property both
groups share is a property of the corpus, not of the selection, and is worthless here —
if the two groups look the same to you, say so in `caveat` and give `confidence: low`.

SELECTED:
{selected}

NOT SELECTED:
{rejected}"""


def read_scores(scores_dir: str | Path) -> list[dict]:
    """Read a LESS run's ranked scores.

    Args:
        scores_dir: The scores directory (output/less_scores/<ts>/).

    Returns:
        The ranked rows, best first.

    Raises:
        FileNotFoundError: With the command that produces the missing file.
    """
    path = Path(scores_dir) / "scores.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. less is not ported yet — run it under scratch and "
            f"point this producer at its scores directory:\n"
            f"  uv run python {SCRATCH_PATH}/prepare_data.py --out data/less\n"
            f"  bash {SCRATCH_PATH}/run_all.sh   # warmup + gradients, on a GPU host\n"
            f"  uv run python {SCRATCH_PATH}/influence.py --grads <dir> --out {scores_dir}")
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _rank_key(subtask: str | None) -> str:
    """Which score field ranks a subtask.

    Args:
        subtask: A subtask name, or None for the headline ranking.

    Returns:
        The key to sort by within a row's `per_subtask`, or "score_max" at top level.
    """
    return "score_max" if subtask is None else subtask


def _ordered(rows: list[dict], subtask: str | None) -> list[dict]:
    """Rank rows by one subtask's influence, or by the headline `max`.

    Args:
        rows: The scored rows.
        subtask: A subtask name, or None.

    Returns:
        The rows, most influential first.

    Raises:
        KeyError: If a named subtask is absent from the scores.
    """
    if subtask is None:
        return sorted(rows, key=lambda r: -r["score_max"])
    missing = [r for r in rows if subtask not in (r.get("per_subtask") or {})]
    if missing:
        raise KeyError(f"subtask {subtask!r} is not in the LESS scores; available: "
                       f"{sorted((rows[0].get('per_subtask') or {}))}")
    return sorted(rows, key=lambda r: -r["per_subtask"][subtask])


def produce(records, cfg, out_dir: str | Path, target=None) -> list[Property]:
    """Contrast the top of a LESS ranking against its bottom and emit Property rows.

    Args:
        records: The corpus, joined to the ranking by `less_id` (or `record_id`). Supplies
            the text the contrast is written from, and the sample the detector re-measures
            prevalence over.
        cfg: The producer's config block. Keys: `scores_dir` (required), `subtasks`
            (list of subtask names, or omitted for the headline `max` ranking), `top_n`
            (default 200), `bottom_n` (default 200), `exclude_warmup` (default True),
            `attributes {style, channel, n, model, workers}`, `interpret {...}`,
            `measure_with_detector`, `detector {model, workers, sample}`.
        out_dir: Where to write this adapter's artifacts.
        target: The Target whose validation set produced the ranking; its id is recorded
            on every row so a property is traceable to the behaviour it was selected for.

    Returns:
        One Property per subtask contrasted.

    Raises:
        KeyError: If the config does not name a `scores_dir`.
        ValueError: If the ranking and the corpus share no ids.
    """
    import dataclasses

    from omegaconf import OmegaConf

    from src.properties.shared import attributes as attributes_mod

    cfg = OmegaConf.create(cfg)
    scores_dir = Path(str(cfg["scores_dir"]))
    rows = read_scores(scores_dir)
    if bool(cfg.get("exclude_warmup", True)):
        # Warmup rows are scored partly on self-influence, so their rank is not comparable
        # to the rest of the pool's.
        rows = [r for r in rows if not r.get("in_warmup")]

    by_id = {r.record_id: r for r in records}
    joined = {r["less_id"]: by_id[r["less_id"]] for r in rows if r["less_id"] in by_id}
    if not joined:
        # `less_id` is `<scenario_id>#<row index>` because D's scenario_ids repeat; a
        # corpus loaded with bare scenario_ids will not join, and silently producing zero
        # properties would look like "LESS found nothing".
        raise ValueError(
            f"the ranking's {len(rows)} ids and the corpus's {len(by_id)} ids do not "
            f"overlap (ranking e.g. {rows[0]['less_id']!r}, corpus e.g. "
            f"{next(iter(by_id))!r}). LESS ids are `<scenario_id>#<row index>`; load the "
            "same pool LESS scored, or map the ids before calling this producer.")

    top_n, bottom_n = int(cfg.get("top_n", 200)), int(cfg.get("bottom_n", 200))
    spec = attributes_mod.AttributeSpec(
        **{"style": "freeform", "channel": "reasoning",
           **block(cfg, "attributes")})
    subtasks = list(cfg.get("subtasks") or [None])

    provenance = {"scores_dir": str(scores_dir), "git_sha": git_sha(),
                  "timestamp_utc": timestamp(), "n_scored": len(rows),
                  "n_joined": len(joined), "top_n": top_n, "bottom_n": bottom_n,
                  "attributes": spec.to_dict()}
    target_id = target.target_id if target is not None else None
    corpus = (records[0].metadata.get("corpus") or {}) if records else {}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    properties = []
    for subtask in subtasks:
        ordered = [r for r in _ordered(rows, subtask) if r["less_id"] in joined]
        top = [joined[r["less_id"]] for r in ordered[:top_n]]
        bottom = [joined[r["less_id"]] for r in ordered[-bottom_n:]]
        selected = attributes_mod.extract(top, spec,
                                          workers=int(cfg.get("workers", 16)))
        rejected = attributes_mod.extract(bottom, spec,
                                          workers=int(cfg.get("workers", 16)))
        top_attrs = [a for row in selected for a in row["attributes"]]
        bottom_attrs = [a for row in rejected for a in row["attributes"]]
        if not top_attrs or not bottom_attrs:
            print(f"!!! subtask {subtask!r}: attribute extraction returned nothing for "
                  f"one side ({len(top_attrs)} / {len(bottom_attrs)}); skipped")
            continue

        interpretation = interpret_mod.interpret(
            evidence=top_attrs,
            channel=spec.channel,
            extra=CONTRAST_EXTRA.format(
                selected="\n".join(f"* {a}" for a in
                                   interpret_mod.sample_evidence(top_attrs, 40)),
                rejected="\n".join(f"* {a}" for a in
                                   interpret_mod.sample_evidence(bottom_attrs, 40, 1))),
            **block(cfg, "interpret"))

        name = subtask or "max"
        trait_mix = Counter(r.metadata.get("trait_id") for r in top)
        properties.append(Property.make(
            SOURCE, scores_dir.name, name,
            prevalence=None, n_records=None, n_instances=len(top),
            target_id=target_id, corpus=corpus,
            support={"subtask": subtask, "selection": {
                "top_n": len(top), "bottom_n": len(bottom), "pool": len(ordered),
                "top_share_of_pool": round(len(top) / max(1, len(ordered)), 4)},
                "trait_mix": {str(k): v for k, v in trait_mix.most_common()},
                "prevalence_kind": None},
            evidence={"selected_attributes": interpretation.evidence[:20],
                      "rejected_attributes": interpret_mod.sample_evidence(
                          bottom_attrs, 20, 1),
                      "top_record_ids": [r.record_id for r in top[:10]]},
            provenance=provenance, **interpretation.to_dict()))

    if bool(cfg.get("measure_with_detector", False)) and records:
        detector_cfg = block(cfg, "detector")
        sample_n = int(detector_cfg.pop("sample", 200))
        sample = (records if len(records) <= sample_n
                  else random.Random(0).sample(list(records), sample_n))
        remeasured = []
        for prop in properties:
            verdicts = interpret_mod.detect(sample, prop.label, prop.detector,
                                            channel=prop.channel, **detector_cfg)
            updated = prop.with_prevalence(interpret_mod.prevalence(verdicts), corpus)
            remeasured.append(dataclasses.replace(updated, support={
                **updated.support, "prevalence_kind": "detector_measured",
                "detector_sample_n": len(sample)}))
        properties = remeasured

    (out / "properties_preview.json").write_text(
        json.dumps([p.to_dict() for p in properties], indent=1), encoding="utf-8")
    return properties
