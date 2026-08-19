# ABOUTME: TURF — trace one case's behaviour back through the training data to the query
# ABOUTME: and reasoning properties that co-occur with responses like it.

"""TURF, adapted (arXiv 2602.05910, *Chunky Post-Training*).

The only producer that starts from an OUTCOME rather than from the corpus. Given a case —
a response we liked or disliked, from an eval — it asks which properties of the TRAINING
data co-occur with training responses similar to that one:

    1. offline: extract query + reasoning attributes (the trigger side) and response
       attributes (the behaviour side) from every training row; cluster the trigger side
    2. blind judge: 10 "The response..." attributes of the case, with no rubric in view
    3. informed judge: pick 3 VERBATIM cruxes against the rubric, guarded so a crux cannot
       merely restate one of the dataset's own generation styles
    4. retrieve the k nearest training response attributes to each crux
    5. route those hits back to their rows' trigger clusters and count the hits
    6. the case's own trigger attribute nearest the winning cluster is the candidate

So its properties come with a claim the other producers cannot make: "these training rows
are followed by responses that are good/bad in the same way as this case". That is a
correlation, not a cause — which is precisely why the property then goes to `ablation/`.

**Port status: the producer code still lives in `scratch/turf/`.** What is here is the
boundary: `produce()` reads a trace result and turns its ranked trigger clusters into
Property rows. Seven files move in when the port lands (`cases`, `prompts`, `extract`,
`index`, `trace`, `unrender`, `plot_turf`), while `common.py` does not: its embedding, its
k-means and its row parsing are already `shared/embed.py`, `shared/grouping.py` and
`sources/base.py`.

A trace result holds, per crux, a ranked table of trigger clusters with their hit counts
and SURF-style one-sentence summaries. Those summaries are what becomes a property: the
summary is the evidence, `shared/interpret.py` turns it into a label and a detector, and
the hit share becomes the row's support. Two things that are NOT smoothed over:

* **`style_echo`.** TURF flags a cluster whose summary merely restates one of the dataset's
  own generation styles. Such a cluster is not a discovered property — it is the corpus
  describing itself — so those rows are DROPPED by default (`keep_style_echoes: true` keeps
  them, flagged, for someone who wants to look).
* **Hits are not prevalence.** A cluster's hit count is retrievals out of k, which measures
  association with the case, not how much of the corpus carries the property. So
  `prevalence` is left None unless `measure_with_detector` runs the detector over the
  corpus, and `support.hit_share` carries the association separately. A merger that sorted
  TURF rows by hit share as if it were prevalence would be comparing different quantities.

Its rubrics move too, and the rule they carry moves with them: a rubric describes the
case's BEHAVIOUR — principle, context, what satisfying and violating look like — and never
candidate properties. `sources/targets.py::from_rubric` enforces that rather than trusting
it, because a rubric that names properties makes the discovery circular.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.properties import block
from src.properties.registry import Property
from src.properties.shared import interpret as interpret_mod
from src.utils import git_sha, timestamp

SOURCE = "turf"
SCRATCH_PATH = "scratch/turf"


def read_trace(trace_dir: str | Path) -> dict:
    """Read one TURF trace run's result.

    Args:
        trace_dir: A trace run directory (output/turf/traces/<ts>_<case-id>/).

    Returns:
        The parsed trace_result.json.

    Raises:
        FileNotFoundError: With the command that produces the missing file.
    """
    path = Path(trace_dir) / "trace_result.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. turf is not ported yet — run it under scratch and "
            f"point this producer at its trace directory:\n"
            f"  uv run python {SCRATCH_PATH}/extract.py --dataset <hf-id> --out <index>\n"
            f"  uv run python {SCRATCH_PATH}/index.py   --dir <index> --k 1000\n"
            f"  uv run python {SCRATCH_PATH}/trace.py   --case <case.json> "
            f"--rubric <rubric.yaml> --index <index>")
    return json.loads(path.read_text(encoding="utf-8"))


def produce(records, cfg, out_dir: str | Path, target=None) -> list[Property]:
    """Turn a TURF trace's trigger clusters into Property rows.

    Args:
        records: The corpus, used only when `measure_with_detector` is set.
        cfg: The producer's config block. Keys: `trace_dir` (required), `top_per_crux`
            (default 3), `keep_style_echoes` (default False), `interpret {...}`,
            `measure_with_detector`, `detector {model, workers, sample}`.
        out_dir: Where to write this producer's preview artifacts.
        target: The Target the trace was run against, when the caller has one. Only its
            id is used — the trace already ran against its rubric.

    Returns:
        Property rows, most associated first.

    Raises:
        KeyError: If the config does not name a `trace_dir`.
    """
    import dataclasses
    import random

    from omegaconf import OmegaConf

    cfg = OmegaConf.create(cfg)
    trace_dir = Path(str(cfg["trace_dir"]))
    result = read_trace(trace_dir)
    top_n = int(cfg.get("top_per_crux", 3))
    keep_echoes = bool(cfg.get("keep_style_echoes", False))

    # One cluster can win under more than one crux; keep the best hit count and record
    # every crux it answered, rather than emitting the same property twice.
    best: dict[int, dict] = {}
    for per_crux in result.get("per_crux", []):
        for entry in per_crux.get("clusters", [])[:top_n]:
            if entry.get("style_echo") and not keep_echoes:
                continue
            cluster = int(entry["cluster"])
            current = best.get(cluster)
            if current is None or entry["hits"] > current["hits"]:
                best[cluster] = {**entry, "cruxes": [per_crux["crux"]]}
            elif per_crux["crux"] not in current["cruxes"]:
                current["cruxes"].append(per_crux["crux"])

    dropped = sum(1 for pc in result.get("per_crux", [])
                  for e in pc.get("clusters", [])[:top_n] if e.get("style_echo"))
    if dropped and not keep_echoes:
        print(f">>> dropped {dropped} style-echo clusters — a cluster that restates one "
              "of the corpus's own generation styles is the corpus describing itself, "
              "not a discovered property (keep_style_echoes: true to see them)")

    interpretations = interpret_mod.interpret_many(
        {c: [best[c]["summary"]] for c in best},
        channel="reasoning",
        extra=(f"This cluster was retrieved because training rows in it are followed by "
               f"responses similar to a case that {result.get('polarity', 'satisfy')}s "
               f"a behaviour rubric. Name the property of the QUERY-AND-REASONING side."),
        **block(cfg, "interpret"))

    provenance = {"trace_dir": str(trace_dir), "git_sha": git_sha(),
                  "timestamp_utc": timestamp(),
                  "producer_git_sha": result.get("git_sha"),
                  "case_id": result.get("case_id"), "polarity": result.get("polarity"),
                  "index": result.get("index"), "k": result.get("k"),
                  "extractor_model": result.get("model")}
    target_id = (target.target_id if target is not None
                 else result.get("case_id"))

    properties = []
    for cluster, entry in best.items():
        interpretation = interpretations.get(cluster)
        if interpretation is None:
            continue
        properties.append(Property.make(
            SOURCE, trace_dir.name, f"t{cluster:04d}",
            # Deliberately None: hits/k is association with the case, not corpus share.
            prevalence=None, n_records=None, n_instances=int(entry["hits"]),
            target_id=target_id,
            corpus={"index": (result.get("index") or {}).get("source_dataset")},
            support={"cluster": cluster, "hits": int(entry["hits"]),
                     "of": int(entry["of"]),
                     "hit_share": round(entry["hits"] / max(1, entry["of"]), 4),
                     "share_reasoning": entry.get("share_reasoning"),
                     "style_echo": bool(entry.get("style_echo")),
                     "producer_summary": entry["summary"],
                     "prevalence_kind": None},
            evidence={"cluster_summary": entry["summary"], "cruxes": entry["cruxes"]},
            provenance=provenance, **interpretation.to_dict()))

    if bool(cfg.get("measure_with_detector", False)) and records:
        detector_cfg = block(cfg, "detector")
        sample_n = int(detector_cfg.pop("sample", 200))
        sample = (records if len(records) <= sample_n
                  else random.Random(0).sample(list(records), sample_n))
        corpus = (records[0].metadata.get("corpus") or {})
        remeasured = []
        for prop in properties:
            verdicts = interpret_mod.detect(sample, prop.label, prop.detector,
                                            channel=prop.channel, **detector_cfg)
            updated = prop.with_prevalence(interpret_mod.prevalence(verdicts), corpus)
            remeasured.append(dataclasses.replace(updated, support={
                **updated.support, "prevalence_kind": "detector_measured",
                "detector_sample_n": len(sample)}))
        properties = remeasured

    properties.sort(key=lambda p: -(p.support.get("hits") or 0))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "properties_preview.json").write_text(
        json.dumps([p.to_dict() for p in properties], indent=1), encoding="utf-8")
    return properties
