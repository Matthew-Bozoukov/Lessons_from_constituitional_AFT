"""Does suppressing the judge's reasoning change what the detector measures?

Three conditions over the SAME records and the SAME 48 real detectors from the published
da716 run:

    batched_off   one call per record, every rubric, reasoning OFF, max_tokens=8000
    batched_on    one call per record, every rubric, reasoning ON,  max_tokens=8000
    single_on     one call per (record, property), reasoning ON, max_tokens=800
                  -- the pre-existing path, the closest thing here to a trusted instrument

What matters is not whether off and on agree with each other, but which of them agrees with
`single_on`. Cheapness is only admissible if it does not move the reading.

Writes its report to a file: a piped stdout was lost to buffering once already.
"""
import json
import pathlib
import random
import sys
import time

sys.path.insert(0, ".")

from src.properties.registry import Property  # noqa: E402
from src.properties.shared import interpret as interpret_mod  # noqa: E402
from src.properties.sources import odcv_rollouts as o  # noqa: E402

REPORT = pathlib.Path(__file__).with_name("ab_reasoning_report.txt")
LINES = []


def emit(line=""):
    LINES.append(line)
    REPORT.write_text("\n".join(LINES) + "\n", encoding="utf-8")


N_RECORDS = 20
props = [Property.from_dict(json.loads(line))
         for line in open("docs/properties/odcv_da716/properties.jsonl", encoding="utf-8")
         if line.strip()]
records = o.load(runs=[
    {"run_dir": "output/odcv_bench/da716_5pct", "arm": "da716_5pct"},
    {"run_dir": "output/odcv_bench/numina_control_0pct", "arm": "numina_control_0pct"}])
by_arm = {}
for r in records:
    by_arm.setdefault(r.metadata["arm"], []).append(r)
rng = random.Random(0)
sample = [r for rows in by_arm.values() for r in rng.sample(rows, N_RECORDS // 2)]
emit(f"{len(props)} real detectors x {len(sample)} records "
     f"({len(sample) * len(props)} cells)")


def _with_reasoning(setting, fn):
    saved = interpret_mod.NO_REASONING
    interpret_mod.NO_REASONING = setting
    started = time.time()
    try:
        return fn(), time.time() - started
    finally:
        interpret_mod.NO_REASONING = saved


OFF = {"reasoning": {"enabled": False}}
ON = {}

results, timings = {}, {}
results["batched_off"], timings["batched_off"] = _with_reasoning(
    OFF, lambda: interpret_mod.detect_many(sample, props, channel="reasoning", workers=8,
                                           max_tokens=8000))
emit(f"  batched_off done in {timings['batched_off']:.0f}s")
results["batched_on"], timings["batched_on"] = _with_reasoning(
    ON, lambda: interpret_mod.detect_many(sample, props, channel="reasoning", workers=8,
                                          max_tokens=8000))
emit(f"  batched_on  done in {timings['batched_on']:.0f}s")
results["single_on"], timings["single_on"] = _with_reasoning(
    ON, lambda: {p.property_id: interpret_mod.detect(sample, p.label, p.detector,
                                                     channel="reasoning", workers=16)
                 for p in props})
emit(f"  single_on   done in {timings['single_on']:.0f}s")


def cells(out):
    return {(pid, v["record_id"]): v["exhibits"]
            for pid, verdicts in out.items() for v in verdicts}


def agreement(a, b):
    ca, cb = cells(a), cells(b)
    both = [(ca[k], cb[k]) for k in ca
            if k in cb and ca[k] is not None and cb[k] is not None]
    return len(both), (sum(1 for x, y in both if x == y) / len(both) if both else None)


def prevalence(out):
    judged = [v for v in cells(out).values() if v is not None]
    return sum(judged) / len(judged) if judged else None


emit()
emit(f"{'condition':14s} {'seconds':>8s} {'prevalence':>11s} {'unjudged':>9s}")
for name, out in results.items():
    unjudged = sum(1 for v in cells(out).values() if v is None)
    emit(f"{name:14s} {timings[name]:8.0f} {(prevalence(out) or 0):11.1%} {unjudged:9d}")

emit()
emit("agreement between conditions (cells where both answered):")
for a, b in (("batched_off", "single_on"), ("batched_on", "single_on"),
             ("batched_off", "batched_on")):
    n, rate = agreement(results[a], results[b])
    emit(f"  {a:12s} vs {b:12s}  {rate:.1%} of {n} cells")

rows = []
for p in props:
    per = {}
    for name in results:
        vals = [v["exhibits"] for v in results[name][p.property_id]
                if v["exhibits"] is not None]
        if not vals:
            break
        per[name] = sum(vals) / len(vals)
    if len(per) == 3:
        rows.append((abs(per["batched_off"] - per["single_on"]), p.label,
                     per["batched_off"], per["batched_on"], per["single_on"]))
rows.sort(reverse=True)
emit()
emit("per-property prevalence, the 12 where batched_off and single_on differ most:")
emit(f"  {'gap':>5s} {'off':>5s} {'on':>5s} {'single':>6s}  property")
for gap, label, off, on, single in rows[:12]:
    emit(f"  {gap:5.0%} {off:5.0%} {on:5.0%} {single:6.0%}  {label[:56]}")

mean_gap_off = sum(r[0] for r in rows) / len(rows) if rows else 0
mean_gap_on = (sum(abs(r[3] - r[4]) for r in rows) / len(rows)) if rows else 0
emit()
emit(f"mean |prevalence gap vs single_on|: batched_off {mean_gap_off:.1%}, "
     f"batched_on {mean_gap_on:.1%}")
emit("DONE")
