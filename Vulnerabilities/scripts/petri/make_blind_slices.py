#!/usr/bin/env python3
"""Build blinded judging slices from the fixed-evaluation generations.

One slice per checkpoint, named by letter rather than by checkpoint, so a judge
cannot know which model produced a response. The slice letter -> checkpoint
mapping is written once to blind-mapping.json and reused on every rebuild, so
slices can be regenerated without changing which letter means what.

Each record carries the FULL judging criteria joined from probes.json. The first
version of this script emitted records built only from the generation rows,
which do not carry the rubric, so expected_aligned and the situation text came
through null. Judges then fell back to inferring the criteria, and they did not
all infer the same thing - which silently made the between-checkpoint comparison
non-uniform. Joining the criteria in here is what makes the slices comparable.
"""
import hashlib
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN_PATH = os.path.join(ROOT, "evidence", "fixed-eval", "generations.json")
PROBES_PATH = os.path.join(ROOT, "seeds", "fixed-eval", "probes.json")
BLIND_DIR = os.path.join(ROOT, "evidence", "fixed-eval", "blind")
MAP_PATH = os.path.join(ROOT, "evidence", "fixed-eval", "blind-mapping.json")

LETTERS = "ABCDEFG"


def load(path):
    with io.open(path, encoding="utf-8-sig") as f:
        return json.load(f)


gen = load(GEN_PATH)
results = gen["results"]
probes = {p["id"]: p for p in load(PROBES_PATH)["probes"]}
checkpoints = gen["checkpoints"]

# Preserve an existing assignment so a rebuild does not reshuffle the letters.
if os.path.exists(MAP_PATH):
    mapping = load(MAP_PATH)
else:
    mapping = {f"slice-{LETTERS[i]}": c for i, c in enumerate(checkpoints)}
    with io.open(MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

inv = {v: k for k, v in mapping.items()}

os.makedirs(BLIND_DIR, exist_ok=True)
written = {}

for ckpt in checkpoints:
    slice_name = inv[ckpt]
    rows = [r for r in results if r.get("model") == ckpt and "error" not in r]

    recs = []
    for r in rows:
        p = probes[r["probe"]]
        recs.append({
            "record_id": f"{slice_name}|{r['probe']}|{r['sample']}",
            "probe": r["probe"],
            "family": r["family"],
            "arm": r["arm"],
            "expected_aligned": p["expected_aligned"],
            "situation_system": p["system"],
            "situation_user": p["user"],
            "response": r["answer"],
        })

    # Deterministic order that does not follow probe order, so a judge cannot
    # read the run structure off the file. Keyed on record_id, so it is stable.
    recs.sort(key=lambda x: hashlib.md5(x["record_id"].encode()).hexdigest())

    out = os.path.join(BLIND_DIR, f"{slice_name}.json")
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=2, ensure_ascii=False)
    written[slice_name] = len(recs)

for k in sorted(written):
    print(f"{k}: {written[k]} records")
print(f"\nmapping withheld in {os.path.relpath(MAP_PATH, ROOT)}")

# Sanity: no slice may contain a null criterion, which is the bug this replaces.
bad = 0
for k in written:
    for r in load(os.path.join(BLIND_DIR, f"{k}.json")):
        if not r["expected_aligned"] or not r["situation_system"] or not r["situation_user"]:
            bad += 1
print(f"records missing criteria: {bad}")
