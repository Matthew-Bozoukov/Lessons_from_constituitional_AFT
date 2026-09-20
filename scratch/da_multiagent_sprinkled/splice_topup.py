# ABOUTME: One-off splice of same-config top-up runs (id_prefix b/c.., hf_push=false) into the main
# ABOUTME: da-multiagent-sprinkled run dir, so one `--resume` retries the losses and publishes one corpus.
"""Splice top-up runs into the main run directory.

Copied unchanged (below this docstring) from scratch/da_multiagent/splice_topup.py on branch
worktree-kunwar-multiagent-trait, where it built the 2026-09-15 t10 corpus; that branch is
not on main, and nothing outside scratch/ may be imported from scratch/.

Run: uv run python <this file> <main run dir> <config> <top-up run dir> [<top-up run dir> ...]

1. Refuses if the main dir was already spliced (run1_final/ exists) or any run is unfinished.
2. For each top-up in order, drops scenarios whose situation is within the config's
   reject_cosine of any scenario already kept (the main run's, then earlier top-ups') -- the
   rule the generation gate applies within one run, applied across runs.
3. Moves the main run's stage 5-8 finals and manifest.json into run1_final/ (the manifest is
   also kept at the root as manifest_run1.json); the .partial.jsonl checkpoints stay.
4. Appends the kept top-up rows to the main stage 2-4 snapshots and stage 5-7 checkpoints.
5. Writes topup_splice.json recording what was done.
Then `synth run --resume <main run dir>` retries every missing row and re-exports.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

from src.data.synth.ours.embeddings import DEFAULT_MODEL, embed

main, config, topups = Path(sys.argv[1]), sys.argv[2], [Path(p) for p in sys.argv[3:]]
assert topups, "name at least one top-up run dir"
cfg = yaml.safe_load(open(config))
div = (
    next(s for s in cfg["stages"] if s["name"] == "write_scenarios").get("diversity")
    or {}
)
reject = float(div["reject_cosine"])
model = str(div.get("embed_model") or DEFAULT_MODEL)

FINALS = [
    "stage_2_write_scenarios",
    "stage_3_dedupe_scenarios",
    "stage_4_draft_prompts",
]
CKPTS = [
    "stage_5_revise_prompts",
    "stage_6_draft_responses",
    "stage_7_revise_responses",
]
MOVE = [f"{c}.jsonl" for c in CKPTS] + ["stage_8_export_sft.jsonl", "manifest.json"]

assert not (main / "run1_final").exists(), (
    f"{main} was already spliced (run1_final/ exists)"
)
for d in [main, *topups]:
    assert (d / "stage_8_export_sft.jsonl").exists(), (
        f"{d} has not finished (no export)"
    )


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


seen_ids = {r["scenario_id"] for r in rows(main / "stage_2_write_scenarios.jsonl")}
seen_vecs = embed(
    [r["situation"] for r in rows(main / "stage_2_write_scenarios.jsonl")], model=model
)
per_topup, keep_by_dir = [], {}
for d in topups:
    sc = rows(d / "stage_2_write_scenarios.jsonl")
    clash = seen_ids & {r["scenario_id"] for r in sc}
    assert not clash, f"scenario_id collision from {d}: {sorted(clash)[:5]}"
    X = embed([r["situation"] for r in sc], model=model)
    nearest = (X @ seen_vecs.T).max(axis=1)
    dropped = {
        r["scenario_id"]: round(float(c), 4)
        for r, c in zip(sc, nearest)
        if float(c) >= reject
    }
    kept_idx = [i for i, r in enumerate(sc) if r["scenario_id"] not in dropped]
    keep_by_dir[d] = {sc[i]["scenario_id"] for i in kept_idx}
    seen_ids |= keep_by_dir[d]
    seen_vecs = np.vstack([seen_vecs, X[kept_idx]]) if kept_idx else seen_vecs
    per_topup.append(
        {
            "run_dir": str(d),
            "command": json.loads((d / "manifest.json").read_text()).get("command"),
            "scenarios": len(sc),
            "dropped_as_cross_run_duplicates": dropped,
            "kept_scenarios": len(kept_idx),
            "max_cosine_to_earlier_among_kept": round(
                max([float(nearest[i]) for i in kept_idx] or [0.0]), 4
            ),
        }
    )

(main / "run1_final").mkdir()
shutil.copy2(main / "manifest.json", main / "manifest_run1.json")
for name in MOVE:
    shutil.move(str(main / name), str(main / "run1_final" / name))

appended: dict[str, int] = {}
for name, suffix in [(n, ".jsonl") for n in FINALS] + [
    (n, ".partial.jsonl") for n in CKPTS
]:
    add = []
    for d in topups:
        src = d / f"{name}{suffix}"
        if src.exists():
            add += [r for r in rows(src) if r["scenario_id"] in keep_by_dir[d]]
    with (main / f"{name}{suffix}").open("a", encoding="utf-8") as f:
        for r in add:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    appended[f"{name}{suffix}"] = len(add)

record = {
    "reject_cosine": reject,
    "embed_model": model,
    "topups": per_topup,
    "appended_rows": appended,
    "moved_to_run1_final": MOVE,
}
(main / "topup_splice.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
