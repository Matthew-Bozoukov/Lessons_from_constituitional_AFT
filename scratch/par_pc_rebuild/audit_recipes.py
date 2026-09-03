#!/usr/bin/env python
# ABOUTME: Read-only audit of the three natural-turn synth recipes against the difficult-advice
# ABOUTME: baseline: front-half prompt parity, whole-constitution injections, and model families.
# Run: uv run python scratch/par_pc_rebuild/audit_recipes.py

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

CONFIGS = {
    "DA": "configs/data/synth/2026-08-01_difficult_advice.yaml",
    "PAR": "configs/data/synth/2026-08-13_post_action_retrospection.yaml",
    "PC": "configs/data/synth/2026-08-13_peer_critique.yaml",
}

# The stages difficult advice owns and the derived recipes are supposed to inherit verbatim.
FRONT_HALF = [
    "chunk_constitution",
    "write_scenarios",
    "corpus_scenarios",
    "dedupe_scenarios",
    # PAR and PC only -- difficult advice has no scenario to keep coherent beyond the
    # situation itself. `compare_front_half` skips a stage DA lacks; the PAR-vs-PC identity
    # check below still covers it.
    "revise_scenarios",
    "draft_prompts",
    "revise_prompts",
]

# Model families this project is allowed to pay for in a constitutional-SFT corpus.
ALLOWED_VENDORS = {"anthropic"}


def load(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def stages(cfg: dict) -> dict[str, dict]:
    return {s["name"]: s for s in cfg["stages"]}


def prompts_of(stage: dict) -> dict[str, str]:
    """Every prompt string a stage can send, including arm-conditioned `variants_by` ones."""
    out: dict[str, str] = {}

    def walk(node, path: str) -> None:
        if isinstance(node, str):
            out[path] = node
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(stage.get("prompts") or {}, "prompts")
    walk(stage.get("variants_by") or {}, "variants_by")
    return out


def whole_constitution_slots(cfg: dict) -> list[tuple[str, str]]:
    """Every stage prompt that interpolates the WHOLE constitution rather than one chunk."""
    hits = []
    for name, stage in stages(cfg).items():
        for key, text in prompts_of(stage).items():
            if re.search(r"\{constitution\}", text):
                hits.append((name, key))
    return hits


def assigned_labels(cfg: dict) -> list[tuple[str, str]]:
    """Every `assign:` label a recipe hands out -- an arm, whatever it is called."""
    hits = []
    for name, stage in stages(cfg).items():
        fields = ((stage.get("assign") or {}).get("fields")) or {}
        for field in fields:
            hits.append((name, field))
    return hits


def arm_conditioned_stages(cfg: dict) -> list[tuple[str, str]]:
    """Stages whose PROMPT is chosen by an arm label -- the obviously-good/obviously-bad tell."""
    hits = []
    for name, stage in stages(cfg).items():
        vb = stage.get("variants_by")
        if isinstance(vb, dict) and vb.get("field"):
            hits.append((name, vb["field"]))
    return hits


def model_vendors(cfg: dict) -> dict[str, str]:
    out = {}
    for slot, spec in (cfg.get("models") or {}).items():
        for field in ("model", "fallback_model"):
            model = (spec or {}).get(field)
            if model:
                out[f"{slot}.{field}" if field != "model" else slot] = model
    return out


def compare_front_half(
    base: dict, other: dict, label: str
) -> tuple[list[str], list[str]]:
    """Stage-by-stage prompt diff of a derived recipe against difficult advice.

    Returns (problems, allowed) -- a difference is ALLOWED when it is the `shortfall`
    additions of 2026-09-02 (`shortfall`, `pushback`), which PAR and PC both make and
    (it has no first reply to get wrong). Any other difference is a problem.
    """
    problems, allowed = [], []
    b, o = stages(base), stages(other)
    for name in FRONT_HALF:
        if name not in o:
            problems.append(f"{label}: stage `{name}` is missing")
            continue
        if name not in b:
            continue
        bp, op = prompts_of(b[name]), prompts_of(o[name])
        if set(bp) != set(op):
            problems.append(
                f"{label}.{name}: prompt keys differ "
                f"(DA {sorted(bp)} vs {label} {sorted(op)})"
            )
            continue
        for key in sorted(bp):
            if bp[key].strip() == op[key].strip():
                continue
            if any(f in op[key] for f in ("shortfall", "pushback")):
                allowed.append(f"{label}.{name}.{key}: + shortfall/pushback")
            else:
                problems.append(f"{label}.{name}.{key}: prompt text differs from DA's")
    return problems, allowed


def main() -> int:
    cfgs = {name: load(path) for name, path in CONFIGS.items()}
    failures: list[str] = []

    print("=" * 78)
    print("1. ALIGNMENT TARGET")
    print("=" * 78)
    import hashlib

    for name, cfg in cfgs.items():
        const = cfg["constitution"]
        sha = hashlib.sha1(Path(const).read_bytes()).hexdigest()[:12]
        print(f"  {name:4s} {const}")
        print(
            f"       sha1 {sha}  n_traits={cfg.get('n_traits')}  chunking={cfg.get('chunking')}"
        )
    shas = {
        hashlib.sha1(Path(c["constitution"]).read_bytes()).hexdigest()
        for c in cfgs.values()
    }
    if len(shas) != 1:
        failures.append("the three recipes do not share one alignment target")
    else:
        print("  -> all three target byte-identical constitution text  OK")
    paths = {c["constitution"] for c in cfgs.values()}
    if len(paths) != 1:
        print(f"  -> NOTE: same bytes, {len(paths)} different paths: {sorted(paths)}")

    print()
    print("=" * 78)
    print("2. WHOLE-CONSTITUTION INJECTIONS (must be none: principle-scoped everywhere)")
    print("=" * 78)
    for name, cfg in cfgs.items():
        hits = whole_constitution_slots(cfg)
        if hits:
            failures.append(
                f"{name} injects the whole constitution in {len(hits)} prompt(s)"
            )
            print(f"  {name:4s} FAIL  {len(hits)} injection(s):")
            for stage, key in hits:
                print(f"          {stage}.{key}")
        else:
            print(f"  {name:4s} OK    no {{constitution}} slot in any stage prompt")

    print()
    print("=" * 78)
    print("3. FRONT-HALF PARITY WITH DIFFICULT ADVICE (DA's, plus shortfall + pushback)")
    print("=" * 78)
    for name in ("PAR", "PC"):
        problems, allowed = compare_front_half(cfgs["DA"], cfgs[name], name)
        if problems:
            failures.extend(problems)
            print(f"  {name:4s} FAIL")
            for pr in problems:
                print(f"          {pr}")
        else:
            print(
                f"  {name:4s} OK    {len(FRONT_HALF)} front-half stages are DA's, "
                f"differing only by shortfall + pushback:"
            )
            for a in allowed:
                print(f"          {a}")

    # ... and the two variants must add it IDENTICALLY, or the attribution contrast they
    # exist to make is confounded by a second difference.
    par_s, pc_s = stages(cfgs["PAR"]), stages(cfgs["PC"])
    drift = [n for n in FRONT_HALF if par_s.get(n) != pc_s.get(n)]
    if drift:
        failures.append(f"PAR and PC front halves have drifted apart: {drift}")
        print(f"  BOTH FAIL PAR and PC differ from each other in {drift}")
    else:
        print("  BOTH OK   PAR and PC front halves are byte-identical to each other")

    print()
    print("=" * 78)
    print("4. MODEL FAMILIES (Anthropic only: Sonnet writes/judges, Haiku generates)")
    print("=" * 78)
    for name, cfg in cfgs.items():
        vendors = model_vendors(cfg)
        bad = {
            s: m for s, m in vendors.items() if m.split("/")[0] not in ALLOWED_VENDORS
        }
        if bad:
            failures.append(f"{name} pays {len(bad)} non-Anthropic model slot(s)")
            print(f"  {name:4s} FAIL  {len(bad)} of {len(vendors)} slots off-family:")
            for slot, model in sorted(bad.items()):
                print(f"          {slot:24s} {model}")
        else:
            print(f"  {name:4s} OK    {len(vendors)} slots, all Anthropic")

    print()
    print("=" * 78)
    print(
        "5. ARM-CONDITIONED PROMPTS (none: an assigned label must not pick the prompt)"
    )
    print("=" * 78)
    for name, cfg in cfgs.items():
        hits = arm_conditioned_stages(cfg)
        if hits:
            failures.append(
                f"{name} picks its prompt from an arm label in {len(hits)} stage(s)"
            )
            print(
                f"  {name:4s} FAIL  {len(hits)} stage(s) branch on an assigned label:"
            )
            for stage, field in hits:
                print(f"          {stage:24s} variants_by: {field}")
        else:
            print(f"  {name:4s} OK    no stage branches its prompt on an arm label")

    print()
    print("=" * 78)
    print(
        "6. ASSIGNED LABELS (PC must hand out none: one arm, grey area, nothing else)"
    )
    print("=" * 78)
    for name, cfg in cfgs.items():
        hits = assigned_labels(cfg)
        if not hits:
            print(f"  {name:4s} OK    no `assign:` label anywhere")
        elif name == "PC":
            failures.append(f"PC assigns {len(hits)} label(s); it must assign none")
            print(f"  {name:4s} FAIL  {len(hits)} label(s):")
            for stage, field in hits:
                print(f"          {stage:24s} {field}")
        else:
            # Reported, not gated: PAR's one label varies the register of the turn that
            # never trains, and PAR is a trained and evaluated arm, not a cleanup target.
            print(f"  {name:4s} NOTE  {len(hits)} label(s), reported not gated:")
            for stage, field in hits:
                print(f"          {stage:24s} {field}")

    print()
    print("=" * 78)
    print("7. GREY-AREA GATE (a rater must keep only genuine grey areas)")
    print("=" * 78)
    for name in ("PAR", "PC"):
        st = stages(cfgs[name])
        has_rater = any(
            "grey" in yaml.safe_dump(s).lower()
            or "grey_area" in yaml.safe_dump(s).lower()
            for s in st.values()
        )
        has_filter = any(
            s.get("kind") == "keep" or n.startswith("filter_") for n, s in st.items()
        )
        if has_rater and has_filter:
            print(f"  {name:4s} OK    grey-area rater present and its filter runs")
        else:
            failures.append(
                f"{name} has no grey-area gate (rater={has_rater}, filter={has_filter})"
            )
            print(f"  {name:4s} FAIL  rater={has_rater} filter={has_filter}")

    print()
    print("=" * 78)
    if failures:
        print(f"AUDIT FAILED: {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(
        "AUDIT PASSED: PAR and PC are both principle-scoped, DA-front-half, Anthropic-only,"
    )
    print("and grey-area gated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
