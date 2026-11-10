# ABOUTME: Builds partial-tools variants of the da-tools-15 mix by dropping the `tools` list from a
# ABOUTME: seeded fraction of the DA rows that carry one; every other byte is left as it was.
"""Dose arms between da-15 (no tools on DA rows) and da-tools-15 (tools on 646 of 650 DA rows).

Only DA rows (`source == "da"`) that carry a `tools` list are candidates. The 110 non-synthetic
rows that carry tools natively are left alone: they are identical in the da-15 mix, so removing
them would change the base blend as well as the dose.

Two selection modes:
  * default (2026-10-19 arms): one seeded order over ALL tooled DA rows, cut at `drop`; doses
    are nested, but the kept share per trait is whatever the shuffle gave (30-39 of 72-73 at
    half).
  * `--by-trait` (2026-11-10): within each trait, exactly `floor(n_trait * keep)` DA rows keep
    their tools, where n_trait counts ALL of that trait's DA rows (the 4 that never had tools
    are always on the no-tools side). Trait ids come from the corpus the mix drew from
    (2026-10-02-da-synth @ 305914d5), matched on the user turn.

Every DA row appears exactly once either way: this edits one mix in place, it never adds rows.

    uv run python scratch/tools_dose/build_tools_dose_mixes.py --drop 0.5 --variant half-tools --date 2026-10-19
    uv run python scratch/tools_dose/build_tools_dose_mixes.py --drop 0.5 --by-trait \
        --variant trait-half-tools --date 2026-11-10 [--push]
"""
import argparse
import collections
import hashlib
import json
import random
import subprocess
from pathlib import Path

from src.infra.huggingface import hf_download, hf_repo_id, push_files, training_data_tags
from src.naming import mix_subject

SRC_REPO = "dougalldeepmind/2026-10-11-da-tools-15-mix"
SRC_REV = "da195751376fea70439730adda75a516aa5bc688"
CORPUS_REPO = "dougalldeepmind/2026-10-02-da-synth"
CORPUS_REV = "305914d58627457002f61d86ba69866f6fa2b9a9"
CONSTITUTION = "constitutions/claude_distilled_09_principles/constitution.md"


def load(path: Path) -> list[dict]:
    """One JSON object per newline. Not str.splitlines: that also splits on U+2028/U+2029 and
    the other Unicode line separators, which occur inside the conversation strings."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def user_turn(row: dict) -> str:
    """The first user message: the key a mix row shares with its corpus row."""
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def trait_of(rows: list[dict]) -> dict[int, str]:
    """Row index -> trait_id for every DA row, read off the pinned corpus. Fails on any miss."""
    corpus = load(hf_download(CORPUS_REPO, "dataset.jsonl", repo_type="dataset", revision=CORPUS_REV))
    by_prompt = {}
    for r in corpus:
        k = user_turn(r)
        assert k not in by_prompt, "corpus has a duplicated user turn; trait lookup is ambiguous"
        by_prompt[k] = r["metadata"]["trait_id"]
    out = {}
    for i, r in enumerate(rows):
        if r.get("source") == "da":
            k = user_turn(r)
            assert k in by_prompt, f"DA row {i} not found in {CORPUS_REPO}@{CORPUS_REV[:8]}"
            out[i] = by_prompt[k]
    return out


def build(rows: list[dict], drop: float, seed: int, by_trait: bool) -> tuple[list[dict], list[int]]:
    """Return the edited rows and the indices whose tools list was removed."""
    candidates = [i for i, r in enumerate(rows) if r.get("source") == "da" and r.get("tools")]
    if by_trait:
        traits = trait_of(rows)
        totals = collections.Counter(traits.values())
        dropped = []
        for t in sorted(totals):
            keep = int(totals[t] * (1 - drop))  # floor, of ALL the trait's DA rows
            tooled = [i for i in candidates if traits[i] == t]
            assert keep <= len(tooled), f"{t}: wants {keep} tooled but only {len(tooled)} have tools"
            order = list(tooled)
            random.Random(f"{seed}:{t}").shuffle(order)
            dropped += order[keep:]
        dropped = sorted(dropped)
    else:
        k = round(drop * len(candidates))
        # One seeded order, cut at k: every dose drops a PREFIX of the same order, so a larger
        # drop always contains a smaller one and the arms differ only in how many rows lose tools.
        order = list(candidates)
        random.Random(seed).shuffle(order)
        dropped = sorted(order[:k])
    out = [dict(r) for r in rows]
    for i in dropped:
        del out[i]["tools"]
    return out, dropped


def main() -> None:
    """Build one variant, verify it, write it under data/mixes/, and optionally push it."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", type=float, required=True, help="fraction of tooled DA rows to strip")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--by-trait", action="store_true", help="stratify the kept share per trait_id")
    ap.add_argument("--push", action="store_true", help="push the mix to HF with its card")
    a = ap.parse_args()

    name = f"{a.date}-{mix_subject('da-tools', 15, a.variant)}-mix"
    src = load(hf_download(SRC_REPO, "mixture.jsonl", repo_type="dataset", revision=SRC_REV))
    rows, dropped = build(src, a.drop, a.seed, a.by_trait)

    # Verification: only the chosen rows changed, and only by losing `tools`.
    assert len(rows) == len(src)
    changed = [i for i, (x, y) in enumerate(zip(src, rows)) if x != y]
    assert changed == dropped, "rows other than the chosen ones changed"
    for i in dropped:
        assert {k: v for k, v in src[i].items() if k != "tools"} == rows[i]
    # No DA scenario twice (once with tools, once without): every DA user turn is unique.
    da_turns = [user_turn(r) for r in rows if r.get("source") == "da"]
    assert len(da_turns) == len(set(da_turns)), "a DA scenario appears more than once"
    da_tooled_before = sum(1 for r in src if r.get("source") == "da" and r.get("tools"))
    da_tooled_after = sum(1 for r in rows if r.get("source") == "da" and r.get("tools"))
    base_tooled = sum(1 for r in rows if r.get("source") != "da" and r.get("tools"))
    per_trait = None
    if a.by_trait:
        traits = trait_of(rows)
        per_trait = {t: {"rows": sum(1 for v in traits.values() if v == t),
                         "with_tools": sum(1 for i, v in traits.items() if v == t and rows[i].get("tools"))}
                     for t in sorted(set(traits.values()))}

    out = Path("data/mixes") / name
    out.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (out / "mixture.jsonl").write_text(body, encoding="utf-8")
    meta = {"name": name, "source_repo": SRC_REPO, "source_revision": SRC_REV,
            "drop_fraction": a.drop, "seed": a.seed, "by_trait": a.by_trait, "rows": len(rows),
            "da_rows": len(da_turns),
            "da_rows_with_tools_before": da_tooled_before,
            "da_rows_with_tools_after": da_tooled_after,
            "da_rows_tools_dropped": len(dropped),
            "non_da_rows_with_tools_untouched": base_tooled,
            "per_trait": per_trait,
            "trait_source": f"{CORPUS_REPO}@{CORPUS_REV}" if a.by_trait else None,
            "dropped_row_indices": dropped,
            "sha256": hashlib.sha256(body.encode()).hexdigest()}
    (out / "build_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{name}: rows={len(rows)}  DA tooled {da_tooled_before} -> {da_tooled_after} "
          f"(dropped {len(dropped)})  non-DA tooled untouched={base_tooled}")
    if per_trait:
        print("per trait (rows, with tools):", {t: (v["rows"], v["with_tools"]) for t, v in per_trait.items()})

    if a.push:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        how = (f"within each trait exactly floor(n/2) DA rows keep `tools` ({per_trait and next(iter(per_trait.values()))['with_tools']} per trait)"
               if a.by_trait else f"a seeded {a.drop:.0%} of the tooled DA rows lose `tools`")
        fields = {
            "experiment": (f"50-50 DA / DA+unused-tools arm: {how}, the rest are plain DA. "
                           f"{da_tooled_after} of {len(da_turns)} DA rows carry tools; every DA scenario "
                           f"appears once. The {base_tooled} non-synthetic rows with native tools are "
                           f"untouched; every other byte equals {SRC_REPO} @ {SRC_REV[:12]}, which equals "
                           "2026-10-05-da-15-mix except for `tools`."),
            "date_generated": a.date,
            "constitution": CONSTITUTION,
            "source_repo": f"https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT.git @ {sha}",
            "models": f"none (pure edit of {SRC_REPO} @ {SRC_REV}; traits from {CORPUS_REPO} @ {CORPUS_REV})",
            "generation_config": (f"seed={a.seed}; drop={a.drop}; by_trait={a.by_trait}; per-trait seeded "
                                  "shuffle `random.Random(f'{seed}:{trait}')`; dropped indices in build_meta.json"),
            "schema": "mixture.jsonl: {messages, source[, tools]} as the source mix",
            "provenance": ("uv run python scratch/tools_dose/build_tools_dose_mixes.py "
                           f"--drop {a.drop} {'--by-trait ' if a.by_trait else ''}--variant {a.variant} "
                           f"--date {a.date} --seed {a.seed} --push"),
        }
        fm = {"configs": [{"config_name": "default", "data_files": "mixture.jsonl"}],
              "tags": training_data_tags("ablation", "da-tools-15", CONSTITUTION, extra=("stage:final",))}
        print(push_files([out / "mixture.jsonl", out / "build_meta.json"], hf_repo_id(name), fields,
                         private=False, front_matter=fm))


if __name__ == "__main__":
    main()
