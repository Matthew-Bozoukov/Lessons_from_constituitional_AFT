# ABOUTME: Builds partial-tools variants of the da-tools-15 mix by dropping the `tools` list from a
# ABOUTME: seeded fraction of the DA rows that carry one; every other byte is left as it was.
"""Dose arms between da-15 (no tools on DA rows) and da-tools-15 (tools on 646 of 650 DA rows).

Only DA rows (`source == "da"`) that carry a `tools` list are candidates. The 110 non-synthetic
rows that carry tools natively are left alone: they are identical in the da-15 mix, so removing
them would change the base blend as well as the dose.

    uv run python scratch/tools_dose/build_tools_dose_mixes.py --drop 0.5 --variant half-tools
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from src.naming import mix_subject

SRC_REPO = "dougalldeepmind/2026-10-11-da-tools-15-mix"
SRC_REV = "da195751376f"
SRC = Path("data/mixes/2026-10-11-da-tools-15-mix/mixture.jsonl")


def load(path: Path) -> list[dict]:
    """One JSON object per newline. Not str.splitlines: that also splits on U+2028/U+2029 and
    the other Unicode line separators, which occur inside the conversation strings."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build(drop: float, seed: int) -> tuple[list[dict], list[int]]:
    """Return the edited rows and the indices whose tools list was removed."""
    rows = load(SRC)
    candidates = [i for i, r in enumerate(rows) if r.get("source") == "da" and r.get("tools")]
    k = round(drop * len(candidates))
    # One seeded order, cut at k: every dose drops a PREFIX of the same order, so a larger drop
    # always contains a smaller one and the arms differ only in how many rows lose their tools.
    order = list(candidates)
    random.Random(seed).shuffle(order)
    dropped = sorted(order[:k])
    out = [dict(r) for r in rows]
    for i in dropped:
        del out[i]["tools"]
    return out, dropped


def main() -> None:
    """Build one variant, verify it, and write it under data/mixes/."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", type=float, required=True, help="fraction of tooled DA rows to strip")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    name = f"{a.date}-{mix_subject('da-tools', 15, a.variant)}-mix"
    rows, dropped = build(a.drop, a.seed)
    src = load(SRC)

    # Verification: only the chosen rows changed, and only by losing `tools`.
    assert len(rows) == len(src)
    changed = [i for i, (x, y) in enumerate(zip(src, rows)) if x != y]
    assert changed == dropped, "rows other than the chosen ones changed"
    for i in dropped:
        assert {k: v for k, v in src[i].items() if k != "tools"} == rows[i]
    da_tooled_before = sum(1 for r in src if r.get("source") == "da" and r.get("tools"))
    da_tooled_after = sum(1 for r in rows if r.get("source") == "da" and r.get("tools"))
    base_tooled = sum(1 for r in rows if r.get("source") != "da" and r.get("tools"))

    out = Path("data/mixes") / name
    out.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (out / "mixture.jsonl").write_text(body, encoding="utf-8")
    meta = {"name": name, "source_repo": SRC_REPO, "source_revision": SRC_REV,
            "drop_fraction": a.drop, "seed": a.seed, "rows": len(rows),
            "da_rows_with_tools_before": da_tooled_before,
            "da_rows_with_tools_after": da_tooled_after,
            "da_rows_tools_dropped": len(dropped),
            "non_da_rows_with_tools_untouched": base_tooled,
            "dropped_row_indices": dropped,
            "sha256": hashlib.sha256(body.encode()).hexdigest()}
    (out / "build_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{name}: rows={len(rows)}  DA tooled {da_tooled_before} -> {da_tooled_after} "
          f"(dropped {len(dropped)})  non-DA tooled untouched={base_tooled}")


if __name__ == "__main__":
    main()
