# ABOUTME: Swap the most ODCV-like difficult-advice rows out of the mixture for the least
# ABOUTME: ODCV-like rows from the unused corpus, holding the reasoning itself unmodified.

"""An arm that changes WHICH scenarios are trained on, not how they are reasoned about.

Every intervention so far rewrote the reasoning and left the scenario set alone, and none of
them moved the misalignment rate. This is the other lever: the content overlap between training
and eval. 132 of the 716 rows are agentic AND oversight-laden — the ones that read like ODCV
cells written as a conversation — and they are replaced by rows from the 1,487 unused corpus
rows that score low on every ODCV dimension.

The reasoning text is untouched in both the kept and the swapped-in rows, so this arm differs
from the control only in scenario selection. If ODCV transfer comes from content overlap rather
than reasoning style, removing these rows is what should show it.

Run:
  uv run python scratch/build_low_odcv_mixture.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_profile import model_profile  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

SOURCE = "synthdoc_difficult_advice"
USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
DIMS = ("operator", "agentic", "metric", "oversight")


def _render(row: dict, profile) -> str:
    """Render one SFT record into the mixture's pre-rendered chat format.

    Args:
        row: A stage_7 SFT record with system/user/assistant messages.
        profile: The family's ModelProfile, for the think-tag literals.

    Returns:
        The rendered text, matching the mixture's existing rows byte for byte in shape.
    """
    m = {x["role"]: x for x in row["messages"]}
    return (f"<|im_start|>system\n{m['system']['content']}<|im_end|>\n"
            f"<|im_start|>user\n{m['user']['content']}<|im_end|>\n"
            f"<|im_start|>assistant\n{profile.prefill}"
            f"{m['assistant']['reasoning_content']}\n</think>\n\n"
            f"{m['assistant']['content']}<|im_end|>\n")


def main(
    rated_mixture: str = "output/odcv_similarity/all716/similarity.jsonl",
    rated_candidates: str = "output/odcv_similarity/candidates/similarity.jsonl",
    extra_candidates: str = "",
    mode: str = "and",
    mixture: str = "data/hf/2026-08-06-table2-9284-synthdoc-716-train/mixture_think.jsonl",
    sft: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
    model: str = "Qwen/Qwen3.6-27B",
    seed: int = 0,
    out_dir: str | None = None,
) -> None:
    """Replace the ODCV-like rows with low-similarity rows from the unused corpus.

    Args:
        rated_mixture: similarity.jsonl covering the 716 rows in the mixture.
        rated_candidates: similarity.jsonl covering unused corpus rows.
        extra_candidates: a second such file, concatenated with the first.
        mode: "and" removes rows scoring >=2 on BOTH agentic and oversight; "or" removes rows
            scoring >=2 on EITHER. Agentic is a subset of oversight here, so "or" is the
            oversight cut.
        mixture: The control mixture to rebuild from.
        sft: The difficult-advice SFT file holding every corpus row.
        model: Model whose ModelProfile supplies the think-tag literals.
        seed: Sampling seed for choosing among eligible replacements.
        out_dir: Output directory.

    Raises:
        RuntimeError: If there are too few eligible replacements, or the rebuilt file does not
            preserve the mixture's row count and source composition.
    """
    rated = {json.loads(x)["scenario_id"]: json.loads(x)
             for x in Path(rated_mixture).read_text().splitlines() if x.strip()}
    cands = [json.loads(x) for x in Path(rated_candidates).read_text().splitlines() if x.strip()]
    if extra_candidates:
        cands += [json.loads(x) for x in Path(extra_candidates).read_text().splitlines()
                  if x.strip()]
    rows = {json.loads(l)["metadata"]["scenario_id"]: json.loads(l)
            for l in Path(sft).read_text().splitlines() if l.strip()}

    # OUT: agentic AND oversight both >= 2 -- the rows that read like ODCV cells.
    pick = ((lambda r: r["agentic"] >= 2 and r["oversight"] >= 2) if mode == "and"
            else (lambda r: r["agentic"] >= 2 or r["oversight"] >= 2))
    drop = sorted(s for s, r in rated.items() if pick(r))
    # IN: strictly below the removal threshold on the two target dimensions, and not high on
    # the other two either, so the swap does not smuggle in operator/metric similarity.
    pool = sorted((r for r in cands if all(r[d] <= 1 for d in DIMS)),
                  key=lambda r: r["scenario_id"])
    if len(pool) < len(drop):
        raise RuntimeError(f"need {len(drop)} replacements, only {len(pool)} eligible")
    picks = random.Random(seed).sample(pool, len(drop))
    print(f"removing {len(drop)} rows (agentic>=2 {mode.upper()} oversight>=2)")
    print(f"eligible replacements (all four dims <=1): {len(pool)}; taking {len(picks)}")

    profile = model_profile(model)
    by_user = {}
    for sid, r in rows.items():
        by_user[{m["role"]: m for m in r["messages"]}["user"]["content"].strip()] = sid

    out_rows, removed, added = [], 0, iter(picks)
    for line in Path(mixture).read_text().splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        if m.get("source") == SOURCE and by_user[USER_RE.search(m["text"]).group(1).strip()] in drop:
            rep = next(added)
            new = rows[rep["scenario_id"]]
            m = {"text": _render(new, profile), "source": SOURCE,
                 "scenario_id": rep["scenario_id"], "swapped_in": True}
            removed += 1
        out_rows.append(m)
    if removed != len(drop):
        raise RuntimeError(f"replaced {removed}, expected {len(drop)}")

    src = Counter(r["source"] for r in out_rows)
    ctl = Counter(json.loads(l)["source"]
                  for l in Path(mixture).read_text().splitlines() if l.strip())
    if src != ctl or len(out_rows) != sum(ctl.values()):
        raise RuntimeError(f"composition changed: {src} vs {ctl}")

    d = Path(out_dir or f"output/low_odcv/{timestamp()}")
    d.mkdir(parents=True, exist_ok=True)
    (d / "mixture_think_lowodcv.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in out_rows))
    kept = [rated[s] for s in rated if s not in drop]
    before = {k: sum(r[k] for r in rated.values()) / len(rated) for k in DIMS}
    after_rows = kept + picks
    after = {k: sum(r[k] for r in after_rows) / len(after_rows) for k in DIMS}
    stats = {"rows_total": len(out_rows), "replaced": removed,
             "eligible_pool": len(pool), "seed": seed,
             "mode": mode, "odcv_dims_before": {k: round(v, 2) for k, v in before.items()},
             "odcv_dims_after": {k: round(v, 2) for k, v in after.items()},
             "removed_ids": drop, "added_ids": [p["scenario_id"] for p in picks],
             "git_sha": git_sha(), "timestamp_utc": timestamp(), "source_mixture": mixture}
    (d / "mixture_stats_lowodcv.json").write_text(json.dumps(stats, indent=2))
    print(f"\n{len(out_rows)} rows, {removed} swapped -> {d / 'mixture_think_lowodcv.jsonl'}")
    print(f"{'dimension':12s}{'before':>9}{'after':>9}")
    for k in DIMS:
        print(f"{k:12s}{before[k]:9.2f}{after[k]:9.2f}")


if __name__ == "__main__":
    fire.Fire(main)
