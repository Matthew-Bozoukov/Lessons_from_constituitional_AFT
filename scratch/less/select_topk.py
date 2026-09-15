# ABOUTME: Cuts the LESS ranking at the top K% of the difficult-advice pool, and draws a
# ABOUTME: seeded random control of the same size, as training-ready jsonl for `uv run mix`.

"""Select the top-K% of D by LESS influence, plus a same-size random control.

    uv run python scratch/less/select_topk.py --frac 0.10 --order score_max

This is the step the 2026-08-14 ranking was built for and deliberately stopped short of:
that run ranked all 2,203 rows and trained nothing, leaving "what fraction to keep" open.
Here K is cut and the two training files are written.

Writes into --out (default data/less/):
    d_top<pct>_<order>.jsonl   the selection
    d_random<K>.jsonl          the control: a seeded draw of K rows from the WHOLE pool
    selection_meta.json        provenance + composition of both, for the dataset cards

The control draws from all 2,203 rows rather than from the pool minus the selection.
"What you would have trained on without LESS" is a plain random draw, and excluding the
top-K would make the control anti-correlated with the arm rather than independent of it.
The incidental overlap is reported rather than engineered away.

DO NOT DRAW THE CONTROL WITH SEED 0. `random.sample` picks positions from the RNG alone,
so seed 0 over a 2,203-row population reproduces prepare_data.py's warmup draw EXACTLY --
caught here on the first run, at in_warmup 220/220 instead of the ~22 chance predicts.
Those rows trained the warmup LoRA that produced the ranking being tested, so a control
made of them is not independent of the treatment. The default seed is 1 for that reason,
and `assert_independent_of_warmup` fails the run rather than trusting the default.

THE JOIN IS BY LINE INDEX, and that is the one thing here that can silently go wrong.
`less_id` is `<scenario_id>#<row index>`, stamped onto the pool at load time by
prepare_data.load_pool() -- the published pool file does NOT carry it, so nothing but
row order ties a score back to a conversation. The rankings/README.md on the Hub says to
join on `metadata.less_id`, which reads as though the field were stored. Every assertion
below exists because a silent off-by-one here would train on the wrong rows and still
produce a plausible loss curve.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scratch.less.prepare_data import load_pool  # noqa: E402
from src.infra.huggingface import hf_api, hf_download  # noqa: E402

SCORES_REPO = "LASR-Callum/2026-08-14-less-selection-difficult-advice"
SCORES_FILE = "scores/scores.jsonl"
POOL_REPO, POOL_FILE = "matboz/synthdoc-v2-difficult-advice", "stage_7_sft.jsonl"

# The orderings scores.jsonl can be cut by. `score_max` is the paper's aggregation (max
# over the validation set) and is the ordering the published `rank` field already carries,
# so it is the only one that can be checked against the Hub artifact.
ORDERS = {
    "score_max": lambda r: r["score_max"],
    "score_mean": lambda r: r["score_mean"],
    "score_min": lambda r: r["score_min"],
    "codebase_resisted": lambda r: r["per_subtask"]["codebase_resisted"],
    "honest_declined": lambda r: r["per_subtask"]["honest_declined"],
    "stayed_ai": lambda r: r["per_subtask"]["stayed_ai"],
}


def load_scores() -> tuple[list[dict], str]:
    """The published ranking, pinned to the exact commit it was read at."""
    sha = hf_api().repo_info(SCORES_REPO, repo_type="dataset").sha
    local = hf_download(SCORES_REPO, SCORES_FILE, repo_type="dataset", revision=sha)
    rows = [json.loads(line) for line in
            Path(local).read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows, sha


def verify_join(pool: list[dict], scores: list[dict]) -> dict[str, dict]:
    """Tie every score to its conversation, refusing anything less than a clean 1:1.

    Returns the pool indexed by less_id. Four independent checks, because each failure
    mode is silent: a length change in the pool file, a reordering, a partial overlap,
    and a trait mismatch would each yield a runnable file trained on the wrong rows.
    """
    by_id = {r["less_id"]: r for r in scores}
    assert len(by_id) == len(scores), "scores.jsonl has duplicate less_ids"
    pool_by_id = {r["metadata"]["less_id"]: r for r in pool}
    assert len(pool_by_id) == len(pool), "pool has duplicate less_ids"
    assert len(pool) == len(scores), (
        f"pool has {len(pool)} rows but the ranking scores {len(scores)} - the pool file "
        f"has changed since the ranking was computed; less_id is positional, so the "
        f"ranking cannot be joined to it")
    assert set(pool_by_id) == set(by_id), (
        "pool and ranking do not cover the same less_ids - the join is positional and "
        "the pool's row order has changed")

    # The id embeds the scenario it was stamped from, so re-deriving it catches a
    # reordering that preserved the id set.
    for row in pool:
        meta = row["metadata"]
        assert meta["less_id"].rsplit("#", 1)[0] == meta.get("scenario_id"), (
            f"{meta['less_id']}: scenario_id checksum failed - the pool was reordered")
        assert by_id[meta["less_id"]]["trait_id"] == meta.get("trait_id"), (
            f"{meta['less_id']}: trait_id disagrees between pool and ranking")
    return pool_by_id


def composition(sel: list[dict], scores_by_id: dict[str, dict]) -> dict:
    """Trait mix, winning subtask and warmup contamination of one selection."""
    ids = [r["metadata"]["less_id"] for r in sel]
    rows = [scores_by_id[i] for i in ids]
    return {
        "n": len(ids),
        "traits": dict(sorted(collections.Counter(r["trait_id"] for r in rows).items())),
        "argmax_subtask": dict(collections.Counter(r["argmax_subtask"] for r in rows)),
        # Warmup rows were trained on before being scored, so their influence is partly
        # self-influence. 10% is the base rate; the 2026-08-14 run measured 25/220 under
        # score_max and read that as "not a confound". Recorded, not corrected for.
        "in_warmup": sum(1 for r in rows if r["in_warmup"]),
    }


def assert_independent_of_warmup(comp: dict, k: int, n_pool: int, seed: int) -> None:
    """Refuse a control that has reproduced the warmup split (see the module docstring).

    The warmup rows are 10% of the pool, so a genuine draw of k lands ~k*k/n of them.
    Bound at 5 sigma of the binomial: wide enough that an unlucky-but-honest draw passes,
    far tighter than the wholesale reproduction seed 0 produces.
    """
    p = k / n_pool
    expected = k * p
    sigma = (k * p * (1 - p)) ** 0.5
    limit = expected + 5 * sigma
    assert comp["in_warmup"] <= limit, (
        f"control (seed {seed}) contains {comp['in_warmup']}/{k} LESS warmup rows, "
        f"above the {limit:.0f} bound (chance is ~{expected:.0f}). random.sample picks "
        f"positions from the RNG alone, so a seed matching prepare_data.py's redraws its "
        f"warmup split -- the rows that trained the LoRA this ranking came from. Pick a "
        f"different --seed.")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
    assert sum(1 for _ in path.open(encoding="utf-8")) == len(rows), f"{path} truncated"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.10, help="top fraction to keep")
    ap.add_argument("--order", default="score_max", choices=sorted(ORDERS))
    # NOT 0: seed 0 redraws the warmup split exactly (see the module docstring).
    ap.add_argument("--seed", type=int, default=1, help="seed for the random control")
    ap.add_argument("--out", type=Path, default=Path("data/less"))
    args = ap.parse_args()

    pool = load_pool()
    scores, sha = load_scores()
    print(f">>> ranking {SCORES_REPO}@{sha[:12]} ({len(scores)} scored rows)")
    pool_by_id = verify_join(pool, scores)
    scores_by_id = {r["less_id"]: r for r in scores}
    print(f">>> join verified 1:1 against {POOL_REPO}/{POOL_FILE} ({len(pool)} rows)")

    k = round(len(pool) * args.frac)
    key = ORDERS[args.order]
    # Tie-break on less_id so the cut is deterministic even where scores coincide.
    ranked = sorted(scores, key=lambda r: (-key(r), r["less_id"]))
    top_ids = [r["less_id"] for r in ranked[:k]]

    # score_max IS the published `rank`, so the recomputed cut must reproduce it exactly.
    # This is the end-to-end check that the ranking we just downloaded, the score field we
    # sorted on and the pool we joined against are the same three things the run produced.
    if args.order == "score_max":
        published = [r["less_id"] for r in sorted(scores, key=lambda r: r["rank"])[:k]]
        assert top_ids == published, (
            "recomputed score_max order does not reproduce the published `rank` field")
        print(f">>> top-{k} reproduces the published `rank` ordering exactly")

    # Sample the pool in its own order, the population prepare_data.py drew from, so the
    # warmup check below compares two draws over the same indexing rather than two
    # coincidentally-aligned ones.
    ctl = random.Random(args.seed).sample(pool, k)
    rnd_ids = [r["metadata"]["less_id"] for r in ctl]

    sel = [pool_by_id[i] for i in top_ids]
    # Every row must carry a real trace: these arms declare `thinking: true`, and a row
    # without one would be supervised as an empty think marker.
    for name, rows in (("selection", sel), ("control", ctl)):
        missing = [
            r["metadata"]["less_id"] for r in rows
            if not str((r["messages"][-1] or {}).get("reasoning_content") or "").strip()
        ]
        assert not missing, f"{name}: {len(missing)} rows carry no reasoning trace"

    ctl_comp = composition(ctl, scores_by_id)
    assert_independent_of_warmup(ctl_comp, k, len(pool), args.seed)

    pct = f"{round(args.frac * 100)}"
    sel_path = args.out / f"d_top{pct}_{args.order}.jsonl"
    ctl_path = args.out / f"d_random{k}.jsonl"
    write_rows(sel_path, sel)
    write_rows(ctl_path, ctl)

    overlap = sorted(set(top_ids) & set(rnd_ids))
    meta = {
        "ranking": {"repo": SCORES_REPO, "file": SCORES_FILE, "revision": sha},
        "pool": {"repo": POOL_REPO, "file": POOL_FILE, "rows": len(pool),
                 "join": "positional: less_id = <scenario_id>#<row index>"},
        "k": k, "frac": args.frac, "order": args.order, "control_seed": args.seed,
        "selection": {"file": sel_path.name, "ids": top_ids,
                      **composition(sel, scores_by_id)},
        "control": {"file": ctl_path.name, "ids": rnd_ids, **ctl_comp},
        # Expected ~k*k/N by chance (~22 at k=220, N=2203); a wild deviation would mean
        # the control was not drawn from the pool it claims to be.
        "incidental_overlap": {"n": len(overlap),
                               "expected_if_independent": round(k * k / len(pool), 1),
                               "ids": overlap},
    }
    (args.out / "selection_meta.json").write_text(json.dumps(meta, indent=2),
                                                  encoding="utf-8")

    print(f"\n>>> selection ({args.order}, top {args.frac:.0%} = {k}) -> {sel_path}")
    print(f"    traits {meta['selection']['traits']}")
    print(f"    argmax {meta['selection']['argmax_subtask']}  "
          f"in_warmup {meta['selection']['in_warmup']}")
    print(f">>> control (seed {args.seed}, {k} rows) -> {ctl_path}")
    print(f"    traits {meta['control']['traits']}")
    print(f"    argmax {meta['control']['argmax_subtask']}  "
          f"in_warmup {meta['control']['in_warmup']}")
    print(f">>> incidental overlap {len(overlap)} rows "
          f"(expected ~{meta['incidental_overlap']['expected_if_independent']})")
    print(f">>> wrote {args.out / 'selection_meta.json'}")


if __name__ == "__main__":
    main()
