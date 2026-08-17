# ABOUTME: LESS step 3 — combines the per-checkpoint gradient datastores into InfAdam
# ABOUTME: scores, ranks D, and runs the checks that decide whether the ranking is real.

"""Score every row of D by influence on the target subtasks.

    uv run python scratch/less/influence.py --grads output/less_grads/<ts> \\
        --out output/less_scores/<ts> [--control-grads output/less_grads_control/<ts>]

The pipeline, in the paper's terms:

    S_i[x, j] = cos( Γ(x, θ_i),  mean_{z in subtask j} ∇l(z; θ_i) )
    I[x, j]   = Σ_i  η_i · S_i[x, j]
    score(x)  = max_j I[x, j]

Two details that are easy to invert. Validation features are averaged WITHIN a subtask
BEFORE the cosine, not after -- mean-then-cosine and cosine-then-mean are different
quantities, and the paper's is the former. And η_i is the learning rate the epoch actually
ran at, read from each checkpoint's meta rather than assumed constant, because a cosine
schedule makes the later checkpoints count for less.

`max` over subtasks is reported as the headline because it is what LESS specifies, but
mean and min ship alongside it. The reason is a real weakness of max worth seeing rather
than hiding: a row that is actively harmful to two subtasks still scores well if it is
excellent for the third. Comparing the three rankings shows whether the subtasks want the
same data or are pulling against each other.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_split(grad_dir: Path, split: str) -> dict[str, dict]:
    """Reassemble every shard of a split into one matrix per checkpoint."""
    shards: dict[str, list[dict]] = {}
    for f in sorted(grad_dir.glob(f"{split}_ckpt_epoch*_shard*.pt")):
        blob = torch.load(f, map_location="cpu", weights_only=False)
        shards.setdefault(blob["checkpoint"], []).append(blob)
    assert shards, f"no {split} shards under {grad_dir}"

    out = {}
    for ckpt, parts in sorted(shards.items()):
        seeds = {p["proj_seed"] for p in parts} | {p["dim"] for p in parts}
        assert len({p["proj_seed"] for p in parts}) == 1 and len({p["dim"] for p in parts}) == 1, (
            f"{ckpt}: shards disagree on projection basis ({seeds}); their cosines would "
            f"be meaningless")
        feats = torch.cat([p["features"] for p in parts])
        ids = [i for p in parts for i in p["less_ids"]]
        subs = [s for p in parts for s in p["subtasks"]]
        assert len(set(ids)) == len(ids), f"{ckpt}: duplicate rows across shards"
        out[ckpt] = {"features": feats, "less_ids": ids, "subtasks": subs,
                     "lr_mean": parts[0]["lr_mean"], "dim": parts[0]["dim"],
                     "proj_seed": parts[0]["proj_seed"]}
    return out


def subtask_means(val: dict) -> tuple[torch.Tensor, list[str]]:
    """Average validation features within each subtask (mean BEFORE cosine)."""
    names = sorted({s for s in val["subtasks"] if s})
    assert names, "validation rows carry no `subtask` metadata"
    rows = []
    for name in names:
        idx = [i for i, s in enumerate(val["subtasks"]) if s == name]
        rows.append(val["features"][idx].mean(dim=0))
    return torch.stack(rows), names


def influence(train: dict[str, dict], val: dict[str, dict]) -> tuple[torch.Tensor, list[str], list[str], dict]:
    """Accumulate lr-weighted cosine similarity across checkpoints.

    Returns:
        (I [N, m], less_ids, subtask names, diagnostics).
    """
    ckpts = sorted(set(train) & set(val))
    assert ckpts, f"no checkpoint appears in both splits: {sorted(train)} vs {sorted(val)}"
    if set(train) != set(val):
        print(f">>> WARNING: using {len(ckpts)} shared checkpoints; train has "
              f"{sorted(set(train) - set(val))} without a validation counterpart")

    ids = train[ckpts[0]]["less_ids"]
    total, per_ckpt = None, {}
    names: list[str] = []
    for ckpt in ckpts:
        assert train[ckpt]["less_ids"] == ids, f"{ckpt}: train row order differs"
        assert train[ckpt]["proj_seed"] == val[ckpt]["proj_seed"], (
            f"{ckpt}: train and validation were projected with different seeds "
            f"({train[ckpt]['proj_seed']} vs {val[ckpt]['proj_seed']}) — their cosines "
            f"compare unrelated bases")
        vmeans, names = subtask_means(val[ckpt])
        tn = torch.nn.functional.normalize(train[ckpt]["features"], dim=1)
        vn = torch.nn.functional.normalize(vmeans, dim=1)
        sim = tn @ vn.T                                   # [N, m]
        eta = float(train[ckpt]["lr_mean"])
        per_ckpt[ckpt] = sim
        total = eta * sim if total is None else total + eta * sim
    return total, ids, names, per_ckpt


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().to(torch.float64)
    rb = b.argsort().argsort().to(torch.float64)
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grads", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--control-grads", type=Path, default=None,
                    help="a second grads dir whose val split is the negative control")
    ap.add_argument("--pool", type=Path, default=Path("data/less/d_full.jsonl"),
                    help="D, for joining trait metadata onto the ranking")
    ap.add_argument("--warmup-ids", type=Path,
                    default=Path("data/less/d_warmup_ids.json"))
    ap.add_argument("--topk", type=int, default=220, help="K for the top-K diagnostics")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train = load_split(args.grads, "train")
    val = load_split(args.grads, "val")
    inf, ids, names, per_ckpt = influence(train, val)
    print(f">>> I: {tuple(inf.shape)} over subtasks {names}")

    score_max, score_mean, score_min = inf.max(1).values, inf.mean(1), inf.min(1).values

    meta_by_id, warm_ids = {}, set()
    if args.pool.exists():
        for line in args.pool.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                meta_by_id[r["metadata"]["less_id"]] = r["metadata"]
    if args.warmup_ids.exists():
        warm_ids = set(json.loads(args.warmup_ids.read_text(encoding="utf-8"))["less_ids"])

    # Nothing here is stored as an aggregate ONLY. `score_max` is what LESS specifies and
    # what the ranking sorts by, but max is lossy in a way that matters: it discards
    # whether a row was picked for helping one subtask a lot or all three a little, and it
    # hides rows that are actively harmful to two subtasks while excelling at a third. So
    # the full m-vector (`per_subtask`) and the per-checkpoint cosines behind it
    # (`per_checkpoint`, m values per warmup epoch) are both written out. Re-deriving any
    # aggregation later is then a file read rather than 10 GPU-hours.
    ck_order = sorted(per_ckpt)
    order = score_max.argsort(descending=True).tolist()
    ranked = []
    for rank, i in enumerate(order):
        md = meta_by_id.get(ids[i], {})
        ranked.append({
            "rank": rank, "less_id": ids[i],
            "score_max": float(score_max[i]), "score_mean": float(score_mean[i]),
            "score_min": float(score_min[i]),
            "per_subtask": {n: float(inf[i, j]) for j, n in enumerate(names)},
            "per_checkpoint": {c: {n: float(per_ckpt[c][i, j]) for j, n in enumerate(names)}
                               for c in ck_order},
            "argmax_subtask": names[int(inf[i].argmax())],
            "trait_id": md.get("trait_id"), "in_warmup": ids[i] in warm_ids,
        })
    (args.out / "scores.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in ranked), encoding="utf-8")

    # The same numbers as dense tensors, for analysis that would rather not parse 2,203
    # JSON lines: I is [N, m], per-checkpoint is [n_ckpt, N, m], and the row/column labels
    # travel with them so the axes cannot be misread.
    torch.save({"influence": inf, "less_ids": ids, "subtasks": names,
                "checkpoints": ck_order,
                "per_checkpoint": torch.stack([per_ckpt[c] for c in ck_order]),
                "lr_weights": {c: float(train[c]["lr_mean"]) for c in ck_order}},
               args.out / "influence.pt")

    # Bare ordered id lists, one per subtask plus the three aggregations. These exist
    # because the headline `max` ranking is ONE choice among several and, as measured here,
    # a lopsided one -- selecting a training set per subtask, or by mean, needs nothing but
    # an ordering, and an ordering is a few tens of KB rather than the 1.2GB datastore it
    # came from. Descending influence; first line is the most useful row.
    rank_dir = args.out / "rankings"
    rank_dir.mkdir(exist_ok=True)
    orderings = {f"by_{n}": inf[:, j] for j, n in enumerate(names)}
    orderings.update({"by_max": score_max, "by_mean": score_mean, "by_min": score_min})
    for label, col in orderings.items():
        idx = col.argsort(descending=True).tolist()
        (rank_dir / f"{label}.txt").write_text(
            "".join(f"{ids[i]}\n" for i in idx), encoding="utf-8")
    (rank_dir / "README.md").write_text(
        "# Rankings\n\nOne file per ordering, most influential row first, one `less_id` per\n"
        "line. Join back to the pool on `metadata.less_id` in\n"
        "`matboz/synthdoc-v2-difficult-advice` (ids are `<scenario_id>#<pool index>`).\n\n"
        + "".join(f"- `{k}.txt`\n" for k in sorted(orderings))
        + "\nThe per-subtask files are not permutations of one another: measured pairwise\n"
          "Spearman between subtasks was 0.45-0.65, and top-220 by `codebase_resisted`\n"
          "shares only 66/220 rows with top-220 by `max`.\n", encoding="utf-8")
    print(f">>> rankings ({len(orderings)}) -> {rank_dir}")

    # --- checks that decide whether any of this is trustworthy -----------------
    diag: dict = {"n_rows": len(ids), "subtasks": names,
                  "checkpoints": {c: train[c]["lr_mean"] for c in sorted(train)},
                  "dim": train[sorted(train)[0]]["dim"]}

    # 1. Do the checkpoints agree? Uncorrelated per-checkpoint rankings mean the warmup
    #    never reached a regime where gradients carry signal, and the sum is averaging noise.
    ck = sorted(per_ckpt)
    diag["checkpoint_rank_agreement"] = {
        f"{a}|{b}": round(_spearman(per_ckpt[a].max(1).values,
                                    per_ckpt[b].max(1).values), 4)
        for x, a in enumerate(ck) for b in ck[x + 1:]}

    # 2. Do the subtasks want the same data, or compete? (the cost of `max`)
    diag["subtask_rank_agreement"] = {
        f"{names[a]}|{names[b]}": round(_spearman(inf[:, a], inf[:, b]), 4)
        for a in range(len(names)) for b in range(a + 1, len(names))}
    diag["aggregation_agreement"] = {
        "max|mean": round(_spearman(score_max, score_mean), 4),
        "max|min": round(_spearman(score_max, score_min), 4)}

    # 3. Is the top-K enriched for anything, or does it look like a random draw?
    top = ranked[:args.topk]
    def _dist(rows):
        c: dict = {}
        for r in rows:
            c[r["trait_id"]] = c.get(r["trait_id"], 0) + 1
        return {k: round(v / len(rows), 4) for k, v in sorted(c.items(), key=lambda kv: -kv[1])}
    diag["topk"] = {
        "k": args.topk,
        "trait_distribution": _dist(top), "pool_trait_distribution": _dist(ranked),
        "argmax_subtask_share": {n: round(sum(r["argmax_subtask"] == n for r in top) / len(top), 4)
                                 for n in names},
        # Base rate for the line above: the share of the SCORED rows that were warmup
        # rows. Dividing the full warmup list by the scored count instead reads >1 whenever
        # a subset is scored, and silently overstates the base rate even when it does not.
        "warmup_rows_in_topk": sum(r["in_warmup"] for r in top),
        "warmup_share_of_scored": round(
            sum(r["in_warmup"] for r in ranked) / max(1, len(ranked)), 4)}

    # 4. Negative control: an unrelated target must NOT produce the same ranking.
    if args.control_grads:
        cval = load_split(args.control_grads, "val")
        cinf, cids, _, _ = influence(train, cval)
        assert cids == ids, "control ranking covers different rows"
        cscore = cinf.max(1).values
        a = set(score_max.argsort(descending=True)[:args.topk].tolist())
        b = set(cscore.argsort(descending=True)[:args.topk].tolist())
        diag["negative_control"] = {
            "topk_overlap": round(len(a & b) / args.topk, 4),
            "spearman_vs_real": round(_spearman(score_max, cscore), 4),
            "expected_overlap_if_random": round(args.topk / len(ids), 4)}

    (args.out / "diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(json.dumps(diag, indent=2))
    print(f"\n>>> {len(ranked)} rows ranked -> {args.out / 'scores.jsonl'}")


if __name__ == "__main__":
    main()
