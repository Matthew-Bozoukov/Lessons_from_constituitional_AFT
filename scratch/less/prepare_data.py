# ABOUTME: Stages the LESS inputs locally — the full pool D, the seeded warmup split
# ABOUTME: Dwarmup, and the Tulu3 negative-control validation set.

"""Stage every dataset the LESS run needs, deterministically.

    uv run python scratch/less/prepare_data.py --out data/less

Writes into --out:
    d_full.jsonl        the scoring pool D (matboz/synthdoc-v2-difficult-advice, 2203 rows)
    d_warmup.jsonl      a seeded --warmup-frac sample of D, the LoRA warmup training set
    d_warmup_ids.json   the sampled row ids, so warmup rows stay identifiable in the
                        final ranking (they are scored like everything else, but their
                        scores are partly self-influence and you may want them excluded)
    dval_control.jsonl  the negative control: Tulu3 rows posing as a target behaviour

Every row carries a stable `less_id`. D's own ids are not unique -- scenario_id repeats
across the trait blocks -- so the id is `<scenario_id>#<row index>`, assigned once here
and used by every later stage to join gradients back to rows.

The negative control exists to answer a question the influence numbers cannot answer
alone: if a deliberately unrelated target produces a similar top-K, then the ranking is
reporting dataset priors rather than influence on OUR behaviour, and nothing downstream
means what it claims to. Tulu3 is the right control precisely because it is competent,
in-format instruction-following data with no relationship to the t2synth behaviours.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

D_REPO = "matboz/synthdoc-v2-difficult-advice"
D_FILE = "stage_7_sft.jsonl"
TULU_REPO = "allenai/tulu-3-sft-mixture"


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def load_pool() -> list[dict]:
    """Fetch D and stamp each row with a unique, stable `less_id`."""
    from src.infra.huggingface import hf_download

    local = hf_download(D_REPO, D_FILE, repo_type="dataset")
    rows = [json.loads(line) for line in
            Path(local).read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, r in enumerate(rows):
        meta = r.setdefault("metadata", {})
        meta["less_id"] = f"{meta.get('scenario_id', 'row')}#{i}"
        meta["pool_index"] = i
    return rows


def load_control(n: int, seed: int, max_turns: int = 2) -> list[dict]:
    """Sample Tulu3 down to `n` rows shaped like a LESS validation subtask.

    Rows are restricted to a single user/assistant exchange so the control matches the
    real Dval's shape; a length filter keeps them inside the same sequence budget.

    Note the one real asymmetry with the true Dval, which is unavoidable rather than an
    oversight: Tulu3 rows carry NO reasoning trace, so they render with an empty think
    marker that the generation-boundary rule masks wholly. The control therefore measures
    influence on a no-CoT target. That is the correct reading of "unrelated behaviour"
    here -- but it means a low overlap with the real ranking is partly attributable to the
    CoT difference, not only the task difference. Stated so the result is not over-read.
    """
    from datasets import load_dataset

    ds = load_dataset(TULU_REPO, split="train", streaming=True)
    picked: list[dict] = []
    for row in ds:
        msgs = row.get("messages") or []
        if len(msgs) != max_turns or msgs[0]["role"] != "user":
            continue
        if msgs[-1]["role"] != "assistant":
            continue
        if sum(len(m.get("content") or "") for m in msgs) > 8000:
            continue
        picked.append({
            "messages": [{"role": m["role"], "content": m["content"]} for m in msgs],
            "metadata": {"subtask": "control_tulu3", "source_file": TULU_REPO,
                         "less_id": f"control#{len(picked)}"},
        })
        if len(picked) >= n * 8:  # oversample, then subsample with the seed
            break
    rng = random.Random(seed)
    rng.shuffle(picked)
    out = picked[:n]
    for i, r in enumerate(out):
        r["metadata"]["less_id"] = f"control#{i}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/less"))
    ap.add_argument("--warmup-frac", type=float, default=0.10,
                    help="fraction of D used to train the warmup LoRA")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control-n", type=int, default=60,
                    help="control set size; match the real Dval (3 subtasks x 20)")
    ap.add_argument("--skip-control", action="store_true",
                    help="skip the Tulu3 download (it streams a large repo)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pool = load_pool()
    _write(args.out / "d_full.jsonl", pool)
    print(f">>> D            {len(pool):>5} rows -> {args.out / 'd_full.jsonl'}")

    n_warm = max(1, round(len(pool) * args.warmup_frac))
    warm = random.Random(args.seed).sample(pool, n_warm)
    warm.sort(key=lambda r: r["metadata"]["pool_index"])  # stable on disk
    _write(args.out / "d_warmup.jsonl", warm)
    (args.out / "d_warmup_ids.json").write_text(
        json.dumps({"seed": args.seed, "frac": args.warmup_frac, "n": len(warm),
                    "less_ids": [r["metadata"]["less_id"] for r in warm]}, indent=2),
        encoding="utf-8")
    print(f">>> Dwarmup      {len(warm):>5} rows ({args.warmup_frac:.0%} of D, seed "
          f"{args.seed}) -> {args.out / 'd_warmup.jsonl'}")

    if args.skip_control:
        print(">>> control      skipped (--skip-control)")
        return
    ctrl = load_control(args.control_n, args.seed)
    _write(args.out / "dval_control.jsonl", ctrl)
    print(f">>> control Dval {len(ctrl):>5} rows (Tulu3) -> "
          f"{args.out / 'dval_control.jsonl'}")


if __name__ == "__main__":
    main()
