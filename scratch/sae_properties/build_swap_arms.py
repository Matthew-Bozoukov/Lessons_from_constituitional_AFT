# ABOUTME: Build matched arms for the domain-concentration swap ablation — verify the
# ABOUTME: unlabeled remainder of the corpus, then stratum-match concentrated vs clean.

"""Prepare a filter/swap ablation of scenario-domain concentration.

The E1 verification labeled 1,000 of difficult-advice v2's 1,968 documents against 15
properties. A naive concentrated-vs-clean split is badly confounded (the clean documents
differ on ethical CONTENT, not just professional setting), so the two arms are built by
exact stratum matching on the properties that must be held constant.

    # 1. label the remaining documents with the same 15-hypothesis detector
    uv run --project scratch/sae_properties python scratch/sae_properties/build_swap_arms.py \
        --run e1_70b --stage extend

    # 2. build matched arms from the full labeled set
    uv run --project scratch/sae_properties python scratch/sae_properties/build_swap_arms.py \
        --run e1_70b --stage match

Outputs `swap_ablation/arm_concentrated.jsonl` and `arm_clean.jsonl` — equal N, matched on
the held-constant properties, differing in scenario domain.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
DIFFING_DIR = Path(__file__).resolve().parent / "third_party" / "interp_embed" / "paper" / "diffing"
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import load_corpus  # noqa: E402

# Hypothesis indices from the E1 run (see diff_difficult_advice/hypotheses.json).
DOMAIN = [1, 2]                 # program-metrics/funder, social-services — the archetype
HOLD_CONSTANT = [12, 13, 14, 0, 5, 6]   # behaviour + ethical content that must not differ


def valid_report(base: Path, n_hyp: int = 15) -> Path:
    cands = []
    for p in glob.glob(str(base / "**" / "verification_report.json"), recursive=True):
        r = json.load(open(p))
        cands.append((r["metadata"]["num_hypotheses"], r["metadata"]["timestamp"], p))
    ok = sorted(c for c in cands if c[0] == n_hyp)
    if not ok:
        raise SystemExit(f"No {n_hyp}-hypothesis verification under {base}")
    return Path(ok[-1][2]).parent


def read_verdicts(vdir: Path, doc_ids: list[str]) -> dict[str, dict[int, bool]]:
    out: dict[str, dict[int, bool]] = defaultdict(dict)
    with open(vdir / "verification_results.csv") as f:
        for row in csv.DictReader(f):
            i = int(row["response_idx"])
            if i < len(doc_ids):
                out[doc_ids[i]][int(row["hypothesis_idx"])] = row["text_verified"].strip().lower() == "true"
    return {k: v for k, v in out.items() if len(v) == 15}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="e1_70b")
    ap.add_argument("--stage", choices=["extend", "match"], required=True)
    ap.add_argument("--config", default="configs/properties/sae_diff.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = OmegaConf.load(REPO_ROOT / args.config)
    run_dir = REPO_ROOT / cfg.embed.out_root / args.run
    diff_dir = run_dir / "diff_difficult_advice"
    out_dir = run_dir / "swap_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = next(OmegaConf.to_container(s) for s in cfg.corpora if s["name"] == "difficult_advice")
    full = load_corpus(spec, "response", limit=None)                     # all 1,968
    done_ids = list(pd.read_csv(run_dir / "datasets/difficult_advice__response.csv").doc_id)

    if args.stage == "extend":
        rest = full[~full.doc_id.isin(done_ids)]
        rest_csv = out_dir / "remainder.csv"
        rest.to_csv(rest_csv, index=False)
        print(f"[swap] labeling the remaining {len(rest)} documents with the same detector")
        subprocess.run([sys.executable, "hypothesis_verifier.py",
                        "-p", str(diff_dir / "hypotheses.json"),
                        "-i", str(rest_csv), "--fields", "text",
                        "-o", str(out_dir / "verify_remainder"),
                        "--judge-model", str(cfg.diff.judge_model),
                        "--max-concurrent", str(cfg.diff.max_concurrency)],
                       cwd=DIFFING_DIR, env=os.environ.copy(), check=True)
        print(f"[swap] done -> {out_dir/'verify_remainder'}")
        return

    # --- match ---
    verdicts = read_verdicts(valid_report(diff_dir / "verify_difficult_advice"), done_ids)
    rest_dir = out_dir / "verify_remainder"
    if rest_dir.exists():
        rest_ids = list(pd.read_csv(out_dir / "remainder.csv").doc_id)
        verdicts.update(read_verdicts(valid_report(rest_dir), rest_ids))
    print(f"[swap] labeled documents: {len(verdicts)}")

    conc, clean = defaultdict(list), defaultdict(list)
    for doc_id, v in verdicts.items():
        stratum = tuple(v[k] for k in HOLD_CONSTANT)
        (conc if any(v[k] for k in DOMAIN) else clean)[stratum].append(doc_id)

    rng = random.Random(args.seed)
    arm_c, arm_k = [], []
    for s in set(conc) | set(clean):
        n = min(len(conc[s]), len(clean[s]))
        if n:
            arm_c += rng.sample(conc[s], n)
            arm_k += rng.sample(clean[s], n)
    print(f"[swap] matched arms: {len(arm_c)} concentrated / {len(arm_k)} clean "
          f"(pools {sum(map(len, conc.values()))} / {sum(map(len, clean.values()))})")

    text = dict(zip(full.doc_id, full.text))
    for name, ids in (("arm_concentrated", arm_c), ("arm_clean", arm_k)):
        rows = [{"doc_id": i, "text": text[i], **{f"prop_{k}": verdicts[i][k] for k in range(15)}} for i in ids]
        (out_dir / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    # Collateral: the held-constant properties must now agree, and the domain ones must not.
    print("\n[swap] balance check (matched arms):")
    for k in range(15):
        pc = sum(verdicts[i][k] for i in arm_c) / max(1, len(arm_c))
        pk = sum(verdicts[i][k] for i in arm_k) / max(1, len(arm_k))
        tag = "DOMAIN (should differ)" if k in DOMAIN else ("held constant" if k in HOLD_CONSTANT else "")
        print(f"   prop {k:2d}  conc {pc:.3f}  clean {pk:.3f}  diff {pc-pk:+.3f}  {tag}")


if __name__ == "__main__":
    main()
