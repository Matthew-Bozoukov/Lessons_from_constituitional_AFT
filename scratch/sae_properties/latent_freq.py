# ABOUTME: Cross-corpus frequency of specific SAE latents — is a correlated latent a
# ABOUTME: property of THIS corpus, or of every corpus? GPU-free, reads embed caches.

"""How often do given latents fire in each corpus's channel?

    uv run --project scratch/sae_properties python scratch/sae_properties/latent_freq.py \
        --run e1_70b --channel response --pairs-from corr_difficult_advice_queryxresponse

Reads the latents named in a correlations run's pairs.jsonl (or an explicit --latents
list) and prints their per-corpus document frequency. A correlation that only matters if
it is corpus-specific is checked here rather than assumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPORA = ["difficult_advice", "peer_critique", "courtroom", "post_action_retrospection"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--channel", default="response")
    ap.add_argument("--pairs-from", help="correlations subdir whose pairs.jsonl names the latents")
    ap.add_argument("--latents", help="comma-separated latent ids (alternative to --pairs-from)")
    ap.add_argument("--top", type=int, default=14, help="how many pairs to take latents from")
    ap.add_argument("--out", help="write JSON here")
    args = ap.parse_args()

    from interp_embed import Dataset

    run_dir = REPO_ROOT / "output/sae_properties" / args.run
    watch: dict[int, str] = {}
    if args.pairs_from:
        for line in (run_dir / args.pairs_from / "pairs.jsonl").read_text().splitlines()[: args.top]:
            p = json.loads(line)
            watch[p["a"]] = p["label_a"]
            watch[p["b"]] = p["label_b"]
    if args.latents:
        for x in args.latents.split(","):
            watch.setdefault(int(x), f"latent {x}")

    freqs: dict[str, dict[int, float]] = {}
    for corpus in CORPORA:
        pkl = run_dir / "datasets" / f"{corpus}__{args.channel}.pkl"
        if not pkl.exists():
            print(f"[latent_freq] skip {corpus}: no {args.channel} cache")
            continue
        ds = Dataset.load_from_file(str(pkl), device="cpu")
        acts = ds.latents() > 0
        freqs[corpus] = {k: float(acts[:, k].mean()) for k in watch}
        print(f"[latent_freq] {corpus}: {acts.shape[0]} docs", flush=True)

    rows = [{"latent": k, "label": v, "freq": {c: round(freqs[c][k], 4) for c in freqs}}
            for k, v in watch.items()]
    rows.sort(key=lambda r: -r["freq"].get("difficult_advice", 0))
    for r in rows:
        f = r["freq"]
        cols = "  ".join(f"{c[:4].upper()} {f[c]:.3f}" for c in f)
        print(f"{cols}  | {r['label'][:72]}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
