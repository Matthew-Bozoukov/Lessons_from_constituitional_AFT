# ABOUTME: Correlations (paper §4.2/§5.2, the Tulu-3 loop) — NPMI between latent pairs
# ABOUTME: across two channels of one corpus, then LLM-judge verification of top pairs.

"""Find unexpected latent-pair correlations, e.g. query-latents × response-latents.

Run (API-only; needs the two channels' embed caches):

    uv run --project scratch/sae_properties python scratch/sae_properties/correlate.py \
        --config configs/properties/sae_diff.yaml run=e1_70b \
        corr.corpus=difficult_advice corr.channel_a=query corr.channel_b=response

Method, following the paper: binarize both channels' doc×latent matrices (aligned by
doc_id); keep latents active in [min_freq, max_freq] of docs; NPMI(i,j) from
co-occurrence counts; drop pairs whose labels look alike (token-Jaccard — "interesting"
means correlated but semantically unrelated); judge-verify the top pairs by reading a
document sample (label_a against channel_a text, label_b against channel_b text) →
verified NPMI + conditional probabilities. Outputs under
<out_root>/<run>/corr_<corpus>_<cha>x<chb>/.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import write_run_meta  # noqa: E402

JUDGE_PROMPT = """You are an expert at analyzing whether text exhibits a specific property.

PROPERTY: {label}

TEXT TO ANALYZE:
{text}

Does the text exhibit the property? Answer with exactly one word: YES or NO."""


def load_channel(ds_dir: Path, corpus: str, channel: str):
    from interp_embed import Dataset

    ds = Dataset.load_from_file(str(ds_dir / f"{corpus}__{channel}.pkl"), device="cpu")
    df = ds.pandas()
    acts = ds.latents()  # docs × d_sae, max-pooled
    labels = ds.feature_labels()
    return df, (acts > 0), labels


def token_jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z]+", a.lower())) - {"the", "a", "of", "in", "and", "or", "to"}
    tb = set(re.findall(r"[a-z]+", b.lower())) - {"the", "a", "of", "in", "and", "or", "to"}
    return len(ta & tb) / max(1, len(ta | tb))


def npmi_pairs(A: np.ndarray, B: np.ndarray, cfg) -> list[dict]:
    """Top NPMI pairs between columns of A (channel_a) and B (channel_b), same doc rows."""
    n = A.shape[0]
    fa, fb = A.mean(0), B.mean(0)
    ka = np.flatnonzero((fa >= cfg.min_freq) & (fa <= cfg.max_freq))
    kb = np.flatnonzero((fb >= cfg.min_freq) & (fb <= cfg.max_freq))
    print(f"[corr] candidate latents: {len(ka)} x {len(kb)} over {n} docs")
    C = A[:, ka].astype(np.float32).T @ B[:, kb].astype(np.float32)  # co-occurrence counts
    pij = C / n
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(pij / np.outer(fa[ka], fb[kb]))
        npmi = pmi / -np.log(pij)
    npmi[C < cfg.min_support] = -1
    flat = np.argsort(npmi, axis=None)[::-1][: cfg.top_k * 40]
    out = []
    for f in flat:
        i, j = np.unravel_index(f, npmi.shape)
        if npmi[i, j] < cfg.npmi_floor:
            break
        out.append({"a": int(ka[i]), "b": int(kb[j]), "npmi": float(npmi[i, j]),
                    "cooc": int(C[i, j]), "freq_a": float(fa[ka[i]]), "freq_b": float(fb[kb[j]])})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/properties/sae_diff.yaml")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = OmegaConf.merge(
        OmegaConf.create({"corr": {"corpus": "difficult_advice", "channel_a": "query",
                                   "channel_b": "response", "min_freq": 0.01, "max_freq": 0.9,
                                   "min_support": 10, "npmi_floor": 0.35, "top_k": 40,
                                   "label_sim_max": 0.34, "verify_pairs": 8,
                                   "verify_docs": 150}}),
        OmegaConf.load(REPO_ROOT / args.config),
        OmegaConf.from_dotlist(args.overrides))
    if not cfg.get("run"):
        raise SystemExit("Pass run=<embed-run-name>.")
    c = cfg.corr
    ds_dir = REPO_ROOT / cfg.embed.out_root / cfg.run / "datasets"
    out_dir = REPO_ROOT / cfg.embed.out_root / cfg.run / f"corr_{c.corpus}_{c.channel_a}x{c.channel_b}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_a, A, labels = load_channel(ds_dir, c.corpus, c.channel_a)
    df_b, B, labels_b = load_channel(ds_dir, c.corpus, c.channel_b)
    labels.update({k: v for k, v in labels_b.items() if k not in labels})

    # Align rows by doc_id — the two channels drop different empty rows.
    ids = sorted(set(df_a.doc_id) & set(df_b.doc_id))
    ia = {d: i for i, d in enumerate(df_a.doc_id)}
    ib = {d: i for i, d in enumerate(df_b.doc_id)}
    A = A[[ia[d] for d in ids]]
    B = B[[ib[d] for d in ids]]
    texts_a = dict(zip(df_a.doc_id, df_a.text))
    texts_b = dict(zip(df_b.doc_id, df_b.text))
    print(f"[corr] aligned docs: {len(ids)}")

    def lab(x):
        return str(labels.get(x, f"latent {x}"))

    pairs = []
    for p in npmi_pairs(A, B, c):
        if token_jaccard(lab(p["a"]), lab(p["b"])) > c.label_sim_max:
            continue  # labels alike -> not "interesting"
        p["label_a"], p["label_b"] = lab(p["a"]), lab(p["b"])
        pairs.append(p)
        if len(pairs) >= c.top_k:
            break
    # Attach one co-activating example per pair.
    for p in pairs:
        ai = [k for k in range(len(ids)) if A[k, p["a"]] and B[k, p["b"]]]
        if ai:
            d = ids[ai[0]]
            p["example"] = {"doc_id": d, "a_text": texts_a[d][:400], "b_text": texts_b[d][:400]}
    (out_dir / "pairs.jsonl").write_text("\n".join(json.dumps(p) for p in pairs))
    print(f"[corr] kept {len(pairs)} interesting pairs -> {out_dir/'pairs.jsonl'}")

    verified = []
    if not args.skip_verify and pairs:
        from interp_embed.llm.utils import get_llm_client

        client = get_llm_client()
        model = str(cfg.diff.judge_model)
        rng = np.random.default_rng(0)
        sample = rng.choice(len(ids), size=min(int(c.verify_docs), len(ids)), replace=False)
        cache: dict[tuple, bool] = {}

        def judge(label: str, text: str) -> bool:
            key = (label, text[:80])
            if key not in cache:
                r = client.chat.completions.create(model=model, max_tokens=4, temperature=0.0,
                    messages=[{"role": "user", "content": JUDGE_PROMPT.format(label=label, text=text[:6000])}])
                cache[key] = "YES" in (r.choices[0].message.content or "").upper()
            return cache[key]

        for p in pairs[: int(c.verify_pairs)]:
            docs = [ids[k] for k in sample]
            with ThreadPoolExecutor(max_workers=16) as ex:
                va = list(ex.map(lambda d: judge(p["label_a"], texts_a[d]), docs))
                vb = list(ex.map(lambda d: judge(p["label_b"], texts_b[d]), docs))
            va, vb = np.array(va), np.array(vb)
            n = len(docs)
            pa, pb, pab = va.mean(), vb.mean(), (va & vb).mean()
            if 0 < pab < 1 and pa > 0 and pb > 0:
                vn = float(np.log(pab / (pa * pb)) / -np.log(pab))
            else:
                vn = None
            verified.append({**{k: p[k] for k in ("a", "b", "npmi", "label_a", "label_b")},
                             "verified_npmi": vn, "p_a": float(pa), "p_b": float(pb),
                             "p_b_given_a": float((va & vb).sum() / max(1, va.sum())),
                             "p_b_given_not_a": float((~va & vb).sum() / max(1, (~va).sum())),
                             "n_docs": n})
            print(f"[corr] verified: {p['label_a'][:40]} x {p['label_b'][:40]} -> NPMI_v={vn}")
        (out_dir / "verified.json").write_text(json.dumps(verified, indent=2))

    write_run_meta(out_dir, OmegaConf.to_container(cfg, resolve=True), extra={"stage": "correlate"})


if __name__ == "__main__":
    main()
