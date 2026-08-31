# ABOUTME: Compare two difficult-advice corpora written by different generators: how
# ABOUTME: separable they are, and on what — length, style, punctuation, structure.

"""Diagnose a generator ablation before training on it.

Run: uv run python scratch/compare_generator_arms.py \
       --arm output/synthdoc_grok_responder_716/<ts>/dataset.jsonl \
       [--baseline_repo LASR-Callum/2026-08-13-haiku45-sonnet45-difficult-advice-diversity-gated-voice-linted]

Why this exists. On 2026-08-17 two corpora in this project were found to leak their arm
label through *who wrote the reply* rather than through quality — peer-critique separated
at AUC 0.9973 (0.85 on length alone) and a model had already been trained on it. A
generator ablation is the same hazard by construction: if a trivial classifier can tell
which model wrote a row, any downstream behavioural difference is attributable to whatever
that classifier found, not to the generator's values.

Four views, in increasing specificity:

  1. SEPARABILITY  5-fold CV AUC from length alone, from bag-of-words, and from both.
     Bag-of-words near 1.0 is EXPECTED between any two models and is not by itself
     disqualifying — the 0.70 gate was built for within-corpus arm leakage. Length is the
     one to worry about: verbosity is a nuisance variable, not a value.
  2. PAIRED FEATURES  When the two corpora answer the SAME questions (as a responder-swap
     arm does), every row pairs by scenario_id. Reported per response AND per 1,000
     characters — the rate column is what separates "longer, so more of everything" from a
     real stylistic difference.
  3. PUNCTUATION  Which apostrophe/quote/dash codepoints each generator emits. Models
     differ here almost perfectly (measured: Sonnet 100% ASCII apostrophes and zero curly
     quotes; grok-4.6 78% Unicode apostrophes and 90% curly quotes), and a model trained
     on one corpus inherits its typography.
  4. LENGTH DECOMPOSITION  Whether the longer arm gets there with more sentences or with
     longer ones, and what the revision stage does to each — the revision step is where
     this project's generators actually diverged.
"""

import json
import re
import statistics as st

import fire
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GATE = 0.70  # the repo's surface_shortcut gate, for reference only

PATTERNS = {
    "contractions": re.compile(r"\b\w+['’](?:s|t|re|ve|ll|d|m)\b", re.I),
    "hedges": re.compile(r"\b(might|could|may|perhaps|possibly|probably|seems?|"
                         r"likely|somewhat|generally)\b", re.I),
    "first person": re.compile(r"\b(i|i'?m|i'?ll|i'?ve|i'?d|my|me)\b", re.I),
    "second person": re.compile(r"\b(you|your|you'?re|you'?ll|yours)\b", re.I),
    "offer phrases": re.compile(r"\b(i can|i could|i'?ll|instead|alternative|"
                                r"you could|we could|option)\b", re.I),
    "refusal phrases": re.compile(r"\b(i (?:will not|won'?t|can'?t|cannot)|"
                                  r"not going to|i'?m not)\b", re.I),
    "numbers": re.compile(r"\b\d[\d,.$%/kK-]*\b"),
    "questions": re.compile(r"\?"),
    "em-dashes": re.compile(r"[—–]|--"),
}
SENT = re.compile(r"[.!?]+(?:\s|$)")
PUNCT = [("ASCII apostrophe '", "'"), ("UNICODE apostrophe ’", "’"),
         ('ASCII quote "', chr(34)), ("curly quote “", "“"),
         ("em-dash —", "—")]


def _read(path_or_repo: str, fn: str | None = None) -> list[dict]:
    p = path_or_repo if fn is None else hf_hub_download(path_or_repo, fn,
                                                        repo_type="dataset")
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def _parts(rows: list[dict]) -> dict:
    """scenario_id -> (response, reasoning). Keyed so two arms can be paired."""
    out = {}
    for r in rows:
        a = [m for m in r["messages"] if m["role"] == "assistant"]
        if a:
            out[r["metadata"]["scenario_id"]] = (
                a[-1].get("content") or "", a[-1].get("reasoning_content") or "")
    return out


def _feats(t: str) -> dict:
    words = re.findall(r"[A-Za-z']+", t)
    sents = [s for s in SENT.split(t) if s.strip()]
    f = {"chars": len(t), "words": len(words), "sentences": len(sents),
         "mean sentence (words)": (len(words) / len(sents)) if sents else 0.0,
         "paragraphs": len([p for p in t.split("\n\n") if p.strip()])}
    f.update({k: len(rx.findall(t)) for k, rx in PATTERNS.items()})
    return f


def _auc(X, y, label: str) -> None:
    clf = make_pipeline(StandardScaler(with_mean=False),
                        LogisticRegression(max_iter=2000))
    s = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    print(f"  {label:22s} AUC {s.mean():.4f} +/- {s.std():.3f}"
          f"   {'SEPARABLE' if s.mean() > GATE else 'ok'}")


def main(arm: str,
         baseline_repo: str = "LASR-Callum/2026-08-13-haiku45-sonnet45-difficult-advice-diversity-gated-voice-linted",
         baseline_file: str = "stage_8_export_sft.jsonl",
         arm_label: str = "arm", base_label: str = "baseline") -> None:
    """Compare a generator arm against the baseline difficult-advice corpus.

    Args:
        arm: Path to the arm's dataset.jsonl.
        baseline_repo: HF dataset repo holding the comparison corpus.
        baseline_file: File within it, in interchange form (messages + metadata).
        arm_label: Display name for the arm.
        base_label: Display name for the baseline.
    """
    load_dotenv()
    a = _parts(_read(arm))
    b = _parts(_read(baseline_repo, baseline_file))
    shared = sorted(set(a) & set(b))
    print(f"{arm_label}: {len(a)} rows | {base_label}: {len(b)} rows | "
          f"paired by scenario_id: {len(shared)}")

    # --- 1. separability -------------------------------------------------------------
    n = min(len(a), len(b))
    rng = np.random.default_rng(0)
    ta = list(rng.permutation([v[0] for v in a.values()])[:n])
    tb = list(rng.permutation([v[0] for v in b.values()])[:n])
    texts, y = ta + tb, np.array([1] * n + [0] * n)
    print(f"\n=== 1. SEPARABILITY ({n} vs {n}, gate {GATE}) ===")
    lens = np.array([[len(t)] for t in texts], dtype=float)
    _auc(lens, y, "length only")
    bow = CountVectorizer(min_df=3, max_features=20000).fit_transform(texts)
    _auc(bow, y, "bag of words")
    print("  (bag-of-words near 1.0 is expected between any two generators; the "
          "length row is the one that confounds a training comparison)")

    # --- 2. paired features ----------------------------------------------------------
    if shared:
        fa = [_feats(a[i][0]) for i in shared]
        fb = [_feats(b[i][0]) for i in shared]
        print(f"\n=== 2. PAIRED FEATURES on {len(shared)} shared questions ===")
        print(f"{'feature':24s} {base_label:>10s} {arm_label:>10s} {'ratio':>7s}"
              f"   {'per-1k ratio':>13s}")
        for k in fa[0]:
            ma, mb = st.median([f[k] for f in fa]), st.median([f[k] for f in fb])
            ra = st.median([f[k] / max(f["chars"], 1) * 1000 for f in fa])
            rb = st.median([f[k] / max(f["chars"], 1) * 1000 for f in fb])
            rat = mb / ma if ma else float("nan")
            rrat = rb / ra if ra else float("nan")
            flag = "  <-- RATE DIFFERS" if ra and (rrat > 1.25 or rrat < 0.8) else ""
            print(f"{k:24s} {mb:10.1f} {ma:10.1f} {rat:7.2f}   {rrat:13.2f}{flag}")

    # --- 3. punctuation fingerprint --------------------------------------------------
    print("\n=== 3. PUNCTUATION FINGERPRINT (% of docs containing) ===")
    print(f"{'character':26s} {base_label:>10s} {arm_label:>10s}")
    for label, ch in PUNCT:
        pa = sum(1 for v in a.values() if ch in v[0]) / max(len(a), 1) * 100
        pb = sum(1 for v in b.values() if ch in v[0]) / max(len(b), 1) * 100
        print(f"{label:26s} {pb:9.1f}% {pa:9.1f}%")

    # --- 4. length decomposition -----------------------------------------------------
    if shared:
        sa = st.median([f["sentences"] for f in fa])
        sb = st.median([f["sentences"] for f in fb])
        wa = st.median([f["mean sentence (words)"] for f in fa])
        wb = st.median([f["mean sentence (words)"] for f in fb])
        print(f"\n=== 4. WHERE THE EXTRA LENGTH COMES FROM ===")
        print(f"  sentence COUNT   {sb:.0f} vs {sa:.0f} = {sb/sa:.2f}x")
        print(f"  sentence LENGTH  {wb:.1f} vs {wa:.1f} words = {wb/wa:.2f}x")
        print(f"  product {sb/sa*wb/wa:.2f}x tracks the char ratio "
              f"{st.median([f['chars'] for f in fb])/st.median([f['chars'] for f in fa]):.2f}x")


if __name__ == "__main__":
    fire.Fire(main)
