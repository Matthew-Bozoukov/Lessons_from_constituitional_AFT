# ABOUTME: Package the train-vs-eval constitution-reference stats into one artifact and push it.
# ABOUTME: Run: uv run python scratch/publish_trait_ref_artifact.py [--no-push]
"""How much of the constitution survives from training traces into eval-time reasoning.

Four arms, three trace populations (the synthetic training corpus each arm was trained on, its
MASK reasoning, its ODCV rollout reasoning), and three measurements per trace: a keyword regex,
an LLM "does it reference the document" pass, and an LLM "does it invoke a SPECIFIC principle"
pass. The answer the artifact exists to record: delib's traces cite specific principles in ~98%
of TRAINING rows and ~15-26% of MASK rows, and ~0.4% of ODCV rows; the DA arms are ~0 throughout.

Nothing here re-runs a model: it reads the label files written by
scratch/classify_constitution_refs.py and the corpora/eval runs they were drawn from.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.infra.huggingface import push_run_dir  # noqa: E402
from src.naming import artifact_name  # noqa: E402

# The append-only JSONL cache, NOT labels.csv: each run rewrites the csv from the arms IT
# sampled, so the delib-only and DA-only runs of the same scheme overwrite each other there.
LABELS = {
    ("reference", "evals"): "output/odcv/2026-09-17_constitution_reference_classification/labels.jsonl",
    ("reference", "training"): "output/odcv/2026-09-17_constref_training/labels.jsonl",
    ("traits", "training"): "output/odcv/2026-09-17_traitref_training/labels.jsonl",
    ("traits", "evals"): "output/odcv/2026-09-17_traitref_evals/labels.jsonl",
}
CORPORA = {  # arm -> (corpus repo, the adapter it trained)
    "da-7": ("dougalldeepmind/2026-09-14-da-synth", "dougalldeepmind/2026-09-15-qwen36-0-da-7"),
    "delib-7": ("dougalldeepmind/2026-09-16-delib-synth", "dougalldeepmind/2026-09-16-qwen36-0-delib-7"),
    "delib-sonnet-7": ("dougalldeepmind/2026-09-16-delib-sonnet-synth", "dougalldeepmind/2026-09-16-qwen36-0-delib-sonnet-7"),
    "da-qwen-7": ("dougalldeepmind/2026-09-16-da-qwen-synth", "dougalldeepmind/2026-09-17-qwen36-0-da-qwen-7"),
}
SCORES = {  # the arms' published eval results, for context
    "da-7": {"mask_honesty": 74.2, "odcv_mr_pct": 8.3},
    "delib-sonnet-7": {"mask_honesty": 72.3, "odcv_mr_pct": 17.1},
    "da-qwen-7": {"mask_honesty": 81.5, "odcv_mr_pct": 17.9},
    "delib-7": {"mask_honesty": 69.3, "odcv_mr_pct": 29.2},
}
CONST = re.compile(r"constitution", re.I)
LEAK = re.compile(r"(no|not|don'?t|without|never|avoid|shouldn'?t)[^.\n]{0,40}"
                  r"(mention|cite|refer|quote|reference)[^.\n]{0,25}constitution", re.I)


def label_frame() -> pd.DataFrame:
    frames = []
    for (scheme, source), path in LABELS.items():
        df = pd.DataFrame([json.loads(l) for l in (REPO / path).open()])
        df["scheme"] = scheme
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["scheme", "eval", "arm", "unit"])


def regex_training() -> dict:
    """The keyword measurement, recomputed here so the artifact is self-contained."""
    from huggingface_hub import hf_hub_download
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    local = {"delib-7": REPO / "output/synth_delib/20260916_181728/dataset.jsonl",
             "da-qwen-7": REPO / "output/synth_da_qwen/20260916_215417/dataset.jsonl"}
    out = {}
    for arm, (repo, _) in CORPORA.items():
        path = local.get(arm) or hf_hub_download(repo, "dataset.jsonl", repo_type="dataset")
        traces = []
        for line in open(path):
            row = json.loads(line)
            a = [m for m in row["messages"] if m["role"] == "assistant"][-1]
            traces.append(a.get("reasoning_content") or "")
        s = pd.Series(traces)
        out[arm] = {"rows": len(s),
                    "mentions_constitution_pct": round(100 * s.str.contains(CONST).mean(), 1),
                    "leak_check_pct": round(100 * s.str.contains(LEAK).mean(), 1)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    out = REPO / "output" / "trait_ref_artifact"
    out.mkdir(parents=True, exist_ok=True)
    df = label_frame()
    df.to_csv(out / "labels.csv", index=False)

    def pct(scheme, ev, label):
        sub = df[(df.scheme == scheme) & (df["eval"] == ev)]
        if sub.empty:
            return {}
        t = sub.groupby("arm")["label"].apply(lambda s: round(100 * (s == label).mean(), 1))
        return t.to_dict()

    summary = {
        "question": "how much of the constitution survives from training traces into eval-time reasoning",
        "n_traces_labelled": {f"{scheme}/{ev}": int(n)
                              for (scheme, ev), n in df.groupby(["scheme", "eval"]).size().items()},
        "references_the_document_pct": {ev: pct("reference", ev, "substantive") for ev in ("training", "mask", "odcv")},
        "invokes_a_specific_principle_pct": {ev: pct("traits", ev, "trait_specific") for ev in ("training", "mask", "odcv")},
        "leak_check_pct_llm": {ev: {k: round(100 * v, 1) for k, v in
                                    df[(df.scheme == "traits") & (df["eval"] == ev)]
                                    .groupby("arm")["leak_check"].mean().dropna().items()}
                               for ev in ("training", "mask", "odcv")},
        "training_corpus_regex": regex_training(),
        "eval_scores": SCORES,
        "corpora": {a: c for a, (c, _) in CORPORA.items()},
        "adapters": {a: m for a, (_, m) in CORPORA.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    ref, ts = summary["references_the_document_pct"], summary["invokes_a_specific_principle_pct"]
    rgx = summary["training_corpus_regex"]
    lines = ["# Constitution references: training traces vs eval traces", "",
             "Every arm trains on 700 synthetic rows and the SAME 9,300 replay rows; only the",
             "synthetic share differs. delib's generation prompt tells the teacher to cite the",
             "constitution's principles, DA's never mentions a constitution.", "",
             "## Invokes a SPECIFIC principle (LLM-labelled, %)", "",
             "| arm | training corpus | MASK | ODCV | MASK honesty | ODCV MR |",
             "|---|---|---|---|---|---|"]
    for arm in ("delib-7", "delib-sonnet-7", "da-7", "da-qwen-7"):
        lines.append(f"| {arm} | {ts['training'].get(arm, 'n/a')} | {ts['mask'].get(arm, 'n/a')} | "
                     f"{ts['odcv'].get(arm, 'n/a')} | {SCORES[arm]['mask_honesty']} | {SCORES[arm]['odcv_mr_pct']}% |")
    lines += ["", "Trait scheme: a near-paraphrase of a principle counts, even with the document never named.",
              "", "## References the document at all (LLM-labelled, %)", "",
              "| arm | training corpus | MASK | ODCV |", "|---|---|---|---|"]
    for arm in ("delib-7", "delib-sonnet-7", "da-7", "da-qwen-7"):
        lines.append(f"| {arm} | {ref['training'].get(arm, 'n/a')} | {ref['mask'].get(arm, 'n/a')} | "
                     f"{ref['odcv'].get(arm, 'n/a')} |")
    lines += ["", "Reference scheme: needs the DOCUMENT invoked as an authority. The DA arms are ~0 here,",
              "so the classifier is not over-firing; their trait-scheme training numbers above are",
              "paraphrase of principle CONTENT in data that never mentions a constitution.",
              "", "## Training corpus, keyword measurement (%)", "",
              "| arm | rows | mentions 'constitution' | of which a don't-leak self-check |",
              "|---|---|---|---|"]
    for arm in ("delib-7", "delib-sonnet-7", "da-7", "da-qwen-7"):
        r = rgx[arm]
        lines.append(f"| {arm} | {r['rows']} | {r['mentions_constitution_pct']} | {r['leak_check_pct']} |")
    lines += ["", "delib-7's teacher wrote a \"No mention of constitution? Checked\" self-check into",
              "most of its traces, a habit trained by the answer-side non-disclosure clause; the",
              "keyword count cannot tell that apart from deliberation, which is why the LLM passes exist.",
              "", "Traces labelled: " + json.dumps(summary["n_traces_labelled"]), ""]
    (out / "summary.md").write_text("\n".join(lines))

    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    fields = {
        "experiment": "constitution references in reasoning traces, training corpus vs eval time "
                      "(MASK, ODCV), for the four constitutional-SFT arms — Callum 2026-09-14: "
                      "'have a look at the inner thoughts of the trained models on these evals, and "
                      "see whether they reference the constitution'",
        "date_generated": "2026-09-17",
        "constitution": "constitutions/abridged/constitution.md",
        "source_repo": f"https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ {sha}",
        "models": {"classifier": "google/gemini-3-flash-preview (temperature 0)",
                   "arms": {a: m for a, (_, m) in CORPORA.items()}},
        "generation_config": {"sampling": "temperature 0.0, max_tokens 250, trace truncated to 40k chars",
                              "sampling_of_traces": "uniform at random over the pooled traces of all arms, "
                                                    "random.Random(0); 1,000 MASK + 1,000 ODCV; training corpora labelled in full",
                              "schemes": {"reference": "none | leak_check | substantive",
                                          "traits": "none | generic | trait_specific, plus leak_check flag and principle numbers"}},
        "schema": "labels.csv: scheme, eval (training|mask|odcv), arm, unit, turn, label, traits, "
                  "leak_check, quote, chars. summary.json: the aggregate rates. summary.md: the headline table.",
        "provenance": "uv run python scratch/classify_constitution_refs.py --source {evals,training} "
                      "--scheme {reference,traits} [--arms delib-7,delib-sonnet-7]; then "
                      "uv run python scratch/publish_trait_ref_artifact.py",
    }
    name = artifact_name("train-vs-eval-trait-ref")
    print(f">>> artifact name: {name}")
    print((out / "summary.md").read_text())
    if args.no_push:
        print(">>> --no-push: not uploading")
        return 0
    url = push_run_dir(out, name, fields)
    print(f">>> pushed {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
