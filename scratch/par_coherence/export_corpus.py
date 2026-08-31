# ABOUTME: Turn a full rewrite run (records.jsonl) into the coherent PAR-716 corpus -- the original
# ABOUTME: five-turn rows with turn 4 replaced -- and push it to HF with the required card.
# Run: uv run python scratch/par_coherence/export_corpus.py --run output/par_coherence/full_<ts> [--push]
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, ".")
from src.huggingface import push_files  # noqa: E402

CORPUS = ("LASR-Callum/2026-08-26-post-action-retrospection-716", "dataset.jsonl")
REPO = "LASR-Callum/2026-08-28-post-action-retrospection-716-coherent"


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def origin() -> str:
    return subprocess.check_output(
        ["git", "remote", "get-url", "origin"], text=True
    ).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    run = Path(args.run)
    recs = [json.loads(l) for l in (run / "records.jsonl").open(encoding="utf-8")]
    meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
    failed = [r["scenario_id"] for r in recs if not r["ok"]]
    assert not failed, (
        f"{len(failed)} rows failed the rewrite; re-run them with --ids first: {failed}"
    )
    assert len(recs) == 716 and len({r["scenario_id"] for r in recs}) == 716, len(recs)

    corpus = {
        json.loads(l)["metadata"]["scenario_id"]: json.loads(l)
        for l in open(hf_hub_download(*CORPUS, repo_type="dataset"), encoding="utf-8")
    }
    prompt_sha = hashlib.sha256(
        (meta["system_prompt"] + meta["user_prompt"]).encode()
    ).hexdigest()[:12]
    out_rows = []
    for r in recs:
        row = json.loads(json.dumps(corpus[r["scenario_id"]]))  # deep copy
        m = row["messages"]
        assert (
            m[4]["role"] == "assistant" and m[4]["content"] == r["before"]["response"]
        )
        m[4]["content"] = r["after"]["response"]
        m[4]["reasoning_content"] = r["after"]["reasoning"]
        row["metadata"]["supervise"] = "final"
        row["metadata"]["rewrite"] = {
            "kind": "coherence_v5",
            "run": run.name,
            "model": meta["args"]["model"],
            "temperature": meta["args"]["temperature"],
            "prompt_sha256_12": prompt_sha,
            "changes": r["changes"],
            "attempts": len(r["attempts"]),
        }
        out_rows.append(row)

    out = run / "dataset.jsonl"
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows),
        encoding="utf-8",
    )
    traits = Counter(r["metadata"]["trait_id"] for r in out_rows)
    ok = [r for r in recs if r["ok"]]
    after_rate = lambda k: 100 * sum(bool(r["props_after"][k]) for r in ok) / len(ok)  # noqa: E731
    before_rate = lambda k: 100 * sum(bool(r["props_before"][k]) for r in ok) / len(ok)  # noqa: E731
    stats = {
        "n": len(out_rows),
        "per_trait": dict(sorted(traits.items())),
        "trace_decides_wide": {
            "before": before_rate("trace_decides_wide"),
            "after": after_rate("trace_decides_wide"),
        },
        "reply_decides_wide": {
            "before": before_rate("reply_decides_wide"),
            "after": after_rate("reply_decides_wide"),
        },
        "reply_firm_strict": {
            "before": before_rate("reply_firm"),
            "after": after_rate("reply_firm"),
        },
        "trace_commits_strict": {
            "before": before_rate("trace_commits"),
            "after": after_rate("trace_commits"),
        },
        "coherent_strict": {
            "before": before_rate("coherent"),
            "after": after_rate("coherent"),
        },
        "decision_lead_formula_after": sum(
            bool(r["props_after"]["decision_lead_formula"]) for r in ok
        ),
        "retries_used_rows": sum(len(r["attempts"]) > 1 for r in recs),
        "tokens_in": sum(r["usage"]["in"] for r in recs),
        "tokens_out": sum(r["usage"]["out"] for r in recs),
    }
    (run / "corpus_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))
    print(f"wrote {out}")
    if not args.push:
        return
    url = push_files(
        [
            out,
            run / "records.jsonl",
            run / "summary.md",
            run / "run_meta.json",
            run / "corpus_stats.json",
        ],
        args.repo,
        {
            "title": "Post-action retrospection 716 -- coherent rewrite (arm 1 of the PAR coherence experiment)",
            "experiment": (
                "The exact 716 five-turn PAR rows that trained LASR-Callum/qwen3.6-27b-lora-t2-9284-par716-r64-dynbatch "
                "(mixture 2026-08-26-table2-9284-par716-train @ 42c8a74), with ONLY the trained turn (turn 4: private "
                "reasoning + reply) rewritten by Sonnet 5 so the reasoning ENDS on a first-person decision (what it "
                "won't do, per action, and what it will do instead) and the reply ENACTS that same decision -- stated "
                "plainly, help framed as what it will do, nothing after the decision reopening it. Turns 1-3, the "
                "constitution, the scenarios, the bare refusal and the pushback are byte-identical to the parent corpus; "
                "length held within ±15% per channel. Motivation: 2026-08-28 diagnostics -- PAR's ODCV deficit vs "
                "difficult advice is a trigger-rate gap (trained voice fires 59% vs 68%, safer when fired), and the "
                "channel-swap result says the voice only protects when the trace's decision is the reply's decision "
                "(P(reply firm | trace commits): grok 94%, PAR 41%)."
            ),
            "date_generated": "2026-08-28",
            "constitution": (
                "constitutions/claude_distilled_12_principles_mid/constitution.md (9 principles), inherited unchanged "
                f"from {CORPUS[0]}; the rewriter sees only the row's target principle, never the document"
            ),
            "source_repo": f"{origin()} @ {git_sha()}",
            "models": f"rewrite: {meta['args']['model']} via OpenRouter (provider pin: anthropic), temperature {meta['args']['temperature']}, hidden reasoning off; parent rows: as in {CORPUS[0]}",
            "generation_config": json.dumps(
                {
                    "script": "scratch/par_coherence/rewrite.py --all",
                    "run": run.name,
                    "prompt_sha256_12": prompt_sha,
                    "lint": "DA trained-turn ban list + decision-lead formula + length ±15%, 4 retries",
                    **{
                        k: v
                        for k, v in meta["args"].items()
                        if k in ("temperature", "tol", "retries", "seed")
                    },
                }
            ),
            "schema": (
                "dataset.jsonl: {messages: [system, user, assistant(bare refusal), user(pushback), assistant{content, "
                "reasoning_content}], metadata: parent metadata + supervise: final + rewrite{kind, run, model, "
                "temperature, prompt_sha256_12, changes, attempts}}. records.jsonl: per row before/after texts, "
                "lint attempts, lexical proxies (props_before/props_after), token usage. summary.md: the proxy table."
            ),
            "provenance": (
                "uv run python scratch/par_coherence/rewrite.py --all (branch par-coherence) then "
                "uv run python scratch/par_coherence/export_corpus.py --run <run> --push"
            ),
            "parent_corpus": f"hf.co/datasets/{CORPUS[0]} ({CORPUS[1]})",
            "parent_mixture": "hf.co/datasets/LASR-Callum/2026-08-26-table2-9284-par716-train @ 42c8a74 (defines the 716 ids)",
            "corpus_stats": json.dumps(stats),
        },
        private=False,
        front_matter={
            "configs": [{"config_name": "default", "data_files": "dataset.jsonl"}],
            "tags": [
                "training-data",
                "pipeline:post_action_retrospection",
                "rewrite:coherence",
            ],
        },
    )
    print(url)


if __name__ == "__main__":
    main()
