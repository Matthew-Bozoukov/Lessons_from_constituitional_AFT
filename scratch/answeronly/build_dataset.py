# ABOUTME: Derive an answer-only corpus from a difficult-advice run: the same prompts and the
# ABOUTME: same replies, with every reasoning trace removed, published as its own HF dataset.
"""The reasoning ablation taken to its limit: no reasoning at all.

    uv run python scratch/answeronly/build_dataset.py [--push] [--source <hf id>]

The `ablation` pipeline restructures the reasoning; this removes it. Rows are otherwise
byte-identical to the source -- same system turn, same user turn, same assistant reply, same
metadata -- so an arm trained on this differs from the da arm in one thing: whether a trace
was supervised at all.

Sizing matters here and is the reason this is a script rather than a one-liner. A mixture
share is taken in SUPERVISED TOKENS, and dropping the trace removes just over half of them
per row, so an answer-only source needs roughly twice the rows to buy the same share. The
script reports the pool it produces against the 15% target so the shortfall is a number
rather than a surprise at mix time.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
# 15% of the published base blend's 4,860,265 supervised tokens -- what `synthetic_pct: 15`
# frees and therefore what a source must be able to refill.
TARGET_15 = 729_241


def strip_reasoning(row: dict) -> dict:
    """The row with every reasoning trace removed, nothing else touched."""
    msgs = []
    for m in row["messages"]:
        m2 = {k: v for k, v in m.items() if k != "reasoning_content"}
        msgs.append(m2)
    return {**row, "messages": msgs}


def main() -> None:
    """Build, measure and optionally publish the answer-only corpus."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="dougalldeepmind/2026-10-02-da-synth")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--repo", default=None, help="target repo; default <date>-da-answeronly-synth")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    env = dotenv_values(REPO / ".env")
    tok = env["HF_TOKEN_MATBOZ"]
    api = HfApi(token=tok)
    src_sha = api.dataset_info(args.source, revision=args.revision).sha
    path = hf_hub_download(args.source, "dataset.jsonl", repo_type="dataset",
                           token=tok, revision=args.revision)
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    out = [strip_reasoning(r) for r in rows]

    # The declaration `reasoning: none` is validated at mix time; assert it here so a miss
    # fails in the place that caused it rather than three steps downstream.
    leftover = sum(1 for r in out for m in r["messages"] if (m.get("reasoning_content") or "").strip())
    assert leftover == 0, f"{leftover} turns still carry reasoning_content"
    assert all(any(m["role"] == "assistant" and m["content"].strip() for m in r["messages"])
               for r in out), "a row lost its reply"

    T = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", token=tok)
    rep = [len(T.encode(next(m["content"] for m in reversed(r["messages"])
                             if m["role"] == "assistant"))) for r in out]
    pool = sum(rep)
    print(f">>> {len(out)} rows from {args.source}@{src_sha[:12]}, reasoning removed")
    print(f">>> answer-only supervised pool: {pool:,} tokens (mean {st.mean(rep):.0f}/row)")
    print(f">>> against the 15% target {TARGET_15:,}: {100*pool/TARGET_15:.1f}% "
          f"({'enough' if pool >= TARGET_15 else f'short by {TARGET_15-pool:,} tok '
             f'= ~{(TARGET_15-pool)/st.mean(rep):.0f} rows'})")

    local = REPO / "output/datasets/da_answeronly"
    local.mkdir(parents=True, exist_ok=True)
    f = local / "dataset.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    print(f">>> wrote {f} ({f.stat().st_size/1e6:.1f} MB)")

    if args.push:
        repo = args.repo or "dougalldeepmind/2026-10-03-da-answeronly-synth"
        api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
        api.upload_file(path_or_fileobj=str(f), path_in_repo="dataset.jsonl",
                        repo_id=repo, repo_type="dataset")
        card = (
            "---\ntags:\n- training-data\n- kind:synth\n- pipeline:da-answeronly\n"
            "- constitution:claude_distilled_09_principles\n---\n\n"
            "# `da-answeronly` — the difficult-advice corpus with no reasoning\n\n"
            f"| field | value |\n| --- | --- |\n"
            f"| `source` | [{args.source}](https://huggingface.co/datasets/{args.source}) "
            f"@ `{src_sha[:12]}` |\n"
            f"| `rows` | {len(out)} (every source row, none dropped) |\n"
            f"| `change` | `reasoning_content` removed from every message; system turn, user "
            f"turn, assistant reply and metadata byte-identical to the source |\n"
            f"| `answer-only supervised tokens` | {pool:,} (mean {st.mean(rep):.0f}/row) |\n"
            f"| `share it can fund` | {100*pool/4_860_265:.2f}% of the published base blend's "
            f"supervised tokens |\n"
            f"| `declare as` | `reasoning: none` in a mixture source |\n\n"
            "Pairs with the source corpus on the same scenarios: an arm trained on this and an "
            "arm trained on the source differ in whether a reasoning trace was supervised at "
            "all. Because dropping the trace removes just over half the supervised tokens per "
            "row, this corpus funds a smaller share than its row count suggests — see the "
            "table above before declaring `synthetic_pct`.\n"
        )
        api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                        repo_id=repo, repo_type="dataset")
        print(f">>> pushed https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
