# ABOUTME: Counts supervised tokens in an interchange mixture under the gpt-oss-120b
# ABOUTME: tokenizer, using the same renderer and masking that Tinker training uses.
"""Report supervised vs total tokens for a mixture, overall and per source.

The dataset cards budget with `Qwen/Qwen3.6-27B`, so their token totals do not
describe the gpt-oss arm. This renders each row through the Harmony renderer at the
pinned tokenizer revision and counts positions carrying loss (`weights > 0`), which
is what `token_mean` loss reduction divides by.

Usage:
    uv run python scratch/gptoss_control/count_supervised_tokens.py \
        dougalldeepmind/2026-10-05-plain-mix dougalldeepmind/2026-10-05-msm-mix
"""
import json
import sys
from collections import defaultdict

from dotenv import dotenv_values
from huggingface_hub import hf_hub_download

from src.infra.endpoints.harmony import make_renderer, supervised_examples, TOKENIZER_REVISION


def count(repo: str, renderer, token: str, max_length: int = 8192) -> dict:
    """Render every row of `repo` and total its supervised and context tokens."""
    path = hf_hub_download(repo, "mixture.jsonl", repo_type="dataset", token=token)
    rows = [json.loads(line) for line in open(path)]
    per_source: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "datums": 0,
                                                                 "supervised": 0, "total": 0})
    for row in rows:
        examples = supervised_examples(renderer, row, max_length)
        bucket = per_source[row.get("source", "?")]
        bucket["rows"] += 1
        bucket["datums"] += len(examples)
        for example in examples:
            assert len(example["weights"]) == len(example["target_tokens"]), "weights/target length mismatch"
            bucket["supervised"] += sum(1 for w in example["weights"] if w)
            bucket["total"] += len(example["input_ids"])
    return per_source


def main() -> None:
    token = dotenv_values(".env")["HF_TOKEN_MATBOZ"]
    renderer = make_renderer("medium")
    print(f"tokenizer revision: {TOKENIZER_REVISION}\n")
    for repo in sys.argv[1:]:
        per_source = count(repo, renderer, token)
        tot = {k: sum(v[k] for v in per_source.values()) for k in ("rows", "datums", "supervised", "total")}
        print(f"{repo}")
        print(f"  {'source':24s} {'rows':>6s} {'datums':>7s} {'supervised':>11s} {'total':>10s}  sup%")
        for name, v in sorted(per_source.items(), key=lambda x: -x[1]["supervised"]):
            print(f"  {name:24s} {v['rows']:6d} {v['datums']:7d} {v['supervised']:11,d} "
                  f"{v['total']:10,d}  {100*v['supervised']/v['total']:5.1f}")
        print(f"  {'TOTAL':24s} {tot['rows']:6d} {tot['datums']:7d} {tot['supervised']:11,d} "
              f"{tot['total']:10,d}  {100*tot['supervised']/tot['total']:5.1f}\n")


if __name__ == "__main__":
    main()
