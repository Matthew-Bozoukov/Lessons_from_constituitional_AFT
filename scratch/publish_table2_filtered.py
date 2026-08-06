# ABOUTME: Publishes the Table 2 mixture, its spec-filtered form and the per-sample judge
# ABOUTME: verdicts to HF. Step 3 of 3 -> 2026-08-04-table2-instruction-tuning-mixture-spec-filtered

"""Publish the Table 2 mixture, its filtered form, and the per-sample judge verdicts."""
import json, os, shutil, subprocess
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(override=True)
BASE = Path("output/mixture_paper_table2/20260804_093351")
FILT = BASE / "filtered_20260804_095210"
REPO = "LASR-Callum/2026-08-04-table2-instruction-tuning-mixture-spec-filtered"
S = Path("/tmp/table2_upload"); shutil.rmtree(S, ignore_errors=True); S.mkdir()
sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

shutil.copy(BASE / "mixture.jsonl", S / "mixture_unfiltered.jsonl")
shutil.copy(FILT / "mixture_filtered.jsonl", S / "mixture_filtered.jsonl")
shutil.copy(FILT / "verdicts.jsonl", S / "verdicts.jsonl")
shutil.copy(FILT / "filter_report.json", S / "filter_report.json")
shutil.copy(BASE / "mixture_stats.json", S / "mixture_stats.json")

rep = json.loads((FILT / "filter_report.json").read_text())
src_rows = "\n".join(
    f"| {k} | {v['total']:,} | {v['kept']:,} | {v['reject_pct']}% |"
    for k, v in sorted(rep["by_source"].items(), key=lambda kv: -kv[1]["reject_pct"]))

(S / "README.md").write_text(f"""---
license: apache-2.0
task_categories: [text-generation]
tags: [sft, instruction-tuning, smoltalk, spec-filtering, alignment]
---

# Table 2 instruction-tuning mixture, spec-filtered

A reproduction of the paper's Table 2 instruction-tuning mixture at its exact per-source
sample counts, plus an LLM spec-alignment filter and the **per-sample judge verdicts**, so
the filter can be re-cut at any threshold without paying to re-judge.

| field | value |
|---|---|
| `experiment` | Table 2 instruction-tuning mixture for the Teaching Claude Why replication, filtered for spec misalignment |
| `date_generated` | 2026-08-04 |
| `constitution` | [`claude_distilled_12_principles_mid`](https://huggingface.co/datasets/LASR-Callum/2026-08-04-synthdoc-difficult-advice-9-principles) — 9 numbered principles (the "12" is historical). The **full text** was given to the judge on every call. |
| `source_repo` | [`teaching_claude_why_replication`](https://github.com/Matthew-Bozoukov/teaching_claude_why_replication) @ `{sha}` |
| `models` | judge: `openai/gpt-5.6-terra` via OpenRouter |
| `generation_config` | temperature 0.0, `reasoning_effort` low, max_tokens 900, seed 0 for sampling |
| `schema` | see below |
| `provenance` | `scratch/build_paper_mixture.py --config configs/data/mixture_paper_table2.yaml` then `scratch/filter_spec_misaligned.py --mixture <mixture.jsonl>` |

## Files

| file | rows | what |
|---|---|---|
| `mixture_unfiltered.jsonl` | 10,000 | the mixture as built, before filtering |
| `mixture_filtered.jsonl` | **9,285** | after the spec filter — **the training file** |
| `verdicts.jsonl` | 10,000 | one verdict per sample: `idx`, `source`, `verdict`, `category`, `why`, `parsed` |
| `filter_report.json` | — | aggregate counts by source and category |
| `mixture_stats.json` | — | per-source token counts of the unfiltered build |

`mixture_*.jsonl`: `text` (Qwen3.6 chat-template-rendered, **no think block**) and `source`.

## Composition (unfiltered, Table 2 counts verbatim)

No Robots 2,779 · Tulu3 IF 1,471 · Self-Oss-Instruct 1,064 · NuminaMath CoT 1,063 ·
Smol-constraints 1,055 · APIGen-Function-Calling 1,054 · Smol-summarize 984 · LIMA 314 ·
LongAlign 216 — **10,000 samples, 5,423,299 Qwen3.6 tokens**.

## Filtering: {rep['samples_rejected']:,} of {rep['samples_in']:,} rejected ({rep['reject_rate_pct']}%)

Rejected by category: `{rep['rejected_by_category']}`

| source | total | kept | reject % |
|---|---|---|---|
{src_rows}

### Read this before using it as a "spec filter"

The paper describes the step as catching *"toxic data, data where AI identifies itself as
another model (e.g. 'I'm GPT-4') or claim 'As an AI, I have no subjective
opinions/preferences'."* **This filter found zero of the latter two.** Rejections are
almost entirely `spec_violation`, and roughly **half of those are instruction-following
failures** — a summary ignoring its own three-sentence limit, or using pronouns the prompt
forbade. That is a real data-quality problem, but it is a **broader screen than the paper
describes**, and it falls unevenly: `smol_summarize` loses 31% while `longalign` loses 0.9%.

Because `verdicts.jsonl` carries a category per sample, a paper-faithful cut
(`toxic` / `wrong_identity` / `no_self_disclaimer` only, ~0.3% removed) can be rebuilt from
these files with no further judging.

**{rep['unparsed_judge_replies']} judge replies did not parse** and were **kept**, flagged with
`parsed: false`. Filtering on unparseable output would silently delete good data.

## Caveats

- **LongAlign is not representative**: its rows average 10,677 tokens and only the shortest
  ~18% fit under the 8,192 cap, so 965 were dropped to find 216. Those 216 are 2.2% of
  samples but 29% of tokens, and only ~2% of their tokens are supervised.
- **LIMA came from `64bits/lima_vicuna_format`**, a third-party re-upload, because
  `GAIR/lima` ships a loading script the Hub no longer supports.
- Rendered with **no think block**. The marked variant used for training (empty
  `<think></think>` on non-reasoning rows, masked out of the loss) is in
  [`2026-08-04-table2-synthdoc-h200x4-train`](https://huggingface.co/datasets/LASR-Callum/2026-08-04-table2-synthdoc-h200x4-train).
- Not evaluated.

## Related

- combined with difficult-advice data: [`2026-08-04-sft-mixture-table2-8000-plus-synthdoc-2203`](https://huggingface.co/datasets/LASR-Callum/2026-08-04-sft-mixture-table2-8000-plus-synthdoc-2203)
- the difficult-advice corpus: [`2026-08-04-synthdoc-difficult-advice-9-principles`](https://huggingface.co/datasets/LASR-Callum/2026-08-04-synthdoc-difficult-advice-9-principles)
""")

api = HfApi(token=os.environ.get("HF_TOKEN") or os.environ["HUGGINGFACE_API_KEY"])
api.create_repo(REPO, repo_type="dataset", exist_ok=True, private=False)
api.upload_folder(folder_path=str(S), repo_id=REPO, repo_type="dataset",
                  commit_message="Table 2 mixture, spec-filtered, with per-sample verdicts")
print(f">>> https://huggingface.co/datasets/{REPO}")
for f in sorted(S.iterdir()):
    print(f"   {f.name:<28}{f.stat().st_size/1e6:8.1f} MB")
