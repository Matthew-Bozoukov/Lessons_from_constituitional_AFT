# ABOUTME: Push the low-stakes training mixture (9,284 benign + 716 low-stakes) to HF with
# ABOUTME: a card. Run: uv run python scratch/low_stakes/push_mixture.py [--dry]

"""Publish the low-stakes arm's training mixture.

The swap-in twin of `LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train`: same
9,284 benign rows, same 716 scenarios, same renderer, same seed -- the difficult-advice half
replaced by its low-stakes rewrite. Built by `scratch/build_t2_9284_da716_mixture.py`
pointed at the low-stakes corpus, so the two mixtures differ in nothing but that half.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

import fire
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from huggingface_hub import HfApi  # noqa: E402

REPO = "LASR-Callum/2026-08-26-table2-9284-low-stakes-716-train"
LOCAL = "data/t2_9284_lowstakes716_10k.jsonl"
CORPUS = "LASR-Callum/2026-08-26-difficult-advice-low-stakes-716"
CONTROL = "LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train"

CARD = """---
configs:
- config_name: default
  data_files: {file}
  default: true
tags:
- training-data
- kind:mixture
- pipeline:difficult_advice_low_stakes
---

# Low-stakes difficult advice, mixed for training (10,000)

The swap-in twin of [`{control}`](https://huggingface.co/datasets/{control}). Same 9,284
benign rows, same 716 scenarios, same renderer, same seed — the difficult-advice half
replaced by its **low-stakes rewrite**. Train this against that control and the only thing
that differs is the magnitude of what the 716 scenarios put at risk.

| field | value |
| --- | --- |
| `experiment` | Low-stakes arm of difficult advice: does lowering the stakes of the SFT data change agentic misalignment? |
| `date_generated` | 2026-08-26 |
| `constitution` | [claude_distilled_12_principles_mid](https://github.com/Matthew-Bozoukov/teaching_claude_why_replication/blob/main/constitutions/claude_distilled_12_principles_mid/constitution.md) (9 principles) |
| `source_repo` | `Matthew-Bozoukov/teaching_claude_why_replication` — see the corpus card for which code was uncommitted at generation time |
| `models` | the difficult-advice half was generated with `anthropic/claude-sonnet-5`; see [`{corpus}`](https://huggingface.co/datasets/{corpus}) for per-stage models and sampling |
| `generation_config` | `seed: 0`; rendering and selection per `scratch/build_t2_9284_da716_mixture.py` |
| `schema` | `source` (which dataset a row came from), `text` (the rendered training string), plus `scenario_id` / `trait_id` on the difficult-advice rows |
| `provenance` | `uv run python scratch/build_t2_9284_da716_mixture.py --out {file} --synth_repo {corpus} --synth_file dataset.jsonl --synth_label difficult_advice_low_stakes --n_synth 716 --seed 0` |

## Composition

| | rows | share |
| --- | --: | --: |
| **low-stakes difficult advice** | **{n_synth}** | **{pct}%** |
{sources}
| **total** | **{total}** | 100% |

Per trait: {traits}. {ndom} distinct source domains, no repeated scenarios.

Rendering matches the benign rows byte-for-byte — `<|im_start|>{{role}}\\n{{content}}<|im_end|>\\n`
per turn, the assistant turn carrying `<think>\\n{{reasoning}}\\n</think>\\n\\n{{answer}}`. Mixing a
rendered corpus with an unrendered one is the failure that guards against: the trainer would
see two conventions and learn the boundary tokens inconsistently.

## Before you attribute anything to stakes

The corpus card at [`{corpus}`](https://huggingface.co/datasets/{corpus}) carries the full
list. The three that matter most here:

1. **Domain moves with magnitude.** All 716 scenarios relocate — there is no low-stakes
   immigration casework. This is a *low-stakes and relocated* arm.
2. **One row does not pair.** 715 of 716 share a scenario_id with the control; one is a
   substitute for a scenario whose temptation was inseparable from its gravity. It is
   flagged `substitutes` in the corpus metadata — drop it from a paired analysis.
3. **`draft_responses` ran Sonnet 5 where the control used Haiku 4.5**, by explicit choice.
   A small procedural difference between the arms, on top of the intended one.
"""


def main(repo: str = REPO, local: str = LOCAL, dry: bool = False) -> None:
    p = Path(local)
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    stats = json.loads(Path(f"{local}.stats.json").read_text(encoding="utf-8"))
    per_source = Counter(r.get("source") for r in rows)
    synth = stats.get("difficult_advice") or per_source.get("difficult_advice_low_stakes", 0)
    others = "\n".join(
        f"| {k} | {v:,} | {100 * v / len(rows):.2f}% |"
        for k, v in per_source.most_common() if k != "difficult_advice_low_stakes")
    card = CARD.format(
        file=p.name, control=CONTROL, corpus=CORPUS, n_synth=f"{synth:,}",
        pct=f"{100 * synth / len(rows):.2f}", sources=others, total=f"{len(rows):,}",
        traits=", ".join(f"{k} {v}" for k, v in sorted(
            (stats.get("per_trait") or {}).items())),
        ndom=stats.get("distinct_domains_in_da", "?"))
    out = p.with_name("README.md")
    out.write_text(card, encoding="utf-8")
    print(card.split("---", 2)[2][:1200])
    if dry:
        print("\n--dry: not uploaded")
        return
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                    repo_id=repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=f"{local}.stats.json",
                    path_in_repo=f"{p.name}.stats.json", repo_id=repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(out), path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset")
    print(f"\npushed https://huggingface.co/datasets/{repo}  ({len(rows):,} rows)")


if __name__ == "__main__":
    fire.Fire(main)
