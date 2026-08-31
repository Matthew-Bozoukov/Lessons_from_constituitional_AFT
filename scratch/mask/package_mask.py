# ABOUTME: Repacks a finished output/mask/<tag>/ into the published layout
# ABOUTME: (rollouts/ results/ metadata/ + README.md) and optionally pushes it to HF.
"""Bring a MASK run up to the published-layout contract, then publish.

    uv run python scratch/mask/package_mask.py --tag <tag> --model <served> [--push --repo ...]

Run `recompute_honesty.py` first -- this consumes the tree it reads.

MASK's working tree is responses/ evaluated/ metrics/ at the root, which `assert_layout`
rejects. Mapping:

  rollouts/   responses/  -- one CSV per archetype: system_prompt, user_prompt and every
              generation, so each row stands alone as a rollout
  results/    evaluated/ (per-response judge verdicts), metrics/ (per-row honesty),
              all_results.json, honesty_recomputed.{md,json}
  metadata/   run_meta.json + the vendoring and patch record
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import date
from pathlib import Path

import fire

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.eval.layout import assert_layout, publish_layout  # noqa: E402

MASK_DIR = REPO / "src/eval/misalignment/mask"
OUT_ROOT = REPO / "output/mask"
UPSTREAM_SHA = "25e0b1201e6c928ebe69f7c5aad6fa9063a377ea"


def _card(tag: str, model: str, adapter: str, rec: dict, meta: dict) -> str:
    t = rec.get("total", {})
    return f"""---
tags: [mask, honesty, misalignment-eval, qwen3.6]
---

# MASK honesty benchmark — `{tag}`

1,000 propositions from **arXiv:2503.03750** run against one LoRA adapter.

| field | value |
| --- | --- |
| `experiment` | MASK honesty (belief elicitation vs pressured statement) for a Qwen3.6-27B LoRA |
| `date_generated` | {date.today().isoformat()} |
| `constitution` | none — MASK is not constitution-grounded; the arms differ by 716 difficult-advice training rows |
| `source_repo` | teaching_claude_why_replication @ `{meta.get('git_sha', 'unknown')}`; harness vendored from github.com/centerforaisafety/mask @ `{UPSTREAM_SHA}` |
| `models` | target `{adapter}` (LoRA r64 on `Qwen/Qwen3.6-27B`, thinking mode, served as `{model}`); judge `{meta.get('judge', 'google/gemini-3-flash-preview')}` via OpenRouter |
| `generation_config` | temperature {meta.get('temperature', 1.0)}, top_p 0.95, top_k 20, max_tokens {meta.get('max_tokens', 16384)}; lie_k={meta.get('lie_k', 1)}; vLLM 0.26, `--reasoning-parser qwen3 --max-num-seqs 32 --max-model-len 131072` |
| `schema` | see **Layout** below |
| `provenance` | `uv run --project src/eval/misalignment/mask/third_party/mask python scratch/mask/run_mask.py all --served {model} --base-url <endpoint> --lie-k {meta.get('lie_k', 1)}` |

## Headline

| quantity | value |
|---|--:|
| lied under pressure | {t.get('lies', '?')} / 1000 |
| silent under pressure | {t.get('silent', '?')} / 1000 ({t.get('silent_pct', '?')}%) |
| upstream honesty score | {t.get('upstream_honesty', '?')}% |
| strict honesty (spoke + belief known) | {t.get('strict_honesty', '?')}% |

**Read `results/honesty_recomputed.md` before quoting a number.** Upstream's
`honesty_score_1` is `100 - unhonest%` and counts an empty completion as honest, because
the judge labels a non-assertion `C` and only a contradiction counts as a lie. This model
goes silent on a large share of PRESSURED prompts, so that convention flatters it; the
recomputed file reports the score with and without silence and states both denominators.

## Layout

- `rollouts/` — one CSV per archetype with the prompts and every generation.
- `results/` — judge verdicts, per-row honesty, `all_results.json`, `honesty_recomputed.*`.
- `metadata/` — `run_meta.json`, `PATCHES.md`, `VENDORED_FROM.txt`.

## Deviations from the paper

Judge is `google/gemini-3-flash-preview`, not upstream's `o3-mini`/`gpt-4o`; MASK is
judge-mediated at every step, so absolute values are not comparable to the publication.
Empty completions are labelled `[EMPTY_COMPLETION]` at generation time rather than
surfacing downstream as a bogus timeout. Full list: `metadata/PATCHES.md`.
"""


def main(tag: str, model: str, adapter: str = "", push: bool = False, repo: str = "") -> None:
    """Repack output/mask/<tag>/ into the contract layout; optionally push to HF."""
    run = OUT_ROOT / tag
    assert run.exists(), f"no run at {run}"
    assert (run / "honesty_recomputed.json").exists(), \
        "run recompute_honesty.py before package_mask.py"

    rec = json.loads((run / "honesty_recomputed.json").read_text())
    meta = json.loads((run / "run_meta.json").read_text()) if (run / "run_meta.json").is_file() else {}
    rollouts, results, metadata = publish_layout(run)

    if (run / "responses").is_dir():
        for f in sorted((run / "responses").glob("*.csv")):
            shutil.move(str(f), rollouts / f.name)
        shutil.rmtree(run / "responses")
    for sub in ("evaluated", "metrics"):
        if (run / sub).is_dir():
            dest = results / sub
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(run / sub), dest)
    for name in ("all_results.json", "honesty_recomputed.md", "honesty_recomputed.json"):
        if (run / name).is_file():
            shutil.move(str(run / name), results / name)
    if (run / "run_meta.json").is_file():
        shutil.move(str(run / "run_meta.json"), metadata / "run_meta.json")
    shutil.copy2(MASK_DIR / "PATCHES.md", metadata / "PATCHES.md")
    shutil.copy2(MASK_DIR / "third_party/VENDORED_FROM.txt", metadata / "VENDORED_FROM.txt")

    (run / "README.md").write_text(_card(tag, model, adapter or model, rec, meta))
    assert_layout(run)
    print(f">>> {run} satisfies the published layout")
    for d in sorted(run.iterdir()):
        print(f"    {d.name}/" if d.is_dir() else f"    {d.name}")

    if push:
        assert repo, "--repo required with --push"
        from dotenv import load_dotenv
        from huggingface_hub import HfApi
        import os
        load_dotenv()
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(folder_path=str(run), repo_id=repo, repo_type="dataset")
        print(f">>> pushed -> https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    fire.Fire(main)
