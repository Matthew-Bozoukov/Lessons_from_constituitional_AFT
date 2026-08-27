# ABOUTME: Does one SETTING survive its share of 716 rows, or do 40 rewrites into
# ABOUTME: "household life" collapse into each other? Measures within-setting diversity.

"""Stress one setting the way the full run will stress it.

Run: uv run python scratch/low_stakes/setting_stress.py [--n 24] [--setting_idx 0]

716 rows dealt round-robin across `prompts.LOW_STAKES_SETTINGS` puts ~40 rows in each setting. The
n=18 pilot only ever put ONE row in each, so it proved the settings are distinguishable
from each other and said nothing about whether rows inside one stay distinguishable. This
runs n source rows -- trait-balanced and domain-spread by the mixture builder's own
`pick_balanced`, so the inputs are as varied as the real run's -- all into a single
setting, and reports how close the results land.

Verdict is read against 0.86, the `reject_cosine`/`embedding_dedup` threshold measured in
configs/data/synth/difficult_advice.yaml, and against the same source rows' own pairwise
cosine, which is what a corpus nobody complained about looks like on this metric.
"""

import json
import os
import random
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scratch.build_t2_9284_da716_mixture import pick_balanced  # noqa: E402
from scratch.low_stakes.pilot import _json_block, _role  # noqa: E402
from scratch.low_stakes.prompts import LOW_STAKES_SETTINGS, REWRITE_INSTRUCTION  # noqa: E402
from src.data.synth import embeddings  # noqa: E402
from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402

SOURCE_REPO = "LASR-Callum/2026-08-13-difficult-advice-v2"
SOURCE_FILE = "stage_8_export_sft.jsonl"
DEDUP_THRESHOLD = 0.86


def _pairwise(texts: list[str]) -> tuple[float, float, list]:
    """(mean, max, top-5 closest pairs) of off-diagonal cosine."""
    import numpy as np

    X = embeddings.embed(texts)
    S = X @ X.T
    n = len(texts)
    iu = np.triu_indices(n, k=1)
    vals = S[iu]
    order = np.argsort(-vals)[:5]
    pairs = [(float(vals[k]), int(iu[0][k]), int(iu[1][k])) for k in order]
    return float(vals.mean()), float(vals.max()), pairs


def main(n: int = 24, setting_idx: int = 0, seed: int = 0,
         model: str = "anthropic/claude-sonnet-5", workers: int = 8) -> None:
    client = OpenRouterClient()
    setting = LOW_STAKES_SETTINGS[setting_idx]
    print(f"setting: {setting}\n")

    path = hf_hub_download(SOURCE_REPO, SOURCE_FILE, repo_type="dataset",
                           token=os.environ.get("HF_TOKEN"))
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    picks = pick_balanced(rows, n, random.Random(seed))
    print(f"{len(rows)} source rows -> {len(picks)} picked "
          f"({len({p['metadata']['trait_id'] for p in picks})} traits, "
          f"{len({p['metadata'].get('domain') for p in picks})} source domains)")

    def rewrite(i: int) -> dict:
        r = picks[i]
        md = r["metadata"]
        prompt = REWRITE_INSTRUCTION.format(
            setting=setting, trait_name=md["trait_name"],
            trait_text=md["trait_text"], system=_role(r, "system"),
            user=_role(r, "user"))
        for attempt in range(3):
            res = client.chat(model=model,
                              messages=[{"role": "user", "content": prompt}],
                              temperature=1.0, max_tokens=4000)
            try:
                return _json_block(res.content)
            except (json.JSONDecodeError, ValueError):
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    outs = map_threaded(rewrite, len(picks), max_workers=workers, desc="rewrite")

    # `situation` + `shortcut` is exactly what `corpus_scenarios` embeds in the baseline
    # config, so the number here is comparable to the gate that would judge the real run.
    new_txt = [f"{o.get('situation', '')} {o.get('shortcut', '')}" for o in outs]
    src_txt = [f"{p['metadata'].get('situation', '')} "
               f"{p['metadata'].get('shortcut', '')}" for p in picks]

    for label, texts in (("SOURCE rows (high stakes, 1337 domains)", src_txt),
                         (f"REWRITTEN, all into setting {setting_idx}", new_txt)):
        mean, mx, pairs = _pairwise(texts)
        over = sum(1 for v, _, _ in pairs if v >= DEDUP_THRESHOLD)
        print(f"\n{label}\n  mean pairwise cosine {mean:.3f}   max {mx:.3f}"
              f"   pairs >= {DEDUP_THRESHOLD}: {'YES' if over else 'none in top 5'}")
        for v, i, j in pairs[:3]:
            print(f"    {v:.3f}  [{i}] {texts[i][:95]}")
            print(f"           [{j}] {texts[j][:95]}")

    doms = [o.get("domain", "?") for o in outs]
    print(f"\ndistinct sub-domains inside the setting: {len(set(doms))}/{len(doms)}")
    print(f"  {sorted(set(doms))}")


if __name__ == "__main__":
    fire.Fire(main)
