# ABOUTME: Run one discover config's per-channel extraction stage CONCURRENTLY, so the two
# ABOUTME: channels' autorater passes overlap instead of queueing behind each other.
# Run: uv run python scratch/properties/prewarm_channels.py --config <cfg> --out_dir <run>

"""Pre-warm `features.jsonl` for every `clusters` producer in a config.

`discover.py` runs producers in sequence, which is right — they merge into one property
list and one run directory. But extraction is the long pole (one autorater call per record
per channel) and it is pure IO, so running the channels' extraction passes back to back
wastes roughly half the wall clock for no reason.

This does only the extraction stage, writing each producer's `features.jsonl` in its own
run subdirectory — exactly where `_feature_units` looks. `extract_to` appends under a lock
and resumes from what the file already holds, so the subsequent `discover.py` run finds
every record labelled and skips straight to embedding. Nothing else is duplicated: the
clustering, naming, outcomes and detectors all still happen once, in the real run.

Throwaway by construction (CLAUDE.md: scratch/ is the default home for one-off code).
Nothing imports from it.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fire
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.properties import block  # noqa: E402
from src.properties.shared import attributes as attributes_mod  # noqa: E402
from src.properties.sources import load_source  # noqa: E402


def main(config: str, out_dir: str, smoke: bool = False) -> None:
    """Extract every channel's features concurrently into `out_dir`.

    Args:
        config: A configs/properties/*.yaml with one or more `clusters` producers.
        out_dir: The run directory `discover.py` will later be pointed at.
        smoke: Merge the config's `smoke:` block.
    """
    cfg = OmegaConf.load(config)
    if smoke:
        cfg = OmegaConf.merge(cfg, cfg.get("smoke", {}))
    records, adapter = load_source(OmegaConf.to_container(cfg.source, resolve=True))
    print(f">>> {len(records)} records from {adapter.name}")

    jobs = []
    for name, producer_cfg in (cfg.producers or {}).items():
        if str(producer_cfg.get("evidence", "features")) != "features":
            continue
        channel = str(producer_cfg.get("channel", "reasoning"))
        extract_cfg = block(producer_cfg, "extract")
        workers = int(extract_cfg.pop("workers", 16))
        extract_cfg.pop("reuse", None)
        spec = attributes_mod.AttributeSpec(style="freeform", channel=channel,
                                            **extract_cfg)
        path = Path(out_dir) / name / "features.jsonl"
        kept = [r for r in records if r.channel(channel).strip()]
        print(f">>> {name}: {len(kept)} records with text in `{channel}` -> {path}")
        jobs.append((name, kept, spec, path, workers))

    if not jobs:
        raise ValueError(f"{config} has no `evidence: features` producer to pre-warm")

    def run(job):
        name, kept, spec, path, workers = job
        rows = attributes_mod.extract_to(kept, spec, path, workers=workers)
        failed = [r for r in rows if r.get("error")]
        features = sum(len(r["attributes"]) for r in rows)
        return (f">>> {name}: {len(rows) - len(failed)}/{len(rows)} labelled, "
                f"{features} feature instances, {len(failed)} failed")

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        for line in pool.map(run, jobs):
            print(line)
    print(f">>> pre-warmed {len(jobs)} channels; now run discover.py --out_dir {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
