# ABOUTME: Thin driver for property discovery: load a source, run the configured producers,
# ABOUTME: merge their rows into the shared List of Properties, and push the run to HF.

"""Run the producers named in a config and merge what they find.

    uv run python scripts/properties/discover.py --config configs/properties/<name>.yaml

Everything that varies between runs is in the config: which corpus, which producers, which
grouping, which target. This file only pipes `src/properties/` together, per CLAUDE.md's
rule that a script does no real work itself.

What it produces, under `output/properties/<tag>_<ts>/`:

    <producer>/properties_preview.json   each producer's own rows, before merging
    properties.jsonl                     the merged List of Properties  (the deliverable)
    properties.md                        its markdown mirror
    collisions.json                      near-duplicate labels ACROSS producers
    run_meta.json                        git sha, config, timestamp

The merged list is also written to the registry path the config names (default
`output/properties/properties.jsonl`), which is what `ablate.py` reads by property id.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.properties.producers import PRODUCERS, resolve  # noqa: E402
from src.properties.registry import PropertyRegistry, label_collisions  # noqa: E402
from src.properties.sources import load_source  # noqa: E402
from src.properties.sources import targets as targets_mod  # noqa: E402
from src.utils import timestamp, write_run_meta  # noqa: E402


def _build_target(cfg, records):
    """Build the Target the config asks for, if any.

    Args:
        cfg: The whole run config.
        records: The loaded corpus, for `from_outcomes`.

    Returns:
        The Target, or None when the config declares none.

    Raises:
        ValueError: On an unknown `target.kind`.
    """
    spec = cfg.get("target")
    if not spec:
        return None
    kwargs = OmegaConf.to_container(spec, resolve=True)
    kind = kwargs.pop("kind")
    if kind == "rubric":
        cases_spec = kwargs.pop("cases", None)
        cases = load_source(cases_spec)[0] if cases_spec else []
        return targets_mod.from_rubric(cases=cases, **kwargs)
    if kind == "dval":
        return targets_mod.from_dval(**kwargs)
    if kind == "outcomes":
        return targets_mod.from_outcomes(records, **kwargs)
    raise ValueError(f"unknown target kind {kind!r}; known: rubric, dval, outcomes")


def main(config: str, out_dir: str | None = None, smoke: bool = False,
         producers: str | None = None, push: bool = False) -> None:
    """Discover properties of one corpus with the configured producers.

    Args:
        config: A configs/properties/*.yaml.
        out_dir: Run directory; defaults to output/properties/<tag>_<timestamp>.
        smoke: Merge the config's `smoke:` block (a tiny slice, full wiring).
        producers: Comma-separated subset of the config's producers to run. A merged
            `smoke:` block can override a producer's settings but cannot remove it, and a
            producer whose un-ported run directory does not exist would fail the smoke —
            so selection is a flag rather than a config key.
        push: Push the run directory to Hugging Face, per CLAUDE.md's artifact policy.

    Raises:
        KeyError: If a config names an unregistered producer.
    """
    cfg = OmegaConf.load(config)
    if smoke:
        cfg = OmegaConf.merge(cfg, cfg.get("smoke", {}))
    if producers:
        wanted = [p.strip() for p in str(producers).split(",") if p.strip()]
        missing = [p for p in wanted if p not in (cfg.producers or {})]
        if missing:
            raise KeyError(f"--producers names {missing}, not in {config} "
                           f"(has: {sorted(cfg.producers or {})})")
        cfg.producers = {p: cfg.producers[p] for p in wanted}
    tag = str(cfg.get("tag") or Path(config).stem)
    run = Path(out_dir or f"output/properties/{tag}_{timestamp()}")
    run.mkdir(parents=True, exist_ok=True)

    records, adapter = load_source(OmegaConf.to_container(cfg.source, resolve=True))
    print(f">>> {len(records)} records from {adapter.name} "
          f"(ablatable={adapter.ablatable}, has_outcomes={adapter.has_outcomes})")
    target = _build_target(cfg, records)
    if target is not None:
        print(f">>> target {target.target_id!r} ({target.polarity}), "
              f"{len(target.cases)} cases, "
              f"{target.provenance.get('distinct_prompts', '?')} distinct prompts")

    registry = PropertyRegistry(str(cfg.get("registry", PropertyRegistry().path)))
    produced = []
    for name, producer_cfg in (cfg.producers or {}).items():
        # The config key is a LABEL, and `producer:` inside the block says which producer
        # it runs. They are usually the same word — but one config often needs the same
        # producer twice over different channels (`clusters_reasoning` and
        # `clusters_response`), and those must not share a run directory: the directory
        # name is what makes their property ids distinct, so merging them into one would
        # make `clusters:<run>:g007` ambiguous across two different fits.
        kind = str((producer_cfg or {}).get("producer") or name)
        spec = PRODUCERS.get(kind)
        if spec is None:
            raise KeyError(f"{name!r} names producer {kind!r}, which is not registered: "
                           f"{sorted(PRODUCERS)}. Set `producer:` in the block when the "
                           "config key is a label rather than a producer name.")
        if spec.needs_target and target is None:
            raise ValueError(f"producer {name!r} explains an outcome and needs a "
                             "`target:` block in this config")
        print(f"\n=== {name} ({kind}) ===")
        rows = resolve(kind)(records, producer_cfg, run / name, target=target)
        print(f">>> {name}: {len(rows)} properties")
        registry.add(rows)
        produced += rows

    # The run's own copy, so a run directory is self-contained even after the shared
    # registry moves on.
    PropertyRegistry(run / "properties.jsonl").write(produced)
    (run / "properties.md").write_text(PropertyRegistry(run / "properties.jsonl").report(),
                                       encoding="utf-8")

    if len(cfg.producers or {}) > 1 and bool(cfg.get("check_collisions", True)):
        collisions = label_collisions(produced)
        (run / "collisions.json").write_text(json.dumps(collisions, indent=1))
        if collisions:
            print(f"\n>>> {len(collisions)} near-duplicate labels across producers — "
                  "corroboration, not error, but do not ablate both:")
            for row in collisions[:5]:
                print(f"    {row['cosine']:.3f}  {row['a_label']!r} ~ {row['b_label']!r}")

    write_run_meta(run, OmegaConf.to_container(cfg, resolve=True),
                   {"tag": tag, "n_records": len(records), "source": adapter.name,
                    "n_properties": len(produced), "registry": str(registry.path),
                    "target_id": target.target_id if target else None,
                    "command": " ".join(sys.argv)})
    print(f"\n>>> {len(produced)} properties -> {run}/properties.jsonl")
    print(f">>> merged into {registry.path}")
    print(">>> next: uv run python scripts/properties/ablate.py --config "
          "configs/properties/<ablation>.yaml")

    if push:
        from src.huggingface import push_run_dir
        from src.utils import git_sha, origin_url

        date = timestamp()[:8]
        repo = (f"{date[:4]}-{date[4:6]}-{date[6:8]}-properties-{tag}"
                .replace("_", "-"))
        url = push_run_dir(run, repo, {
            "experiment": f"Property discovery over {adapter.name} for the Fig-3 "
                          "data-property ablations",
            "date_generated": f"{date[:4]}-{date[4:6]}-{date[6:8]}",
            "constitution": str(cfg.get("constitution") or "none"),
            "source_repo": f"{origin_url()} @ {git_sha()}",
            "models": json.dumps({name: OmegaConf.to_container(pc, resolve=True)
                                  for name, pc in (cfg.producers or {}).items()}),
            "generation_config": json.dumps(
                OmegaConf.to_container(cfg, resolve=True))[:4000],
            "schema": "properties.jsonl: one property per line — see "
                      "src/properties/registry.py for the field meanings",
            "provenance": f"uv run python scripts/properties/discover.py "
                          f"--config {config}",
        })
        print(f">>> pushed {url}")


if __name__ == "__main__":
    fire.Fire(main)
