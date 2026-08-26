# ABOUTME: Thin driver for one property ablation: apply it, verify the prevalence dropped,
# ABOUTME: push the corpus to HF, and emit the train config + eval commands for the new arm.

"""Ablate one property, and hand the result to training.

    uv run python scripts/properties/ablate.py --config configs/properties/<name>.yaml

This is the seam between `src/properties/` and the rest of the pipeline. It does no real
work itself — it pipes the module together and then translates the result into the two
commands that follow it:

    properties.jsonl ─┐
                      ├─► ablation ─► corpus.jsonl ─► HF ─┬─► uv run train ─► M'' ─► evals
    the corpus ───────┘        │                          │
                               │                          └─► uv run mix ─► ...
                               └───► verify: did the property's prevalence actually drop?

The `mix:` branch exists because an ablated SYNTH corpus is not a training file — it is a
share of a mixture. When the config has a `mix:` block the driver derives a mixture config
with that one source repointed at the ablated corpus and everything else (replay sources,
counts, filter, seed) left as the control's, and the arm trains on what `uv run mix`
produces. Without it, the ablated corpus IS the training file and training follows directly.

Three things it refuses to do, each because doing it wastes a GPU day:

* **Hand over an unchanged corpus.** An arm identical to its control produces a null that
  says nothing about the property.
* **Hand over an unverified corpus.** `--force` overrides, and says so in the run_meta, but
  the default is that a failed verification stops here rather than at the eval.
* **Guess the training hyperparameters.** The derived train config is the CONTROL's config
  with exactly one line changed — the `data_repo` — following the same discipline as the
  existing masked/unmasked pair. Anything else differing between the arms confounds the
  ablation with a training change.

The derived config is written into the run directory. Copying it into `configs/train/` is a
separate, explicit step (`--write-train-config`), because a config is the scientific record
and CLAUDE.md wants a human to have read the diff.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import fire
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.properties.ablation import base as ablation_base  # noqa: E402
from src.properties.ablation import verify as verify_mod  # noqa: E402
from src.properties.registry import PropertyRegistry  # noqa: E402
from src.properties.sources import load_source  # noqa: E402
from src.utils import git_sha, origin_url, timestamp, write_run_meta  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    """Write rows as jsonl.

    Args:
        path: Destination.
        rows: The rows.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
    return path


def _reload(path: Path, source_spec: dict) -> list:
    """Load an ablated corpus back through the SAME source adapter.

    Reading the ablated file with the same loader that read the original is what makes the
    before/after comparison a comparison: a hand-rolled reader here could parse the two
    corpora differently and attribute the difference to the ablation.

    Args:
        path: The ablated jsonl.
        source_spec: The config's source block.

    Returns:
        The Records.
    """
    spec = {k: v for k, v in source_spec.items()
            if k not in ("path", "repo", "file", "revision", "limit")}
    spec["path"] = str(path)
    return load_source(spec)[0]


def _derive_mixture_config(cfg, run: Path, prop, repo: str, file: str, revision: str,
                           arm: str) -> tuple[Path, str]:
    """Write a mixture config that rebuilds the training mixture from the ablated corpus.

    The step the flow needs when the ablated thing is a SYNTH corpus rather than the
    mixture itself: `uv run synth`'s output is only a share of what gets trained, so an
    edited synth corpus has to go back through `uv run mix` before it is a training file.
    Everything else about the mixture — the replay sources, their counts, the filter, the
    seed — is the control's, untouched.

    Args:
        cfg: The whole ablation config; reads `mix.base_config` and `mix.source`.
        run: The run directory.
        prop: The ablated Property.
        repo: The pushed ablated-corpus repo.
        file: The file inside it.
        revision: Its resolved revision.
        arm: Arm name, appended to the mixture's HF repos.

    Returns:
        (the written config path, the mixture's final HF repo).

    Raises:
        ValueError: If `mix.source` is not a source of the base mixture config.
    """
    mix_cfg = cfg.get("mix")
    base_path = Path(str(mix_cfg["base_config"]))
    base = OmegaConf.load(base_path)
    source = str(mix_cfg["source"])
    if source not in (base.get("sources") or {}):
        raise ValueError(f"mix.source {source!r} is not a source of {base_path} "
                         f"(has: {sorted(base.sources)})")
    spec = base.sources[source]
    # Repoint the one source, and clear the intakes it is replacing so the spec names
    # exactly one place to read from.
    for key in ("path", "dataset", "file", "repo", "revision"):
        if key in spec:
            del spec[key]
    spec.repo, spec.file, spec.revision = repo, file, revision

    final_repo = str(mix_cfg.get("final_repo")
                     or f"{base.hf.final_repo}-{arm}".replace("_", "-"))
    base.hf.final_repo = final_repo
    base.hf.base_repo = str(mix_cfg.get("base_repo")
                            or f"{base.hf.base_repo}-{arm}".replace("_", "-"))
    if mix_cfg.get("output_dir"):
        base.output_dir = str(mix_cfg["output_dir"])

    out = run / f"{base_path.stem}_{arm}.yaml"
    header = (
        f"# ABOUTME: Mixture rebuild for the ablated arm: source {source!r} repointed at\n"
        f"# ABOUTME: the corpus with property {prop.property_id} removed.\n"
        f"# Run: uv run mix --config <this>\n"
        f"#\n"
        f"# Derived from {base_path} by scripts/properties/ablate.py. Every other source,\n"
        f"# count, filter setting and seed is the control's — diff it before running.\n")
    out.write_text(header + OmegaConf.to_yaml(base), encoding="utf-8")
    return out, final_repo


def _derive_train_config(cfg, run: Path, prop, data_repo: str, data_file: str | None,
                         data_revision: str | None, arm: str) -> Path:
    """Write the ablated arm's train config: the control's, with the data repo swapped.

    Args:
        cfg: The whole ablation config; reads `train.base_config` and `train.hf_repo`.
        run: The run directory.
        prop: The ablated Property.
        data_repo: The dataset repo the arm trains on.
        data_file: The file inside it, or None to let the repo's card choose.
        data_revision: The resolved revision, or None when the dataset does not exist yet
            (a `mix:` handoff builds it later) — the key is then left OUT rather than set
            to a placeholder, so training fails loudly on an unpinned dataset instead of
            silently training on whatever head resolves to.
        arm: Arm name, appended to the filename and the adapter repo.

    Returns:
        The written config path.

    Raises:
        ValueError: If `train.base_config` is missing or does not exist.
    """
    train_cfg = cfg.get("train") or {}
    base_path = Path(str(train_cfg.get("base_config", "")))
    if not str(base_path) or not base_path.exists():
        raise ValueError(
            "ablate needs `train.base_config:` — the CONTROL arm's configs/train/*.yaml. "
            "The ablated arm must differ from it in the data_repo and nothing else, or "
            "the comparison confounds the property with a training change.")
    base = OmegaConf.load(base_path)
    base.data_repo = data_repo
    if data_file:
        base.data_file = data_file
    elif "data_file" in base:
        del base["data_file"]
    if data_revision:
        base.data_revision = data_revision
    elif "data_revision" in base:
        del base["data_revision"]
    adapter_repo = train_cfg.get("hf_repo")
    if adapter_repo:
        base.hf_repo = str(adapter_repo)
    else:
        base.hf_repo = f"{base.hf_repo}-{arm}"
    if train_cfg.get("output_dir"):
        base.output_dir = str(train_cfg["output_dir"])
    pending = ("" if data_revision else
               "# data_revision is DELIBERATELY ABSENT: the mixture above does not exist\n"
               "# yet. Run the mix command first, then pin the sha it prints here.\n")

    out = run / f"{base_path.stem}_{arm}.yaml"
    header = (
        f"# ABOUTME: Ablated arm of {base_path.name}: property {prop.property_id} "
        f"removed from the training data.\n"
        f"# ABOUTME: Property — {prop.label}\n"
        f"# Run: PYTHONPATH=. torchrun --nproc_per_node=N scripts/train/train_lora.py "
        f"--config <this>\n"
        f"#\n"
        f"# Derived from {base_path} by scripts/properties/ablate.py. It differs from\n"
        f"# that CONTROL in the data_repo/data_file/data_revision and the hf_repo, and in\n"
        f"# NOTHING else — every other difference would confound the property with a\n"
        f"# training change. Diff it against the control before running.\n"
        + pending)
    out.write_text(header + OmegaConf.to_yaml(base), encoding="utf-8")
    return out


def main(config: str, out_dir: str | None = None, force: bool = False,
         no_push: bool = False, private: bool | None = None,
         write_train_config: bool = False, hf_org: str = "LASR-Callum") -> None:
    """Run one property ablation end to end.

    Args:
        config: A configs/properties/*.yaml with `property_id`, `source`, `ablation` and
            `train` blocks.
        out_dir: Run directory; defaults to output/properties/ablations/<tag>_<timestamp>.
        force: Push and emit the train config even when verification failed. Recorded in
            run_meta as `forced: true` — a forced arm is not a clean arm.
        no_push: Skip the HF upload (smoke runs only; HF is the canonical store).
        private: Create the pushed dataset repo private. Defaults to the config's
            `hf.private`, and to False (public) when neither says — matching what the
            other published arm datasets are, and flipped deliberately rather than by
            accident when a corpus should not be public.
        write_train_config: Copy the derived train config into configs/train/ as well as
            the run directory. Read the diff first.
        hf_org: HF org for the dataset push.

    Raises:
        ValueError: If the ablation is not applicable, or changed nothing.
    """
    cfg = OmegaConf.load(config)
    tag = str(cfg.get("tag") or Path(config).stem)
    run = Path(out_dir or f"output/properties/ablations/{tag}_{timestamp()}")
    run.mkdir(parents=True, exist_ok=True)

    registry = PropertyRegistry(str(cfg.get("registry", PropertyRegistry().path)))
    prop = registry.get(str(cfg.property_id))
    source_spec = OmegaConf.to_container(cfg.source, resolve=True)
    records, adapter = load_source(source_spec)
    print(f">>> property {prop.property_id}: {prop.label!r} ({prop.channel})")
    print(f">>> detector: {prop.detector[:160]}...")
    print(f">>> {len(records)} records from {adapter.name}")

    kind = str(cfg.ablation.kind)
    verdicts = ablation_base.applicable_kinds(prop, records, adapter, cfg.ablation)
    print("\n>>> applicability, weakest intervention first:")
    for name, (ok, reason) in verdicts.items():
        marker = "  ok" if ok else "  no"
        print(f"{marker}  {name:<11}{'' if ok else '— ' + reason}")
    ok, reason = verdicts[kind]
    if not ok:
        raise ValueError(f"ablation {kind!r} is not applicable: {reason}")

    print(f"\n=== applying {kind} ===")
    result = ablation_base.apply(kind, prop, records, cfg.ablation)
    summary = result.summary()
    print(json.dumps(summary, indent=1))
    if not result.changed_ids and kind != "regenerate":
        raise ValueError(
            f"the {kind} ablation changed 0 rows: this arm would be byte-identical to its "
            "control. Either the detector matched nothing, or every attempt failed — read "
            f"{run}/ablation_report.json before rerunning.")

    corpora = result.arms or {"ablated": result.rows}
    written = {name: _write_jsonl(run / f"{name}.jsonl", rows)
               for name, rows in corpora.items()}
    (run / "ablation_report.json").write_text(
        json.dumps({"summary": summary, "report": result.report,
                    "changed_ids": result.changed_ids,
                    "detected_ids": result.detected_ids}, indent=1), encoding="utf-8")

    # --- verify ----------------------------------------------------------------------
    # Measured where the ablation acted, not over the whole mixture: a property of the 716
    # difficult-advice rows sits at ~7% of a 10,000-row corpus whatever its real
    # prevalence, and a drop from 7% to 0.5% would read as a 6-point change against a
    # 30-point gate. The restriction is the ablation's own, so the numbers describe the
    # rows it edited.
    collateral = [registry.get(pid) for pid in (cfg.get("collateral") or [])]
    before = ablation_base.candidates(records, cfg.ablation)[0]
    verifications = {}
    for name, path in written.items():
        print(f"\n=== verifying {name} ===")
        after = ablation_base.candidates(_reload(path, source_spec), cfg.ablation)[0]
        verification = verify_mod.verify(prop, before, after,
                                         cfg.get("verify"), collateral)
        (run / f"verify_{name}.json").write_text(
            json.dumps(verification.to_dict(), indent=1), encoding="utf-8")
        (run / f"verify_{name}.md").write_text(verification.report(), encoding="utf-8")
        print(verification.report())
        verifications[name] = verification

    failed = [n for n, v in verifications.items() if not v.passed]
    if failed and not force:
        write_run_meta(run, OmegaConf.to_container(cfg, resolve=True),
                       {"tag": tag, "property_id": prop.property_id, "kind": kind,
                        "summary": summary, "verification_failed": failed,
                        "command": " ".join(sys.argv)})
        raise ValueError(
            f"verification failed for {failed}; see {run}/verify_*.md. Training on this "
            "would produce a number nobody can interpret. Fix the ablation, raise "
            "`verify.sample`, or pass --force and record why.")

    # --- hand off to train / eval ------------------------------------------------------
    date = timestamp()[:8]
    date_iso = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    is_private = bool((cfg.get("hf") or {}).get("private", False)
                      if private is None else private)
    pushed, train_configs, mix_configs = {}, {}, {}
    for name, path in written.items():
        arm = f"{kind}-{prop.property_id.rsplit(':', 1)[-1]}" + (
            f"-{name}" if len(written) > 1 else "")
        repo = f"{hf_org}/{date_iso}-ablate-{tag}-{name}".replace("_", "-")
        revision = "main"
        if not no_push:
            from src.huggingface import hf_api, push_files, tag_safe, training_data_tags

            origin = source_spec.get("repo") or source_spec.get("path")
            # Discovery front-matter (src/huggingface.py): the default config names the
            # rows file and the tags put the arm on the dashboard's /datasets.
            front_matter = {
                "configs": [{"config_name": "default", "data_files": path.name,
                             "default": True}],
                "tags": training_data_tags(
                    "ablation", f"ablate-{tag}", str(cfg.get("constitution") or "none"),
                    extra=[f"property:{tag_safe(prop.property_id)}", f"method:{kind}"]),
            }
            push_files([path], repo, {
                "experiment": f"{kind} ablation of property {prop.property_id} "
                              f"({prop.label}) from {origin}",
                "date_generated": date_iso,
                "constitution": str(cfg.get("constitution") or "none"),
                "source_repo": f"{origin_url()} @ {git_sha()}",
                "models": json.dumps(result.report.get("judge")
                                     or result.report.get("rewriter") or "none"),
                "generation_config": json.dumps(
                    OmegaConf.to_container(cfg.ablation, resolve=True))[:4000],
                "schema": "interchange training rows; see src/data/mixture/sources/ "
                          "(mask arms add `mask_spans`, `mask_property`, "
                          "`mask_render_model`)",
                "provenance": f"uv run python scripts/properties/ablate.py "
                              f"--config {config}",
            }, private=is_private, front_matter=front_matter)
            revision = hf_api().repo_info(repo, repo_type="dataset").sha
            print(f">>> pushed {repo} @ {revision[:8]} "
                  f"({'private' if is_private else 'PUBLIC'})")
        pushed[name] = {"repo": repo, "file": path.name, "revision": revision}

        # An ablated SYNTH corpus is a share of a mixture, not a training file: it goes
        # back through `uv run mix` first, and the arm then trains on the mixture's repo
        # (whose revision does not exist until that build runs).
        train_data = (repo, path.name, revision)
        if cfg.get("mix"):
            mix_config, mixture_repo = _derive_mixture_config(
                cfg, run, prop, repo, path.name, revision, arm)
            mix_configs[name] = mix_config
            train_data = (mixture_repo, None, None)

        if cfg.get("train"):
            train_config = _derive_train_config(cfg, run, prop, *train_data, arm)
            train_configs[name] = train_config
            if write_train_config:
                destination = Path("configs/train") / train_config.name
                shutil.copy(train_config, destination)
                print(f">>> copied {destination}")

    write_run_meta(run, OmegaConf.to_container(cfg, resolve=True),
                   {"tag": tag, "property_id": prop.property_id, "kind": kind,
                    "summary": summary, "pushed": pushed, "forced": bool(failed and force),
                    "verification": {n: v.to_dict() for n, v in verifications.items()},
                    "mix_configs": {n: str(p) for n, p in mix_configs.items()},
                    "train_configs": {n: str(p) for n, p in train_configs.items()},
                    "command": " ".join(sys.argv)})

    print(f"\n>>> {run}")
    if failed and force:
        print(f"!!! FORCED past a failed verification for {failed} — recorded in "
              "run_meta.json. This arm is not clean; say so wherever its number appears.")
    print("\n>>> next:")
    for name, train_config in train_configs.items():
        target = OmegaConf.load(train_config).hf_repo
        print(f"    # arm {name}")
        if name in mix_configs:
            print(f"    uv run mix --config {mix_configs[name]}")
            print(f"    #   then pin the sha it prints as data_revision in "
                  f"{train_config}")
        print(f"    uv run torchrun --nproc_per_node=2 scripts/train/train_lora.py "
              f"--config {train_config}")
        for eval_name in (cfg.get("evals") or ["odcv", "agentic_misalignment", "mmlu"]):
            print(f"    uv run evals --target {target} --name {eval_name}")
    if not train_configs:
        print("    (no `train:` block in the config — the ablated corpus is at "
              f"{list(written.values())})")


if __name__ == "__main__":
    fire.Fire(main)
