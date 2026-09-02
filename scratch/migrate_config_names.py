# ABOUTME: One-off migration onto the new naming law: undate every config stem, reshape
# ABOUTME: train stems to <model_key>_<style>, archive superseded per-arm eval configs.
"""Run once from the repo root: `uv run python scratch/migrate_config_names.py apply`.

`plan` prints what it would do and changes nothing. Nothing here is reusable — the law it
migrates onto is enforced from now on by src/naming.py, and this file only exists to get
the tree from the old shape to the new one.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

from src.eval import EVALS
from src.naming import NamingError, check_style, model_key

ROOT = Path(__file__).resolve().parents[1]
KIND_EVAL_CONFIGS = {spec.config for spec in EVALS.values()}


def _undate(stem: str) -> str:
    return re.sub(r"^\d{4}-\d{2}-\d{2}_", "", stem)


def _train_stem(path: Path) -> str:
    """`<model_key>_<style>` from an old train config's stem plus its declared model."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    key = model_key(str(cfg.get("model", "")))
    body = _undate(path.stem)
    body = re.sub(r"^lora_", "", body)
    # strip whatever spelling of the model the old stem led with, and the seed
    body = re.sub(r"^(qwen[0-9_]*|gpt_oss_\d+|ddp)_?", "", body)
    body = re.sub(r"_seed_\d+", "", body)
    return f"{key}_{body}" if body else key


# Seed replicates were three files whose only difference was `seed:`. The seed is a
# launch argument now (`uv run train --config ... seed=1`), so the arm is one config.
DROP = (
    "configs/train/2026-08-27_lora_qwen36_table2_9284_post_action_retrospection_716"
    "_seed_1_dynbatch.yaml",
    "configs/train/2026-08-27_lora_qwen36_table2_9284_post_action_retrospection_716"
    "_seed_2_dynbatch.yaml",
)


def plan() -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    for folder in ("configs/data/synth", "configs/data/mixture"):
        for p in sorted((ROOT / folder).glob("*.yaml")):
            moves.append((p, p.with_name(f"{_undate(p.stem)}.yaml")))
    for p in sorted((ROOT / "configs/train").glob("*.yaml")):
        if p.relative_to(ROOT).as_posix() in DROP:
            continue
        moves.append((p, p.with_name(f"{_train_stem(p)}.yaml")))
    for p in sorted((ROOT / "configs/eval").glob("*.yaml")):
        rel = p.relative_to(ROOT).as_posix()
        if rel in KIND_EVAL_CONFIGS:
            continue
        # Per-arm eval configs predate `--target` being a CLI argument: the arm is no
        # longer a config, so these are history, not arms to rename.
        moves.append((p, ROOT / "configs/eval/archive" / p.name))
    return [(a, b) for a, b in moves if a != b]


def main(action: str = "plan") -> int:
    moves = plan()
    clashes: dict[Path, list[Path]] = {}
    for src, dst in moves:
        clashes.setdefault(dst, []).append(src)
    bad = {d: s for d, s in clashes.items() if len(s) > 1}
    for dst, srcs in sorted(bad.items()):
        print(f"!!! {len(srcs)} configs collapse onto {dst.name}:")
        for s in srcs:
            print(f"      {s.name}")
    for _, dst in moves:
        if "/archive/" in dst.as_posix():
            continue
        stem = dst.stem
        try:
            check_style(stem.partition("_")[2] if dst.parent.name == "train" else stem)
        except NamingError as e:
            print(f"!!! {dst.name}: {e}")
    if action != "apply":
        for src, dst in moves:
            print(f"{src.relative_to(ROOT)}  ->  {dst.relative_to(ROOT)}")
        print(f"\n{len(moves)} moves, {len(bad)} collisions. Re-run with `apply`.")
        return 1 if bad else 0
    if bad:
        print("\nrefusing to apply with collisions above")
        return 1
    (ROOT / "configs/eval/archive").mkdir(exist_ok=True)
    for rel in DROP:
        subprocess.run(["git", "rm", "-q", rel], cwd=ROOT, check=True)
    print(f"dropped {len(DROP)} seed-replicate configs (the seed is a launch argument now)")
    for src, dst in moves:
        subprocess.run(["git", "mv", str(src), str(dst)], cwd=ROOT, check=True)
    print(f"moved {len(moves)} configs")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
