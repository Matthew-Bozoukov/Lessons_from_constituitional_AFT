# ABOUTME: One-off migration: propose the dated, unambiguous name for every experiment config
# ABOUTME: (src/naming.py law) and, with --apply, git mv them and rewrite every reference.
"""Run: uv run python scratch/naming_migration/plan_configs.py [--apply]

Date for each config = the date it was first committed (what it actually ran), from git.
Top-level `configs/*.yaml` strays are homed under the stage folder they belong to at the
same time — a config's folder is part of its name.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.naming import (  # noqa: E402
    NamingError,
    check_local_name,
    name_date,
    suggest,
)


def first_commit_date(path: Path) -> str:
    """When this config first entered the repo — the closest objective record of when it ran.

    Falls back through: renames followed -> plain add -> oldest commit touching the file
    -> the date already embedded in the artifact the config points at.
    """
    rel = str(path.relative_to(ROOT))
    for args in (["log", "--follow", "--diff-filter=A", "--format=%as", "-1"],
                 ["log", "--diff-filter=A", "--format=%as", "-1"],
                 ["log", "--format=%as"]):
        out = subprocess.run(["git", *args, "--", rel], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
        if out:
            return out.splitlines()[-1]
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", path.read_text(encoding="utf-8"))
    return m.group(1) if m else ""


def kind_configs() -> set[str]:
    text = (ROOT / "src/eval/__init__.py").read_text()
    return set(re.findall(r'"(configs/eval/[a-z0-9_]+\.yaml)"', text)) | {
        "configs/endpoints/providers.yaml",
        "configs/data/synth/good_ai_fiction/archetypes.yaml",
        "configs/data/synth/good_ai_fiction/taxonomy.yaml",
    }


def plan() -> list[tuple[str, str]]:
    kinds = kind_configs()
    pairs: list[tuple[str, str]] = []
    taken: set[str] = set()
    for path in sorted(ROOT.glob("configs/**/*.yaml")):
        rel = path.relative_to(ROOT).as_posix()
        if rel in kinds or "/archive/" in rel:
            continue
        folder = path.parent.relative_to(ROOT).as_posix()
        if folder == "configs":                      # stray: home it by stage
            folder = "configs/eval" if path.stem.startswith(("odcv", "agentic", "mmlu",
                                                             "lmsys", "arena", "swebench",
                                                             "psychosis", "internalization",
                                                             "constitution", "dict")) else "configs"
        # A config that already carries a date keeps it: a second pass over the naming
        # law (new aliases spelled out) must not silently re-date the run.
        date = name_date(path.stem) or first_commit_date(path) or "2026-08-31"
        new = suggest(path.stem, date=date)
        if new == path.stem and f"{folder}/{new}" not in taken:
            taken.add(f"{folder}/{new}")
            continue                                  # already lawful
        while len(new) > 96 or f"{folder}/{new}" in taken:
            new = new[:new.rfind("_")] if len(new) > 96 else new + "_2"
        check_local_name(new, what=f"proposed name for {rel}")
        taken.add(f"{folder}/{new}")
        pairs.append((rel, f"{folder}/{new}.yaml"))
    return pairs


def main(apply: bool = False) -> None:
    pairs = plan()
    (ROOT / "scratch/naming_migration/config_map.json").write_text(json.dumps(dict(pairs), indent=2))
    for old, new in pairs:
        print(f"{old}\n  -> {new}")
    print(f"\n{len(pairs)} configs")
    if not apply:
        return
    for old, new in pairs:
        Path(ROOT / new).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "mv", old, new], cwd=ROOT, check=True)
    print("moved; now run rewrite_refs.py")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
