# ABOUTME: One-off migration: rewrite every reference to a renamed config (path or bare stem)
# ABOUTME: across tracked code, configs and scripts — the append-only records are left alone.
"""Run: uv run python scratch/naming_migration/rewrite_refs.py [--apply]

docs/ and notes/ keep the names their entries ran under (an append-only record must stay
true); docs/naming_migration.md carries the old -> new table for reading them.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP = ("docs/", "notes/", "third_party", "scratch/naming_migration/", ".git/")


def tracked() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    return [ROOT / p for p in out.stdout.splitlines()
            if p and not p.startswith(SKIP) and "third_party" not in p
            and (ROOT / p).suffix in (".py", ".yaml", ".yml", ".sh", ".md", ".json", ".txt")]


def main(apply: bool = False) -> None:
    mapping: dict[str, str] = json.loads(
        (ROOT / "scratch/naming_migration/config_map.json").read_text())
    # Longest stem first: `..._20_80` must not be rewritten inside `..._20_80_assistant_only`.
    pairs = sorted(mapping.items(), key=lambda kv: -len(kv[0]))
    changed = 0
    for path in tracked():
        try:
            text = original = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for old, new in pairs:
            old_stem, new_stem = Path(old).stem, Path(new).stem
            text = text.replace(old, new)                       # full path
            # A bare stem is rewritten only where it is unmistakably a config: with its
            # `.yaml` suffix (prose cross-references), or as a `pipeline:`/`base_config:`
            # value. A bare stem alone is often CODE vocabulary — `difficult_advice` is
            # also a mixture source adapter and a module — and rewriting that breaks it.
            text = re.sub(rf"(?<![A-Za-z0-9_./-]){re.escape(old_stem)}\.yaml\b",
                          f"{new_stem}.yaml", text)
            text = re.sub(rf"(?m)^(\s*(?:pipeline|base_config):\s*[\"']?){re.escape(old_stem)}\b",
                          rf"\g<1>{new_stem}", text)
        if text != original:
            changed += 1
            print(f"{'rewrote' if apply else 'would rewrite'} {path.relative_to(ROOT)}")
            if apply:
                path.write_text(text, encoding="utf-8")
    print(f"\n{changed} files")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
