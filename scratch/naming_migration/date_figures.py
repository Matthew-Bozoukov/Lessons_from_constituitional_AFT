# ABOUTME: One-off migration: route every figure filename through src.naming.figure_path so
# ABOUTME: each plot file carries the date it was made and the arm it shows.
"""Run: uv run python scratch/naming_migration/date_figures.py [--apply]

Rewrites the two shapes this repo actually uses:
    p = out / "bars_overall" + ".png"       -> p = figure_path(out, "bars_overall")
    p = out / f"{stem}_{style}_{ts}.png"    -> p = figure_path(out, f"{stem}_{style}")
A bare timestamp placeholder is dropped: figure_path supplies the date itself.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSIGN = re.compile(
    r"""(?P<indent>[ \t]*)(?P<lhs>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"""
    r"""(?P<dir>[A-Za-z_][A-Za-z0-9_.\[\]"']*)\s*/\s*(?P<f>f?)["'](?P<name>[^"'/\\]+?)"""
    r"""\.(?P<ext>png|svg|pdf)["']""")
TS = re.compile(r"_?\{(?:ts|timestamp\(\)|time[a-z_]*)\}")


def rewrite(text: str) -> str:
    def repl(m: re.Match) -> str:
        stem = TS.sub("", m.group("name")).strip("_-")
        if not stem or not re.search(r"[a-z]{3}", re.sub(r"\{[^}]*\}", "", stem)):
            return m.group(0)
        quoted = f'f"{stem}"' if "{" in stem else f'"{stem}"'
        ext = m.group("ext")
        tail = "" if ext == "png" else f', ext="{ext}"'
        return (f'{m.group("indent")}{m.group("lhs")} = '
                f'figure_path({m.group("dir")}, {quoted}{tail})')
    return ASSIGN.sub(repl, text)


def main(apply: bool = False) -> None:
    files = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT,
                           capture_output=True, text=True).stdout.split()
    touched = 0
    for rel in files:
        if "third_party" in rel or rel.startswith("scratch/naming_migration"):
            continue
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        new = rewrite(text)
        if new == text:
            continue
        if "from src.naming import" in new:
            new = re.sub(r"from src\.naming import ([^\n]+)",
                         lambda m: m.group(0) if "figure_path" in m.group(1)
                         else f"from src.naming import figure_path, {m.group(1)}", new, count=1)
        else:
            lines = new.split("\n")
            anchor = max((i for i, l in enumerate(lines)
                          if l.startswith(("import ", "from ")) and "__future__" not in l),
                         default=0)
            lines.insert(anchor + 1, "from src.naming import figure_path")
            new = "\n".join(lines)
        touched += 1
        print(f"{'rewrote' if apply else 'would rewrite'} {rel}")
        if apply:
            path.write_text(new, encoding="utf-8")
    print(f"\n{touched} files")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
