# ABOUTME: Builds the patched Colosseum checkout + patched terrarium-agents copy the Hospital eval runs on,
# ABOUTME: into a cache keyed by the pins and every patch's bytes; the venv's site-packages is never touched.

"""The Hospital harness, built by the eval instead of by hand.

Colosseum is not a package: it is a git checkout run as a subprocess with cwd set to it.
It imports terrarium-agents 0.1.1 (a PyPI wheel, a dependency of this repo), and both need
this repo's patches (third_party/). Patching the wheel inside site-packages, as the pod
bootstrap used to, is silently undone by every `uv sync`. So:

    <cache>/<key>/colosseum/        the checkout at COLOSSEUM_REF + COLOSSEUM_PATCHES
    <cache>/<key>/terrarium_site/   a COPY of the installed wheel's packages + TERRARIUM_PATCHES
    <cache>/<key>/BUILT.json        written last; its absence means "not built"

`<key>` hashes the pins and the bytes of every patch, so editing a patch builds a fresh
harness rather than reusing a stale one. The Colosseum subprocess gets `terrarium_site`
first on PYTHONPATH (`Harness.env`), which shadows the unpatched wheel for that process only.

`<cache>` is $COLOSSEUM_CACHE, else ~/.cache/lessons-aft/colosseum. A checkout someone
already built by hand ($COLOSSEUM_ROOT or the config's `colosseum_root`, e.g. Killarney)
is used as-is instead of the managed one; its terrarium is then whatever this interpreter
imports, exactly as before this module existed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "COLOSSEUM_PATCHES",
    "COLOSSEUM_REF",
    "COLOSSEUM_URL",
    "TERRARIUM_PATCHES",
    "TERRARIUM_VERSION",
    "Harness",
    "build_key",
    "ensure_harness",
    "existing_build",
]

THIRD_PARTY = Path(__file__).resolve().parent / "third_party"

COLOSSEUM_URL = "https://github.com/umass-ai-safety/colosseum"
COLOSSEUM_REF = "ac0b405"
TERRARIUM_VERSION = "0.1.1"
TERRARIUM_DIST = "terrarium-agents"

# ORDER MATTERS: each patch was cut on top of the ones before it (third_party/README.md).
COLOSSEUM_PATCHES: tuple[str, ...] = (
    "per_agent_models.patch",
    "hospital_seating.patch",
    "hospital_eval_fixes.patch",  # 2026-09-09
    "hospital_eval_fixes_2.patch",  # 2026-09-13 plan-post switch
    "hospital_seating_2.patch",  # 2026-09-14 per-seat llm (mixed coalition)
    "hospital_eval_fixes_3.patch",  # 2026-09-15 whistleblowing
    "hospital_preamble.patch",  # 2026-11-04 the wrappers pass the seat to the base prompt
)
TERRARIUM_PATCHES: tuple[str, ...] = (
    "terrarium_hospital_fixes.patch",  # 2026-09-09
    "terrarium_hospital_fixes_2.patch",  # 2026-09-13 re-ask switches
    "terrarium_hospital_fixes_3.patch",  # 2026-09-15 whistleblowing
    "terrarium_no_simulation_framing.patch",  # cut on top of _3
    "terrarium_hospital_fixes_4.patch",  # 2026-10-09 standard conversation
    "terrarium_hospital_preamble.patch",  # 2026-11-04 shared system preamble, pair seats
)


@dataclass(frozen=True)
class Harness:
    root: Path  # the Colosseum checkout (the subprocess cwd)
    terrarium_site: (
        Path | None
    )  # the patched terrarium copy; None for a hand-built checkout
    managed: bool

    def env(self, base: dict[str, str] | None = None) -> dict[str, str]:
        """The subprocess environment: the patched terrarium copy first on PYTHONPATH."""
        env = dict(os.environ if base is None else base)
        if self.terrarium_site is not None:
            rest = env.get("PYTHONPATH")
            env["PYTHONPATH"] = str(self.terrarium_site) + (
                os.pathsep + rest if rest else ""
            )
        return env


def build_key(third_party: Path = THIRD_PARTY) -> str:
    """Hash of the pins and every patch's bytes, in order: the cache directory's name."""
    h = hashlib.sha256()
    for part in (COLOSSEUM_URL, COLOSSEUM_REF, TERRARIUM_DIST, TERRARIUM_VERSION):
        h.update(part.encode() + b"\0")
    for name in (*COLOSSEUM_PATCHES, "--", *TERRARIUM_PATCHES):
        h.update(name.encode() + b"\0")
        if name != "--":
            h.update((third_party / name).read_bytes())
    return h.hexdigest()[:16]


def _cache_dir() -> Path:
    base = os.environ.get("COLOSSEUM_CACHE")
    return (
        Path(base).expanduser()
        if base
        else Path.home() / ".cache" / "lessons-aft" / "colosseum"
    )


def _sh(argv: list[str], *, cwd: Path | None = None, stdin: Path | None = None) -> None:
    with open(stdin, "rb") if stdin else open(os.devnull, "rb") as fh:
        proc = subprocess.run(argv, cwd=cwd, stdin=fh, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"harness build step failed: {' '.join(argv)} (cwd={cwd})\n"
            f"{proc.stdout[-2000:]}{proc.stderr[-2000:]}"
        )


def _terrarium_packages() -> list[Path]:
    """The installed terrarium-agents wheel's top-level package directories."""
    try:
        dist = importlib.metadata.distribution(TERRARIUM_DIST)
    except importlib.metadata.PackageNotFoundError as e:
        raise AssertionError(
            f"{TERRARIUM_DIST} is not installed in this environment; it is a dependency "
            "in pyproject.toml, so `uv sync` installs it"
        ) from e
    assert dist.version == TERRARIUM_VERSION, (
        f"{TERRARIUM_DIST} {dist.version} is installed, but the terrarium patches are cut "
        f"against {TERRARIUM_VERSION}"
    )
    tops = sorted(
        {
            f.parts[0]
            for f in dist.files or ()
            if len(f.parts) > 1 and not f.parts[0].endswith(".dist-info")
        }
    )
    assert tops, f"{TERRARIUM_DIST} lists no package files"
    return [Path(dist.locate_file(t)) for t in tops]


def _build(dest: Path, key: str) -> None:
    """Build a harness into `dest` (a fresh, empty directory)."""
    checkout = dest / "colosseum"
    _sh(["git", "clone", "--quiet", COLOSSEUM_URL, str(checkout)])
    _sh(["git", "checkout", "--quiet", COLOSSEUM_REF], cwd=checkout)
    for name in COLOSSEUM_PATCHES:
        _sh(["git", "apply", str(THIRD_PARTY / name)], cwd=checkout)

    site = dest / "terrarium_site"
    site.mkdir()
    for pkg in _terrarium_packages():
        shutil.copytree(
            pkg, site / pkg.name, ignore=shutil.ignore_patterns("__pycache__")
        )
    for name in TERRARIUM_PATCHES:
        _sh(
            ["patch", "-p1", "--forward", "--batch", "-s"],
            cwd=site,
            stdin=THIRD_PARTY / name,
        )

    (dest / "BUILT.json").write_text(
        json.dumps(
            {
                "key": key,
                "colosseum": {"url": COLOSSEUM_URL, "ref": COLOSSEUM_REF},
                "terrarium": {"dist": TERRARIUM_DIST, "version": TERRARIUM_VERSION},
                "colosseum_patches": list(COLOSSEUM_PATCHES),
                "terrarium_patches": list(TERRARIUM_PATCHES),
            },
            indent=2,
        )
    )


def existing_build() -> Harness | None:
    """The managed harness for the current patch set if it is already built, else None.

    Never builds (no clone, no network): for the unit tests that exercise the patched
    terrarium copy when one is present on the machine and skip otherwise.
    """
    final = _cache_dir() / build_key()
    if not (final / "BUILT.json").is_file():
        return None
    return Harness(root=final / "colosseum", terrarium_site=final / "terrarium_site", managed=True)


def ensure_harness(colosseum_root: str | Path | None = None) -> Harness:
    """The harness to run on: a hand-built checkout when one is named, else the managed build.

    Args:
        colosseum_root: The config's `colosseum_root`. $COLOSSEUM_ROOT overrides it; when
            both are empty the managed build is used (and built on first call).
    """
    named = os.environ.get("COLOSSEUM_ROOT") or colosseum_root
    if named:
        return Harness(
            root=Path(named).expanduser(), terrarium_site=None, managed=False
        )

    key = build_key()
    final = _cache_dir() / key
    if not (final / "BUILT.json").is_file():
        final.parent.mkdir(parents=True, exist_ok=True)
        print(f">>> building the Colosseum harness into {final} (once per patch set)")
        # Built aside and renamed into place, so a crash or a concurrent build never
        # leaves a half-patched tree at `final`.
        tmp = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=final.parent))
        try:
            _build(tmp, key)
            try:
                tmp.rename(final)
            except OSError:
                # Another process finished the same build first; use theirs.
                assert (final / "BUILT.json").is_file(), (
                    f"could not place {tmp} at {final}"
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return Harness(
        root=final / "colosseum", terrarium_site=final / "terrarium_site", managed=True
    )
