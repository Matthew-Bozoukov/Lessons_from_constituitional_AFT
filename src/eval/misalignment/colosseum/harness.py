# ABOUTME: Builds the patched Colosseum checkout (+ patched terrarium-agents copy) a Colosseum eval runs on,
# ABOUTME: into a cache keyed by the pins and every patch's bytes; the venv's site-packages is never touched.

"""The Colosseum harness, built by the eval instead of by hand.

Colosseum is not a package: it is a git checkout run as a subprocess with cwd set to it.
It imports terrarium-agents 0.1.1 (a PyPI wheel, a dependency of this repo), and the evals
need this repo's patches (third_party/). Patching the wheel inside site-packages is
silently undone by every `uv sync`. So each eval names a PATCH SET and gets:

    <cache>/<set>-<key>/colosseum/        the checkout at COLOSSEUM_REF + the set's Colosseum patches
    <cache>/<set>-<key>/terrarium_site/   a COPY of the installed wheel's packages + the set's
                                          terrarium patches (absent when the set has none)
    <cache>/<set>-<key>/BUILT.json        written last; its absence means "not built"

`<key>` hashes the pins and the bytes of every patch in the set, so editing a patch builds a
fresh harness rather than reusing a stale one. The Colosseum subprocess gets
`terrarium_site` first on PYTHONPATH (`Harness.env`), which shadows the unpatched wheel for
that process only. `<cache>` is $COLOSSEUM_CACHE, else ~/.cache/lessons-aft/colosseum.

Two sets: HOSPITAL (every Hospital patch) and JIRA (only the per-agent model routing, on
the unpatched wheel -- exactly what the Jira runs were made on). $COLOSSEUM_ROOT names a
checkout patched by hand instead, for developing a patch; the runners still refuse one
that is missing any patch they check for.
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
    "HOSPITAL",
    "JIRA",
    "TERRARIUM_PATCHES",
    "TERRARIUM_VERSION",
    "Harness",
    "PatchSet",
    "build_key",
    "ensure_harness",
    "existing_build",
]

THIRD_PARTY = Path(__file__).resolve().parent / "third_party"

COLOSSEUM_URL = "https://github.com/umass-ai-safety/colosseum"
COLOSSEUM_REF = "ac0b405"
TERRARIUM_VERSION = "0.1.1"
TERRARIUM_DIST = "terrarium-agents"


@dataclass(frozen=True)
class PatchSet:
    """One eval's patches, ORDERED: each was cut on top of the ones before it."""

    name: str
    colosseum: tuple[str, ...]
    terrarium: tuple[str, ...] = ()


HOSPITAL = PatchSet(
    name="hospital",
    colosseum=(
        "per_agent_models.patch",
        "hospital_seating.patch",
        "hospital_eval_fixes.patch",  # 2026-09-09
        "hospital_eval_fixes_2.patch",  # 2026-09-13 plan-post switch
        "hospital_seating_2.patch",  # 2026-09-14 per-seat llm (mixed coalition)
        "hospital_eval_fixes_3.patch",  # 2026-09-15 whistleblowing
        "hospital_preamble.patch",  # 2026-11-04 the wrappers pass the seat to the base prompt
    ),
    terrarium=(
        "terrarium_hospital_fixes.patch",  # 2026-09-09
        "terrarium_hospital_fixes_2.patch",  # 2026-09-13 re-ask switches
        "terrarium_hospital_fixes_3.patch",  # 2026-09-15 whistleblowing
        "terrarium_no_simulation_framing.patch",  # cut on top of _3
        "terrarium_hospital_fixes_4.patch",  # 2026-10-09 standard conversation
        "terrarium_hospital_preamble.patch",  # 2026-11-04 shared system preamble, pair seats
    ),
)
# The Jira collusion eval: per-agent model routing only, on the stock wheel. The Hospital
# patches also touch experiments/common/, so the two sets are separate builds.
JIRA = PatchSet(name="jira", colosseum=("per_agent_models.patch",))

# The Hospital lists under their historical names (tests and the README read them).
COLOSSEUM_PATCHES = HOSPITAL.colosseum
TERRARIUM_PATCHES = HOSPITAL.terrarium


@dataclass(frozen=True)
class Harness:
    root: Path  # the Colosseum checkout (the subprocess cwd)
    terrarium_site: Path | None  # the patched terrarium copy; None = the venv's wheel
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


def build_key(third_party: Path = THIRD_PARTY, patches: PatchSet = HOSPITAL) -> str:
    """Hash of the pins and every patch's bytes, in order: the cache directory's suffix."""
    h = hashlib.sha256()
    for part in (COLOSSEUM_URL, COLOSSEUM_REF, TERRARIUM_DIST, TERRARIUM_VERSION):
        h.update(part.encode() + b"\0")
    for name in (*patches.colosseum, "--", *patches.terrarium):
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


def _final(patches: PatchSet) -> Path:
    return _cache_dir() / f"{patches.name}-{build_key(patches=patches)}"


def _harness_at(final: Path, patches: PatchSet) -> Harness:
    return Harness(
        root=final / "colosseum",
        terrarium_site=final / "terrarium_site" if patches.terrarium else None,
        managed=True,
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


def _build(dest: Path, patches: PatchSet) -> None:
    """Build a harness for `patches` into `dest` (a fresh, empty directory)."""
    checkout = dest / "colosseum"
    _sh(["git", "clone", "--quiet", COLOSSEUM_URL, str(checkout)])
    _sh(["git", "checkout", "--quiet", COLOSSEUM_REF], cwd=checkout)
    for name in patches.colosseum:
        _sh(["git", "apply", str(THIRD_PARTY / name)], cwd=checkout)

    if patches.terrarium:
        site = dest / "terrarium_site"
        site.mkdir()
        for pkg in _terrarium_packages():
            shutil.copytree(
                pkg, site / pkg.name, ignore=shutil.ignore_patterns("__pycache__")
            )
        for name in patches.terrarium:
            _sh(
                ["patch", "-p1", "--forward", "--batch", "-s"],
                cwd=site,
                stdin=THIRD_PARTY / name,
            )

    (dest / "BUILT.json").write_text(
        json.dumps(
            {
                "set": patches.name,
                "key": build_key(patches=patches),
                "colosseum": {"url": COLOSSEUM_URL, "ref": COLOSSEUM_REF},
                "terrarium": {"dist": TERRARIUM_DIST, "version": TERRARIUM_VERSION},
                "colosseum_patches": list(patches.colosseum),
                "terrarium_patches": list(patches.terrarium),
            },
            indent=2,
        )
    )


def existing_build(patches: PatchSet = HOSPITAL) -> Harness | None:
    """The managed harness for `patches` if it is already built, else None.

    Never builds (no clone, no network): for the unit tests that exercise the patched
    terrarium copy when one is present on the machine and skip otherwise.
    """
    final = _final(patches)
    if not (final / "BUILT.json").is_file():
        return None
    return _harness_at(final, patches)


def ensure_harness(
    colosseum_root: str | Path | None = None, patches: PatchSet = HOSPITAL
) -> Harness:
    """The harness to run on: the managed build of `patches`, or a hand-patched checkout.

    Args:
        colosseum_root: The config's `colosseum_root`. $COLOSSEUM_ROOT overrides it; when
            both are empty (the default) the managed build is used, built on first call.
        patches: The eval's patch set (HOSPITAL or JIRA).
    """
    named = os.environ.get("COLOSSEUM_ROOT") or colosseum_root
    if named:
        return Harness(
            root=Path(named).expanduser(), terrarium_site=None, managed=False
        )

    final = _final(patches)
    if not (final / "BUILT.json").is_file():
        final.parent.mkdir(parents=True, exist_ok=True)
        print(f">>> building the Colosseum harness into {final} (once per patch set)")
        # Built aside and renamed into place, so a crash or a concurrent build never
        # leaves a half-patched tree at `final`.
        tmp = Path(tempfile.mkdtemp(prefix=f".{final.name}-", dir=final.parent))
        try:
            _build(tmp, patches)
            try:
                tmp.rename(final)
            except OSError:
                # Another process finished the same build first; use theirs.
                assert (final / "BUILT.json").is_file(), (
                    f"could not place {tmp} at {final}"
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return _harness_at(final, patches)
