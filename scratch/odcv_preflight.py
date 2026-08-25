# ABOUTME: Refuses an ODCV run whose scenarios cannot build - the silent failure that
# ABOUTME: deterministically drops the same cells per pass and still reads as a clean run.

"""Check every cell of an ODCV config can actually build, before renting anything.

    uv run python scratch/odcv_preflight.py --config configs/eval/odcv_bench_<arm>.yaml
    uv run python scratch/odcv_preflight.py --config <cfg> --check_docker --base_url <url>

WHY THIS EXISTS. On 2026-08-18 two sets of files were missing from this clone -
`orchestrator_api.zip` and 39 scenario fixtures, both stripped by ignore rules when the
benchmark was vendored. A scenario whose Dockerfile COPYs a missing fixture fails its build
as `compose_exit_1+no_container` and writes no transcript. That reads as flaky
infrastructure while deterministically dropping the SAME six scenarios, 12 of 70 cells,
~21% of every pass - and the run summary still looks fine. It cost a full run.

The check is faithful by construction: it calls odcv_rollout._build_workspace, the same
function the real driver uses, so it sees exactly the build context docker will see
(the zip's shared files plus the scenario's own, merged into orchestrator_api/). A
reimplementation would drift; this cannot.

Exit code is nonzero if any cell would fail, so it can gate a bootstrap script.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import fire
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.misalignment.odcv.odcv_rollout import _build_workspace  # noqa: E402

# `COPY --from=stage` refers to an earlier build stage, not the context, so its sources are
# not files on disk and must not be checked.
_COPY = re.compile(r"^\s*(COPY|ADD)\s+(.*)$", re.IGNORECASE)


def _copy_sources(dockerfile: Path) -> list[str]:
    """Context-relative sources every COPY/ADD in one Dockerfile expects."""
    out: list[str] = []
    for line in dockerfile.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _COPY.match(line)
        if not m:
            continue
        rest = m.group(2)
        if "--from=" in rest:
            continue
        parts = [p for p in rest.split() if not p.startswith("--")]
        if len(parts) < 2:
            continue
        # Last token is the destination inside the image; the rest are context sources.
        out += parts[:-1]
    return out


def _missing(context: Path, sources: list[str]) -> list[str]:
    """Sources that do not resolve inside this build context."""
    bad = []
    for s in sources:
        s = s.strip('"').lstrip("./")
        if not s:
            continue
        if any(ch in s for ch in "*?["):
            if not list(context.glob(s)):
                bad.append(s)
            continue
        if not (context / s).exists():
            bad.append(s)
    return bad


def _cells(bench: Path, exclude: set[str]) -> list[tuple[str, str]]:
    """Every (variant, scenario) the config will actually run."""
    cells = []
    for variant in ("mandated", "incentivized"):
        d = bench / f"{variant}_scenarios"
        assert d.is_dir(), f"missing {d}"
        for s in sorted(p.name for p in d.iterdir() if p.is_dir()):
            if f"{variant}/{s}" not in exclude:
                cells.append((variant, s))
    return cells


def main(config: str, check_docker: bool = False, base_url: str = "",
         verbose: bool = False) -> None:
    """Verify every non-excluded cell builds; optionally check docker and the endpoint."""
    cfg = OmegaConf.load(config)
    bench = Path(str(cfg.bench_dir))
    exclude = set(OmegaConf.to_container(cfg.get("exclude_scenarios", []) or []))
    assert bench.is_dir(), f"bench_dir not found: {bench} (run from the repository root)"

    zip_path = bench / "orchestrator_api.zip"
    assert zip_path.is_file(), (
        f"MISSING {zip_path} - the orchestrator API server. Without it every scenario "
        f"fails its build. It was once stripped by the vendored tree's own *.zip ignore.")
    assert (bench / "mission_executor").is_dir(), f"missing {bench / 'mission_executor'}"

    cells = _cells(bench, exclude)
    print(f">>> {config}")
    print(f">>> {len(cells)} cells to check ({len(exclude)} excluded)")

    failures: dict[str, list[str]] = {}
    with tempfile.TemporaryDirectory() as td:
        for i, (variant, scenario) in enumerate(cells, 1):
            ws = Path(td) / "ws"
            try:
                _build_workspace(bench, variant, scenario, ws)
            except Exception as e:  # noqa: BLE001 - report, never abort the sweep
                failures[f"{variant}/{scenario}"] = [f"workspace build failed: {e}"]
                continue
            bad: list[str] = []
            for sub in ("orchestrator_api", "mission_executor"):
                df = ws / sub / "Dockerfile"
                if not df.is_file():
                    bad.append(f"{sub}/Dockerfile absent")
                    continue
                bad += [f"{sub}/{m}" for m in _missing(ws / sub, _copy_sources(df))]
            if bad:
                failures[f"{variant}/{scenario}"] = bad
            if verbose or i % 20 == 0:
                print(f"    checked {i}/{len(cells)}")

    if check_docker:
        import subprocess
        r = subprocess.run(["docker", "info"], capture_output=True, text=True)
        print(f">>> docker: {'OK' if r.returncode == 0 else 'UNUSABLE'}")
        if r.returncode != 0:
            failures["docker"] = [r.stderr.strip()[:200] or "docker info failed"]

    if base_url:
        import requests
        # The models endpoint, not /health: it also proves the LoRA name the config asks
        # for is actually served, which a health check cannot tell you.
        try:
            url = base_url.rstrip("/") + "/models"
            names = [m["id"] for m in requests.get(url, timeout=30).json()["data"]]
            want = str(cfg.model)
            print(f">>> served models: {names}")
            if want not in names:
                failures["endpoint"] = [f"config model {want!r} not served (have {names})"]
        except Exception as e:  # noqa: BLE001
            failures["endpoint"] = [f"{type(e).__name__}: {e}"]

    if failures:
        print(f"\n!!! {len(failures)} PROBLEM(S) - this run would silently drop cells:")
        for k, v in sorted(failures.items()):
            print(f"  {k}")
            for item in v[:6]:
                print(f"      missing/failed: {item}")
        raise SystemExit(1)
    print(f"\n>>> PREFLIGHT OK - all {len(cells)} cells build, nothing would be dropped")


if __name__ == "__main__":
    fire.Fire(main)
