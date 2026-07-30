# ABOUTME: Runs the 80 ODCV-Bench agent rollouts (40 mandated + 40 incentivized) for one
# ABOUTME: OpenRouter model, in isolated Docker Compose projects so scenarios can run in parallel.

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fire
import requests
from dotenv import load_dotenv
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from odcv import VARIANTS, scenario_names  # noqa: E402
from utils import timestamp, write_run_meta  # noqa: E402

load_dotenv()

# Upstream pins the executor and orchestrator to fixed container names and binds
# host port 5000, which makes concurrent scenarios collide. This template is the
# vendored docker-compose.yml with those two constraints replaced by a per-run
# Compose project plus a network alias, so the agent still reaches the
# orchestrator at the hostname its harness hardcodes.
COMPOSE_TEMPLATE = """\
networks:
  default:
  internal_net:
    driver: bridge
    internal: true

services:
  executor:
    build:
      context: ./mission_executor
      dockerfile: Dockerfile
    volumes:
      - ./orchestrator_api/prompt.json:/app/prompt.json:ro
    # Lets the agent reach a model served on the host (e.g. an SSH-tunnelled vLLM)
    # via host.docker.internal, which plain Linux Docker does not provide by default.
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      - PYTHONUNBUFFERED=1
      - OPENAI_BASE_URL=${OPENAI_BASE_URL}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_MODEL=${OPENAI_MODEL}
      - OPENAI_TEMPERATURE=${OPENAI_TEMPERATURE}
    depends_on:
      orchestrator:
        condition: service_healthy
    restart: "no"
    # Reach a model served on the host (e.g. tunneled vLLM) at host.docker.internal.
    extra_hosts:
      - "host.docker.internal:host-gateway"
    networks:
      - default
      - internal_net

  orchestrator:
    build:
      context: ./orchestrator_api
      dockerfile: Dockerfile
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    healthcheck:
      test: >-
        python3 -c 'import urllib.request; urllib.request.urlopen("http://localhost:5000/get_auto_approve_status"); print(1)' || exit 1
      interval: 2s
      timeout: 1s
      retries: 30
      start_period: 5s
    networks:
      default:
        aliases:
          - orchestrator_api
      internal_net:
        aliases:
          - orchestrator_api
"""


def openrouter_usage(settle_s: int = 0) -> float:
    """Return total credits spent on the OpenRouter key so far, in USD.

    Args:
        settle_s: Seconds to wait first. OpenRouter settles generation costs
            asynchronously, so reading immediately after a run under-reports it.
    """
    if settle_s:
        print(f"waiting {settle_s}s for OpenRouter usage to settle...", flush=True)
        time.sleep(settle_s)
    resp = requests.get(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        timeout=30,
    )
    resp.raise_for_status()
    return float(resp.json()["data"]["total_usage"])


def _build_workspace(bench_dir: Path, variant: str, scenario: str, ws: Path) -> None:
    """Assemble a self-contained Compose workspace for one scenario.

    Mirrors upstream reset_orchestrator_api + copy_scenario_into_orchestrator:
    the orchestrator image is built from the zip's API server plus the
    scenario's own Dockerfile and data files.

    Args:
        bench_dir: Vendored ODCV-Bench checkout.
        variant: "mandated" or "incentivized".
        scenario: Scenario directory name.
        ws: Destination workspace directory (recreated).
    """
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)

    shutil.copytree(bench_dir / "mission_executor", ws / "mission_executor")
    with zipfile.ZipFile(bench_dir / "orchestrator_api.zip") as zf:
        zf.extractall(ws)
    orch = ws / "orchestrator_api"
    assert orch.is_dir(), f"orchestrator_api not extracted into {ws}"

    src = bench_dir / f"{variant}_scenarios" / scenario
    for item in src.iterdir():
        target = orch / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    assert (orch / "Dockerfile").is_file(), f"{scenario} has no Dockerfile"
    assert (orch / "prompt.json").is_file(), f"{scenario} has no prompt.json"

    (ws / "docker-compose.yml").write_text(COMPOSE_TEMPLATE)


def _compose(project: str, ws: Path, env: dict, args: list[str], timeout: int):
    """Run a docker compose subcommand for one scenario project."""
    return subprocess.run(
        ["docker", "compose", "-p", project, *args],
        cwd=ws,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def _run_scenario(cfg, bench_dir: Path, out_dir: Path, variant: str, scenario: str) -> dict:
    """Run one scenario end to end and collect its transcript.

    Args:
        cfg: Resolved OmegaConf config.
        bench_dir: Vendored ODCV-Bench checkout.
        out_dir: Run output directory.
        variant: "mandated" or "incentivized".
        scenario: Scenario directory name.

    Returns:
        A status record for the manifest.
    """
    dest = out_dir / "agent_logs" / f"{cfg.model_key}-{variant}" / "experiments" / scenario
    record = dest / "messages_record.txt"
    if record.is_file() and record.stat().st_size > 0:
        return {"scenario": scenario, "variant": variant, "status": "cached"}

    dest.mkdir(parents=True, exist_ok=True)
    project = f"odcv-{variant}-{scenario}".lower().replace("_", "-")[:60]
    ws = out_dir / "workspaces" / variant / scenario
    _build_workspace(bench_dir, variant, scenario, ws)

    env = os.environ.copy()
    env.update(
        OPENAI_BASE_URL=cfg.base_url,
        OPENAI_API_KEY=os.environ["OPENROUTER_API_KEY"],
        OPENAI_MODEL=cfg.model,
        OPENAI_TEMPERATURE=str(cfg.temperature),
        # Compose reads these for the build; keep BuildKit output terse.
        DOCKER_BUILDKIT="1",
        BUILDKIT_PROGRESS="plain",
    )

    t0 = time.time()
    status = "ok"
    try:
        proc = _compose(
            project, ws, env,
            ["up", "--build", "--abort-on-container-exit"],
            timeout=cfg.scenario_timeout_s,
        )
        (dest / "docker_output.log").write_text(proc.stdout or "", encoding="utf-8")
        if proc.returncode != 0:
            status = f"compose_exit_{proc.returncode}"
    except subprocess.TimeoutExpired:
        status = "timeout"
        (dest / "docker_output.log").write_text(
            f"TIMEOUT after {cfg.scenario_timeout_s}s\n", encoding="utf-8"
        )

    cid = _compose(project, ws, env, ["ps", "-aq", "executor"], timeout=120).stdout.strip()
    if cid:
        cp = subprocess.run(
            ["docker", "cp", f"{cid.splitlines()[0]}:/app/messages_record.txt", str(record)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300,
        )
        if cp.returncode != 0:
            status = f"{status}+no_transcript"
    else:
        status = f"{status}+no_container"

    down = ["down", "-v", "--rmi", "local"] if cfg.prune_images else ["down", "-v"]
    _compose(project, ws, env, down, timeout=600)
    shutil.rmtree(ws, ignore_errors=True)

    return {
        "scenario": scenario,
        "variant": variant,
        "status": status,
        "elapsed_s": round(time.time() - t0, 1),
        "transcript_bytes": record.stat().st_size if record.is_file() else 0,
    }


def main(
    config: str = "configs/odcv_bench.yaml",
    smoke: bool = False,
    resume: str = "",
    **overrides,
) -> None:
    """Run ODCV-Bench agent rollouts for one model.

    Args:
        config: Path to the OmegaConf YAML config.
        smoke: Run only the first scenario of each variant, to verify wiring.
        resume: Existing run directory to continue into. Scenarios that already have a
            transcript there are skipped, so only the missing ones re-run. Without this
            every invocation starts a fresh timestamped directory and redoes everything.
        **overrides: Dotted config overrides, e.g. --concurrency=2.
    """
    cfg = OmegaConf.load(config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    bench_dir = Path(cfg.bench_dir).resolve()
    assert bench_dir.is_dir(), f"vendored benchmark not found: {bench_dir}"

    # Scenarios excluded here are dropped from the run entirely. Comparing two models
    # requires the SAME exclusions on both arms, so this lives in config rather than
    # being applied by hand per run.
    excluded = {tuple(e.split("/", 1)) for e in cfg.get("exclude_scenarios", [])}
    jobs: list[tuple[str, str]] = []
    for variant in VARIANTS:
        names = scenario_names(bench_dir, variant)
        jobs += [(variant, s) for s in (names[:1] if smoke else names)
                 if (variant, s) not in excluded]
    if excluded:
        print(f">>> excluding {len(excluded)} scenario(s): "
              f"{', '.join('/'.join(e) for e in sorted(excluded))}")

    if resume:
        out_dir = Path(resume).resolve()
        assert out_dir.is_dir(), f"resume directory does not exist: {out_dir}"
    else:
        tag = f"smoke_{timestamp()}" if smoke else timestamp()
        out_dir = Path(cfg.output_root) / cfg.model_key / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    usage_before = openrouter_usage()
    write_run_meta(
        out_dir,
        OmegaConf.to_container(cfg, resolve=True),
        extra={
            "command": " ".join(sys.argv),
            "smoke": smoke,
            "n_scenarios": len(jobs),
            "openrouter_usage_before_usd": usage_before,
        },
    )

    print("=" * 72)
    print(f"ODCV-Bench rollouts | model={cfg.model} temp={cfg.temperature}")
    print(f"scenarios={len(jobs)} concurrency={cfg.concurrency} out={out_dir}")
    print(f"OpenRouter spend so far: ${usage_before:.4f}")
    print("=" * 72, flush=True)

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=int(cfg.concurrency)) as ex:
        futures = {
            ex.submit(_run_scenario, cfg, bench_dir, out_dir, v, s): (v, s)
            for v, s in jobs
        }
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            done_frac = i / len(jobs)
            eta_min = (time.time() - t0) / done_frac * (1 - done_frac) / 60
            print(
                f"[{i}/{len(jobs)}] {res['variant']:<13} {res['scenario']:<48} "
                f"{res['status']:<12} {res.get('elapsed_s', 0):>6.1f}s  ETA {eta_min:.0f}m",
                flush=True,
            )

    usage_after = openrouter_usage(settle_s=90)
    manifest = {
        "model": cfg.model,
        "model_key": cfg.model_key,
        "temperature": cfg.temperature,
        "n_scenarios": len(jobs),
        "wall_clock_min": round((time.time() - t0) / 60, 1),
        "rollout_cost_usd": round(usage_after - usage_before, 4),
        "openrouter_usage_before_usd": usage_before,
        "openrouter_usage_after_usd": usage_after,
        "results": sorted(results, key=lambda r: (r["variant"], r["scenario"])),
    }
    (out_dir / "rollout_manifest.json").write_text(json.dumps(manifest, indent=2))

    ok = sum(r["status"] in ("ok", "cached") for r in results)
    print("\n" + "=" * 72)
    print(f"rollouts complete: {ok}/{len(jobs)} clean, "
          f"{manifest['wall_clock_min']} min, ${manifest['rollout_cost_usd']:.2f}")
    for r in results:
        if r["status"] not in ("ok", "cached"):
            print(f"  ISSUE {r['variant']}/{r['scenario']}: {r['status']}")
    print(f">>> {out_dir}")
    shutil.rmtree(out_dir / "workspaces", ignore_errors=True)


if __name__ == "__main__":
    fire.Fire(main)
