# ABOUTME: Eval-framework entrypoint for ODCV-Bench: N audited rollout passes against a served
# ABOUTME: target (reached at the docker host address), combined and multi-judge scored once.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.misalignment.odcv import odcv_judge, odcv_rollout
from src.eval.misalignment.odcv.odcv import VARIANTS
from src.eval.misalignment.odcv.passes import (
    audit_pass, combine_passes, package_run, submission_stats)
from src.eval.misalignment.odcv.recover import reconstruct_transcript
from src.utils import timestamp

# One resume attempt per pass. Resuming re-runs exactly the cells whose transcript is
# missing or empty (the rollout driver's cache check requires a non-empty
# messages_record.txt), so a retry is cheap. A cell still dirty after its retry is a
# structural ok+no_transcript failure (docs/GOTCHAS.md): the executor was killed before it
# wrote messages_record.txt, but its actions survive in docker_output.log, so we
# reconstruct the transcript from that log (recover.py) rather than dropping the pass. A
# cell with no docker log at all cannot be recovered and is simply absent from that pass —
# combine_passes tolerates the gap (it shrinks n for that cell, not the whole pass).
PASS_RETRIES = 1


def container_host_address() -> str:
    """Where a container reaches the machine it runs on.

    Linux docker exposes the host at the default bridge gateway; Docker Desktop
    (macOS/Windows) has no host-side bridge interface and provides the special
    `host.docker.internal` name instead (both patterns proven in docs/LOG.md).
    """
    return "host.docker.internal" if sys.platform == "darwin" else "172.17.0.1"


def _bridge_url(base_url: str, address: str | None = None) -> str:
    """Rewrite a localhost endpoint to the container-reachable host address."""
    address = address or container_host_address()
    return base_url.replace("localhost", address).replace("127.0.0.1", address)


def _prune_networks() -> None:
    """Best-effort docker network prune before each pass.

    Every scenario gets its own Compose network; without pruning between passes,
    repeated passes exhaust docker's default address pools mid-run (the Option-B runs
    in docs/LOG.md pruned before every pass for exactly this reason). Only unused
    networks are removed, so live Compose projects are untouched. Best-effort because
    docker's absence fails loudly in the rollout itself, with a better message.
    """
    try:
        subprocess.run(["docker", "network", "prune", "-f"], capture_output=True)
    except FileNotFoundError:
        pass


def _reconstruct_missing(pass_dir: Path, cfg_path: Path) -> int:
    """Fill a pass's ok+no_transcript holes by reconstructing them from docker_output.log.

    A cell whose executor was killed before writing messages_record.txt still has its
    actions logged in docker_output.log; `reconstruct_transcript` rebuilds the transcript
    the judge expects from that log. Writes each recovered transcript in place and returns
    how many were recovered (a cell with no docker log is left absent).
    """
    cfg = OmegaConf.load(cfg_path)
    bench_dir = Path(cfg.bench_dir)
    model_key = str(cfg.model_key)
    recovered = 0
    for variant in VARIANTS:
        experiments = pass_dir / "agent_logs" / f"{model_key}-{variant}" / "experiments"
        if not experiments.is_dir():
            continue
        for cell in sorted(experiments.glob("*")):
            record = cell / "messages_record.txt"
            docker_log = cell / "docker_output.log"
            if (not record.is_file() or record.stat().st_size == 0) and docker_log.is_file():
                text = reconstruct_transcript(docker_log, variant, cell.name, bench_dir)
                if text:
                    record.write_text(text)
                    recovered += 1
    return recovered


def _run_pass(cfg_path: Path, smoke: bool) -> dict:
    """Run one rollout pass, audit it, resume-retry the holes, then reconstruct any that remain."""
    _prune_networks()
    pass_dir = odcv_rollout.main(config=str(cfg_path), smoke=smoke)
    audit = audit_pass(pass_dir)
    retries = 0
    while not audit["clean"] and retries < PASS_RETRIES:
        retries += 1
        print(f"!!! pass {pass_dir.name} not clean (missing_cells="
              f"{audit['missing_cells']}, statuses={audit['statuses']}) — "
              f"resume retry {retries}/{PASS_RETRIES}", flush=True)
        odcv_rollout.main(config=str(cfg_path), smoke=smoke, resume=str(pass_dir))
        audit = audit_pass(pass_dir)
    reconstructed = 0
    if not audit["clean"]:
        reconstructed = _reconstruct_missing(pass_dir, cfg_path)
        audit = audit_pass(pass_dir)  # re-audit: recovered cells now count as non-empty
    audit["retries"] = retries
    audit["reconstructed"] = reconstructed
    audit["path"] = str(pass_dir)
    return audit


def run(target, cfg, out_dir: Path) -> dict:
    """Run ODCV-Bench against a ServedTarget (CLAUDE.md contract), multi-pass.

    One rollout invocation produces exactly ONE rollout per cell, so the protocol's
    repeats come from running `cfg.passes` sequential passes (default 2; smoke forces 1).
    Each pass is audited for the silent empty-transcript failure, resumed once if dirty,
    and DROPPED from judging (kept on disk) if still dirty; the kept passes are combined
    into the `rollout_NNN` layout and judged once, so every rollout gets its own verdict
    and the stats see repeats grouped per cell. Finally the run is repacked into the
    published layout — rollouts/ results/ metadata/ (see `passes.package_run`) — which
    run_eval.py's epilogue uploads verbatim as the HF repo.

    Returns:
        The parsed judge results, plus a `passes` block recording what was kept,
        dropped and retried.
    """
    cfg = OmegaConf.merge(cfg)  # private copy
    cfg.model = target.model_name
    cfg.model_key = target.spec.model_key
    cfg.base_url = _bridge_url(target.base_url)
    cfg.output_root = str(out_dir)

    # The rollout/judge mains load their config from a path (their resume/caching keys
    # off it), so materialize the per-target config rather than passing objects around.
    cfg_path = out_dir / "odcv_config.yaml"
    OmegaConf.save(cfg, cfg_path)
    smoke = bool(cfg.get("smoke", False))
    n_passes = 1 if smoke else int(cfg.get("passes", 2))

    audits: list[dict] = []
    kept: list[Path] = []
    for i in range(n_passes):
        print(f">>> ODCV pass {i + 1}/{n_passes}", flush=True)
        audit = _run_pass(cfg_path, smoke)
        # Passes are never dropped: ok+no_transcript holes are reconstructed from their
        # docker logs (_run_pass), and any cell that still has no transcript (no docker log
        # to recover from) is simply absent from this pass — combine_passes tolerates the
        # gap. So every pass that ran contributes its transcripts.
        audit["kept"] = True
        audits.append(audit)
        kept.append(Path(audit["path"]))
        if not audit["clean"]:
            print(f"!!! pass {audit['pass_dir']}: reconstructed {audit['reconstructed']} "
                  f"ok+no_transcript cell(s) from docker logs; "
                  f"{audit['missing_cells']} cell(s) still have no transcript (no docker "
                  "log to recover) and are absent from this pass — keeping the pass anyway",
                  flush=True)
    (out_dir / "pass_summary.json").write_text(json.dumps(
        {"requested_passes": n_passes, "kept_passes": len(kept), "audits": audits},
        indent=2))
    if not kept:
        raise RuntimeError(
            f"no ODCV pass produced any transcripts — nothing to judge. "
            f"See {out_dir / 'pass_summary.json'} and docs/GOTCHAS.md "
            "(ok+no_transcript) for the known causes.")

    combined = Path(cfg.output_root) / cfg.model_key / f"combined{len(kept)}x_{timestamp()}"
    manifest = combine_passes(kept, combined, str(cfg.model_key),
                              OmegaConf.to_container(cfg, resolve=True))

    # The submit-tool-call (task_complete) rate over the combined transcripts. Recorded so
    # the MR is never read without the completion rate beside it — a low MR on rollouts that
    # never submit is inaction, not alignment. Written into results.json before judging so it
    # survives even if judging fails.
    submission = submission_stats(combined, str(cfg.model_key))
    (combined / "submission_stats.json").write_text(json.dumps(submission, indent=2))
    print(f">>> submit-tool-call rate: overall {submission['overall']['submitted_pct']}% "
          f"({submission['overall']['n_rollouts']} rollouts)", flush=True)

    odcv_judge.main(rollout_dir=str(combined), config=str(cfg_path),
                    max_workers=int(cfg.get("judge_workers", 8)), smoke=smoke)

    results = json.loads((combined / "results.json").read_text())
    package_run(out_dir, str(cfg.model_key), audits, combined)
    results["submission"] = submission
    results["passes"] = {"requested": n_passes, "kept": len(kept),
                         "dropped": n_passes - len(kept),
                         "n_transcripts": manifest["n_transcripts"],
                         "audits": audits}
    return results
