# ABOUTME: Eval-framework entrypoint for ODCV-Bench: N audited rollout passes against a served
# ABOUTME: target (reached at the docker host address), combined and multi-judge scored once.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.misalignment.odcv import odcv_judge, odcv_rollout
from src.eval.misalignment.odcv.passes import (
    audit_pass, combine_passes, package_run, submission_stats)
from src.utils import timestamp

# One resume attempt per pass. Resuming re-runs exactly the cells whose transcript is
# missing or empty (the rollout driver's cache check requires a non-empty
# messages_record.txt), so a retry is cheap. A pass still dirty after its retry is
# excluded from judging rather than retried forever: transient causes (executor timeout,
# dropped tunnel) clear on one retry, and anything that survives it is structural —
# docs/GOTCHAS.md "ok+no_transcript" — and needs a human, not more spend.
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


def _run_pass(cfg_path: Path, smoke: bool) -> dict:
    """Run one rollout pass, audit it, and resume-retry the holes up to PASS_RETRIES."""
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
    audit["retries"] = retries
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
        audit["kept"] = audit["clean"]
        audits.append(audit)
        if audit["clean"]:
            kept.append(Path(audit["path"]))
        else:
            print(f"!!! DROPPING pass {audit['pass_dir']}: still missing "
                  f"{audit['missing_cells']} cell(s) after {audit['retries']} "
                  "resume retry — kept on disk, excluded from judging", flush=True)
    (out_dir / "pass_summary.json").write_text(json.dumps(
        {"requested_passes": n_passes, "kept_passes": len(kept), "audits": audits},
        indent=2))
    if not kept:
        raise RuntimeError(
            f"all {n_passes} ODCV pass(es) failed their audit — nothing to judge. "
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
