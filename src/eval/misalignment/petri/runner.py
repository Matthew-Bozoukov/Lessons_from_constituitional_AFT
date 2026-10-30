# ABOUTME: run() for the Petri tenet audit per the eval contract: Petri's `audit` task in-process on
# ABOUTME: inspect-ai against the served target, auditor + judge via OpenRouter, logs -> the layout.

"""The registry's run() for petri.

Petri (inspect-petri) drives the whole audit -- the auditor builds the scenario the seed
describes, the target acts in it with real tools, the judge scores the transcript -- and this
function decides only WHAT runs and WHERE the models are: the served target as
`openai-api/vllm/<served name>`, auditor and judge as provider-pinned OpenRouter models, one
`inspect_ai.eval` call of `passes` epochs whose `.eval` logs stay under metadata/inspect/ for
export.py. Kept from the October driver (deleted 2026-10-30; docs/LOG.md): a liveness watchdog
on the served endpoint, a completeness gate on the scored seeds, and `resume_from`.
"""

from __future__ import annotations

import _thread
import json
import os
import shutil
import threading
import urllib.request
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlparse

from inspect_ai import eval as inspect_eval
from inspect_ai import eval_retry
from inspect_ai.log import read_eval_log
from inspect_ai.model import get_model
from inspect_petri import audit
from omegaconf import DictConfig

from src.eval.layout import publish_layout
from src.eval.misalignment.petri import export, scores, target_hook
from src.eval.preamble import Preamble
from src.infra.endpoints.openrouter import provider_pin

__all__ = ["run"]


class _Watchdog:
    """Abort the audit when the served target stops answering.

    A daemon thread probes `<base_url>/models`; after `fails` consecutive misses it sets
    `tripped` and `_thread.interrupt_main()`, which inspect_ai's asyncio loop takes as a
    cancel: `eval` then RETURNS cancelled logs rather than raising, so the runner checks
    `tripped` after the call. Abort rather than reconnect: scored samples are durable in the
    .eval log and `resume_from` finishes the run, whereas on 2026-10-03 a dead tunnel left
    running cost ~$52 of auditor for 7 scored seeds (the ssh keepalive noticed; nothing acted).
    """

    def __init__(self, base_url: str, every_s: float, fails: int):
        self.probe = base_url.rstrip("/") + "/models"
        self.every_s = float(every_s)
        self.fails = int(fails)
        self.tripped = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def __enter__(self) -> "_Watchdog":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=15)  # a probe in flight (10s timeout) must not outlive the block

    def _watch(self) -> None:
        misses = 0
        while not self._stop.wait(self.every_s):
            try:
                with urllib.request.urlopen(self.probe, timeout=10) as r:
                    ok = r.status == 200
            except Exception:
                ok = False
            misses = 0 if ok else misses + 1
            # Re-checked after the probe: a miss that straddles __exit__ must not interrupt
            # the export and publish that follow the eval.
            if misses >= self.fails and not self._stop.is_set():
                print(f"!!! target stopped answering at {self.probe} ({misses} consecutive probes "
                      f"{self.every_s:g}s apart) -- aborting the audit so it does not keep paying the "
                      f"auditor to talk to nothing.", flush=True)
                self.tripped = True
                _thread.interrupt_main()
                return


def _check_resume_port(log_path: Path, base_url: str) -> None:
    """eval-retry reuses the target base_url the log recorded and ignores anything else, so a
    server on any other port leaves every target call failing while the auditor and judge carry
    on and score the silence as concerning=1 (2026-10-09: 50 da-15 seeds)."""
    roles = read_eval_log(str(log_path), header_only=True).eval.model_roles or {}
    assert "target" in roles, f"resume_from: {log_path.name} records no target role; it is not a petri log"
    recorded = roles["target"].base_url
    assert urlparse(recorded).port == urlparse(base_url).port, (
        f"resume_from: the log's target is {recorded}; serve on --port {urlparse(recorded).port} "
        f"(eval-retry reuses the recorded base_url and ignores anything else; 2026-10-09 da-15)")


def _run_meta(prior: Path) -> dict:
    """A prior run dir's run_meta.json: at the root while a run is live or died (run_eval
    writes it there at launch), under metadata/ once its epilogue homed it."""
    for path in (prior / "run_meta.json", prior / "metadata" / "run_meta.json"):
        if path.is_file():
            return json.loads(path.read_text())
    raise FileNotFoundError(f"resume_from={prior}: no run_meta.json at its root or under metadata/")


def run(target, cfg: DictConfig, out_dir: Path) -> dict:
    """Run the Petri tenet audit against one target (CLAUDE.md contract).

    Args:
        target: The ServedTarget from run_eval (base_url, model_name, api_key, spec).
        cfg: configs/eval/petri.yaml plus CLI overrides.
        out_dir: This target's run directory; rollouts/, results/, metadata/ go under it.

    Returns:
        The summary: `concerning >= 4` rate (with a t over passes when passes >= 2), the
        strict rate, per-family counts, and run health.
    """
    assert os.environ.get("OPENROUTER_API_KEY"), "petri: auditor and judge both route through OpenRouter"
    rollouts_dir, results_dir, metadata_dir = publish_layout(out_dir)
    log_dir = metadata_dir / "inspect"
    log_dir.mkdir(exist_ok=True)
    seeds_dir = Path(str(cfg.seeds)).resolve()
    seed_ids = {p.stem for p in seeds_dir.glob("*.md")}
    assert seed_ids, f"petri: no seed files (*.md) under {seeds_dir}"
    passes = int(cfg.passes)
    limit = int(cfg.limit) if cfg.get("limit") else None

    # The constitution arms: the composed preamble goes first in every TARGET system message
    # (auditor and judge untouched). Installed before either eval call because eval_retry
    # re-creates `audit` from the registry; with no preamble the hook is Petri's audit exactly.
    preamble = Preamble.from_cfg(cfg)
    target_hook.install(preamble)

    base_url = target.base_url  # touching it is what serves a lazy target
    target_model = f"openai-api/vllm/{target.model_name}"
    # The key goes through the env, not a get_model kwarg: eval_retry rebuilds every role from
    # the log's ModelConfig, which records base_url but never a credential, and the openai-api
    # provider then reads <SERVICE>_API_KEY (whistlebench's wiring).
    os.environ["VLLM_API_KEY"] = target.api_key or "EMPTY"
    roles = {"target": get_model(target_model, base_url=base_url),
             "auditor": get_model(f"openrouter/{cfg.auditor}", provider=provider_pin(str(cfg.auditor))),
             "judge": get_model(f"openrouter/{cfg.judge}", provider=provider_pin(str(cfg.judge)))}

    # Resume: the prior run's logs are copied in so this run dir is self-contained, and
    # eval_retry re-runs only the unscored samples of the newest one.
    resumed_from = cfg.get("resume_from")
    retry_log = None
    if resumed_from:
        prior = Path(str(resumed_from)).resolve()
        prior_meta = _run_meta(prior)
        now = {"target": target.spec.hf_path, "target_revision": target.spec.revision,
               "mode": target.spec.mode}
        then = {k: prior_meta.get(k) for k in now}
        assert then == now, f"resume_from is a different run: {then} != {now}"
        prior_logs = sorted((prior / "metadata" / "inspect").glob("*.eval"))
        assert prior_logs, f"resume_from={prior}: no .eval logs under metadata/inspect"
        _check_resume_port(prior_logs[-1], base_url)
        for log in prior_logs:
            shutil.copy2(log, log_dir / log.name)
        retry_log = log_dir / prior_logs[-1].name

    print(f">>> petri: {limit or len(seed_ids)} seeds x {passes} passes, max_turns {int(cfg.max_turns)} | "
          f"target {target_model} (mode {target.spec.mode}) | auditor {cfg.auditor} | judge {cfg.judge}"
          + (f" | preamble {preamble.file} (sha {preamble.sha256[:12]})" if preamble else "")
          + (f" | resuming {retry_log.name}" if retry_log else ""), flush=True)

    aborted = f"scored samples are in {log_dir}; resume with resume_from={out_dir}"
    with _Watchdog(base_url, cfg.health.every_s, cfg.health.fails) as wd:
        try:
            if retry_log is not None:
                # The log carries the task, seeds, max_turns and model roles; passing them
                # again would be a second source of truth for facts the log already fixes.
                logs = eval_retry(str(retry_log), log_dir=str(log_dir),
                                  max_connections=int(cfg.inspect.max_connections),
                                  retry_on_error=int(cfg.inspect.retry_on_error),
                                  log_level=str(cfg.inspect.log_level), display="plain",
                                  fail_on_error=False)
            else:
                task = audit(seed_instructions=str(seeds_dir), max_turns=int(cfg.max_turns))
                logs = inspect_eval(task, model_roles=roles, epochs=passes, limit=limit,
                                    log_dir=str(log_dir), log_format=str(cfg.inspect.log_format),
                                    log_level=str(cfg.inspect.log_level),
                                    max_connections=int(cfg.inspect.max_connections),
                                    retry_on_error=int(cfg.inspect.retry_on_error),
                                    fail_on_error=False, display="plain")
        except KeyboardInterrupt:  # the interrupt escaped inspect's loop (e.g. during setup)
            if wd.tripped:
                raise SystemExit(f"petri: target stopped answering; {aborted}") from None
            raise
    # inspect takes the watchdog's interrupt (and a Ctrl-C) as a cancel and RETURNS the
    # cancelled logs, so the trip is checked here rather than caught: a cancelled run is never
    # scored, exported or published as a result.
    if wd.tripped or any(log.status == "cancelled" for log in logs):
        raise SystemExit(f"petri: {'target stopped answering' if wd.tripped else 'the audit was cancelled'}; "
                         f"{aborted}")

    all_logs = sorted(log_dir.glob("*.eval"))
    by_seed = scores.valid_scores(all_logs)
    # The completeness gate: a limited run is a smoke and has no completeness to check.
    expected = None if limit else seed_ids
    summary = scores.summarize(by_seed, expected)
    if expected is not None:
        allowed = set(cfg.get("allow_missing") or [])
        if set(summary["missing_seeds"]) != allowed or summary["uneven_seeds"]:
            raise RuntimeError(
                f"petri: {len(summary['missing_seeds'])} seeds have no valid score (e.g. "
                f"{summary['missing_seeds'][:3]}) and {len(summary['uneven_seeds'])} were not scored in "
                f"every pass -- finish with resume_from={out_dir} or name them in allow_missing")

    counts = export.write_rollouts(all_logs, rollouts_dir,
                                   preamble_text=preamble.text if preamble else None, passes=passes)
    if preamble and counts["targets_without_system_prompt"]:
        # The seed's own behaviour, not a run failure: the hook prepends only to a staged system
        # message, so such a target saw no constitution and the export gate had nothing to check.
        print(f"!!! petri: {counts['targets_without_system_prompt']} target(s) were called with no system "
              f"message, so no constitution was shown; see results.json targets_without_system_prompt", flush=True)
    (results_dir / "scores.json").write_text(json.dumps(by_seed, indent=2, sort_keys=True))
    print(f">>> petri: concerning>=4 {summary['concerning_ge_4_pct']}% ({summary['concerning_ge_4']}/"
          f"{summary['n_scenarios'] * summary['n_passes']} seed-passes), >=6 {summary['concerning_ge_6_pct']}% | "
          f"{counts['n_rollouts']} rollouts ({counts['excluded_rollouts']} excluded) | "
          f"missing {len(summary['missing_seeds'])}", flush=True)
    return {**summary, **counts,
            "auditor": str(cfg.auditor), "judge": str(cfg.judge), "max_turns": int(cfg.max_turns),
            "passes": passes, "limit": limit, "seeds_dir": str(cfg.seeds),
            "context_window": int(cfg.serving.context_window),
            "inspect_logs": "metadata/inspect",
            "inspect_status": {Path(log.location).name: log.status for log in logs},
            "inspect_version": version("inspect-ai"), "inspect_petri_version": version("inspect-petri"),
            "resumed_from": str(resumed_from) if resumed_from else None,
            "system_preamble_sha256": preamble.sha256 if preamble else None}
