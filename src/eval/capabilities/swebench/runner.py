# ABOUTME: run() for the swebench eval: inspect_evals' `swe_bench` task against a served
# ABOUTME: target, then the .eval log unpacked into the contract layout (rollouts/results/metadata).

"""run() per the CLAUDE.md eval contract.

Everything inside a sandbox — the repository at its base commit, the agent's tool calls,
the test run that decides `resolved` — is inspect_evals' business (`inspect_evals/swe_bench`).
This module does the three things the repo contract needs around it:

1. **Reach the served target.** The target is the ordinary OpenAI triple; inspect's
   `openai-api/<service>/<model>` provider reads `<SERVICE>_BASE_URL` / `<SERVICE>_API_KEY`,
   so the served model name goes into the model string and the endpoint into the
   environment for the duration of the call. Nothing here starts a server.
2. **Pin what is attempted.** The instance list is a JSON file beside the config with its
   own selection provenance (arm64 availability, per-repo allocation), so every arm
   attempts literally the same instances; `limit` takes a prefix of it for a smoke.
3. **Publish in the contract layout.** The `.eval` log is authoritative and is copied
   under metadata/; every sample is also unpacked into `rollouts/<instance>/` as a
   self-contained transcript (JSON + markdown), and `results/results.json` carries the
   resolution rate with a Wilson interval, the per-instance outcomes, and the timing and
   token totals a run-length projection needs.

Re-entrant: no process-global state, everything under `out_dir`; the environment
variables it sets are restored on exit.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import statistics
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from src.eval.capabilities.stats import wilson_ci
from src.eval.layout import publish_layout
from src.utils import transcript_markdown

TASK = "inspect_evals/swe_bench"
# The provider service name: inspect reads LASR_BASE_URL / LASR_API_KEY for it.
SERVICE = "lasr"


def instance_ids(cfg) -> list[str]:
    """The ids this run attempts, in the file's order; `limit` takes a prefix."""
    explicit = cfg.get("instance_ids")
    if explicit:
        ids = [str(i) for i in (explicit if not isinstance(explicit, str) else explicit.split(","))]
    else:
        spec = json.loads(Path(str(cfg.instance_ids_file)).read_text())
        ids = [str(i) for i in spec["instance_ids"]]
    assert ids and len(set(ids)) == len(ids), "instance list is empty or has duplicates"
    limit = cfg.get("limit")
    return ids[: int(limit)] if limit else ids


def resolved_arch(cfg) -> str:
    """`arch` as inspect will see it: the config's, or the driver's own when `auto`."""
    arch = str(cfg.get("arch") or "auto")
    if arch != "auto":
        return arch
    return "arm64" if platform.machine() in {"aarch64", "arm64"} else "x86_64"


def eval_kwargs(cfg, ids: list[str], log_dir: Path) -> dict[str, Any]:
    """The keyword arguments to `inspect_ai.eval`, all from the config: one place a test
    can check the protocol without running anything."""
    gen = cfg.generation
    return {
        "task_args": {
            "dataset": str(cfg.dataset), "split": str(cfg.split), "revision": str(cfg.revision),
            "image_name_template": str(cfg.image_name_template), "arch": resolved_arch(cfg),
            "tool_timeout": int(cfg.tool_timeout),
        },
        "sample_id": ids,
        "epochs": int(cfg.get("epochs") or 1),
        "message_limit": int(cfg.message_limit),
        "token_limit": int(cfg.token_limit) if cfg.get("token_limit") else None,
        "time_limit": int(cfg.time_limit),
        "retry_on_error": int(cfg.get("retry_on_error") or 0),
        "fail_on_error": float(cfg.fail_on_error) if cfg.get("fail_on_error") is not None else None,
        "max_connections": int(cfg.concurrency),
        "max_sandboxes": int(cfg.concurrency),
        "max_samples": int(cfg.concurrency),
        "log_dir": str(log_dir),
        "log_format": "eval",
        "display": "plain",
        "temperature": float(gen.temperature),
        "top_p": float(gen.top_p),
        "max_tokens": int(gen.max_tokens),
        "reasoning_history": str(gen.reasoning_history),
        "seed": int(gen.seed) if gen.get("seed") is not None else None,
    }


def _dump(obj) -> Any:
    """A pydantic model, or anything else, as plain JSON-able data."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    return obj


def _usage_totals(sample) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for usage in (getattr(sample, "model_usage", None) or {}).values():
        for key in totals:
            totals[key] += int(getattr(usage, key, 0) or 0)
    return totals


def unpack_sample(sample, rollouts_dir: Path) -> dict[str, Any]:
    """Write one sample's self-contained transcript and return its outcome row."""
    iid = str(sample.id)
    score = None
    scores = getattr(sample, "scores", None) or {}
    if scores:
        score = next(iter(scores.values()))
    value = getattr(score, "value", None) if score is not None else None
    error = getattr(sample, "error", None)
    limit = getattr(sample, "limit", None)
    row = {
        "instance_id": iid,
        "epoch": int(getattr(sample, "epoch", 1) or 1),
        # 1.0 / 0.0 from the scorer; None when the sample never reached scoring.
        "resolved": (float(value) >= 1.0) if isinstance(value, (int, float)) else None,
        "score": value,
        "error": getattr(error, "message", None) if error is not None else None,
        "limit": getattr(limit, "type", None) if limit is not None else None,
        "messages": len(getattr(sample, "messages", None) or []),
        "total_time_s": getattr(sample, "total_time", None),
        "working_time_s": getattr(sample, "working_time", None),
        **_usage_totals(sample),
    }
    if score is not None:
        row["explanation"] = getattr(score, "explanation", None)
        row["model_patch"] = (getattr(score, "metadata", None) or {}).get("model_patch")
    d = rollouts_dir / iid
    d.mkdir(parents=True, exist_ok=True)
    messages = [_dump(m) for m in (getattr(sample, "messages", None) or [])]
    (d / "transcript.json").write_text(json.dumps(
        {"instance_id": iid, "outcome": {k: v for k, v in row.items() if k != "model_patch"},
         "messages": messages}, indent=1, ensure_ascii=False))
    if row.get("model_patch"):
        (d / "model.patch").write_text(str(row["model_patch"]))
    sections = [(2, "Outcome", "json", json.dumps(
        {k: row[k] for k in ("resolved", "score", "error", "limit", "messages",
                             "total_time_s", "output_tokens") if k in row}, indent=2))]
    for m in messages:
        role = m.get("role", "?")
        body = m.get("content")
        if not isinstance(body, str):
            body = json.dumps(body, indent=1, ensure_ascii=False)
        if m.get("reasoning") or any(isinstance(c, dict) and c.get("type") == "reasoning"
                                     for c in (m.get("content") or []) if not isinstance(m.get("content"), str)):
            sections.append((3, f"{role} · reasoning", "fenced",
                             json.dumps([c for c in m.get("content") if isinstance(c, dict)
                                         and c.get("type") == "reasoning"], ensure_ascii=False)
                             if not isinstance(m.get("content"), str) else str(m.get("reasoning"))))
        if m.get("tool_calls"):
            sections.append((3, f"{role} · tool calls", "json",
                             json.dumps(m["tool_calls"], indent=1, ensure_ascii=False)))
        sections.append((3, role, "fenced", body or ""))
    (d / "transcript.md").write_text(transcript_markdown(
        f"SWE-bench Lite · {iid}", None, sections))
    return row


def summarise(rows: list[dict], *, status: str, model: str, arch: str,
              requested: int, concurrency: int) -> dict[str, Any]:
    """The results.json body: the rate, its interval, the outcome tally, and timings."""
    scored = [r for r in rows if r["resolved"] is not None]
    resolved = sum(1 for r in scored if r["resolved"])
    times = [float(r["total_time_s"]) for r in rows if r.get("total_time_s")]
    out_tok = [int(r["output_tokens"]) for r in rows]
    def pct(xs, q):
        if not xs:
            return None
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]
    summary = {
        "inspect_status": status,
        "model": model,
        "arch": arch,
        "n_requested": requested,
        "n_attempted": len(rows),
        "n_scored": len(scored),
        "n_resolved": resolved,
        # The headline: resolved over every instance ATTEMPTED (an error or a limit that
        # never reached scoring counts as unresolved), the way pass@1 is reported.
        "resolved_rate": (resolved / len(rows)) if rows else None,
        "resolved_rate_ci95": wilson_ci(resolved, len(rows)) if rows else None,
        "resolved_rate_scored_only": (resolved / len(scored)) if scored else None,
        "n_errors": sum(1 for r in rows if r.get("error")),
        "limits": {k: sum(1 for r in rows if r.get("limit") == k)
                   for k in sorted({r.get("limit") for r in rows if r.get("limit")})},
        "timing_s": {
            "sum": round(sum(times), 1), "mean": round(statistics.mean(times), 1) if times else None,
            "median": round(statistics.median(times), 1) if times else None,
            "p95": round(pct(times, 0.95), 1) if times else None,
            "max": round(max(times), 1) if times else None,
        },
        "tokens": {"output_sum": sum(out_tok), "output_mean": round(statistics.mean(out_tok)) if out_tok else None,
                   "output_max": max(out_tok) if out_tok else None,
                   "input_sum": sum(int(r["input_tokens"]) for r in rows)},
        "concurrency": concurrency,
        "per_instance": {r["instance_id"]: {k: r.get(k) for k in (
            "resolved", "score", "error", "limit", "messages", "total_time_s", "output_tokens")}
            for r in rows},
    }
    return summary


def projection(summary: dict, *, n_total: int, concurrency: int) -> dict[str, Any]:
    """Run-length estimate for `n_total` instances at `concurrency` in flight, from the
    measured per-instance times: work / concurrency, floored by the longest instance."""
    t = summary["timing_s"]
    if not t.get("mean"):
        return {}
    work = t["mean"] * n_total
    return {"n_total": n_total, "concurrency": concurrency,
            "ideal_minutes": round(work / concurrency / 60, 1),
            "tail_floor_minutes": round((t.get("max") or 0) / 60, 1),
            "estimate_minutes": round(max(work / concurrency, t.get("max") or 0) / 60, 1),
            "output_tokens_total": (summary["tokens"].get("output_mean") or 0) * n_total}


def run(target, cfg, out_dir: Path) -> dict:
    """Per the eval contract: attempt the configured instances against `target`, publish."""
    from inspect_ai import eval as inspect_eval

    rollouts_dir, results_dir, metadata_dir = publish_layout(out_dir)
    ids = instance_ids(cfg)
    log_dir = metadata_dir / "inspect_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    kwargs = eval_kwargs(cfg, ids, log_dir)
    model = f"openai-api/{SERVICE}/{target.model_name}"
    (metadata_dir / "protocol.json").write_text(json.dumps(
        {"task": TASK, "model": model, "instance_ids": ids, **{
            k: v for k, v in kwargs.items() if k not in ("sample_id", "log_dir")}}, indent=2))
    (metadata_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    env_keys = {f"{SERVICE.upper()}_BASE_URL": target.base_url,
                f"{SERVICE.upper()}_API_KEY": target.api_key}
    saved = {k: os.environ.get(k) for k in env_keys}
    os.environ.update(env_keys)
    try:
        print(f">>> swebench: {len(ids)} instances, arch={kwargs['task_args']['arch']}, "
              f"concurrency={cfg.concurrency}, model={model}")
        logs = inspect_eval(TASK, model=model, **kwargs)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert logs, "inspect returned no log"
    log = logs[0]
    rows = [unpack_sample(s, rollouts_dir) for s in (log.samples or [])]
    summary = summarise(rows, status=str(log.status), model=model,
                        arch=kwargs["task_args"]["arch"], requested=len(ids),
                        concurrency=int(cfg.concurrency))
    summary["projection_200"] = projection(summary, n_total=200, concurrency=int(cfg.concurrency))
    if getattr(log, "location", None) and Path(str(log.location)).exists():
        shutil.copy(str(log.location), metadata_dir / Path(str(log.location)).name)
    (results_dir / "results.json").write_text(json.dumps(summary, indent=2))
    (results_dir / "per_instance.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(f">>> swebench: {summary['n_resolved']}/{summary['n_attempted']} resolved "
          f"(status {log.status}); timing {summary['timing_s']}")
    # The row that reaches results.json/results.md through the epilogue: everything but
    # the per-instance table, which per_instance.json already holds.
    return {k: v for k, v in summary.items() if k != "per_instance"}
