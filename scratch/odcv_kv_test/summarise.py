# ABOUTME: Summarise one ODCV subset run for the prefix-caching A/B: pass wall clock, per-cell
# ABOUTME: elapsed, MR from results.json, and vLLM's cache/throughput lines from a log slice.
"""usage: uv run python scratch/odcv_kv_test/summarise.py <run_dir> [vllm_log_slice]"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path


def run_summary(run_dir: Path) -> dict:
    out: dict = {"run_dir": str(run_dir), "passes": []}
    for m in sorted(run_dir.glob("**/*rollout_manifest.json")):
        d = json.loads(m.read_text())
        el = [r["elapsed_s"] for r in d["results"]
              if r.get("status") == "ok" and r.get("elapsed_s")]
        if not el:
            continue
        out["passes"].append({
            "dir": str(m.parent.relative_to(run_dir)),
            "wall_clock_min": d["wall_clock_min"], "n": len(el),
            "elapsed_mean_s": round(statistics.mean(el), 1),
            "elapsed_median_s": round(statistics.median(el), 1),
            "elapsed_max_s": round(max(el), 1),
            "statuses": sorted({r["status"] for r in d["results"]}),
        })
    published = run_dir / "rollouts"  # the layout contract, once the run has repacked
    records = (published.glob("**/messages_record.txt") if published.is_dir()
               else run_dir.glob("**/combined*/**/messages_record.txt"))
    steps = [sum(1 for line in t.read_text(errors="replace").splitlines()
                 if line.startswith("== Step")) for t in records]
    if steps:
        out["steps_per_rollout"] = {"n": len(steps), "mean": round(statistics.mean(steps), 1),
                                    "max": max(steps)}
    res = sorted(run_dir.glob("**/results.json"))
    if res:
        r = json.loads(res[-1].read_text())
        out["results"] = {k: r.get(k) for k in ("ours", "n_judged", "submission")}
    return out


def log_summary(text: str) -> dict:
    hits = [float(x) for x in re.findall(r"Prefix cache hit rate: ([0-9.]+)%", text)]
    kv = [float(x) for x in re.findall(r"GPU KV cache usage: ([0-9.]+)%", text)]
    gen = [float(x) for x in re.findall(r"Avg generation throughput: ([0-9.]+) tokens/s", text)]
    pre = [float(x) for x in re.findall(r"Avg prompt throughput: ([0-9.]+) tokens/s", text)]
    busy = lambda xs: [x for x in xs if x > 0]  # noqa: E731  (idle ticks log 0.0)
    # vLLM logs the stats every ~10 s, so throughput x 10 s approximates tokens moved.
    return {
        "gen_tokens_est": int(sum(gen) * 10),
        "prompt_tokens_est": int(sum(pre) * 10),
        "busy_intervals": len(busy(gen)),
        "prefix_hit_rate_last": hits[-1] if hits else None,
        "kv_usage_max_pct": max(kv) if kv else None,
        "gen_tok_s_mean_busy": round(statistics.mean(busy(gen)), 1) if busy(gen) else None,
        "prompt_tok_s_mean_busy": round(statistics.mean(busy(pre)), 1) if busy(pre) else None,
        "n_stat_lines": len(gen),
    }


if __name__ == "__main__":
    summary = run_summary(Path(sys.argv[1]))
    if len(sys.argv) > 2:
        summary["vllm"] = log_summary(Path(sys.argv[2]).read_text())
    print(json.dumps(summary, indent=2))
