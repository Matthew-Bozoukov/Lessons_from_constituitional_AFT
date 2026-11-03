# ABOUTME: Compares auditor token billing between Petri run dirs (auditor_cache on vs off) from their
# ABOUTME: Inspect logs: uncached input, cache reads/writes, output, per seed. Run: uv run python scratch/petri_cache_compare.py <run_dir>...
import glob, json, sys
from inspect_ai.log import read_eval_log

def summarise(run_dir: str) -> dict:
    meta = json.load(open(f"{run_dir}/metadata/run_meta.json"))
    cfg = meta.get("config") or {}
    tot = {"in": 0, "cache_read": 0, "cache_write": 0, "out": 0, "calls": 0, "seeds": 0, "scored": 0}
    for p in glob.glob(f"{run_dir}/metadata/inspect/*.eval"):
        log = read_eval_log(p)
        for s in log.samples or []:
            tot["seeds"] += 1
            tot["scored"] += bool(s.scores)
            for e in s.events:
                if e.event != "model" or "claude" not in e.model or not e.output or not e.output.usage:
                    continue
                u = e.output.usage
                tot["calls"] += 1
                tot["in"] += u.input_tokens or 0
                tot["cache_read"] += u.input_tokens_cache_read or 0
                tot["cache_write"] += u.input_tokens_cache_write or 0
                tot["out"] += u.output_tokens or 0
    prompt = tot["in"] + tot["cache_read"] + tot["cache_write"]
    tot["prompt_total"] = prompt
    tot["cache_read_share"] = round(tot["cache_read"] / prompt, 3) if prompt else None
    # Anthropic billing weights: uncached 1.0, cache write 1.25, cache read 0.1 (of the input price)
    tot["billed_input_equiv"] = round(tot["in"] + 1.25 * tot["cache_write"] + 0.1 * tot["cache_read"])
    tot["auditor_cache"] = cfg.get("auditor_cache")
    tot["limit"] = cfg.get("limit")
    tot["run"] = run_dir.rstrip("/").split("/")[-1]
    return tot

rows = [summarise(d) for d in sys.argv[1:]]
keys = ["run", "auditor_cache", "seeds", "scored", "calls", "in", "cache_write", "cache_read", "out", "prompt_total", "cache_read_share", "billed_input_equiv"]
print("| " + " | ".join(keys) + " |"); print("|" + "---|" * len(keys))
for r in rows:
    print("| " + " | ".join(str(r[k]) for k in keys) + " |")
