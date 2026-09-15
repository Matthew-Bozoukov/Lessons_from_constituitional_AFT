# ABOUTME: Throwaway collector: every fact the legacy-name table needs, per repo in the org,
# ABOUTME: dumped to one JSON. Resumable and rate-limit aware (HF: 1000 req / 5 min).
from __future__ import annotations
import json, re, sys, time
from pathlib import Path
from huggingface_hub.errors import HfHubHTTPError
from src.infra.huggingface import hf_api, hf_download
from src.naming import NamingError, derive_artifact_name_from_legacy, mix_subject_from

ORG = "LASR-Callum"; api = hf_api()
OUT = Path(sys.argv[1]); facts = json.loads(OUT.read_text()) if OUT.exists() else {}

def retry(fn, *a, **k):
    for _ in range(4):
        try: return fn(*a, **k)
        except HfHubHTTPError as e:
            if "429" in str(e): print("rate limited; sleeping 310s", flush=True); time.sleep(310); continue
            raise
    return fn(*a, **k)
def dl(repo, f, rt):
    try: return retry(hf_download, repo, f, repo_type=rt)
    except Exception: return None
def jl(p): return json.loads(Path(p).read_text()) if p else {}
def card(repo, rt):
    p = dl(repo, "README.md", rt); text = Path(p).read_text(errors="ignore") if p else ""
    fields = dict(re.findall(r"^\| `?([a-z_]+)`? \| (.*?) \|$", text, re.M))
    return {k: v[:300] for k, v in fields.items()}, text[:600]
def rows_subject(repo, files):
    for f in sorted((f for f in files if f.endswith(".jsonl")), key=lambda x: ("partial" in x, x)):
        p = dl(repo, f, "dataset")
        if not p: continue
        rows, srcs = [], {}
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line); m = r.get("metadata") or {}
                    s = r.get("source") or m.get("source"); rows.append({"source": s, "supervise": r.get("supervise") or m.get("supervise")})
                    srcs[s] = srcs.get(s, 0) + 1
        if rows and all(r["source"] for r in rows):
            try: return {"file": f, "sources": srcs, "subject": derive_artifact_name_from_legacy(rows), "refused": None}
            except NamingError as e: return {"file": f, "sources": srcs, "subject": None, "refused": str(e)[:200]}
        stats = dl(repo, f + ".stats.json", "dataset") or dl(repo, "mixture_stats.json", "dataset")
        if stats: return {"file": f, "sources": None, "subject": None, "refused": "no source column", "stats": jl(stats).get("by_source")}
    return None

todo = [(d.id, "dataset") for d in retry(api.list_datasets, author=ORG)] + [(m.id, "model") for m in retry(api.list_models, author=ORG)]
print(f"{len(todo)} repos, {len(facts)} already collected", flush=True)
for i, (repo, rt) in enumerate(todo):
    if repo in facts: continue
    info = retry(api.repo_info, repo, repo_type=rt, files_metadata=False)
    files = sorted(s.rfilename for s in (info.siblings or []))
    names = {Path(f).name for f in files}
    fields, head = card(repo, rt)
    rec = {"type": rt, "files": files[:40], "tags": list(info.tags or []), "card": fields, "readme_head": head,
           "created": str(getattr(info, "created_at", ""))[:10], "lawful": bool(mix_subject_from(repo)) if rt == "dataset" else False}
    if rt == "dataset":
        if "metadata/run_meta.json" in files:
            rec["kind"] = "eval"; meta = jl(dl(repo, "metadata/run_meta.json", "dataset"))
            rec["run_meta"] = {k: (meta.get(k) if k != "config" else None) for k in ("target", "base_model", "mode", "command")}
        elif any(re.match(r"stage_\d+_.*\.jsonl", n) for n in names) or "manifest.json" in names:
            rec["kind"] = "synth"; man = jl(dl(repo, "manifest.json", "dataset")) if "manifest.json" in names else {}
            rec["manifest"] = {k: man.get(k) for k in ("pipeline", "run_id", "counts", "git_sha")}
        elif any(n.endswith(".jsonl") for n in names):
            rs = rows_subject(repo, files); rec["kind"] = "mixture" if rs else "other"; rec["rows"] = rs
        else: rec["kind"] = "other"
    else:
        if "adapter_config.json" in files:
            rec["kind"] = "model"; ac = jl(dl(repo, "adapter_config.json", "model")); tm = jl(dl(repo, "training_meta.json", "model"))
            tc = tm.get("train_config"); seed = tc.get("seed") if isinstance(tc, dict) else tm.get("seed")
            if seed is None:
                mm = re.search(r'"seed":\s*(\d+)', head); seed = int(mm.group(1)) if mm else None
            rec["stamp"] = {"base": ac.get("base_model_name_or_path") or tm.get("base_model"), "r": ac.get("r"),
                            "dataset": (tm.get("dataset") or {}).get("repo") if isinstance(tm.get("dataset"), dict) else None,
                            "seed": seed, "thinking": tm.get("thinking"), "stamped": bool(tm),
                            "train_config": (tc if isinstance(tc, str) else (tm.get("train_config_name") if tm else None))}
        else: rec["kind"] = "other"
    facts[repo] = rec
    if i % 20 == 0: OUT.write_text(json.dumps(facts, indent=1)); print(f"  {i+1}/{len(todo)} {repo.split('/')[1][:50]} -> {rec['kind']}", flush=True)
OUT.write_text(json.dumps(facts, indent=1)); print(f"done: {len(facts)} repos -> {OUT}")
