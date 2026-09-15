# ABOUTME: Throwaway probe: classify every repo in the LASR-Callum org and report what the
# ABOUTME: law could name a NEW artifact built from it. Bulk file listings (rate limit).
from __future__ import annotations
import json, re
from collections import Counter
from pathlib import Path
from src.infra.huggingface import hf_api, hf_download
from src.model_profile import model_key
from src.naming import NamingError, derive_artifact_name_from_legacy, mix_subject_from, undated

ORG = "LASR-Callum"; api = hf_api()
def dl(repo, f, rt):
    try: return hf_download(repo, f, repo_type=rt)
    except Exception: return None
def jl(p): return json.loads(Path(p).read_text()) if p else {}

def _files(repo, rt):
    info = api.repo_info(repo, repo_type=rt, files_metadata=False)
    return {s.rfilename for s in (info.siblings or [])}
datasets = {d.id: _files(d.id, "dataset") for d in api.list_datasets(author=ORG)}
models = {m.id: _files(m.id, "model") for m in api.list_models(author=ORG)}
print(f"{len(datasets)} datasets, {len(models)} models", flush=True)

cache = {}
def mix_subject_for(repo):
    if repo in cache: return cache[repo]
    s = mix_subject_from(repo)
    if s: cache[repo] = (s, "name"); return cache[repo]
    fs = sorted((f for f in datasets.get(repo, ()) if f.endswith(".jsonl")), key=lambda x: ("partial" in x, x))
    for f in fs:
        p = dl(repo, f, "dataset")
        if not p: continue
        rows = []
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line); m = r.get("metadata") or {}
                    rows.append({"source": r.get("source") or m.get("source"), "supervise": r.get("supervise") or m.get("supervise")})
        if rows and all(r["source"] for r in rows):
            try: cache[repo] = (derive_artifact_name_from_legacy(rows), "rows")
            except NamingError as e: cache[repo] = (None, "REFUSED: " + str(e).split(". ")[0].split(": ", 1)[-1][:80])
            return cache[repo]
    cache[repo] = (None, "REFUSED: no jsonl with a source column"); return cache[repo]

kinds = Counter(); status = {k: Counter() for k in ("synth", "mixture", "eval", "model")}; notes = []
for repo, fs in datasets.items():
    names = {Path(f).name for f in fs}
    if "metadata/run_meta.json" in fs:
        kinds["eval run"] += 1; meta = jl(dl(repo, "metadata/run_meta.json", "dataset"))
        t = meta.get("target") or ""
        cmd = meta.get("command") or ""; mt = re.search(r"--target\s+(\S+)", cmd)
        if t: status["eval"]["derived from run_meta.target"] += 1
        elif mt: status["eval"]["derived from run_meta.command"] += 1
        else: status["eval"]["REFUSED: no target in run_meta"] += 1; notes.append(("eval", repo, sorted(meta)[:5]))
    elif any(re.match(r"stage_\d+_.*\.jsonl", n) for n in names) or "manifest.json" in names:
        kinds["synth corpus"] += 1
        pipe = jl(dl(repo, "manifest.json", "dataset")).get("pipeline") if "manifest.json" in names else None
        if not pipe:
            try: pipe = next((x[9:] for x in (api.repo_info(repo, repo_type="dataset").tags or []) if x.startswith("pipeline:")), None)
            except Exception: pipe = None
        status["synth"][f"needs map: {pipe}" if pipe else "REFUSED: no pipeline anywhere"] += 1
    elif any(n.endswith(".jsonl") for n in names):
        s, how = mix_subject_for(repo)
        if s: kinds["mixture"] += 1; status["mixture"][f"derived/{how}"] += 1
        elif "source column" in how: kinds["other dataset"] += 1
        else: kinds["mixture"] += 1; status["mixture"][how] += 1
    else: kinds["other dataset"] += 1
for repo, fs in models.items():
    if "adapter_config.json" not in fs: kinds["other model"] += 1; continue
    kinds["adapter"] += 1
    tm = jl(dl(repo, "training_meta.json", "model")) if "training_meta.json" in fs else {}
    if not tm: status["model"]["REFUSED: unstamped (no training_meta.json)"] += 1; continue
    ds = (tm.get("dataset") or {}).get("repo") if isinstance(tm.get("dataset"), dict) else None
    if not ds: status["model"]["REFUSED: stamp names no dataset repo"] += 1; continue
    s, how = mix_subject_for(ds)
    if not s: status["model"][f"REFUSED via its mixture: {how[9:60]}"] += 1; continue
    tc = tm.get("train_config"); seed = tc.get("seed") if isinstance(tc, dict) else tm.get("seed")
    if seed is None:
        rd = dl(repo, "README.md", "model"); mm = re.search(r'"seed":\s*(\d+)', Path(rd).read_text()) if rd else None
        seed = int(mm.group(1)) if mm else None
    if seed is None: status["model"]["REFUSED: no seed in stamp or card"] += 1; continue
    try: base = model_key(jl(dl(repo, "adapter_config.json", "model")).get("base_model_name_or_path") or tm.get("base_model") or "")
    except ValueError: status["model"]["REFUSED: base model not in MODEL_KEYS"] += 1; continue
    status["model"][f"derived (mix by {how})"] += 1

print("\nkinds:", dict(kinds))
for k, c in status.items():
    print(f"\n{k}:"); [print(f"  {n:3}  {s}") for s, n in c.most_common()]
