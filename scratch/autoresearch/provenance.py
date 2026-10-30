# ABOUTME: Pinned provenance for every Qwen3.6 DA eval run: the adapter REVISION the eval served, the mix that
# ABOUTME: revision trained on and that mix's DA source corpus (repo names are reused, so heads cannot be trusted).
"""    uv run python scratch/autoresearch/provenance.py   -> output/autoresearch/provenance.json

An adapter repo name is `<date>-qwen36-<seed>-<subject>` and a same-day push of the same subject overwrites it
(docs/GOTCHAS.md 2026-09-22). 2026-09-28-qwen36-0-da-{5,25} each hold TWO different adapters: a morning one trained
on the 25 Sep corpus and an evening one trained on the 28 Sep corpus. So every score here is attributed through the
eval's own `metadata/run_meta.json: target_revision`, never through the adapter repo's head.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

load_dotenv(".env")
ORG = "dougalldeepmind"
api = HfApi()


def j(repo, f, kind="dataset", rev=None):
    try:
        return json.load(open(hf_hub_download(repo, f, repo_type=kind, revision=rev)))
    except Exception:
        return None


_cache = {}


def trained_on(adapter, rev):
    if (adapter, rev) in _cache:
        return _cache[(adapter, rev)]
    out = {"adapter": adapter.split("/")[-1], "adapter_revision": (rev or "")[:8]}
    tm = j(adapter, "training_meta.json", "model", rev)
    if tm:
        d = tm.get("dataset") or {}
        if not isinstance(d, dict) or not d.get("repo"):   # older adapters record the data another way
            tc = tm.get("train_config") if isinstance(tm.get("train_config"), dict) else {}
            d = {"repo": tc.get("data_repo") or tm.get("data_repo") or "", "revision": tc.get("data_revision") or tm.get("data_revision")}
        if not d.get("repo"):
            _cache[(adapter, rev)] = out
            return out
        tc = tm.get("train_config") if isinstance(tm.get("train_config"), dict) else {}
        out.update(mix=d["repo"].split("/")[-1], mix_revision=str(d.get("revision"))[:8], seed=tc.get("seed"))
        try:
            cfg = yaml.safe_load(open(hf_hub_download(d["repo"], "mixture_config.yaml", repo_type="dataset", revision=d["revision"])))
            out["synthetic_pct"] = cfg.get("synthetic_pct"); out["share_unit"] = cfg.get("share_unit", "rows")
            out["sources"] = {k: f"{str(v.get('dataset') or v.get('repo') or v.get('path')).split('/')[-1]}@{str(v.get('revision'))[:8]}"
                              for k, v in (cfg.get("sources") or {}).items()
                              if isinstance(v, dict) and (v.get("dataset") or v.get("repo") or v.get("path")) and re.match(r"da|delib|nonmoral", k)}
        except Exception as exc:
            out["sources"] = {"?": str(exc)[:80]}
    _cache[(adapter, rev)] = out
    return out


runs = []
for d in api.list_datasets(author=ORG, search="qwen36", limit=4000):
    m = re.match(rf"{ORG}/(\d{{4}}-\d\d-\d\d)-(odcv|mask)-(qwen36-\d-.*)$", d.id)
    if not m:
        continue
    res = j(d.id, "results/results.json"); meta = j(d.id, "metadata/run_meta.json")
    if not res or not meta:
        continue
    date, ev, arm = m.groups()
    score = ((res.get("ours") or {}).get("overall") or {}).get("mr_pct") if ev == "odcv" else res.get("overall_honesty_score")
    n = ((res.get("ours") or {}).get("overall") or {}).get("n_rollouts") if ev == "odcv" else res.get("n_rows")
    runs.append({"run": d.id.split("/")[1], "eval": ev, "date": date, "score": score, "n": n, "mode": res.get("mode"),
                 **trained_on(meta.get("target"), meta.get("target_revision"))})
runs.sort(key=lambda r: (r.get("mix") or "", r["adapter"], r["eval"], r["date"]))
Path("output/autoresearch/provenance.json").write_text(json.dumps(runs, indent=1))
for r in runs:
    print(f"{r['run']:52} {r['eval']:4} {str(r['score']):>5} n={r['n']} | {r['adapter']}@{r['adapter_revision']} | {r.get('mix')}@{r.get('mix_revision')} pct={r.get('synthetic_pct')} {r.get('share_unit')} | {r.get('sources')}")
