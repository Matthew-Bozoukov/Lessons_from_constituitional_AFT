# ABOUTME: Inventory every published Qwen3.6 ODCV / MASK run since 2026-09-20 with its score, the adapter's
# ABOUTME: training mix and that mix's DA source, as one table (output/autoresearch/inventory.{json,md}).
"""    uv run python scratch/autoresearch/inventory.py"""
from __future__ import annotations

import json
import re
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

load_dotenv(".env")
ORG = "dougalldeepmind"
api = HfApi()


def j(repo, f, kind="dataset"):
    try:
        return json.load(open(hf_hub_download(repo, f, repo_type=kind)))
    except Exception:
        return None


runs = {}
for d in api.list_datasets(author=ORG, search="qwen36", limit=2000):
    m = re.match(rf"{ORG}/(\d{{4}}-\d\d-\d\d)-(odcv|mask)-(qwen36-.*)$", d.id)
    if not m or m.group(1) < "2026-09-20":
        continue
    date, ev, arm = m.groups()
    r = j(d.id, "results/results.json")
    if not r:
        continue
    e = runs.setdefault(arm, {"arm": arm, "odcv": [], "mask": []})
    if ev == "odcv":
        o = (r.get("ours") or {}).get("overall") or {}
        e["odcv"].append({"run": d.id.split("/")[1], "mr": o.get("mr_pct"), "n": o.get("n_rollouts"), "sev": o.get("severity_mean", o.get("sev")),
                          "mode": r.get("mode"), "submit": ((r.get("submission") or {}).get("overall") or {}).get("submitted_pct"),
                          "target": r.get("target")})
    else:
        e["mask"].append({"run": d.id.split("/")[1], "honesty": r.get("overall_honesty_score"), "n": r.get("n_rows"), "mode": r.get("mode"),
                          "target": r.get("target")})
_mix = {}


def mix_of(tgt):
    """adapter -> (mix@rev, synthetic pct, {source: corpus@rev} for the synthetic sources)."""
    if tgt in _mix or not tgt or ":" in tgt:
        return _mix.get(tgt)
    tm = j(tgt, "training_meta.json", "model") or {}
    ds = tm.get("dataset") or {}
    info = {"adapter": tgt, "mix": f"{ds.get('repo')}@{str(ds.get('revision'))[:8]}"}
    if ds.get("repo"):
        st = j(ds["repo"], "mixture_stats.json") or {}
        info["synthetic_pct"] = st.get("synthetic_pct")
        try:
            import yaml
            cfg = yaml.safe_load(open(hf_hub_download(ds["repo"], "mixture_config.yaml", repo_type="dataset", revision=ds.get("revision"))))
            info["sources"] = {k: (f"{str(v.get('dataset') or v.get('repo') or v.get('path')).split('/')[-1]}@{str(v.get('revision'))[:8]}")
                               for k, v in (cfg.get("sources") or {}).items() if isinstance(v, dict) and (v.get("dataset") or v.get("repo") or v.get("path"))
                               and k.split("-")[0] in ("da", "dat", "delib", "nonmoral", "daa")}
        except Exception as exc:
            info["sources"] = {"?": str(exc)[:60]}
    _mix[tgt] = info
    return info


rows = []
for arm, e in runs.items():
    by = {}
    for x in e["odcv"]:
        by.setdefault(x["target"], {"odcv": [], "mask": []})["odcv"].append(x)
    for x in e["mask"]:
        by.setdefault(x["target"], {"odcv": [], "mask": []})["mask"].append(x)
    for tgt, v in by.items():
        rows.append({"adapter": str(tgt).split("/")[-1], **(mix_of(tgt) or {}), **v})
rows.sort(key=lambda r: r["adapter"])
out = Path("output/autoresearch"); out.mkdir(parents=True, exist_ok=True)
(out / "inventory.json").write_text(json.dumps(rows, indent=1))
lines = ["| adapter | MASK | ODCV MR % (runs) | synth % | synthetic source corpus |", "|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['adapter']} | " + " / ".join(str(x["honesty"]) for x in r["mask"]) + " | "
                 + " / ".join(f"{x['mr']}" for x in r["odcv"]) + f" | {r.get('synthetic_pct', '')} | "
                 + "; ".join(f"{k}: {v}" for k, v in (r.get("sources") or {}).items()) + " |")
(out / "inventory.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
