# ABOUTME: Check the da-tools mixture against da-15's: same rows in the same order, the only change the
# ABOUTME: synthetic rows' `tools`; then run training's own loader, renderer and mask gate on it locally.
# Run: uv run python scratch/da_tools/compare_mixes.py <org/new-mix@rev> [--old org/da-15-mix@rev]
import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from src.infra.huggingface import hf_download
from src.model_profile import model_profile, render_chat
from src.train.mask_gate import gate_generation_boundary

OLD = "dougalldeepmind/2026-09-25-da-15-mix@73f66648dc1c1f4e12d887dca5780bb065ddd385"


def fetch(ref: str, name: str) -> str:
    repo, _, rev = ref.partition("@")
    return hf_download(repo, name, repo_type="dataset", **({"revision": rev} if rev else {}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("new")
    ap.add_argument("--old", default=OLD)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-gate", action="store_true")
    a = ap.parse_args()
    new_path, old_path = fetch(a.new, "mixture.jsonl"), fetch(a.old, "mixture.jsonl")
    new = [json.loads(line) for line in open(new_path)]
    old = [json.loads(line) for line in open(old_path)]
    rep: dict = {"new": a.new, "old": a.old, "rows_new": len(new), "rows_old": len(old)}
    rep["sources_new"] = dict(Counter(r.get("source") for r in new))
    rep["sources_old"] = dict(Counter(r.get("source") for r in old))
    same_msgs = sum(json.dumps(x["messages"], ensure_ascii=False) == json.dumps(y["messages"], ensure_ascii=False)
                    for x, y in zip(new, old))
    rep["same_messages_same_position"] = same_msgs
    rep["tools_rows_new"] = sum(bool(r.get("tools")) for r in new)
    rep["tools_rows_old"] = sum(bool(r.get("tools")) for r in old)
    rep["tools_rows_by_source_new"] = dict(Counter(r.get("source") for r in new if r.get("tools")))
    synth_new = [r for r in new if r.get("source") not in rep["sources_old"] or r.get("source") == "da-tools"]
    rep["synthetic_rows_new"] = len(synth_new)
    rep["synthetic_rows_without_tools"] = sum(not r.get("tools") for r in synth_new)
    rep["supervise_equal"] = sum(x.get("supervise") == y.get("supervise") for x, y in zip(new, old))
    for tag, ref in (("new", a.new), ("old", a.old)):
        try:
            stats = json.load(open(fetch(ref, "mixture_stats.json")))
            rep[f"stats_{tag}"] = {k: stats[k] for k in stats
                                   if any(s in k for s in ("share", "token", "rows", "examples"))
                                   and not isinstance(stats[k], (dict, list))}
        except Exception as e:  # stats are informative only
            rep[f"stats_{tag}"] = f"unavailable: {type(e).__name__}"

    # Training's own path: HF's json loader (it unions every row's nested tool schemas into
    # one Arrow type -- the step a heterogeneous tool set could break), then render_chat.
    ds = load_dataset("json", data_files=new_path, split="train")
    rep["load_dataset_rows"] = len(ds)
    prof = model_profile("qwen36")
    tok = AutoTokenizer.from_pretrained(prof.model)
    texts = [render_chat(tok, r["messages"], r.get("tools"), render_kwargs=prof.render_kwargs)
             for r in ds]
    direct = [render_chat(tok, r["messages"], r.get("tools"), render_kwargs=prof.render_kwargs)
              for r in new]
    rep["loader_render_equals_direct_render"] = sum(x == y for x, y in zip(texts, direct))
    rep["rendered_with_tools_block"] = sum("<tools>" in t for t in texts)
    if not a.no_gate:
        census = gate_generation_boundary(texts, tok, 8192, prof, bool(prof.thinking),
                                          supervise=ds["supervise"] if "supervise" in ds.column_names else None)
        rep["mask_gate"] = "passed"
        rep["census"] = {k: v for k, v in census.items() if not isinstance(v, (list, dict))}
    # A synthetic row without tools is one the corpus marked `tools_status: exhausted`
    # (reported, not a failure); tools anywhere OUTSIDE the synthetic rows would be.
    ok = (len(new) == len(old) and same_msgs == len(new)
          and set(rep["tools_rows_by_source_new"]) <= {"da-tools"} and rep["tools_rows_old"] == 0
          and rep["loader_render_equals_direct_render"] == len(new))
    rep["PASS"] = bool(ok)
    text = json.dumps(rep, indent=1, default=str)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
