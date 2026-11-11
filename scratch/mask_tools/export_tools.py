# ABOUTME: Turns a mask-tools synth corpus (local dataset.jsonl or a pinned HF repo) into the
# ABOUTME: {task_id: [tools]} JSON the MASK eval's inject_tools switch reads, and reports counts.
"""Run: uv run python scratch/mask_tools/export_tools.py <dataset.jsonl | org/repo@revision> --out FILE

Every one of MASK's 1,000 task_ids must be in the corpus (the eval refuses a missing row). A row
whose tools never passed the judges (`tools_status: exhausted`) keeps its empty list, so the eval
sends it without tools; the count is printed.
"""
import argparse
import collections
import json
from pathlib import Path


def load(src: str) -> list[dict]:
    if Path(src).is_file():
        return [json.loads(line) for line in open(src, encoding="utf-8")]
    from huggingface_hub import hf_hub_download
    repo, _, rev = src.partition("@")
    assert rev, "pin the corpus: org/repo@revision"
    path = hf_hub_download(repo, "dataset.jsonl", repo_type="dataset", revision=rev)
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = load(a.src)
    table = {r["metadata"]["scenario_id"]: r["tools"] or [] for r in rows}
    assert len(table) == len(rows), "duplicate task_id in the corpus"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(table, ensure_ascii=False, indent=0), encoding="utf-8")
    status = collections.Counter(r["metadata"]["tools_status"] for r in rows)
    n = collections.Counter(len(v) for v in table.values())
    names = collections.Counter(t["function"]["name"] for v in table.values() for t in v)
    print(f"{a.out}: {len(table)} rows | status {dict(status)} | tools per row {dict(sorted(n.items()))}")
    print(f"distinct tool names {len(names)}; most common: {names.most_common(8)}")


if __name__ == "__main__":
    main()
