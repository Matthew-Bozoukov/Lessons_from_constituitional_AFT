# ABOUTME: Share of rows and supervised tokens carrying tool definitions in the da-tools-15 mixes,
# ABOUTME: DA rows (unused tools) against the base's own agentic rows. Run: uv run python scratch/tools_share/da_tools_share.py
"""Counts the way the mixture builder counts (`supervised_tokens`: Qwen3.6 render + build_labels at 8,192).
A row "has tools" when its `tools` list is non-empty; "called" when any assistant turn has tool_calls."""
import collections
import json
from pathlib import Path

from transformers import AutoTokenizer

from src.data.mixture.build_mixture import supervised_tokens
from src.infra.huggingface import hf_download
from src.model_profile import model_profile

MIXES = {
    "a": ("dougalldeepmind/2026-10-11-da-tools-15-mix", "da195751376fea70439730adda75a516aa5bc688"),
    "b": ("dougalldeepmind/2026-11-06-da-tools-15-b-mix", "1e9bbde6765187647b5fd61f44eb9d651e05055a"),
    "c": ("dougalldeepmind/2026-11-06-da-tools-15-c-mix", "a943ed3381478e637e08ccc5c029b83c9c159377"),
}
QWEN = "Qwen/Qwen3.6-27B"
MAX_SEQ_LEN = 8192


def load(path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def status(row: dict) -> str:
    if any(m.get("tool_calls") for m in row["messages"]):
        return "called"
    return "unused" if row.get("tools") else "none"


def main() -> None:
    profile = model_profile(QWEN)
    tok = AutoTokenizer.from_pretrained(QWEN, revision=profile.revision) if getattr(profile, "revision", None) \
        else AutoTokenizer.from_pretrained(QWEN)
    out = {}
    for arm, (repo, rev) in MIXES.items():
        rows = load(hf_download(repo, "mixture.jsonl", repo_type="dataset", revision=rev))
        cells = collections.defaultdict(lambda: [0, 0])  # (source, status) -> [rows, supervised tokens]
        for r in rows:
            n = supervised_tokens(tok, profile, r, MAX_SEQ_LEN)
            c = cells[(r["source"], status(r))]
            c[0] += 1
            c[1] += n
        out[arm] = {"repo": repo, "revision": rev,
                    "cells": {f"{s}|{st}": {"rows": v[0], "supervised_tokens": v[1]} for (s, st), v in sorted(cells.items())}}
        print(arm, repo, rev[:8])
        for k, v in out[arm]["cells"].items():
            print(f"  {k:40s} {v['rows']:6d} {v['supervised_tokens']:10d}")
    dst = Path("output/tools_share")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "2026-11-18_da_tools_15_tool_share.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
