# ABOUTME: The MSM Table 2 base blend with its on-policy Qwen reasoning traces removed: the published
# ABOUTME: 2026-09-29-nosynth-mix rows, byte-identical except that no assistant turn carries reasoning_content.
# Run: uv run python scratch/nosynth_no_traces/build.py   (local only: writes output/, never pushes)
"""Tests whether the base blend ever needed on-policy traces, now that history turns earn no loss.

`dougalldeepmind/2026-09-29-nosynth-mix` carries Qwen3.6-27B traces on 1,130 of its 10,000 rows (the
reasoning backfill of 2026-09-08). This drops them and changes nothing else -- same rows, same order, same
answers, same constitution-filter verdicts, same native tool rows -- so the only difference between an arm
trained on this and one trained on the published mix is whether those traces were there.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from src.data.mixture.build_mixture import supervised_tokens
from src.infra.huggingface import resolve_dataset
from src.model_profile import model_profile, render_chat
from src.train.mask_gate import gate_generation_boundary
from src.utils import timestamp, write_run_meta

REPO, REVISION, FILE = "dougalldeepmind/2026-09-29-nosynth-mix", "058e163e7d02bfa01503506653e9e360e2e5798c", "mixture.jsonl"
QWEN, MAX_SEQ_LEN = "Qwen/Qwen3.6-27B", 8192


def main() -> None:
    path, ref = resolve_dataset(REPO, FILE, REVISION)
    rows = [json.loads(line) for line in Path(path).open(encoding="utf-8")]
    removed = Counter()
    for r in rows:
        for m in r["messages"]:
            if m.pop("reasoning_content", None):
                removed[r["source"]] += 1
    tok, profile = AutoTokenizer.from_pretrained(QWEN), model_profile(QWEN)
    texts = [render_chat(tok, r["messages"], r.get("tools"), render_kwargs=profile.render_kwargs) for r in rows]
    census = gate_generation_boundary(texts, tok, MAX_SEQ_LEN, profile, thinking=True,
                                      supervise=[r.get("supervise") for r in rows])
    by_source: dict[str, Counter] = {}
    for r, text in zip(rows, texts):
        s = by_source.setdefault(r["source"], Counter())
        s["examples"] += 1
        s["tokens"] += len(tok(text, add_special_tokens=False).input_ids)
        s["supervised_tokens"] += supervised_tokens(tok, profile, r, MAX_SEQ_LEN)
    total = sum(by_source.values(), Counter())
    out_dir = Path("output/mixture_base") / f"{timestamp()}_nosynth_no_traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "mixture.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"total": dict(total), "by_source": {k: dict(v) for k, v in by_source.items()},
              "trace_turns_removed": dict(removed), "census": census, "source": ref,
              "max_seq_len": MAX_SEQ_LEN, "tokenizer": QWEN, "qwen_render_kwargs": profile.render_kwargs}
    (out_dir / "mixture_stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_run_meta(out_dir, {"source": ref}, extra={"command": " ".join(sys.argv), "stats": report})
    print(json.dumps(report, indent=2))
    print(f">>> wrote {out_dir / 'mixture.jsonl'}")


if __name__ == "__main__":
    main()
