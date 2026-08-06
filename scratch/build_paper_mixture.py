# ABOUTME: Builds the 10,000-sample instruction-tuning mixture of Table 2 (No Robots, Tulu3 IF,
# ABOUTME: NuminaMath, Self-Oss-Instruct, Smol-*, APIGen, LIMA, LongAlign) at its exact counts.

"""Reproduce the paper's instruction-tuning mixture.

Sources span three schemas -- `messages`, LIMA's vicuna `conversations`, and
Self-Oss-Instruct's `instruction`/`response` -- so each is normalised to a chat
conversation before rendering with the Qwen chat template.

Rows are rendered with NO think block; `src/data/add_empty_think_multi.py` adds the empty
`<think></think>` marker afterwards for whichever sources should carry it, and training
masks those marker tokens out of the loss.

    uv run python scratch/build_paper_mixture.py --config configs/data/mixture_paper_table2.yaml
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import fire
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mixture.build_mixture import _render_without_think, _usable  # noqa: E402
from src.utils import timestamp, write_run_meta  # noqa: E402

_VICUNA_ROLE = {"human": "user", "gpt": "assistant", "system": "system"}


def _to_messages(row: dict) -> list[dict]:
    """Normalise one dataset row to a chat conversation, whatever schema it uses.

    Args:
        row: A raw dataset row.

    Returns:
        Messages with `role`/`content`, or [] when the row carries no usable conversation.
    """
    if row.get("messages"):
        return row["messages"]
    if row.get("conversations"):  # LIMA, vicuna format
        return [{"role": _VICUNA_ROLE.get(c.get("from"), c.get("from")),
                 "content": c.get("value", "")} for c in row["conversations"]]
    if row.get("instruction") and row.get("response"):  # Self-Oss-Instruct
        return [{"role": "user", "content": row["instruction"]},
                {"role": "assistant", "content": row["response"]}]
    return []


def _take(tok, spec: dict, name: str, n_want: int, max_len: int, seed: int,
          shuffle_buffer: int) -> tuple[list[dict], dict]:
    """Stream one source until `n_want` renderable rows fit within `max_len`.

    Args:
        tok: Tokenizer.
        spec: Source spec with `repo`, optional `config`/`split`.
        name: Label recorded on each row.
        n_want: Exact number of rows to take.
        max_len: Drop conversations longer than this rather than truncating mid-answer.
        seed: Shuffle seed.
        shuffle_buffer: Streaming shuffle buffer; kept modest because a large buffer over
            long rows (LongAlign) exhausts memory and the process is OOM-killed.

    Returns:
        (rows, stats). Falls short of `n_want` only when the source runs dry, which the
        caller reports rather than silently accepting.
    """
    args = [spec["repo"]] + ([spec["config"]] if spec.get("config") else [])
    ds = load_dataset(*args, split=spec.get("split", "train"),
                      streaming=True).shuffle(seed=seed, buffer_size=shuffle_buffer)
    out, seen, too_long, unusable = [], 0, 0, 0
    for row in ds:
        msgs = _to_messages(row)
        if not _usable(msgs):
            unusable += 1
            continue
        try:
            text = _render_without_think(tok, msgs)
        except (AssertionError, ValueError):
            unusable += 1
            continue
        seen += 1
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        if n > max_len:
            too_long += 1
            continue
        out.append({"text": text, "source": name, "n_tokens": n})
        if len(out) >= n_want:
            break
    stats = {"requested": n_want, "taken": len(out), "scanned": seen,
             "dropped_too_long": too_long, "unusable": unusable,
             "tokens": sum(r["n_tokens"] for r in out)}
    flag = "" if len(out) == n_want else f"  <-- SHORT by {n_want - len(out)}"
    print(f"  {name:<26}{len(out):>6}/{n_want:<6} tok {stats['tokens']:>9,}  "
          f"dropped>{max_len}: {too_long}{flag}")
    return out, stats


def main(config: str) -> None:
    """Build the mixture named in a config and write it with per-source statistics.

    Args:
        config: OmegaConf YAML with `sources: {name: {repo, config?, split?, samples}}`.
    """
    cfg = OmegaConf.load(config)
    seed = int(cfg.seed)
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
    max_len = int(cfg.max_seq_len)

    print(f">>> building mixture, max_seq_len {max_len}")
    rows: list[dict] = []
    stats: dict[str, dict] = {}
    for name, spec in OmegaConf.to_container(cfg.sources, resolve=True).items():
        got, st = _take(tok, spec, name, int(spec["samples"]), max_len, seed,
                        int(cfg.get("shuffle_buffer", 1000)))
        rows += got
        stats[name] = st
    random.Random(seed).shuffle(rows)

    out_dir = Path(cfg.output_dir) / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mixture.jsonl"
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["text"], "source": r["source"]},
                               ensure_ascii=False) + "\n")

    assert len(list(out_path.open())) == len(rows), "mixture file is truncated"
    grand = sum(r["n_tokens"] for r in rows)
    summary = {"total": {"examples": len(rows), "tokens": grand},
               "by_source": stats, "mixture_path": str(out_path),
               "max_seq_len": max_len}
    (out_dir / "mixture_stats.json").write_text(json.dumps(summary, indent=2))
    write_run_meta(out_dir, OmegaConf.to_container(cfg, resolve=True),
                   extra={"stats": summary, "command": " ".join(sys.argv)})

    short = {k: v for k, v in stats.items() if v["taken"] != v["requested"]}
    print(f"\n>>> {len(rows):,} examples, {grand:,} tokens -> {out_path}")
    if short:
        print(">>> SHORT SOURCES (a source ran dry or its rows exceed max_seq_len):")
        for k, v in short.items():
            print(f"      {k}: {v['taken']}/{v['requested']}, "
                  f"{v['dropped_too_long']} dropped for length")
    with_think = sum(1 for r in rows if "<think>" in r["text"])
    print(f">>> rows containing any <think>: {with_think} (MUST be 0)")
    assert with_think == 0, "a think block would train the model to stop reasoning"

    # Breaking out of a streaming dataset mid-shard leaves HF's reader threads to crash
    # during interpreter finalization; the artifact above is written and verified.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    fire.Fire(main)
