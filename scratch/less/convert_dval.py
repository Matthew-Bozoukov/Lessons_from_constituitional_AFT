# ABOUTME: Converts the exported t2synth top-20 conversation dumps into the repo's
# ABOUTME: interchange format, one LESS validation subtask per source document.

"""Build the LESS validation set (Dval) from exported conversation dumps.

    uv run python scratch/less/convert_dval.py \
        --src "C:/Users/nikak/Downloads/t2synth_codebase_resisted_top20_conversations.jsonl" \
        --src "C:/Users/nikak/Downloads/t2synth_honest_declined_top20_conversations.jsonl" \
        --out data/less

One source FILE is one LESS subtask (the m in `max_j I[x, j]`), so the subtask name is
derived from the filename rather than from any per-row field -- the exports disagree on
their id column (`scenario_id` vs `prompt_id`) and two of the three carry no grouping
field at all.

The one transformation that matters scientifically: these dumps carry the assistant's
chain-of-thought in a TOP-LEVEL `reasoning` key, while the repo's interchange format (and
dataset D) put it in `reasoning_content` ON the assistant message. Rendered as-is, the
assistant turn would get an empty `<think>\\n\\n</think>\\n\\n` marker, which the
generation-boundary rule masks WHOLLY (src/train/masking.py) -- we would silently compute
validation gradients for a non-thinking response and then select training data to match
it. Nothing errors; the answer is just wrong. Hence the remap, and hence the assertions.

Reads and writes are pinned to UTF-8: these files contain em-dashes and greek letters, and
`Path.read_text`/`write_text` default to the locale codec (cp1252 on Windows).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

# Provenance fields worth carrying through when a source has them. The exports are not
# schema-consistent, so every one of these is optional and absence is not an error.
CARRY = ("scenario_id", "prompt_id", "sample", "arm", "difficulty",
         "judge_score", "judge_rationale")

# Filenames look like `t2synth_codebase_resisted_top20_conversations.jsonl`; the subtask
# is the middle -- drop the arm prefix and the export suffix.
_TRIM = re.compile(r"^(?P<arm>[a-z0-9]+)_(?P<name>.+?)_top\d+_conversations$")


def subtask_name(path: Path) -> str:
    """Derive a subtask id from an export filename.

    A browser re-download suffix (`... (1).jsonl`) is stripped so a duplicated file
    cannot masquerade as a distinct subtask.
    """
    stem = re.sub(r"\s*\(\d+\)$", "", path.stem)
    m = _TRIM.match(stem)
    return m.group("name") if m else stem


def convert_row(raw: dict, subtask: str, source: str, index: int) -> dict:
    """Map one exported conversation onto an interchange row.

    Raises:
        AssertionError: the row's shape would make its gradient meaningless -- no
            assistant turn to supervise, reasoning already inlined as a `<think>` block
            (which the remap would then duplicate), or an empty trace (which would render
            as an empty marker and be wholly masked).
    """
    msgs = raw["messages"]
    assert msgs and msgs[-1]["role"] == "assistant", (
        f"{source}[{index}]: last message must be the assistant turn we take gradients "
        f"on, got roles {[m['role'] for m in msgs]}")

    reasoning = (raw.get("reasoning") or "").strip()
    assert reasoning, (
        f"{source}[{index}]: empty `reasoning` -- this row would render an empty think "
        f"marker, which the generation-boundary rule masks entirely, leaving the "
        f"validation gradient blind to the reasoning we are selecting for")

    out_msgs = []
    for m in msgs:
        content = m.get("content") or ""
        assert "<think>" not in content, (
            f"{source}[{index}]: reasoning is already inlined as a <think> block; the "
            f"remap would duplicate it")
        out_msgs.append({"role": m["role"], "content": content})
    out_msgs[-1]["reasoning_content"] = reasoning

    # `less_id` is the join key every later stage uses to tie a gradient row back to its
    # source, so it must be stamped here as well as in prepare_data.py -- the pool and the
    # control get theirs there, and a validation row without one crashes feature extraction.
    # Keyed on subtask, because the exports disagree on their own id column and one
    # (stayed_ai) has none at all.
    meta = {"less_id": f"{subtask}#{index}", "subtask": subtask,
            "source_file": source, "source_index": index}
    meta.update({k: raw[k] for k in CARRY if raw.get(k) is not None})
    return {"messages": out_msgs, "metadata": meta}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", action="append", required=True, type=Path,
                    help="an exported conversation dump; repeat once per subtask")
    ap.add_argument("--out", required=True, type=Path,
                    help="output directory (a gitignored data/ path)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    seen: dict[str, Path] = {}
    manifest: list[dict] = []

    for src in args.src:
        digest = hashlib.md5(src.read_bytes()).hexdigest()
        if digest in seen:
            raise SystemExit(
                f"{src.name} is byte-identical to {seen[digest].name} (md5 {digest[:8]}) "
                f"-- one document is one subtask, so a duplicate would double-weight it")
        seen[digest] = src

        subtask = subtask_name(src)
        text = src.read_text(encoding="utf-8")
        raws = [json.loads(line) for line in text.splitlines() if line.strip()]
        converted = [convert_row(r, subtask, src.name, i) for i, r in enumerate(raws)]
        rows += converted

        chars = [sum(len(m["content"]) for m in r["messages"])
                 + len(r["messages"][-1]["reasoning_content"]) for r in converted]
        manifest.append({
            "subtask": subtask, "source_file": src.name, "md5": digest,
            "n_examples": len(converted),
            "chars_median": sorted(chars)[len(chars) // 2], "chars_max": max(chars),
            "id_field": next((k for k in ("scenario_id", "prompt_id")
                              if any(k in r["metadata"] for r in converted)), None),
        })
        print(f">>> {subtask:<24} {len(converted):>3} examples  "
              f"median {sorted(chars)[len(chars) // 2]:>7,} chars  "
              f"max {max(chars):>7,}  ({src.name})")

    dval = args.out / "dval.jsonl"
    dval.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
    (args.out / "dval_manifest.json").write_text(
        json.dumps({"subtasks": manifest, "n_examples": len(rows),
                    "m": len(manifest)}, indent=2), encoding="utf-8")

    print(f"\n>>> wrote {len(rows)} examples across m={len(manifest)} subtasks -> {dval}")
    if len(manifest) < 3:
        print(f">>> NOTE: m={len(manifest)}; a third subtask document was expected. "
              f"Re-run with an extra --src to add it.")


if __name__ == "__main__":
    main()
