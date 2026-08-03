# ABOUTME: Adds Qwen3.6's empty <think></think> non-thinking marker to chosen sources of a
# ABOUTME: mixture, exactly where apply_chat_template would place it.

from __future__ import annotations

import collections
import json
from pathlib import Path

import fire
from transformers import AutoTokenizer

ASSISTANT_HEADER = "<|im_start|>assistant\n"
EMPTY_THINK = "<think>\n\n</think>\n\n"


def _add_marker(text: str) -> str:
    """Insert the empty think marker on the FINAL assistant turn.

    Qwen3.6's template renders a think block only for the final assistant turn and omits it
    on historical ones, so inserting only on the last turn reproduces exactly what
    `apply_chat_template` emits for the same conversation.

    Args:
        text: A conversation rendered with no think block anywhere.

    Returns:
        The same text with the marker after the last assistant header.
    """
    i = text.rindex(ASSISTANT_HEADER) + len(ASSISTANT_HEADER)
    assert not text.startswith(EMPTY_THINK, i), "marker already present"
    return text[:i] + EMPTY_THINK + text[i:]


def main(src: str, out: str, sources, tokenizer: str = "Qwen/Qwen3.6-27B") -> None:
    """Rewrite the named sources of a mixture to carry the empty think marker.

    Args:
        src: Existing mixture.jsonl (fields: text, source).
        out: Destination jsonl.
        sources: Source names to mark; Fire passes a tuple when commas are used.
        tokenizer: Tokenizer used to verify the surgery matches the chat template.
    """
    names = set(sources) if isinstance(sources, (list, tuple)) else \
        {s.strip() for s in str(sources).split(",")}
    tok = AutoTokenizer.from_pretrained(tokenizer)

    # Prove the surgery equals the template's own output before touching real data.
    probe = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    plain = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=False)
    padded = tok.apply_chat_template(probe + [{"role": "user", "content": "X"}],
                                     tokenize=False, add_generation_prompt=False)
    no_think = padded[: padded.rindex("<|im_start|>user")]
    assert _add_marker(no_think) == plain, "surgery does not reproduce the chat template"
    print(f">>> verified: inserting {EMPTY_THINK!r} reproduces apply_chat_template exactly")

    rows = [json.loads(line) for line in Path(src).open()]
    changed = untouched = 0
    for r in rows:
        if r["source"] in names:
            assert "<think>" not in r["text"], f"{r['source']} row already has a think block"
            r["text"] = _add_marker(r["text"])
            changed += 1
        else:
            untouched += 1

    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["text"], "source": r["source"]},
                               ensure_ascii=False) + "\n")

    written = [json.loads(line) for line in dest.open()]
    assert len(written) == len(rows), "output is truncated"
    marked = collections.Counter(w["source"] for w in written if EMPTY_THINK in w["text"])
    per_row = {w["text"].count(EMPTY_THINK) for w in written if w["source"] in names}
    print(json.dumps({"rows": len(written), "marked": changed, "untouched": untouched,
                      "marked_by_source": dict(marked), "markers_per_marked_row": sorted(per_row)},
                     indent=2))
    assert sum(marked.values()) == changed, "not every selected row received the marker"
    assert per_row == {1}, f"expected exactly one marker per row, got {per_row}"
    assert all(EMPTY_THINK not in w["text"] for w in written if w["source"] not in names), \
        "a marker leaked into an unselected source"
    print(f">>> wrote {dest}")


if __name__ == "__main__":
    fire.Fire(main)
