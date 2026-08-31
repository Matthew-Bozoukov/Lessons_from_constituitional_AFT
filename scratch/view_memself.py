# ABOUTME: Split the memself training mixture into readable per-turn rows and build a
# ABOUTME: self-contained HTML browser, so the rendered chat text can be read as a conversation.

"""Make the memself mixture readable.

`mixture.jsonl` stores each example as one fully-rendered Qwen3.6 chat string, which is
unreadable as-is. This splits every mem_self row into its turns (system, the original ask,
the first reply, the pushback, and the supervised final reply with its reasoning trace),
writes them as JSONL for grepping, and renders a sample as browsable HTML.

Only the FINAL assistant turn is supervised (`supervise: "final"`), so that turn plus its
think trace is what the model actually learns; the rest is context.

Run:
  uv run python scratch/view_memself.py --mixture data/memself/mixture/mixture.jsonl
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TURN_RE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.S)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


def split_turns(text: str) -> list[dict]:
    """Split one rendered chat string into role/content/think turns.

    Args:
        text: Fully rendered Qwen3.6 chat text.

    Returns:
        One dict per turn: role, content (think block stripped), think (may be empty).
    """
    out = []
    for role, body in TURN_RE.findall(text):
        think = THINK_RE.search(body)
        out.append({
            "role": role,
            "think": think.group(1).strip() if think else "",
            "content": THINK_RE.sub("", body).strip(),
        })
    assert out, "no turns parsed; the render format may have changed"
    return out


def main(mixture: str, out_dir: str = "data/memself", samples: int = 40) -> None:
    """Write parsed mem_self rows plus an HTML browser.

    Args:
        mixture: Path to mixture.jsonl.
        out_dir: Where to write mem_self_parsed.jsonl and memself_viewer.html.
        samples: How many examples to render into the HTML.
    """
    rows = [json.loads(x) for x in Path(mixture).read_text().splitlines() if x.strip()]
    ms = [r for r in rows if r["source"] == "mem_self"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    parsed = [{"index": i, "supervise": r.get("supervise"), "turns": split_turns(r["text"])}
              for i, r in enumerate(ms)]
    with (out / "mem_self_parsed.jsonl").open("w") as fh:
        for p in parsed:
            fh.write(json.dumps(p) + "\n")

    def esc(s: str) -> str:
        return html.escape(s)

    cards = []
    for p in parsed[:samples]:
        blocks = []
        for j, t in enumerate(p["turns"]):
            last = j == len(p["turns"]) - 1
            think = (f"<div class=think><b>reasoning trace</b> "
                     f"({len(t['think'].split())} words)<pre>{esc(t['think'])}</pre></div>"
                     if t["think"] else
                     ("<div class=empty>empty think block (masked, not trained)</div>"
                      if t["role"] == "assistant" else ""))
            blocks.append(
                f"<div class='turn {t['role']}{' supervised' if last else ''}'>"
                f"<div class=role>{t['role']}{' — SUPERVISED (only this turn trains)' if last else ''}</div>"
                f"{think}<pre>{esc(t['content'])}</pre></div>")
        cards.append(f"<details><summary>#{p['index']} — "
                     f"{esc(p['turns'][1]['content'][:110])}…</summary>{''.join(blocks)}</details>")

    (out / "memself_viewer.html").write_text(f"""<!doctype html><meta charset=utf-8>
<title>memself training data</title><style>
body{{font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;margin:0 auto;max-width:980px;
padding:26px;background:#fafafa;color:#1c1c1e}}
details{{background:#fff;border:1px solid #e3e3e6;border-radius:10px;margin:10px 0;padding:10px 14px}}
summary{{cursor:pointer;font-weight:600}}
.turn{{border-left:4px solid #ccc;padding:6px 12px;margin:12px 0}}
.system{{border-color:#999}} .user{{border-color:#4C72B0}} .assistant{{border-color:#55A868}}
.supervised{{border-color:#DD8452;background:#fff8f3}}
.role{{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#666;margin-bottom:4px}}
pre{{white-space:pre-wrap;font:13px/1.55 ui-monospace,Menlo,monospace;margin:4px 0}}
.think{{background:#f2f4f8;border-radius:8px;padding:8px 10px;margin:6px 0;font-size:13px}}
.empty{{color:#999;font-size:12px;font-style:italic;margin:4px 0}}
</style>
<h1>memself training data</h1>
<p>{len(ms)} mem_self rows from <code>{esc(mixture)}</code> (the 20% side of the
table2-80 / memself-20 mixture). Showing the first {min(samples, len(parsed))}.
Every row is: system → the user's hard ask → a first reply → a pushback →
the <b>supervised</b> final reply carrying the real reasoning trace.</p>
{''.join(cards)}
""")
    tw = [len(p["turns"][-1]["think"].split()) for p in parsed]
    print(f"mem_self rows: {len(ms)}")
    print(f"final-turn reasoning traces: mean {sum(tw) / len(tw):.0f} words, "
          f"min {min(tw)}, max {max(tw)}")
    print(f"wrote {out}/mem_self_parsed.jsonl and {out}/memself_viewer.html")


if __name__ == "__main__":
    fire.Fire(main)
