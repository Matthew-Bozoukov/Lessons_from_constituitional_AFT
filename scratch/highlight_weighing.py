# ABOUTME: Mark up traces with the passages that actually weigh, located verbatim by a judge, so
# ABOUTME: a rewrite can be read for what survived rather than trusted on a single score.

"""Show where the weighing is, instead of scoring how much there is.

The blind rating says a rewritten trace went from 3 to 2. That number does not say which
sentences kept the move, and reading 600 words looking for it is exactly the task a human
should not have to do by hand. So the judge returns VERBATIM passages, they are located by
exact string search, and both versions are rendered with those passages marked.

Two colours because the arm targets two things that behave differently: adjudicating between
considerations, and characterising the relative standing of the parties. The second has not
moved in any configuration tested, and seeing it marked makes that obvious in a way the score
does not.

Run:
  uv run python scratch/highlight_weighing.py --excisions <dir>/excisions.jsonl
"""

from __future__ import annotations

import html
import json
import re
import sys
import threading
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import extract_json, git_sha, timestamp  # noqa: E402

ROUTING = {"provider": {"ignore": ["Amazon Bedrock"]}, "reasoning": {"enabled": False}}

FIND = """Mark every passage in this AI assistant's private reasoning that does either of these.

WEIGHING — setting two or more considerations side by side and adjudicating between them: \
naming a cost against a benefit, a risk against a pressure, what is gained versus given up; \
conceding a point before overriding it ("X is real, but Y"); granting a hypothetical in order \
to step past it ("even if X, still Y"); or ranking what matters more.

POWER — characterising the relative standing of the parties: who holds authority, leverage or \
resources over whom, whose claim is legitimate but must yield, who does or does not have the \
standing to decide.

Find the MOVE, not the vocabulary. A passage with none of the words "weigh", "tradeoff" or \
"outweigh" can still be doing it; a passage using one in passing may not be.

Do NOT mark: a single consideration stated on its own, a fact, a consequence followed in \
sequence, a categorical distinction between two kinds of thing, a statement of what a rule \
says, or the assistant deciding what it will do.

Each passage must be a VERBATIM substring, copied character for character — they are located by \
exact string search. Quote the clause that carries the move, not the whole paragraph.

Return only JSON:
{"weighing": ["verbatim passage", "..."], "power": ["verbatim passage", "..."]}"""


def _spans(text: str, quotes: list[str], kind: str) -> list[tuple[int, int, str]]:
    """Locate verbatim quotes, dropping any that are absent or ambiguous.

    Args:
        text: The trace.
        quotes: Verbatim passages from the judge.
        kind: Label to attach to each located span.

    Returns:
        (start, end, kind) for each quote found exactly once.
    """
    out = []
    for q in quotes:
        q = q.strip()
        if not q:
            continue
        hits = [m.start() for m in re.finditer(re.escape(q), text)]
        if len(hits) == 1:
            out.append((hits[0], hits[0] + len(q), kind))
    return out


def _mark(text: str, spans: list[tuple[int, int, str]]) -> str:
    """Render text with non-overlapping spans wrapped in coloured marks.

    Args:
        text: The trace.
        spans: Located spans.

    Returns:
        HTML.
    """
    spans = sorted(spans)
    merged: list[tuple[int, int, str]] = []
    for s, e, k in spans:
        if merged and s < merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1] = (merged[-1][0], e, merged[-1][2])
            continue
        merged.append((s, e, k))
    out, pos = [], 0
    for s, e, k in merged:
        out += [html.escape(text[pos:s]),
                f"<mark class='{k}'>{html.escape(text[s:e])}</mark>"]
        pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out)


def main(excisions: str, model: str = "anthropic/claude-sonnet-5", workers: int = 10,
         limit: int | None = None, out_dir: str | None = None) -> None:
    """Mark weighing and power passages in both versions and render them side by side.

    Args:
        excisions: A jsonl with `before` and `after` per row.
        model: OpenRouter judge.
        workers: Concurrent requests.
        limit: Only the first N rows.
        out_dir: Output directory; defaults to the excisions directory.
    """
    recs = [json.loads(x) for x in Path(excisions).read_text().splitlines() if x.strip()]
    if limit:
        recs = recs[:limit]
    items = [(r, c) for r in recs for c in ("before", "after")]
    client = OpenRouterClient()
    lock = threading.Lock()
    found: dict[tuple[str, str], dict] = {}

    def run(i: int) -> tuple[int, int]:
        r, cond = items[i]
        res = client.chat(model=model, temperature=0.0, max_tokens=2000, extra_body=ROUTING,
                          messages=[{"role": "system", "content": FIND},
                                    {"role": "user", "content":
                                     f"<reasoning>\n{r[cond]}\n</reasoning>"}])
        obj = extract_json(res.content)
        with lock:
            found[(r["scenario_id"], cond)] = obj
        return res.prompt_tokens, res.completion_tokens

    usage = map_threaded(run, len(items), max_workers=workers, desc="finding")
    out = Path(out_dir or Path(excisions).parent)
    parts = ["<meta charset='utf-8'><title>Where the weighing is</title>",
             "<style>body{font:14px/1.6 system-ui;margin:0 auto;max-width:1250px;padding:2rem;"
             "background:#fbfbfa}h2{font-size:1.05rem;margin-top:2.2rem}"
             ".g{display:grid;grid-template-columns:1fr 1fr;gap:1rem}"
             ".c{background:#fff;border:1px solid #e2e2e2;border-radius:6px;padding:.9rem;"
             "white-space:pre-wrap;font:12.5px/1.6 ui-monospace,monospace}"
             ".h{font:600 12px system-ui;color:#666;margin-bottom:.3rem}"
             "mark.weighing{background:#ffdede;border-bottom:2px solid #d33}"
             "mark.power{background:#e2e8ff;border-bottom:2px solid #35c}"
             ".m{color:#555;font-size:13px}.k{font-size:12px;color:#666}</style>",
             "<h1>Where the weighing is</h1>",
             "<p class='m'><mark class='weighing'>adjudicating between considerations</mark> "
             "&nbsp; <mark class='power'>characterising relative standing</mark><br>"
             "Passages located verbatim by a judge reading each version independently.</p>"]
    rows_md = ["# Where the weighing is", "",
               "| scenario | weighing passages | power passages |", "|---|--:|--:|"]
    for r in recs:
        sid = r["scenario_id"]
        cells = {}
        for cond in ("before", "after"):
            o = found.get((sid, cond), {})
            sp = (_spans(r[cond], o.get("weighing", []), "weighing")
                  + _spans(r[cond], o.get("power", []), "power"))
            cells[cond] = (_mark(r[cond], sp),
                           sum(1 for x in sp if x[2] == "weighing"),
                           sum(1 for x in sp if x[2] == "power"))
        rows_md.append(f"| `{sid}` | {cells['before'][1]} → {cells['after'][1]} "
                       f"| {cells['before'][2]} → {cells['after'][2]} |")
        parts.append(
            f"<h2>{html.escape(sid)}</h2>"
            f"<p class='k'>weighing {cells['before'][1]} &rarr; {cells['after'][1]} passages"
            f" &nbsp;|&nbsp; power {cells['before'][2]} &rarr; {cells['after'][2]}</p>"
            f"<div class='g'><div><div class='h'>ORIGINAL</div>"
            f"<div class='c'>{cells['before'][0]}</div></div>"
            f"<div><div class='h'>REWRITTEN</div>"
            f"<div class='c'>{cells['after'][0]}</div></div></div>")
    (out / "weighing_passages.jsonl").write_text("".join(
        json.dumps({"scenario_id": r["scenario_id"], "cond": cond,
                    "weighing": [r[cond][a:b] for a, b, k in
                                 _spans(r[cond], found.get((r["scenario_id"], cond), {})
                                        .get("weighing", []), "weighing")],
                    "power": [r[cond][a:b] for a, b, k in
                              _spans(r[cond], found.get((r["scenario_id"], cond), {})
                                     .get("power", []), "power")]}) + "\n"
        for r in recs for cond in ("before", "after")))
    (out / "weighing_highlighted.html").write_text("\n".join(parts))
    (out / "weighing_highlighted.md").write_text("\n".join(rows_md) + "\n")
    tin, tout = sum(u[0] for u in usage), sum(u[1] for u in usage)
    (out / "run_meta_highlight.json").write_text(json.dumps(
        {"git_sha": git_sha(), "timestamp_utc": timestamp(), "model": model,
         "rows": len(recs), "cost_usd": tin / 1e6 * 3 + tout / 1e6 * 15}, indent=2))
    print("\n".join(rows_md))
    print(f"\ncost ${tin / 1e6 * 3 + tout / 1e6 * 15:.2f} -> {out / 'weighing_highlighted.html'}")


if __name__ == "__main__":
    fire.Fire(main)
