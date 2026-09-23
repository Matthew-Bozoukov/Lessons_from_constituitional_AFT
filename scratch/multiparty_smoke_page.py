# ABOUTME: Renders a da-multiparty synth run's exported rows as one HTML page for hand review:
# ABOUTME: the four scenario fields, system prompt, tools, user turn, reasoning and reply.
"""Usage: uv run python scratch/multiparty_smoke_page.py <run_dir_or_jsonl> <out.html>

Reads the run's `stage_*_export_sft.jsonl` (or a `dataset.jsonl` / any export jsonl given
directly) and writes a self-contained page: one card per row, rows grouped by principle so
equal representation is visible at a glance, with a count of the scenario fields on top.
The page is a review surface, not a result; nothing is judged here.
"""

from __future__ import annotations

import collections
import html
import json
import sys
from pathlib import Path

CSS = """
:root{--bg:#F4F5F8;--card:#fff;--ink:#1A1926;--muted:#5D5B70;--rule:#D8D7E4;--acc:#0B7A6B;
--acc2:#6A4BC4;--warn:#B85F08;--think:#EEEEF4}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#131219;--card:#1C1B26;
--ink:#ECEAF6;--muted:#A3A1B8;--rule:#33314A;--acc:#56C9B5;--acc2:#A994F2;--warn:#F0A24E;--think:#24232F}}
:root[data-theme="dark"]{--bg:#131219;--card:#1C1B26;--ink:#ECEAF6;--muted:#A3A1B8;--rule:#33314A;
--acc:#56C9B5;--acc2:#A994F2;--warn:#F0A24E;--think:#24232F}
body{background:var(--bg);color:var(--ink);font:15px/1.55 "IBM Plex Sans","Segoe UI",system-ui,sans-serif;
padding-inline:20px;padding-block:32px 64px;margin:0}
.page{max-width:1040px;margin-inline:auto;display:flex;flex-direction:column;gap:28px}
h1{font-size:1.9rem;margin:0;line-height:1.15}h2{font-size:1.25rem;margin:24px 0 0}h3{font-size:1.05rem;margin:0}
.muted{color:var(--muted)}.mono{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-size:.85em}
table{border-collapse:collapse;font-size:.9rem}th,td{text-align:left;vertical-align:top;padding:6px 14px 6px 0;
border-bottom:1px solid var(--rule)}th{font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.scroll{overflow-x:auto}
.card{background:var(--card);border:1px solid var(--rule);border-radius:6px;display:flex;flex-direction:column}
.head{padding:14px 18px;border-bottom:1px solid var(--rule);display:flex;flex-direction:column;gap:6px}
.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{font-family:"IBM Plex Mono",monospace;font-size:.72rem;
padding:2px 8px;border:1px solid var(--rule);border-radius:99px;color:var(--muted);white-space:nowrap}
.chip.a{border-color:var(--acc);color:var(--acc)}.chip.b{border-color:var(--acc2);color:var(--acc2)}
.chip.w{border-color:var(--warn);color:var(--warn)}
.turn{padding:14px 18px;border-bottom:1px solid var(--rule);display:flex;flex-direction:column;gap:4px;font-size:.93rem}
.turn:last-child{border-bottom:0}.role{font-family:"IBM Plex Mono",monospace;font-size:.7rem;letter-spacing:.08em;
text-transform:uppercase;color:var(--muted)}.think{background:var(--think)}.think .body{color:var(--muted);font-style:italic}
.reply{border-left:4px solid var(--acc)}.body{white-space:pre-wrap;overflow-wrap:anywhere}
pre{margin:0;font:.78rem/1.5 "IBM Plex Mono",ui-monospace,Menlo,monospace;background:var(--think);
border:1px solid var(--rule);border-radius:4px;padding:10px 12px;overflow-x:auto}
dl{margin:0;display:grid;grid-template-columns:max-content 1fr;gap:3px 12px;font-size:.88rem}
dt{font-family:"IBM Plex Mono",monospace;font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);padding-top:2px}
dd{margin:0}
"""

FIELDS = ["trait_id", "asker", "agenticness"]


def load_rows(path: Path) -> list[dict]:
    if path.is_dir():
        cands = sorted(path.glob("stage_*_export_sft.jsonl"))
        if not cands:
            cands = [path / "dataset.jsonl"]
        path = cands[-1]
    print("reading", path)
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def field_table(rows: list[dict]) -> str:
    parts = [
        "<div class='scroll'><table><thead><tr><th>field</th><th>counts</th></tr></thead><tbody>"
    ]
    for a in FIELDS:
        c = collections.Counter(str(r["metadata"].get(a, "")) for r in rows)
        cells = ", ".join(f"{esc(k)} <b>{v}</b>" for k, v in sorted(c.items()))
        parts.append(f"<tr><td class='mono'>{esc(a)}</td><td>{cells}</td></tr>")
    n_tools = sum(1 for r in rows if r.get("tools"))
    parts.append(
        f"<tr><td class='mono'>tools present</td><td><b>{n_tools}</b> / {len(rows)} rows</td></tr>"
    )
    parts.append("</tbody></table></div>")
    return "".join(parts)


def card(r: dict) -> str:
    m = r["metadata"]
    msgs = {x["role"]: x for x in r["messages"]}
    tools = r.get("tools") or []
    chips = [
        f"<span class='chip b'>{esc(m.get('trait_id'))} · {esc(m.get('trait_name'))}</span>",
        f"<span class='chip a'>asker: {esc(m.get('asker'))}</span>",
        f"<span class='chip w'>assistant: {esc(m.get('agenticness'))}</span>",
    ]
    struct = "".join(
        f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>"
        for k, v in [
            ("parties", m.get("parties")),
            ("turns on", m.get("dynamic")),
            ("case for it", m.get("case_for")),
            ("workaround closed", m.get("workaround_closed")),
            ("shortcut", m.get("shortcut")),
        ]
    )
    a = msgs.get("assistant", {})
    out = [
        "<article class='card'>",
        f"<div class='head'><h3>{esc(m.get('scenario_id'))} · {esc(m.get('domain'))}</h3>",
        f"<div class='chips'>{''.join(chips)}</div><dl>{struct}</dl></div>",
        f"<div class='turn'><span class='role'>system</span><div class='body'>{esc(msgs.get('system', {}).get('content'))}</div></div>",
    ]
    if tools:
        out.append(
            f"<div class='turn'><span class='role'>tools ({len(tools)})</span>"
            f"<pre>{esc(json.dumps(tools, indent=1))}</pre></div>"
        )
    out += [
        f"<div class='turn'><span class='role'>user</span><div class='body'>{esc(msgs.get('user', {}).get('content'))}</div></div>",
        f"<div class='turn think'><span class='role'>assistant · reasoning ({len(a.get('reasoning_content') or '')} chars)</span>"
        f"<div class='body'>{esc(a.get('reasoning_content'))}</div></div>",
        f"<div class='turn reply'><span class='role'>assistant · reply ({len(a.get('content') or '')} chars)</span>"
        f"<div class='body'>{esc(a.get('content'))}</div></div>",
        "</article>",
    ]
    return "".join(out)


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    rows = load_rows(src)
    by_trait: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_trait[r["metadata"].get("trait_id", "?")].append(r)
    parts = [
        "<title>Multi-Party DA Smoke</title>",
        f"<style>{CSS}</style>",
        "<div class='page'>",
        "<div><p class='muted mono'>da-multiparty · smoke review · hand-read, nothing judged</p>",
        "<h1>Multi-party difficult advice, smoke rows</h1>",
        f"<p class='muted'>{len(rows)} rows from <span class='mono'>{esc(src)}</span>; "
        f"{len(by_trait)} principles. Grouped by principle.</p></div>",
        field_table(rows),
    ]
    for tid in sorted(by_trait):
        group = by_trait[tid]
        name = group[0]["metadata"].get("trait_name", "")
        parts.append(
            f"<h2>{esc(tid)} · {esc(name)} <span class='muted'>({len(group)})</span></h2>"
        )
        parts.extend(card(r) for r in group)
    parts.append("</div>")
    dst.write_text("\n".join(parts), encoding="utf-8")
    print("wrote", dst, f"({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
