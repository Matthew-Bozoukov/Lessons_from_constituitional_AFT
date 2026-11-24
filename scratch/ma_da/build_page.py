# ABOUTME: Build a single reading page (HTML) of every row in a mada synth run, for publishing as an artifact.
# ABOUTME: Usage: uv run python scratch/ma_da/build_page.py <run_dir> <out.html>
import html
import json
import re
import sys
from pathlib import Path

CHECKS = (
    (
        "advice",
        "Advice",
        "Does the person ask for advice or a judgement, and not for a piece of work?",
    ),
    (
        "hard",
        "Hard",
        "Is it hard: no fact-check settles it, and no answer is obvious or one-sided?",
    ),
    ("real", "Real", "Could a real person have sent this message to this assistant?"),
    (
        "row",
        "Reply",
        "Does the reply commit, weigh the people fairly, name its cost and invent nothing?",
    ),
)
SEED = (
    ("situation", "Situation"),
    ("parties", "The other people"),
    ("tension", "Tension"),
    ("principle_link", "Why this principle"),
    ("delivery", "How it reaches the assistant"),
)

STYLE = """
/* Layout: one reading column; each conversation is a chapter with its checks in the heading. */
:root{--bg:#f5f7f8;--surface:#ffffff;--fg:#16202a;--muted:#5a6773;--line:#d9e0e6;--accent:#1c6a86;
--pass:#1b7548;--fail:#b2362e;--tint:#eaf1f4;
--read:"Source Serif 4",Georgia,serif;--ui:Inter,system-ui,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#11161b;--surface:#192027;--fg:#e5eaee;
--muted:#9aa6b2;--line:#2a343e;--accent:#72c0da;--pass:#62c794;--fail:#f08b81;--tint:#1f2a33;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#11161b;--surface:#192027;--fg:#e5eaee;--muted:#9aa6b2;--line:#2a343e;
--accent:#72c0da;--pass:#62c794;--fail:#f08b81;--tint:#1f2a33;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--ui);font-size:15px;line-height:1.55}
.wrap{max-width:46rem;margin:0 auto;padding-block:2.5rem 5rem;
padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));
display:flex;flex-direction:column;gap:2.25rem}
h1{font-size:1.9rem;line-height:1.15;margin:0;text-wrap:balance;letter-spacing:-.01em}
h2{font-size:1.3rem;line-height:1.25;margin:0;text-wrap:balance}
h3{font-size:.74rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);margin:0;font-weight:600}
p{margin:0}
a{color:var(--accent)}
a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.lede{color:var(--muted);max-width:40rem}
.stack{display:flex;flex-direction:column;gap:.75rem}
.defs{display:grid;grid-template-columns:auto 1fr;gap:.35rem .9rem;align-items:baseline}
.defs dt{font-weight:600}.defs dd{margin:0;color:var(--muted)}
.tablebox{overflow-x:auto;border:1px solid var(--line);background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.86rem;font-variant-numeric:tabular-nums}
th,td{padding:.5rem .65rem;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
td.wide{white-space:normal;min-width:11rem}
th{font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);font-weight:600}
tr:last-child td{border-bottom:0}
.num{text-align:right}
.pill{display:inline-block;font-size:.72rem;font-weight:600;padding:.08rem .45rem;border:1px solid currentColor}
.pass{color:var(--pass)}.fail{color:var(--fail)}
article{display:flex;flex-direction:column;gap:1.4rem;border-top:3px solid var(--fg);padding-top:1.4rem}
.head{display:flex;flex-direction:column;gap:.6rem}
.eyebrow{font-size:.78rem;color:var(--muted)}
.checks{display:flex;flex-direction:column;gap:.35rem;font-size:.88rem}
.checks div{display:grid;grid-template-columns:4.6rem 1fr;gap:.6rem;align-items:baseline}
.checks span.note{color:var(--muted)}
.block{display:flex;flex-direction:column;gap:.55rem}
.text{font-family:var(--read);font-size:1.06rem;line-height:1.62;white-space:pre-wrap;overflow-wrap:anywhere}
.shown{background:var(--surface);border:1px solid var(--line);padding:1rem 1.1rem}
.think{border-left:2px solid var(--line);padding-left:1.1rem;color:var(--fg)}
.system{font-family:var(--ui);font-size:.9rem;background:var(--tint);padding:.8rem 1rem;white-space:pre-wrap}
details{border:1px solid var(--line);background:var(--surface);padding:.7rem 1rem}
summary{cursor:pointer;font-weight:600;font-size:.9rem}
details .stack{margin-top:.8rem}
details .text{font-size:.98rem}
.count{font-weight:400;color:var(--muted);text-transform:none;letter-spacing:0}
"""


def esc(text: str) -> str:
    out = html.escape(text.strip())
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)


def pill(verdict: str) -> str:
    v = str(verdict).strip().lower()
    return f'<span class="pill {"pass" if v == "pass" else "fail"}">{html.escape(v)}</span>'


def words(text: str) -> int:
    return len(text.split())


def main(run_dir: str, out_path: str) -> None:
    run = Path(run_dir)
    rows = [
        json.loads(ln)
        for ln in (run / "dataset.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    parts = [
        '<meta name="claude:viewport-fit-sides" content="cover">',
        "<title>MA-DA smoke rows</title>",
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700'
        '&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">',
        f"<style>{STYLE}</style>",
        '<div class="wrap">',
        '<header class="stack">',
        f"<h1>{len(rows)} practice conversations from the new multi-party advice recipe</h1>",
        '<p class="lede">These were written by AI models on 24 November 2026 as a small trial of a recipe for '
        "training data. In each one, a person asks an assistant for advice about a conflict among several "
        "people. There is one conversation for each of the nine principles in the constitution the assistant "
        "is being taught. Nothing here has been used for training.</p>",
        '<p class="lede">Each conversation shows what the assistant was given (a short instruction from the '
        "company running it, then the person's message), what it thought privately, and what it replied. "
        "After that come four yes-or-no checks answered by a different AI model, Gemini, and the outline the "
        "conversation was written from.</p>",
        "</header>",
        '<section class="stack"><h3>The four checks</h3><dl class="defs">',
    ]
    parts += [f"<dt>{label}</dt><dd>{html.escape(q)}</dd>" for _, label, q in CHECKS]
    parts.append("</dl></section>")

    parts.append(
        '<section class="stack"><h3>All nine at a glance</h3><div class="tablebox"><table><tr>'
    )
    parts.append(
        '<th>Principle</th><th>Setting</th><th class="num">Message words</th>'
        '<th class="num">Thinking</th><th class="num">Reply</th>'
    )
    parts += [f"<th>{label}</th>" for _, label, _ in CHECKS]
    parts.append("</tr>")
    for i, row in enumerate(rows, 1):
        meta = row["metadata"]
        msg = {m["role"]: m for m in row["messages"]}
        a = msg["assistant"]
        parts.append(
            f'<tr><td class="wide"><a href="#c{i}">{html.escape(meta["trait_name"])}</a></td>'
            f'<td class="wide">{html.escape(meta.get("domain", ""))}</td>'
            f'<td class="num">{words(msg["user"]["content"])}</td>'
            f'<td class="num">{words(a.get("reasoning_content", ""))}</td>'
            f'<td class="num">{words(a["content"])}</td>'
            + "".join(
                f"<td>{pill(meta.get(k + '_verdict', '?'))}</td>" for k, _, _ in CHECKS
            )
            + "</tr>"
        )
    parts.append("</table></div></section>")

    for i, row in enumerate(rows, 1):
        meta = row["metadata"]
        msg = {m["role"]: m for m in row["messages"]}
        a = msg["assistant"]
        system, user = msg["system"]["content"], msg["user"]["content"]
        reasoning, reply = a.get("reasoning_content", ""), a["content"]
        parts.append(f'<article id="c{i}"><div class="head">')
        parts.append(
            f'<p class="eyebrow">Conversation {i} of {len(rows)} · setting: '
            f"{html.escape(meta.get('domain', ''))}</p>"
        )
        parts.append(f"<h2>Principle: {html.escape(meta['trait_name'])}</h2></div>")
        parts.append(
            '<div class="block"><h3>The instruction the assistant runs under</h3>'
            f'<div class="system">{esc(system)}</div></div>'
        )
        parts.append(
            f'<div class="block"><h3>The person\'s message <span class="count">({words(user)} words)'
            f'</span></h3><div class="text shown">{esc(user)}</div></div>'
        )
        parts.append(
            f'<div class="block"><h3>The assistant\'s private thinking <span class="count">'
            f'({words(reasoning)} words)</span></h3><div class="text think">{esc(reasoning)}</div></div>'
        )
        parts.append(
            f'<div class="block"><h3>The assistant\'s reply <span class="count">({words(reply)} words)'
            f'</span></h3><div class="text shown">{esc(reply)}</div></div>'
        )
        parts.append(
            '<div class="block"><h3>What Gemini said about it</h3><div class="checks">'
        )
        for key, label, _ in CHECKS:
            parts.append(
                f"<div><span>{label} {pill(meta.get(key + '_verdict', '?'))}</span>"
                f'<span class="note">{html.escape(str(meta.get(key + "_note", "")))}</span></div>'
            )
        parts.append("</div></div>")
        parts.append(
            "<details><summary>The outline this conversation was written from</summary>"
            '<div class="stack">'
        )
        for key, label in SEED:
            parts.append(
                f'<div class="block"><h3>{label}</h3><div class="text">{esc(str(meta.get(key, "")))}'
                "</div></div>"
            )
        parts.append("</div></details></article>")
    parts.append("</div>")
    Path(out_path).write_text("\n".join(parts))
    print("wrote", out_path, Path(out_path).stat().st_size, "bytes")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
