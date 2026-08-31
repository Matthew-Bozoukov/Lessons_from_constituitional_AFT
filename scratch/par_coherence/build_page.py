# ABOUTME: Builds the human-review page for a PAR coherence rewrite run: every record's full five turns,
# ABOUTME: BEFORE/AFTER trained turn side by side with the decision sentences highlighted, proxies, notes.
# Run: uv run python scratch/par_coherence/build_page.py --run output/par_coherence/smoke_<ts> [--notes notes.json] --out page.html
from __future__ import annotations

import argparse
import html
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

import sys

sys.path.insert(0, ".")
from scratch.par_coherence import props as P  # noqa: E402

SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'“(])")


def esc(s: str) -> str:
    return html.escape(s or "", quote=False)


def inline(s: str) -> str:
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", s)
    return s


def mark_sentences(par: str) -> str:
    out = []
    for sent in SENT.split(par):
        n = P.norm(sent)
        if P.WONT_WIDE.search(n):
            out.append(f'<mark class="wont">{inline(sent)}</mark>')
        elif P.WILL_WIDE.search(n):
            out.append(f'<mark class="will">{inline(sent)}</mark>')
        else:
            out.append(inline(sent))
    return " ".join(out)


def render(text: str, highlight: bool = True) -> str:
    """Paragraphs, simple bullets/numbered lists, bold; decision sentences marked."""
    blocks, buf, items = [], [], []
    f = mark_sentences if highlight else inline

    def flush_items():
        nonlocal items
        if items:
            blocks.append("<ul>" + "".join(f"<li>{f(i)}</li>" for i in items) + "</ul>")
            items = []

    for line in (text or "").split("\n"):
        s = line.strip()
        if not s:
            flush_items()
            if buf:
                blocks.append(f"<p>{f(' '.join(buf))}</p>")
                buf = []
            continue
        m = re.match(r"^(?:[-*•]|\d+[.)])\s+(.*)", s)
        if m:
            if buf:
                blocks.append(f"<p>{f(' '.join(buf))}</p>")
                buf = []
            items.append(m.group(1))
        else:
            flush_items()
            buf.append(s)
    flush_items()
    if buf:
        blocks.append(f"<p>{f(' '.join(buf))}</p>")
    return "\n".join(blocks)


def chip(label: str, before, after) -> str:
    def fmt(v):
        return "yes" if v is True else "no" if v is False else str(v)

    same = before == after
    cls = "same" if same else ("up" if after and not before else "chg")
    return f'<span class="chip {cls}"><span class="k">{esc(label)}</span> {fmt(before)} <span class="arr">→</span> {fmt(after)}</span>'


def build(run: Path, notes: dict, out: Path) -> None:
    recs = [json.loads(l) for l in (run / "records.jsonl").open(encoding="utf-8")]
    meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
    ok = [r for r in recs if r["ok"]]
    tin = sum(r["usage"]["in"] for r in recs)
    tout = sum(r["usage"]["out"] for r in recs)
    cost = (tin * 2 + tout * 10) / 1e6

    def rate(key, which):
        return 100 * st.mean(bool(r[f"props_{which}"][key]) for r in ok)

    tc_b = [r for r in ok if r["props_before"]["trace_commits"]]
    tc_a = [r for r in ok if r["props_after"]["trace_commits"]]
    cond_b = (
        100 * st.mean(r["props_before"]["reply_firm"] for r in tc_b)
        if tc_b
        else float("nan")
    )
    cond_a = (
        100 * st.mean(r["props_after"]["reply_firm"] for r in tc_a)
        if tc_a
        else float("nan")
    )
    lead_a = Counter(
        r["props_after"]["decision_lead_formula"]
        for r in ok
        if r["props_after"]["decision_lead_formula"]
    )
    dr = [
        100
        * (len(r["after"]["reasoning"]) - len(r["before"]["reasoning"]))
        / len(r["before"]["reasoning"])
        for r in ok
    ]
    dp = [
        100
        * (len(r["after"]["response"]) - len(r["before"]["response"]))
        / len(r["before"]["response"])
        for r in ok
    ]

    summary_rows = [
        (
            "Reasoning ends on a stated decision (wide lexicon)",
            f"{rate('trace_decides_wide', 'before'):.0f}%",
            f"{rate('trace_decides_wide', 'after'):.0f}%",
        ),
        (
            "Reply states what it won't and will do (wide lexicon)",
            f"{rate('reply_decides_wide', 'before'):.0f}%",
            f"{rate('reply_decides_wide', 'after'):.0f}%",
        ),
        (
            "Reply's last sentence carries the decision",
            f"{rate('reply_last_sentence_decides', 'before'):.0f}%",
            f"{rate('reply_last_sentence_decides', 'after'):.0f}%",
        ),
        (
            "Trace commits — four-MO strict lexicon (grok 28% · Sonnet 23% · PAR 24%)",
            f"{rate('trace_commits', 'before'):.0f}%",
            f"{rate('trace_commits', 'after'):.0f}%",
        ),
        (
            "Reply firm-refusal composite — strict (grok 72% · Sonnet 20% · PAR 24%)",
            f"{rate('reply_firm', 'before'):.0f}%",
            f"{rate('reply_firm', 'after'):.0f}%",
        ),
        (
            "P(reply firm | trace commits) — strict (grok 94% · Sonnet 28% · PAR 41%)",
            f"{cond_b:.0f}% (n={len(tc_b)})",
            f"{cond_a:.0f}% (n={len(tc_a)})",
        ),
        (
            "Decision paragraph opens with a stock formula",
            "0/%d" % len(ok),
            f"{sum(lead_a.values())}/{len(ok)}"
            + (
                " — " + ", ".join(f"“{k}” ×{v}" for k, v in lead_a.most_common())
                if lead_a
                else ""
            ),
        ),
        (
            "Reply mentions the earlier refusal",
            f"{rate('reply_mentions_earlier_refusal', 'before'):.0f}%",
            f"{rate('reply_mentions_earlier_refusal', 'after'):.0f}%",
        ),
        (
            "Length change, reasoning (chars) — median [min, max]",
            "",
            f"{st.median(dr):+.0f}% [{min(dr):+.0f}, {max(dr):+.0f}]",
        ),
        (
            "Length change, reply (chars) — median [min, max]",
            "",
            f"{st.median(dp):+.0f}% [{min(dp):+.0f}, {max(dp):+.0f}]",
        ),
    ]

    css = """
:root{--bg:#F5F4F0;--panel:#FFFFFF;--ink:#1F2226;--muted:#646A72;--line:#D9D7D0;--line-strong:#B9B6AC;
--wont:#0E6B6B;--wont-bg:#D8ECEA;--will:#9A6A12;--will-bg:#F4E7C8;--refusal:#9B3B39;--refusal-bg:#F5E1DF;
--user-bg:#ECEBE6;--chip-same:#E8E7E2;--chip-chg:#DCE9E8;--chip-up:#0E6B6B;--chip-up-ink:#FFFFFF;
--verdict-bg:#E7F0EC;--verdict-ink:#1F4F3F;}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#16181C;--panel:#1E2126;--ink:#E7E5DF;--muted:#9AA0A8;--line:#31353C;--line-strong:#4A4F58;
--wont:#5FC2C0;--wont-bg:#163634;--will:#E3B15A;--will-bg:#3A2E14;--refusal:#E58C88;--refusal-bg:#3B2222;
--user-bg:#23262C;--chip-same:#262A30;--chip-chg:#1C3634;--chip-up:#5FC2C0;--chip-up-ink:#0F1A1A;--verdict-bg:#1C2E27;--verdict-ink:#BFE3D2;}}
:root[data-theme="dark"]{--bg:#16181C;--panel:#1E2126;--ink:#E7E5DF;--muted:#9AA0A8;--line:#31353C;--line-strong:#4A4F58;
--wont:#5FC2C0;--wont-bg:#163634;--will:#E3B15A;--will-bg:#3A2E14;--refusal:#E58C88;--refusal-bg:#3B2222;
--user-bg:#23262C;--chip-same:#262A30;--chip-chg:#1C3634;--chip-up:#5FC2C0;--chip-up-ink:#0F1A1A;--verdict-bg:#1C2E27;--verdict-ink:#BFE3D2;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;font-size:15.5px;line-height:1.55}
.wrap{max-width:1180px;margin:0 auto;padding:32px 24px 80px}
h1,h2,h3{font-family:"Fraunces","Iowan Old Style",Georgia,serif;font-weight:500;letter-spacing:-0.01em;text-wrap:balance;margin:0}
h1{font-size:2.2rem;line-height:1.1}
h2{font-size:1.35rem;margin:0 0 12px}
h3{font-size:1.05rem}
.sub{color:var(--muted);margin:8px 0 0;font-size:.95rem}
.mono{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.82rem}
.label{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
header{display:grid;gap:14px;padding-bottom:24px;border-bottom:1px solid var(--line-strong)}
.verdict{background:var(--verdict-bg);color:var(--verdict-ink);border-radius:6px;padding:14px 18px;max-width:72ch}
.verdict p{margin:0 0 6px}.verdict p:last-child{margin:0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:28px;margin-top:28px}
@media (max-width:900px){.grid2{grid-template-columns:1fr}}
table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:500}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.legend{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;margin-top:10px;font-size:.9rem}
mark{border-radius:3px;padding:0 3px;color:inherit}
mark.wont{background:var(--wont-bg);box-shadow:inset 0 -2px 0 var(--wont)}
mark.will{background:var(--will-bg);box-shadow:inset 0 -2px 0 var(--will)}
.howto{max-width:72ch}.howto ol{padding-left:22px;margin:8px 0}.howto li{margin:4px 0}
nav.idx{display:flex;flex-wrap:wrap;gap:8px;margin:28px 0 8px}
nav.idx a{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.78rem;color:var(--ink);text-decoration:none;border:1px solid var(--line-strong);border-radius:999px;padding:4px 10px;background:var(--panel)}
nav.idx a:hover,nav.idx a:focus-visible{border-color:var(--wont);outline:none}
article{margin-top:44px;padding-top:28px;border-top:2px solid var(--line-strong)}
article .head{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:baseline}
article .head .n{font-family:"Fraunces",Georgia,serif;font-size:1.6rem;font-weight:500}
.meta{display:flex;flex-wrap:wrap;gap:6px 14px;color:var(--muted);font-size:.85rem;margin-top:4px}
details{margin-top:12px}summary{cursor:pointer;color:var(--muted);font-size:.9rem}
.sys{background:var(--user-bg);border-radius:6px;padding:12px 14px;font-size:.9rem;margin-top:8px;max-width:80ch}
.convo{display:grid;gap:10px;margin-top:16px;max-width:80ch}
.turn{border-radius:6px;padding:12px 14px}
.turn .label{display:block;margin-bottom:4px}
.turn.user{background:var(--user-bg)}
.turn.refusal{background:var(--refusal-bg);color:var(--ink)}
.turn.refusal .label{color:var(--refusal)}
.turn p{margin:0 0 8px}.turn p:last-child{margin:0}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-top:20px}
@media (max-width:900px){.pair{grid-template-columns:1fr}}
.col{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px 18px;min-width:0}
.col .label{display:block;margin-bottom:8px}
.col.after{border-color:var(--wont)}
.col h4{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:14px 0 6px;font-weight:500}
.col h4:first-of-type{margin-top:0}
.col p{margin:0 0 10px}.col ul{margin:0 0 10px;padding-left:20px}.col li{margin:3px 0}
.reason{font-size:.93rem;color:var(--ink);opacity:.92}
.changes{margin-top:14px;padding:10px 14px;border-left:3px solid var(--will);background:var(--panel);font-size:.92rem;max-width:80ch}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}
.chip{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.74rem;border-radius:4px;padding:3px 8px;background:var(--chip-same);white-space:nowrap}
.chip .k{color:var(--muted)}.chip .arr{color:var(--muted)}
.chip.chg{background:var(--chip-chg)}.chip.up{background:var(--chip-up);color:var(--chip-up-ink)}.chip.up .k,.chip.up .arr{color:inherit;opacity:.8}
.note{margin-top:14px;padding:12px 16px;border-radius:6px;background:var(--verdict-bg);color:var(--verdict-ink);max-width:80ch;font-size:.93rem}
.note .label{color:inherit;opacity:.75;display:block;margin-bottom:4px}
.prompt{margin-top:40px}
.prompt pre{white-space:pre-wrap;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px;font-size:.8rem;line-height:1.45;overflow-x:auto}
a{color:var(--wont)}
@media (prefers-reduced-motion: no-preference){html{scroll-behavior:smooth}}
"""

    parts = [
        "<title>PAR Coherence Smoke</title>",
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">',
        f"<style>{css}</style>",
        '<div class="wrap">',
        "<header>",
        '<div><span class="label">Post-action retrospection · arm 1 · coherent trace+reply rewrite</span>',
        "<h1>PAR Coherence Smoke</h1>",
        f'<p class="sub">{len(ok)} of {len(recs)} rows rewritten and lint-clean · run <span class="mono">{esc(run.name)}</span> · '
        f"{esc(meta['args']['model'])} at T={meta['args']['temperature']} · {tin:,} in / {tout:,} out tokens · ≈ ${cost:.2f} · git {esc(meta['git_sha'])}</p></div>",
        '<div class="verdict">' + notes.get("_verdict", "") + "</div>",
        "</header>",
        '<div class="grid2">',
        "<section><h2>What moved</h2><table><thead><tr><th>property (10 rows)</th><th>before</th><th>after</th></tr></thead><tbody>",
    ]
    for label, b, a in summary_rows:
        parts.append(
            f"<tr><td>{esc(label)}</td><td class='num'>{esc(b)}</td><td class='num'>{esc(a)}</td></tr>"
        )
    parts.append("</tbody></table>")
    parts.append(
        '<div class="legend"><span class="label">Highlights</span>'
        '<span><mark class="wont">what it won\'t do</mark> — a first-person decline of a specific action</span>'
        '<span><mark class="will">what it will do</mark> — a first-person commitment to help</span></div>'
    )
    parts.append("</section>")
    parts.append(
        '<section class="howto"><h2>How to read each record</h2>'
        + notes.get("_howto", "")
        + "</section>"
    )
    parts.append("</div>")
    parts.append(
        '<nav class="idx">'
        + "".join(
            f'<a href="#r{i}">{i}. {esc(r["trait_id"])} · {esc(r["scenario_id"])}</a>'
            for i, r in enumerate(recs, 1)
        )
        + "</nav>"
    )

    for i, r in enumerate(recs, 1):
        b, a = r["props_before"], r["props_after"]
        parts.append(f'<article id="r{i}">')
        parts.append(
            f'<div class="head"><span class="n">{i}.</span><h3>{esc(r["trait_name"])}</h3></div>'
        )
        parts.append(
            f'<div class="meta"><span class="mono">{esc(r["scenario_id"])}</span><span>refusal register: {esc(r["refusal_register"])}</span>'
            f"<span>domain: {esc(r['domain'] or '')}</span><span>attempts: {len(r['attempts'])}</span></div>"
        )
        parts.append(
            f'<details><summary>Deployment system prompt</summary><div class="sys">{render(r["system"], highlight=False)}</div></details>'
        )
        parts.append('<div class="convo">')
        parts.append(
            f'<div class="turn user"><span class="label">Turn 1 · user</span>{render(r["user"], highlight=False)}</div>'
        )
        parts.append(
            f'<div class="turn refusal"><span class="label">Turn 2 · assistant — bare refusal (context only, never trained)</span>{render(r["first_turn"], highlight=False)}</div>'
        )
        parts.append(
            f'<div class="turn user"><span class="label">Turn 3 · user — pushback</span>{render(r["followup"], highlight=False)}</div>'
        )
        parts.append("</div>")
        if r["ok"]:
            parts.append('<div class="pair">')
            for which, cls, lab in (
                ("before", "before", "Before — turn 4 as trained (PAR-716)"),
                ("after", "after", "After — coherent rewrite"),
            ):
                t = r[which]
                parts.append(
                    f'<div class="col {cls}"><span class="label">{esc(lab)}</span>'
                )
                parts.append(
                    f'<h4>Private reasoning · {len((t["reasoning"] or "").split())} words</h4><div class="reason">{render(t["reasoning"])}</div>'
                )
                parts.append(
                    f"<h4>Reply · {len((t['response'] or '').split())} words</h4><div>{render(t['response'])}</div>"
                )
                parts.append("</div>")
            parts.append("</div>")
            parts.append(
                f'<div class="changes"><span class="label">Rewriter\'s own account of the change</span><p style="margin:4px 0 0">{inline(r["changes"] or "")}</p></div>'
            )
            parts.append(
                '<div class="chips">'
                + chip(
                    "reasoning ends on a decision",
                    b["trace_decides_wide"],
                    a["trace_decides_wide"],
                )
                + chip(
                    "reply won't+will", b["reply_decides_wide"], a["reply_decides_wide"]
                )
                + chip(
                    "last sentence decides",
                    b["reply_last_sentence_decides"],
                    a["reply_last_sentence_decides"],
                )
                + chip("closer", b["reply_closer"], a["reply_closer"])
                + chip("strict: trace commits", b["trace_commits"], a["trace_commits"])
                + chip("strict: reply firm", b["reply_firm"], a["reply_firm"])
                + chip(
                    "formula",
                    b["decision_lead_formula"] or "none",
                    a["decision_lead_formula"] or "none",
                )
                + "</div>"
            )
        else:
            parts.append(
                f'<p class="note">Rewrite FAILED after {len(r["attempts"])} attempts: {esc(str(r["attempts"][-1]["errors"]))}</p>'
            )
        if r["scenario_id"] in notes:
            parts.append(
                f'<div class="note"><span class="label">Reviewer note (Claude)</span>{inline(notes[r["scenario_id"]])}</div>'
            )
        parts.append("</article>")

    parts.append(
        '<section class="prompt"><h2>The rewrite prompt, verbatim</h2><p class="sub">System: '
        + esc(meta["system_prompt"])
        + "</p><pre>"
        + esc(meta["user_prompt"])
        + "</pre></section>"
    )
    parts.append("</div>")
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument(
        "--notes",
        help="json: scenario_id -> reviewer note; _verdict and _howto are HTML blocks",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    notes = (
        json.loads(Path(args.notes).read_text(encoding="utf-8")) if args.notes else {}
    )
    build(Path(args.run), notes, Path(args.out))


if __name__ == "__main__":
    main()
