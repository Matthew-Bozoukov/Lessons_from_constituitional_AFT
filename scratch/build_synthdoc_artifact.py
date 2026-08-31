# ABOUTME: Renders the synthdoc v2 corpus (2,203 trait-grounded SFT documents) into one
# ABOUTME: self-contained HTML browser for publishing as an Artifact.
"""Build the constitution-trait corpus browser.

    uv run python scratch/build_synthdoc_artifact.py [--out <path.html>]

Every document is written into the markup in full. The one thing NOT repeated per row is
`trait_text`: it is 4.98 MB of the source file but only 9 distinct values, so it is
rendered once in the trait index and referenced by id. That alone is what brings the
corpus under the 16 MB artifact ceiling.
"""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

import fire

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl"

# One hue per trait, carried as an inline `--h` custom property. Saturation and lightness
# come from theme tokens, so a trait keeps its identity on both grounds without declaring
# nine colours twice.
HUES = {"t1": 205, "t2": 265, "t3": 158, "t4": 18, "t5": 45,
        "t6": 320, "t7": 188, "t8": 96, "t9": 32}


def _esc(s, attr: bool = False) -> str:
    """Escape for element text by default; pass attr=True inside an attribute.

    quote=True also encodes `"` as `&quot;` -- six bytes for one. Across this corpus that
    is 0.95 MB of pure escaping, which matters against a 16 MB ceiling, and it is only
    needed inside attribute values.
    """
    return html.escape(str(s if s is not None else ""), quote=attr)


def _words(text) -> int:
    return len(str(text or "").split())


def _paras(text: str) -> str:
    out = []
    for block in str(text).split("\n\n"):
        block = block.strip()
        if block:
            out.append("<p>" + _esc(block).replace("\n", "<br>") + "</p>")
    return "".join(out) or "<p class='void'>(empty)</p>"


def build(rows: list[dict], part: str = "", other: str = "", all_rows: int = 0) -> str:
    traits: dict[str, dict] = {}
    for r in rows:
        m = r["metadata"]
        traits.setdefault(m["trait_id"], {"name": m["trait_name"], "text": m["trait_text"]})
    tids = sorted(traits)
    tcount = Counter(r["metadata"]["trait_id"] for r in rows)
    doms = Counter(r["metadata"]["domain"].strip().lower() for r in rows)
    n_dom = len(doms)

    index = "".join(
        f"""<details class="trait" style="--h:{HUES.get(t, 200)}">
<summary><span class="tid">{_esc(t)}</span><span class="tname">{_esc(traits[t]['name'])}</span>
<span class="tn">{tcount[t]}</span></summary>
<div class="ttext">{_paras(traits[t]['text'])}</div></details>"""
        for t in tids)

    rmap = [next((x.get("reasoning_content") or "") for x in r["messages"]
                 if x["role"] == "assistant") for r in rows]
    cards = []
    for i, r in enumerate(rows, 1):
        m = r["metadata"]
        msg = {x["role"]: x["content"] for x in r["messages"]}
        t = m["trait_id"]
        cards.append(
            f'<article class="doc" data-t="{_esc(t, True)}" data-dom="{_esc(m["domain"].strip().lower(), True)}"'
            f' style="--h:{HUES.get(t, 200)}" id="d{i}">'
            f'<header class="dh"><span class="sid">{_esc(m["scenario_id"])}</span>'
            f'<span class="tchip">{_esc(t)}</span>'
            f'<span class="dom">{_esc(m["domain"].strip())}</span></header>'
            f'<div class="setup"><div class="sit"><span class="k">situation</span>'
            f'{_esc(m["situation"])}</div>'
            f'<div class="cut"><span class="k">shortcut on offer</span>'
            f'{_esc(m["shortcut"])}</div></div>'
            f'<div class="turn sys"><div class="who">system<span class="wc">'
            f'{_words(msg.get("system")):,}w</span></div>'
            f'<div class="prose">{_paras(msg.get("system"))}</div></div>'
            f'<div class="turn usr"><div class="who">user<span class="wc">'
            f'{_words(msg.get("user")):,}w</span></div>'
            f'<div class="prose">{_paras(msg.get("user"))}</div></div>'
            f'<div class="turn rea"><div class="who">reasoning<span class="wc">'
            f'{_words(rmap[i - 1]):,}w</span></div>'
            f'<div class="prose">{_paras(rmap[i - 1])}</div></div>'
            f'<div class="turn ast"><div class="who">assistant<span class="wc">'
            f'{_words(msg.get("assistant")):,}w</span></div>'
            f'<div class="prose">{_paras(msg.get("assistant"))}</div></div>'
            f'</article>')

    topdoms = "".join(
        f'<button data-dom="{_esc(d, True)}" aria-pressed="false">{_esc(d)} <i>{c}</i></button>'
        for d, c in doms.most_common(8))
    topts = "".join(
        f'<button data-t="{_esc(t, True)}" aria-pressed="false" style="--h:{HUES.get(t,200)}">'
        f'{_esc(t)} <i>{tcount[t]}</i></button>' for t in tids)
    med_a = sorted(_words({x["role"]: x["content"] for x in r["messages"]}.get("assistant"))
                   for r in rows)[len(rows) // 2]

    return f"""<title>Constitution Trait Corpus</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Literata:opsz,wght@7..72,400;7..72,600&display=swap">
<style>
:root{{--paper:#F6F7F9;--surface:#FFF;--sunk:#EDF0F4;--ink:#131A21;--ink-2:#39434D;
--muted:#6B7682;--line:#DCE2E8;--accent:#1A6B72;--accent-soft:#E2EEEF;
--ts:38%;--tl:34%;--tbg:96%;
--sans:Archivo,ui-sans-serif,system-ui,sans-serif;
--serif:Literata,Georgia,serif;--mono:"JetBrains Mono",ui-monospace,Menlo,monospace}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
--paper:#0F1417;--surface:#161C21;--sunk:#1B2228;--ink:#E4E9ED;--ink-2:#BCC5CD;
--muted:#8B959E;--line:#252D34;--accent:#5FB3B8;--accent-soft:#14282A;
--ts:42%;--tl:66%;--tbg:14%}}}}
:root[data-theme="dark"]{{--paper:#0F1417;--surface:#161C21;--sunk:#1B2228;--ink:#E4E9ED;
--ink-2:#BCC5CD;--muted:#8B959E;--line:#252D34;--accent:#5FB3B8;--accent-soft:#14282A;
--ts:42%;--tl:66%;--tbg:14%}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
font-size:16px;line-height:1.55;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1060px;margin:0 auto;padding:0 20px 90px}}
:focus-visible{{outline:2px solid var(--accent);outline-offset:2px;border-radius:3px}}
header.mast{{padding:54px 0 26px}}
.eyebrow{{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
color:var(--muted);margin:0 0 12px}}
h1{{font-family:var(--serif);font-weight:600;font-size:clamp(29px,4.3vw,44px);line-height:1.12;
margin:0 0 13px;letter-spacing:-.015em;text-wrap:balance}}
.stand{{margin:0;max-width:62ch;color:var(--ink-2);font-family:var(--serif);font-size:17.5px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(126px,1fr));gap:1px;
background:var(--line);border:1px solid var(--line);margin:28px 0 34px}}
.stat{{background:var(--surface);padding:13px 15px}}
.stat .k{{font-family:var(--mono);font-size:22px;display:block;line-height:1.2;
font-variant-numeric:tabular-nums}}
.stat .l{{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}}
h2.sec{{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
color:var(--muted);margin:0 0 12px;font-weight:500}}
.trait{{border:1px solid var(--line);background:var(--surface);margin-bottom:-1px}}
.trait summary{{display:flex;align-items:baseline;gap:12px;padding:11px 14px;cursor:pointer;
list-style:none}}
.trait summary::-webkit-details-marker{{display:none}}
.trait[open] summary{{border-bottom:1px solid var(--line)}}
.tid{{font-family:var(--mono);font-size:12px;font-weight:500;
color:hsl(var(--h) var(--ts) var(--tl))}}
.tname{{font-family:var(--serif);font-size:15.5px;color:var(--ink);flex:1;min-width:0}}
.tn{{font-family:var(--mono);font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums}}
.ttext{{padding:13px 14px;font-family:var(--serif);font-size:15px;color:var(--ink-2);
max-width:70ch;line-height:1.6}}
.ttext p{{margin:0 0 .7em}}.ttext p:last-child{{margin:0}}
.controls{{position:sticky;top:0;z-index:5;background:var(--paper);padding:12px 0 10px;
border-bottom:1px solid var(--line);margin:26px 0 6px;display:flex;flex-wrap:wrap;gap:8px;
align-items:center}}
.controls .grp{{display:flex;flex-wrap:wrap;gap:5px}}
.controls button{{font-family:var(--mono);font-size:11.5px;color:var(--ink-2);
background:var(--surface);border:1px solid var(--line);border-radius:2px;padding:5px 9px;
cursor:pointer}}
.controls button i{{font-style:normal;color:var(--muted);font-size:10px}}
.controls button[data-t]{{color:hsl(var(--h) var(--ts) var(--tl))}}
.controls button[aria-pressed="true"]{{background:var(--accent-soft);border-color:var(--accent);
font-weight:500}}
input[type=search]{{font-family:var(--sans);font-size:14px;color:var(--ink);
background:var(--surface);border:1px solid var(--line);border-radius:2px;padding:7px 9px;
min-width:200px;flex:1 1 200px}}
.count{{font-family:var(--mono);font-size:11.5px;color:var(--muted);
font-variant-numeric:tabular-nums;margin-left:auto}}
.doc{{background:var(--surface);border:1px solid var(--line);
border-left:3px solid hsl(var(--h) var(--ts) var(--tl));margin:18px 0;padding:18px 20px 15px;
content-visibility:auto;contain-intrinsic-size:0 900px}}
.doc[hidden]{{display:none}}
.dh{{display:flex;align-items:baseline;gap:11px;flex-wrap:wrap;padding-bottom:10px;
border-bottom:1px solid var(--line)}}
.sid{{font-family:var(--mono);font-size:11.5px;color:var(--muted)}}
.tchip{{font-family:var(--mono);font-size:10.5px;font-weight:500;padding:2px 7px;border-radius:2px;
color:hsl(var(--h) var(--ts) var(--tl));background:hsl(var(--h) 30% var(--tbg))}}
.dom{{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}}
.setup{{display:grid;gap:9px;padding:13px 0}}
.sit,.cut{{font-family:var(--serif);font-size:15.5px;line-height:1.55;max-width:72ch}}
.sit{{color:var(--ink)}}
.cut{{color:var(--ink-2);border-left:2px solid hsl(var(--h) var(--ts) var(--tl));
padding-left:11px}}
.k{{display:block;font-family:var(--mono);font-size:9.5px;letter-spacing:.13em;
text-transform:uppercase;color:var(--muted);margin-bottom:3px}}
.turn{{border-top:1px solid var(--line);padding:12px 0}}
.turn.usr{{background:var(--sunk);margin:0 -20px;padding:12px 20px}}
.who{{font-family:var(--mono);font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;
color:var(--muted);margin-bottom:7px;display:flex;gap:10px}}
.wc{{margin-left:auto;font-variant-numeric:tabular-nums;opacity:.85}}
.prose{{font-family:var(--serif);font-size:16px;line-height:1.62;max-width:68ch}}
.prose p{{margin:0 0 .8em}}.prose p:last-child{{margin:0}}
.turn.sys .prose{{font-size:14.5px;color:var(--muted)}}
.turn.rea{{border-left:2px solid hsl(var(--h) var(--ts) var(--tl));padding-left:14px;
margin-left:1px}}
.turn.rea .prose{{font-size:15px;color:var(--ink-2);font-style:italic;line-height:1.6}}
.void{{color:var(--muted);font-style:italic}}
.empty{{padding:56px 0;text-align:center;color:var(--muted);font-family:var(--serif);font-size:18px}}
footer.col{{margin-top:40px;padding-top:18px;border-top:1px solid var(--line);font-size:13px;
color:var(--muted);max-width:70ch}}
footer.col code{{font-family:var(--mono);font-size:12px;color:var(--ink-2)}}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}}}
@media (max-width:640px){{.doc{{padding:15px 14px 12px}}.turn.usr{{margin:0 -14px;padding:12px 14px}}}}
</style>
<div class="wrap">
<header class="mast">
<p class="eyebrow">synthetic training corpus &middot; synthdoc v2 &middot; 2026-08-03{part}</p>
<h1>Constitution Trait Corpus</h1>
<p class="stand">{len(rows):,} generated documents, each built around a constitution
trait. Every document sets up a situation, names the shortcut that would betray the trait,
then shows the model reasoning about it and the reply that declines it.{other}</p>
<div class="stats">
<div class="stat"><span class="k">{len(rows):,}</span><span class="l">documents</span></div>
<div class="stat"><span class="k">{len(tids)}</span><span class="l">traits here</span></div>
<div class="stat"><span class="k">{n_dom}</span><span class="l">domains</span></div>
<div class="stat"><span class="k">{tcount[tids[0]]}</span><span class="l">per trait</span></div>
<div class="stat"><span class="k">{med_a}</span><span class="l">median reply words</span></div>
</div>
<h2 class="sec">traits in this part &mdash; open one to read its full text</h2>
{index}
</header>
<div class="controls">
<div class="grp">{topts}</div>
<div class="grp">{topdoms}</div>
<input type="search" id="q" placeholder="search every word&hellip;" aria-label="search the corpus">
<span class="count" id="count">{len(rows):,} shown</span>
</div>
<main id="list">{''.join(cards)}</main>
<p class="empty" id="empty" hidden>No documents match those filters.</p>
<footer class="col">
<p>Source: <code>output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl</code>. Each row is a
three-turn SFT example (system, user, assistant) plus the scenario metadata shown above it.
Trait text appears once in the index rather than on all {len(rows):,} rows &mdash; it is
identical within a trait, and repeating it would put this page over the size limit.</p>
<p>The corpus is {all_rows:,} documents in total and does not fit one page: its text alone
is 17.4&nbsp;MB against a 16&nbsp;MB ceiling, so it is split by trait across two artifacts
with every field intact rather than truncated.</p>
<p>Domain labels are the generator's own free text and arrive with inconsistent casing
({len(Counter(r['metadata']['domain'] for r in rows))} raw spellings collapse to {n_dom});
the quick filters use the normalised form, and search matches either.</p>
</footer>
</div>
<script>
(function(){{
var docs=[].slice.call(document.querySelectorAll('.doc'));
var q=document.getElementById('q'),count=document.getElementById('count'),
empty=document.getElementById('empty');
var hay=docs.map(function(d){{return d.textContent.toLowerCase()}});
var st={{t:'',dom:'',text:''}};
function apply(){{
var n=0;
for(var i=0;i<docs.length;i++){{
var d=docs[i];
var ok=(!st.t||d.dataset.t===st.t)&&(!st.dom||d.dataset.dom===st.dom)
&&(!st.text||hay[i].indexOf(st.text)!==-1);
d.hidden=!ok;if(ok)n++;
}}
count.textContent=n.toLocaleString()+' shown';empty.hidden=n!==0;
}}
document.querySelectorAll('.controls .grp').forEach(function(g){{
g.addEventListener('click',function(e){{
var b=e.target.closest('button');if(!b)return;
var key='t' in b.dataset?'t':'dom';
var on=b.getAttribute('aria-pressed')==='true';
g.querySelectorAll('button').forEach(function(x){{x.setAttribute('aria-pressed','false')}});
b.setAttribute('aria-pressed',String(!on));
st[key]=on?'':(b.dataset[key]);
apply();
}});
}});
var t;q.addEventListener('input',function(){{clearTimeout(t);
t=setTimeout(function(){{st.text=q.value.trim().toLowerCase();apply()}},140)}});
}})();
</script>
"""


def main(out: str = "", traits="", part: str = "", other: str = "") -> None:
    """Render the trait-corpus browser to a self-contained HTML file.

    Args:
        out: Output path.
        traits: Comma-separated trait ids to include (default: all).
        part: Label such as "part 1 of 2", shown in the eyebrow.
        other: URL of the companion artifact, linked from the standfirst.
    """
    all_json = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    # fire turns `--traits t1,t2` into a tuple and `--traits t1` into a str.
    seq = traits if isinstance(traits, (list, tuple)) else str(traits).split(",")
    keep = {str(t).strip() for t in seq if str(t).strip()}
    rows = [r for r in all_json if not keep or r["metadata"]["trait_id"] in keep]
    assert rows, f"no rows for traits={traits!r}"
    path = Path(out) if out else REPO / "output/synthdoc_v2/trait_corpus.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    part_s = f" &middot; {part}" if part else ""
    other_s = (f' <a href="{other}">The other {len(all_json) - len(rows):,} documents '
               f'are here.</a>') if other else ""
    path.write_text(build(rows, part_s, other_s, len(all_json)), encoding="utf-8")
    mb = path.stat().st_size / 1048576
    print(f">>> {path}  ({len(rows):,} documents, {mb:.1f} MB)")
    assert mb < 16, f"{mb:.1f} MB exceeds the 16 MB artifact limit"


if __name__ == "__main__":
    fire.Fire(main)
