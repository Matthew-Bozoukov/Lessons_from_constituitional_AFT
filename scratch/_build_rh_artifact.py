# ABOUTME: Render scratch/reward_hacking_da_seeds.jsonl as a review page — the corpus rows
# ABOUTME: laid out as a critical edition: apparatus in the rail, transcript in the text column.
import html
import json
from pathlib import Path

SRC = Path("scratch/reward_hacking_da_seeds.jsonl")
OUT = Path("/tmp/claude-1000/-home-matthewb-git-repos-teaching-claude-why-replication/"
           "3340b89e-6b54-4cc7-bd0e-ab0946cd182a/scratchpad/refusing-the-shortcut.html")

def paras(text: str) -> str:
    return "\n".join(f"<p>{html.escape(b.strip())}</p>"
                     for b in text.split("\n\n") if b.strip())

def main() -> None:
    rows = [json.loads(l) for l in SRC.open(encoding="utf-8")]
    cards, nav = [], []
    for i, r in enumerate(rows, 1):
        m = r["metadata"]
        sysm, userm, asst = r["messages"]
        nav.append(f'<a href="#ex{i}"><span class="n">{i}</span>'
                   f'<span class="t">{html.escape(m["domain"])}</span>'
                   f'<span class="k">{m["trait_id"]}</span></a>')
        cards.append(f"""
<article class="ex" id="ex{i}">
  <aside class="rail">
    <div class="apex">{i}</div>
    <dl>
      <dt>scenario_id</dt><dd class="mono">{html.escape(m["scenario_id"])}</dd>
      <dt>principle</dt><dd><span class="chip">{m["trait_id"]}</span>
        <span class="tname">{html.escape(m["trait_name"])}</span></dd>
      <dt>domain</dt><dd>{html.escape(m["domain"])}</dd>
      <dt>asking</dt><dd><span class="who who-{m['asker']}">{m["asker"]}</span></dd>
    </dl>
    <div class="shortcut">
      <span class="lbl">the shortcut being refused</span>
      <p>{html.escape(m["shortcut"])}</p>
    </div>
  </aside>
  <div class="text">
    <div class="turn sys"><span class="role">system</span>{paras(sysm["content"])}</div>
    <div class="turn usr"><span class="role">user</span>{paras(userm["content"])}</div>
    <div class="turn think" data-reasoning>
      <span class="role">reasoning_content
        <span class="ct mono">{len(asst["reasoning_content"]):,} ch</span></span>
      {paras(asst["reasoning_content"])}
    </div>
    <div class="turn ast"><span class="role">content
      <span class="ct mono">{len(asst["content"]):,} ch</span></span>{paras(asst["content"])}</div>
  </div>
</article>""")

    doc = f"""<title>Refusing the Shortcut</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,300;0,400;0,600;1,400&family=Archivo+Narrow:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --paper:#EDEFEE; --surface:#F7F8F7; --sunk:#E4E8E5;
  --ink:#16201E; --muted:#5C6B66; --rule:#D2D8D4;
  --accent:#1B6B58; --accent-soft:#DCE9E4;
  --flag:#9E3826; --flag-soft:#F0DED9;
  --serif:"Spectral",Georgia,"Times New Roman",serif;
  --sans:"Archivo Narrow","Helvetica Neue",Arial,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Menlo,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#111615; --surface:#181F1D; --sunk:#141A19;
    --ink:#E3E9E6; --muted:#94A39D; --rule:#2A3431;
    --accent:#5CB59B; --accent-soft:#1B2B27;
    --flag:#D9836E; --flag-soft:#2E211E;
  }}
}}
:root[data-theme="dark"] {{
  --paper:#111615; --surface:#181F1D; --sunk:#141A19;
  --ink:#E3E9E6; --muted:#94A39D; --rule:#2A3431;
  --accent:#5CB59B; --accent-soft:#1B2B27;
  --flag:#D9836E; --flag-soft:#2E211E;
}}
*{{box-sizing:border-box}}
body{{background:var(--paper);color:var(--ink);font-family:var(--serif);
  font-weight:300;line-height:1.62;margin:0;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1180px;margin:0 auto;padding:0 28px 96px}}
header.top{{padding:56px 0 30px;border-bottom:1px solid var(--rule)}}
h1{{font-size:clamp(2.1rem,4.4vw,3.1rem);font-weight:600;letter-spacing:-.018em;
  line-height:1.06;margin:0 0 14px;text-wrap:balance}}
.dek{{font-size:1.12rem;color:var(--muted);max-width:64ch;margin:0 0 26px}}
.prov{{display:flex;flex-wrap:wrap;gap:8px 26px;font-family:var(--sans);
  font-size:.83rem;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}}
.prov b{{color:var(--ink);font-weight:600}}
.prov code{{font-family:var(--mono);text-transform:none;letter-spacing:0;font-size:.86em}}
nav.index{{display:grid;grid-template-columns:repeat(auto-fit,minmax(186px,1fr));
  gap:0;border-bottom:1px solid var(--rule);margin-bottom:8px}}
nav.index a{{display:flex;align-items:baseline;gap:9px;padding:16px 14px 16px 0;
  text-decoration:none;color:var(--ink);border-right:1px solid var(--rule)}}
nav.index a:last-child{{border-right:0}}
nav.index a:hover .t{{color:var(--accent)}}
nav.index .n{{font-family:var(--mono);font-size:.78rem;color:var(--accent)}}
nav.index .t{{font-family:var(--sans);font-size:.93rem;font-weight:600;line-height:1.25}}
nav.index .k{{font-family:var(--mono);font-size:.72rem;color:var(--muted);margin-left:auto}}
.controls{{display:flex;justify-content:flex-end;padding:14px 0 0}}
button#tog{{font-family:var(--sans);font-size:.82rem;font-weight:600;letter-spacing:.05em;
  text-transform:uppercase;color:var(--ink);background:var(--surface);
  border:1px solid var(--rule);border-radius:2px;padding:8px 14px;cursor:pointer}}
button#tog:hover{{border-color:var(--accent);color:var(--accent)}}
button#tog:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
.ex{{display:grid;grid-template-columns:255px minmax(0,1fr);gap:44px;
  padding:52px 0;border-bottom:1px solid var(--rule);scroll-margin-top:18px}}
.rail{{position:sticky;top:24px;align-self:start}}
.apex{{font-family:var(--mono);font-size:.8rem;color:var(--accent);margin-bottom:18px}}
.rail dl{{margin:0 0 22px;display:grid;gap:13px}}
.rail dt{{font-family:var(--sans);font-size:.72rem;letter-spacing:.09em;
  text-transform:uppercase;color:var(--muted);margin-bottom:3px}}
.rail dd{{margin:0;font-size:.93rem;line-height:1.45}}
.mono{{font-family:var(--mono);font-size:.84rem}}
.chip{{font-family:var(--mono);font-size:.76rem;color:var(--accent);
  background:var(--accent-soft);padding:2px 6px;border-radius:2px;margin-right:6px}}
.tname{{font-size:.9rem}}
.who{{font-family:var(--sans);font-size:.78rem;font-weight:600;letter-spacing:.06em;
  text-transform:uppercase;padding:3px 8px;border-radius:2px}}
.who-human{{background:var(--sunk);color:var(--ink)}}
.who-model{{background:var(--accent-soft);color:var(--accent)}}
.shortcut{{border-left:2px solid var(--flag);background:var(--flag-soft);padding:13px 15px}}
.shortcut .lbl{{display:block;font-family:var(--sans);font-size:.7rem;letter-spacing:.09em;
  text-transform:uppercase;color:var(--flag);margin-bottom:7px;font-weight:600}}
.shortcut p{{margin:0;font-size:.9rem;line-height:1.5}}
.text{{min-width:0;max-width:68ch}}
.turn{{margin-bottom:30px}}
.turn .role{{display:flex;align-items:baseline;gap:10px;font-family:var(--mono);
  font-size:.76rem;color:var(--muted);margin-bottom:9px}}
.turn .ct{{color:var(--muted);opacity:.72;font-size:.72rem}}
.turn p{{margin:0 0 .95em}}
.turn p:last-child{{margin-bottom:0}}
.sys p,.usr p{{font-size:.98rem}}
.sys{{color:var(--muted);border-bottom:1px solid var(--rule);padding-bottom:20px}}
.usr p{{border-left:2px solid var(--rule);padding-left:17px;margin-left:0}}
.think{{background:var(--sunk);border-left:2px solid var(--accent);
  padding:18px 20px;font-size:.95rem}}
.think p{{color:var(--muted)}}
.ast p{{font-size:1.045rem}}
.ast .role{{color:var(--accent)}}
body.hide-reasoning [data-reasoning]{{display:none}}
footer{{padding:40px 0 0;font-family:var(--sans);font-size:.85rem;color:var(--muted);
  max-width:70ch}}
footer code{{font-family:var(--mono);font-size:.86em;color:var(--ink)}}
@media (max-width:900px){{
  .ex{{grid-template-columns:1fr;gap:26px}}
  .rail{{position:static}}
  nav.index a{{border-right:0;border-bottom:1px solid var(--rule)}}
}}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}}}
</style>

<div class="wrap">
<header class="top">
  <h1>Refusing the Shortcut</h1>
  <p class="dek">Five difficult-advice rows where someone wants to cheat to get ahead &mdash;
  a place at Harvard, a first-author line, an offer that doubles their pay, a passing eval
  score &mdash; and their reasoning is half right.</p>
  <div class="prov">
    <span><b>5</b> rows</span>
    <span><b>5</b> principles &middot; t1 t3 t4 t8 t9</span>
    <span><b>3</b> human &middot; <b>2</b> model</span>
    <span>schema of <code>2026-08-21-difficult-advice-v2-chunk-only-716</code></span>
  </div>
</header>

<nav class="index">{"".join(nav)}</nav>
<div class="controls"><button id="tog" type="button">Hide reasoning</button></div>

{"".join(cards)}

<footer>
  <p>Written to the source corpus&rsquo;s own shape: a system prompt establishing the
  deployment, one user turn carrying the ask, and an assistant turn with a separate
  <code>reasoning_content</code> trace. Lengths track the 708-row corpus &mdash; reply
  median 2,611&nbsp;characters against its 2,751, reasoning 2,953 against 2,920.
  <code>trait_text</code> is read from that corpus at build time rather than retyped, so each
  row is written against the same principle string the originals were.</p>
  <p>Generated by <code>scratch/_gen_reward_hacking_da.py</code> &rarr;
  <code>scratch/reward_hacking_da_seeds.jsonl</code>. Candidate training data, not yet reviewed
  or mixed.</p>
</footer>
</div>

<script>
const b=document.body,t=document.getElementById('tog');
t.addEventListener('click',()=>{{
  const hidden=b.classList.toggle('hide-reasoning');
  t.textContent=hidden?'Show reasoning':'Hide reasoning';
  t.setAttribute('aria-pressed',String(hidden));
}});
</script>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT} ({len(doc):,} chars, {len(rows)} rows)")

if __name__ == "__main__":
    main()
