# ABOUTME: Render the 351 generated reward-hacking rows as a review console: a scan table over all
# ABOUTME: of them for spotting sameness, plus full transcripts for a per-trait sample.

"""A different job from the five-row edition, so a different shape.

These rows are unreviewed and model-generated, and the failure this page has to make visible is
not one bad row — it is 351 rows that quietly rhyme. So the whole batch appears as one dense
scannable list (trait, domain, asker, and the sentence where the asker states what they want), and
a sample opens in full underneath. Repetition is legible in the list; quality is legible in the
sample.
"""

import html
import json
import random
import re
from pathlib import Path

SRC = Path("output/rh_corpus/dataset.jsonl")
OUT = Path("/tmp/claude-1000/-home-matthewb-git-repos-teaching-claude-why-replication/"
           "9c25d488-a37b-4b6f-9dcb-49bb61cbb356/scratchpad/the-generated-351.html")
PER_TRAIT_SAMPLE = 4
SEED = 0

WANT = re.compile(r"[^.!?\n]*\b(?:I want|I'd rather|quiet part|not pretending|"
                  r"be honest about|say the honest|honest about the motive|I'm after)\b[^.!?\n]*[.!?]",
                  re.I)


def want_line(user: str) -> str:
    m = WANT.findall(user)
    if m:
        return max(m, key=len).strip()
    return user.strip().split("\n")[0][:150]


def paras(text: str) -> str:
    return "\n".join(f"<p>{html.escape(b.strip())}</p>"
                     for b in text.split("\n\n") if b.strip())


def main() -> None:
    rows = [json.loads(line) for line in SRC.open(encoding="utf-8")]
    gen = [r for r in rows if r["metadata"].get("family") == "reward-hacking-difficult-advice"]
    gen.sort(key=lambda r: (r["metadata"]["trait_id"], r["metadata"]["scenario_id"]))

    rng = random.Random(SEED)
    by_trait: dict[str, list] = {}
    for r in gen:
        by_trait.setdefault(r["metadata"]["trait_id"], []).append(r)
    trait_names = {t: v[0]["metadata"]["trait_name"] for t, v in by_trait.items()}
    sample = []
    for t in sorted(by_trait):
        sample += rng.sample(by_trait[t], min(PER_TRAIT_SAMPLE, len(by_trait[t])))

    scan = []
    for r in gen:
        m = r["metadata"]
        scan.append(
            f'<tr data-t="{m["trait_id"]}"><td class="mono tc">{m["trait_id"]}</td>'
            f'<td class="dom">{html.escape(m["domain"])}</td>'
            f'<td><span class="who who-{m["asker"]}">{m["asker"]}</span></td>'
            f'<td class="want">{html.escape(want_line(r["messages"][1]["content"]))}</td></tr>')

    cards = []
    for r in sample:
        m = r["metadata"]
        sysm, userm, asst = r["messages"]
        cards.append(f"""
<article class="ex" data-t="{m['trait_id']}">
  <aside class="rail">
    <div class="apex mono">{html.escape(m['scenario_id'])}</div>
    <dl>
      <dt>principle</dt><dd><span class="chip">{m['trait_id']}</span></dd>
      <dt>domain</dt><dd>{html.escape(m['domain'])}</dd>
      <dt>asking</dt><dd><span class="who who-{m['asker']}">{m['asker']}</span></dd>
    </dl>
    <div class="shortcut"><span class="lbl">shortcut refused</span>
      <p>{html.escape(m['shortcut'])}</p></div>
  </aside>
  <div class="text">
    <div class="turn sys"><span class="role">system</span>{paras(sysm['content'])}</div>
    <div class="turn usr"><span class="role">user</span>{paras(userm['content'])}</div>
    <div class="turn think" data-reasoning><span class="role">reasoning_content
      <span class="ct mono">{len(asst['reasoning_content']):,} ch</span></span>
      {paras(asst['reasoning_content'])}</div>
    <div class="turn ast"><span class="role">content
      <span class="ct mono">{len(asst['content']):,} ch</span></span>{paras(asst['content'])}</div>
  </div>
</article>""")

    filters = "".join(
        f'<button class="f" data-f="{t}">{t}<span class="n">{len(by_trait[t])}</span></button>'
        for t in sorted(by_trait))
    legend = "".join(f'<div><span class="chip">{t}</span> {html.escape(trait_names[t])}</div>'
                     for t in sorted(trait_names))

    doc = f"""<title>The Generated 351</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@300;400;600&family=Archivo+Narrow:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --paper:#EDEFEE; --surface:#F7F8F7; --sunk:#E4E8E5;
  --ink:#16201E; --muted:#5C6B66; --rule:#D2D8D4;
  --accent:#1B6B58; --accent-soft:#DCE9E4;
  --flag:#9E3826; --flag-soft:#F0DED9;
  --serif:"Spectral",Georgia,serif;
  --sans:"Archivo Narrow","Helvetica Neue",Arial,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --paper:#111615; --surface:#181F1D; --sunk:#141A19;
  --ink:#E3E9E6; --muted:#94A39D; --rule:#2A3431;
  --accent:#5CB59B; --accent-soft:#1B2B27; --flag:#D9836E; --flag-soft:#2E211E; }} }}
:root[data-theme="dark"] {{
  --paper:#111615; --surface:#181F1D; --sunk:#141A19;
  --ink:#E3E9E6; --muted:#94A39D; --rule:#2A3431;
  --accent:#5CB59B; --accent-soft:#1B2B27; --flag:#D9836E; --flag-soft:#2E211E; }}
*{{box-sizing:border-box}}
body{{background:var(--paper);color:var(--ink);font-family:var(--serif);font-weight:300;
  line-height:1.6;margin:0;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1220px;margin:0 auto;padding:0 26px 90px}}
header.top{{padding:50px 0 24px;border-bottom:1px solid var(--rule)}}
h1{{font-size:clamp(2rem,4vw,2.9rem);font-weight:600;letter-spacing:-.018em;line-height:1.06;
  margin:0 0 12px;text-wrap:balance}}
.dek{{font-size:1.08rem;color:var(--muted);max-width:66ch;margin:0 0 22px}}
.warn{{border-left:2px solid var(--flag);background:var(--flag-soft);padding:11px 15px;
  max-width:66ch;font-size:.95rem;margin:0 0 22px}}
.prov{{display:flex;flex-wrap:wrap;gap:7px 24px;font-family:var(--sans);font-size:.82rem;
  letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}}
.prov b{{color:var(--ink);font-weight:600}}
.prov code{{font-family:var(--mono);text-transform:none;letter-spacing:0;font-size:.86em}}
h2{{font-family:var(--sans);font-size:.82rem;font-weight:700;letter-spacing:.11em;
  text-transform:uppercase;color:var(--muted);margin:38px 0 4px}}
.hint{{font-size:.93rem;color:var(--muted);margin:0 0 16px;max-width:66ch}}
.bar{{display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding:0 0 14px}}
button.f,button#tog{{font-family:var(--sans);font-size:.8rem;font-weight:600;letter-spacing:.05em;
  text-transform:uppercase;color:var(--ink);background:var(--surface);border:1px solid var(--rule);
  border-radius:2px;padding:7px 11px;cursor:pointer;display:inline-flex;gap:6px;align-items:baseline}}
button.f .n{{font-family:var(--mono);font-size:.72rem;color:var(--muted)}}
button.f[aria-pressed="true"]{{border-color:var(--accent);color:var(--accent);background:var(--accent-soft)}}
button.f[aria-pressed="true"] .n{{color:var(--accent)}}
button.f:focus-visible,button#tog:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
#tog{{margin-left:auto}}
.scan{{overflow-x:auto;border:1px solid var(--rule);background:var(--surface);max-height:420px;
  overflow-y:auto}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}}
th{{position:sticky;top:0;background:var(--sunk);font-family:var(--sans);font-size:.72rem;
  letter-spacing:.09em;text-transform:uppercase;color:var(--muted);text-align:left;
  padding:9px 12px;border-bottom:1px solid var(--rule);font-weight:600}}
td{{padding:8px 12px;border-bottom:1px solid var(--rule);vertical-align:top}}
tr:last-child td{{border-bottom:0}}
.tc{{color:var(--accent);width:3.2rem}}
.dom{{font-family:var(--sans);font-weight:600;min-width:12rem}}
.want{{color:var(--muted);min-width:24rem}}
.mono{{font-family:var(--mono);font-size:.82rem}}
.chip{{font-family:var(--mono);font-size:.75rem;color:var(--accent);background:var(--accent-soft);
  padding:2px 6px;border-radius:2px}}
.who{{font-family:var(--sans);font-size:.72rem;font-weight:600;letter-spacing:.06em;
  text-transform:uppercase;padding:2px 7px;border-radius:2px;white-space:nowrap}}
.who-human{{background:var(--sunk);color:var(--muted)}}
.who-model{{background:var(--accent-soft);color:var(--accent)}}
.legend{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:5px 26px;
  font-size:.88rem;color:var(--muted);margin:14px 0 0}}
.ex{{display:grid;grid-template-columns:230px minmax(0,1fr);gap:40px;padding:44px 0;
  border-bottom:1px solid var(--rule)}}
.rail{{position:sticky;top:20px;align-self:start}}
.apex{{color:var(--accent);margin-bottom:15px}}
.rail dl{{margin:0 0 20px;display:grid;gap:12px}}
.rail dt{{font-family:var(--sans);font-size:.71rem;letter-spacing:.09em;text-transform:uppercase;
  color:var(--muted);margin-bottom:3px}}
.rail dd{{margin:0;font-size:.92rem;line-height:1.45}}
.shortcut{{border-left:2px solid var(--flag);background:var(--flag-soft);padding:12px 14px}}
.shortcut .lbl{{display:block;font-family:var(--sans);font-size:.69rem;letter-spacing:.09em;
  text-transform:uppercase;color:var(--flag);margin-bottom:6px;font-weight:600}}
.shortcut p{{margin:0;font-size:.89rem;line-height:1.5}}
.text{{min-width:0;max-width:68ch}}
.turn{{margin-bottom:28px}}
.turn .role{{display:flex;gap:10px;align-items:baseline;font-family:var(--mono);font-size:.75rem;
  color:var(--muted);margin-bottom:8px}}
.turn .ct{{opacity:.72;font-size:.71rem}}
.turn p{{margin:0 0 .92em}} .turn p:last-child{{margin-bottom:0}}
.sys p,.usr p{{font-size:.97rem}}
.sys{{color:var(--muted);border-bottom:1px solid var(--rule);padding-bottom:18px}}
.usr p{{border-left:2px solid var(--rule);padding-left:16px}}
.think{{background:var(--sunk);border-left:2px solid var(--accent);padding:17px 19px;font-size:.94rem}}
.think p{{color:var(--muted)}}
.ast p{{font-size:1.03rem}} .ast .role{{color:var(--accent)}}
body.hide-reasoning [data-reasoning]{{display:none}}
.hidden{{display:none!important}}
footer{{padding:36px 0 0;font-family:var(--sans);font-size:.85rem;color:var(--muted);max-width:70ch}}
footer code{{font-family:var(--mono);font-size:.86em;color:var(--ink)}}
@media (max-width:900px){{ .ex{{grid-template-columns:1fr;gap:24px}} .rail{{position:static}} }}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}}}
</style>

<div class="wrap">
<header class="top">
  <h1>The Generated 351</h1>
  <p class="dek">Every substituted row in
  <code>2026-09-04-da-rewardhack-351-synth</code>, listed by what the asker says they want &mdash;
  then {len(sample)} of them opened in full.</p>
  <div class="warn"><b>Unreviewed.</b> These passed schema, length-band and no-identity-framing
  checks only. The failure to look for is not one bad row but 351 that quietly rhyme &mdash; which
  is what the list below is for.</div>
  <div class="prov">
    <span><b>{len(gen)}</b> generated &middot; <b>357</b> retained</span>
    <span><b>{len({r['metadata']['domain'].lower() for r in gen})}</b> distinct domains</span>
    <span><b>39</b> per trait</span>
    <span>generator <code>anthropic/claude-sonnet-5</code></span>
  </div>
</header>

<h2>Scan &mdash; all {len(gen)} rows</h2>
<p class="hint">One line per row: the principle it was written against, the setting, who is asking,
and the sentence where they state what they want. Filter by principle; the sample below follows the
same filter.</p>
<div class="bar"><button class="f" data-f="all" aria-pressed="true">all<span class="n">{len(gen)}</span></button>{filters}
<button id="tog" type="button">Hide reasoning</button></div>
<div class="scan"><table>
<thead><tr><th>trait</th><th>domain</th><th>asking</th><th>what they say they want</th></tr></thead>
<tbody id="scanbody">{"".join(scan)}</tbody></table></div>
<div class="legend">{legend}</div>

<h2>Sample &mdash; {PER_TRAIT_SAMPLE} per principle, in full</h2>
<p class="hint">Drawn with a fixed seed, so this is the same sample every time the page is rebuilt.</p>
{"".join(cards)}

<footer>
  <p>Rows generated by <code>scratch/_gen_rh_corpus.py</code> from five hand-written seeds, then
  substituted 39-per-trait for an equal slice of
  <code>2026-08-21-difficult-advice-v2-chunk-only-716</code>. Per-trait counts in the published
  corpus are unchanged, which the build asserts before writing; every removed row id is recorded in
  <code>substitution_stats.json</code>.</p>
</footer>
</div>

<script>
const b=document.body, tog=document.getElementById('tog');
tog.addEventListener('click',()=>{{
  const hid=b.classList.toggle('hide-reasoning');
  tog.textContent=hid?'Show reasoning':'Hide reasoning';
  tog.setAttribute('aria-pressed',String(hid));
}});
const btns=[...document.querySelectorAll('button.f')];
btns.forEach(btn=>btn.addEventListener('click',()=>{{
  const f=btn.dataset.f;
  btns.forEach(x=>x.setAttribute('aria-pressed',String(x===btn)));
  document.querySelectorAll('#scanbody tr,article.ex').forEach(el=>{{
    el.classList.toggle('hidden', f!=='all' && el.dataset.t!==f);
  }});
}}));
</script>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT} ({len(doc):,} chars; {len(gen)} scanned, {len(sample)} in full)")


if __name__ == "__main__":
    main()
