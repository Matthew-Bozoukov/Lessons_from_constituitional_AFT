# ABOUTME: Renders the difficult-advice corpus (all 2,155 graded samples) into a single
# ABOUTME: self-contained HTML case-file browser for publishing as an Artifact.
"""Build the difficult-advice corpus browser.

    uv run python scratch/build_da_corpus_artifact.py [--out <path.html>]

Every sample is written into the markup in full -- no lazy loading, no click-to-expand --
so the whole corpus is present in the DOM and findable with the browser's own search.
`content-visibility: auto` on each card lets the renderer skip off-screen work without
removing anything from the document.
"""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

import fire

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "output/difficult_advice_gen/run1p5m_20260727_105526"
# The reasoning traces live in a SEPARATE export of the same generation: the graded
# corpus stores only the final answer. Verified 2026-08-26 -- joining on user_message
# matches all 2,119 accepted rows and their final answers are byte-identical, so this is
# the same sample with its deliberation restored. The 36 rejected rows were never
# carried into that export and have no trace; they are shown saying so.
THINK = REPO / "output/difficult_advice_gen/think_20260727_135323/sft_dataset_thinking.jsonl"

CRITERIA = [
    ("engages", "engages", True),
    ("declines_violation", "declines", True),
    ("deliberates_values", "deliberates", True),
    ("offers_alternative", "alternative", True),
    ("respects_autonomy", "autonomy", True),
    ("preachy", "preachy", False),   # desirable value is False
]


def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""), quote=True)


def _paras(text: str) -> str:
    out = []
    for block in str(text).split("\n\n"):
        block = block.strip()
        if block:
            out.append("<p>" + _esc(block).replace("\n", "<br>") + "</p>")
    return "\n".join(out) or "<p class='void'>(empty)</p>"


def _words(text) -> int:
    """Whitespace-delimited word count -- the same thing a reader would eyeball."""
    return len(str(text or "").split())


def load_reasoning() -> dict[str, str]:
    """{user_message: reasoning_content} from the thinking export."""
    out = {}
    for line in THINK.read_text().splitlines():
        if not line.strip():
            continue
        msgs = json.loads(line)["messages"]
        user = next(m for m in msgs if m["role"] == "user")["content"]
        asst = next(m for m in msgs if m["role"] == "assistant")
        if asst.get("reasoning_content"):
            out[user.strip()] = asst["reasoning_content"]
    return out


def build(rows: list[dict], meta: dict, reasoning: dict[str, str]) -> str:
    cfg = meta.get("config", {})
    domains = sorted({r["domain"] for r in rows})
    dom_counts = Counter(r["domain"] for r in rows)
    n_acc = sum(1 for r in rows if r.get("accepted"))
    scores = [r["grade"].get("overall_score") for r in rows if r.get("grade")]
    scores = [s for s in scores if isinstance(s, int)]
    n_preachy = sum(1 for r in rows if (r.get("grade") or {}).get("preachy"))
    n_think = sum(1 for r in rows if reasoning.get(r["user_message"].strip()))
    w_resp = sorted(_words(r.get("response")) for r in rows)
    w_think = sorted(_words(t) for t in
                     (reasoning.get(r["user_message"].strip()) for r in rows) if t)
    med_resp = w_resp[len(w_resp) // 2]
    med_think = w_think[len(w_think) // 2] if w_think else 0
    n_nodecline = sum(1 for r in rows
                      if not (r.get("grade") or {}).get("declines_violation", True))

    cards = []
    for i, r in enumerate(rows, 1):
        g = r.get("grade") or {}
        score = g.get("overall_score")
        acc = bool(r.get("accepted"))
        chips = []
        for key, label, want in CRITERIA:
            val = g.get(key)
            good = (val is want)
            state = "pass" if good else "fail"
            mark = "✓" if good else "✕"
            chips.append(
                f'<span class="chip {state}"><span class="mark">{mark}</span>{_esc(label)}</span>')
        trace = reasoning.get(r["user_message"].strip())
        if trace:
            think_block = f"""<div class="turn think">
    <div class="who">deliberation &mdash; trains as reasoning<span class="wc">{_words(trace):,} words</span></div>
    <div class="prose">{_paras(trace)}</div>
  </div>"""
        else:
            think_block = ("""<div class="turn think none">
    <div class="who">deliberation</div>
    <p class="prose void">Not captured &mdash; this sample was rejected, and only accepted
    samples were carried into the reasoning export.</p>
  </div>""")
        band = "hi" if isinstance(score, int) and score >= 9 else (
            "mid" if isinstance(score, int) and score >= 7 else "lo")
        cards.append(f"""
<article class="case" data-domain="{_esc(r['domain'])}" data-acc="{'y' if acc else 'n'}"
         data-band="{band}" data-think="{'y' if trace else 'n'}" id="c{i}">
  <header class="case-head">
    <div class="case-id">
      <span class="num">{i:04d}</span>
      <span class="dom">{_esc(r['domain'].replace('_', ' '))}</span>
    </div>
    <div class="case-marks">
      <span class="score {band}" title="rubric overall score">{_esc(score if score is not None else '—')}<span class="of">/10</span></span>
      <span class="verdict {'acc' if acc else 'rej'}">{'accepted' if acc else 'rejected'}</span>
    </div>
  </header>
  <h2 class="label">{_esc(r.get('label'))}</h2>
  <p class="temptation"><span class="tlabel">pressure</span>{_esc(r.get('temptation'))}</p>
  <div class="turn user">
    <div class="who">the person asking<span class="wc">{_words(r.get('user_message')):,} words</span></div>
    <div class="prose">{_paras(r.get('user_message'))}</div>
  </div>
  {think_block}
  <div class="turn asst">
    <div class="who">the reply<span class="wc">{_words(r.get('response')):,} words</span></div>
    <div class="prose">{_paras(r.get('response'))}</div>
  </div>
  <footer class="grade">
    <div class="chips">{''.join(chips)}</div>
    <p class="reason">{_esc(g.get('reason'))}</p>
  </footer>
</article>""")

    opts = "\n".join(
        f'<option value="{_esc(d)}">{_esc(d.replace("_", " "))} ({dom_counts[d]})</option>'
        for d in domains)

    return f"""<title>Difficult Advice Corpus</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>
:root {{
  --paper:#F1F3EF; --surface:#FFFFFF; --sunk:#E9ECE6;
  --ink:#191D1B; --ink-2:#3C433E; --muted:#6C736E; --line:#DCE1D8;
  --accent:#7C6A3A; --accent-soft:#EFE9D9;
  --pass:#3D6B54; --fail:#A34433; --flag:#94701C;
  --pass-bg:#E4EDE7; --fail-bg:#F6E3DF; --flag-bg:#F5ECD8;
  --sans:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;
  --serif:"Source Serif 4",Georgia,"Times New Roman",serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#101311; --surface:#171B18; --sunk:#1D221E;
    --ink:#E6EAE5; --ink-2:#C3CAC3; --muted:#939A94; --line:#282E29;
    --accent:#C0A868; --accent-soft:#2A2519;
    --pass:#7FB496; --fail:#DE8873; --flag:#D6AC55;
    --pass-bg:#1B2721; --fail-bg:#2C1D19; --flag-bg:#2A2314;
  }}
}}
:root[data-theme="dark"] {{
  --paper:#101311; --surface:#171B18; --sunk:#1D221E;
  --ink:#E6EAE5; --ink-2:#C3CAC3; --muted:#939A94; --line:#282E29;
  --accent:#C0A868; --accent-soft:#2A2519;
  --pass:#7FB496; --fail:#DE8873; --flag:#D6AC55;
  --pass-bg:#1B2721; --fail-bg:#2C1D19; --flag-bg:#2A2314;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.55;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1080px;margin:0 auto;padding:0 20px 96px}}
a{{color:var(--accent)}}
:focus-visible{{outline:2px solid var(--accent);outline-offset:2px;border-radius:3px}}

header.masthead{{padding:56px 0 28px;border-bottom:1px solid var(--line)}}
.eyebrow{{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted);margin:0 0 14px}}
h1{{font-family:var(--serif);font-weight:600;font-size:clamp(30px,4.4vw,46px);line-height:1.1;
  margin:0 0 14px;text-wrap:balance;letter-spacing:-.01em}}
.standfirst{{margin:0;max-width:63ch;color:var(--ink-2);font-family:var(--serif);font-size:18px}}

.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));gap:1px;
  background:var(--line);border:1px solid var(--line);margin:30px 0 0}}
.stat{{background:var(--surface);padding:14px 16px}}
.stat .k{{font-family:var(--mono);font-size:23px;font-variant-numeric:tabular-nums;
  color:var(--ink);display:block;line-height:1.2}}
.stat .l{{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}}

.controls{{position:sticky;top:0;z-index:5;background:var(--paper);
  border-bottom:1px solid var(--line);padding:12px 0;margin-bottom:8px;
  display:flex;flex-wrap:wrap;gap:10px;align-items:center}}
.controls label{{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}}
select,input[type=search]{{font-family:var(--sans);font-size:14px;color:var(--ink);
  background:var(--surface);border:1px solid var(--line);border-radius:2px;padding:7px 9px}}
input[type=search]{{min-width:220px;flex:1 1 220px}}
.seg{{display:flex;border:1px solid var(--line);border-radius:2px;overflow:hidden}}
.seg button{{font-family:var(--sans);font-size:13px;color:var(--ink-2);background:var(--surface);
  border:0;padding:7px 12px;cursor:pointer;border-right:1px solid var(--line)}}
.seg button:last-child{{border-right:0}}
.seg button[aria-pressed="true"]{{background:var(--accent-soft);color:var(--ink);font-weight:600}}
.count{{margin-left:auto;font-family:var(--mono);font-size:12px;color:var(--muted);
  font-variant-numeric:tabular-nums}}

.case{{background:var(--surface);border:1px solid var(--line);margin:20px 0;padding:22px 24px 18px;
  content-visibility:auto;contain-intrinsic-size:0 980px}}
.case[hidden]{{display:none}}
.case-head{{display:flex;justify-content:space-between;align-items:baseline;gap:14px;
  flex-wrap:wrap;padding-bottom:10px;border-bottom:1px solid var(--line)}}
.case-id{{display:flex;align-items:baseline;gap:12px;min-width:0}}
.num{{font-family:var(--mono);font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}}
.dom{{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);font-weight:600}}
.case-marks{{display:flex;align-items:center;gap:10px}}
.score{{font-family:var(--mono);font-size:15px;font-variant-numeric:tabular-nums;font-weight:500}}
.score .of{{font-size:11px;color:var(--muted)}}
.score.hi{{color:var(--pass)}} .score.mid{{color:var(--flag)}} .score.lo{{color:var(--fail)}}
.verdict{{font-size:10px;letter-spacing:.1em;text-transform:uppercase;padding:3px 8px;border-radius:2px}}
.verdict.acc{{background:var(--pass-bg);color:var(--pass)}}
.verdict.rej{{background:var(--fail-bg);color:var(--fail)}}
h2.label{{font-family:var(--serif);font-size:21px;font-weight:600;line-height:1.28;
  margin:14px 0 6px;text-wrap:balance}}
.temptation{{margin:0 0 18px;font-size:14px;color:var(--muted)}}
.tlabel{{font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--accent);margin-right:9px}}
.turn{{padding:14px 0;border-top:1px solid var(--line)}}
.who{{font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--muted);margin-bottom:8px;display:flex;align-items:baseline;gap:12px}}
.wc{{margin-left:auto;letter-spacing:.06em;color:var(--muted);opacity:.85;
  font-variant-numeric:tabular-nums;white-space:nowrap}}
.turn.user .prose{{color:var(--ink-2)}}
.turn.think{{border-left:2px solid var(--accent);padding-left:16px}}
.turn.think .prose{{font-size:15.5px;color:var(--muted);font-style:italic;line-height:1.58}}
.turn.think.none .prose{{font-size:14px}}
.turn.user{{background:var(--sunk);margin:0 -24px;padding:14px 24px}}
.prose{{font-family:var(--serif);font-size:16.5px;line-height:1.62;max-width:66ch}}
.prose p{{margin:0 0 .82em}} .prose p:last-child{{margin-bottom:0}}
.prose .void{{color:var(--muted);font-style:italic}}
.grade{{border-top:1px solid var(--line);padding-top:13px;margin-top:4px}}
.chips{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:9px}}
.chip{{font-size:11px;letter-spacing:.03em;padding:3px 8px;border-radius:2px;
  display:inline-flex;align-items:center;gap:5px}}
.chip .mark{{font-size:10px}}
.chip.pass{{background:var(--pass-bg);color:var(--pass)}}
.chip.fail{{background:var(--fail-bg);color:var(--fail)}}
.reason{{margin:0;font-size:13.5px;color:var(--muted);max-width:74ch}}
.empty{{padding:60px 0;text-align:center;color:var(--muted);font-family:var(--serif);font-size:18px}}
footer.colophon{{margin-top:44px;padding-top:20px;border-top:1px solid var(--line);
  font-size:13px;color:var(--muted);max-width:70ch}}
footer.colophon code{{font-family:var(--mono);font-size:12px;color:var(--ink-2)}}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}}}
@media (max-width:640px){{
  .turn.user{{margin:0 -24px;padding:14px 24px}}
  .case{{padding:18px 16px 14px}} .turn.user{{margin:0 -16px;padding:13px 16px}}
}}
</style>

<div class="wrap">
<header class="masthead">
  <p class="eyebrow">synthetic training corpus · generated {_esc(meta.get('timestamp_utc', '')[:10])}</p>
  <h1>Difficult Advice</h1>
  <p class="standfirst">Someone under real professional pressure asks for help doing
  something they shouldn't. {len(rows):,} generated exchanges across {len(domains)} regulated
  domains, each graded on whether the reply engages with the person, declines the
  violation, and offers them somewhere else to go.</p>

  <div class="stats">
    <div class="stat"><span class="k">{len(rows):,}</span><span class="l">samples</span></div>
    <div class="stat"><span class="k">{len(domains)}</span><span class="l">domains</span></div>
    <div class="stat"><span class="k">{n_acc:,}</span><span class="l">accepted</span></div>
    <div class="stat"><span class="k">{len(rows)-n_acc}</span><span class="l">rejected</span></div>
    <div class="stat"><span class="k">{sum(scores)/len(scores):.1f}</span><span class="l">mean score</span></div>
    <div class="stat"><span class="k">{n_nodecline}</span><span class="l">didn't decline</span></div>
    <div class="stat"><span class="k">{n_preachy}</span><span class="l">flagged preachy</span></div>
    <div class="stat"><span class="k">{n_think:,}</span><span class="l">with reasoning</span></div>
    <div class="stat"><span class="k">{med_think}</span><span class="l">median reasoning words</span></div>
    <div class="stat"><span class="k">{med_resp}</span><span class="l">median reply words</span></div>
  </div>
</header>

<div class="controls">
  <label for="dom">domain</label>
  <select id="dom"><option value="">all {len(domains)}</option>{opts}</select>
  <div class="seg" role="group" aria-label="verdict">
    <button data-acc="" aria-pressed="true">all</button>
    <button data-acc="y" aria-pressed="false">accepted</button>
    <button data-acc="n" aria-pressed="false">rejected</button>
  </div>
  <div class="seg" role="group" aria-label="score band">
    <button data-band="" aria-pressed="true">any score</button>
    <button data-band="hi" aria-pressed="false">9–10</button>
    <button data-band="mid" aria-pressed="false">7–8</button>
    <button data-band="lo" aria-pressed="false">≤6</button>
  </div>
  <div class="seg" role="group" aria-label="reasoning">
    <button data-think="" aria-pressed="true">any</button>
    <button data-think="y" aria-pressed="false">has reasoning</button>
    <button data-think="n" aria-pressed="false">no reasoning</button>
  </div>
  <input type="search" id="q" placeholder="search every word of every sample…"
         aria-label="search the corpus">
  <span class="count" id="count">{len(rows):,} shown</span>
</div>

<main id="list">
{''.join(cards)}
</main>
<p class="empty" id="empty" hidden>No samples match those filters.</p>

<footer class="colophon">
  <p>Generated with <code>{_esc(cfg.get('gen_model', '?'))}</code> and graded by
  <code>{_esc(cfg.get('grade_model', '?'))}</code>, {_esc(cfg.get('scenarios_per_domain', '?'))}
  scenarios per domain, seed {_esc(cfg.get('seed', '?'))}, response temperature
  {_esc((cfg.get('response') or {{}}).get('temperature', '?'))}. Source:
  <code>output/difficult_advice_gen/run1p5m_20260727_105526/all_records.jsonl</code>.</p>
  <p>The six chips under each sample are the grader's rubric. <em>preachy</em> is the one
  criterion where a mark means the reply went wrong — it is shown in the failing colour
  when true, so a row of green always reads as good.</p>
</footer>
</div>

<script>
(function () {{
  var cases = Array.prototype.slice.call(document.querySelectorAll('.case'));
  var dom = document.getElementById('dom');
  var q = document.getElementById('q');
  var count = document.getElementById('count');
  var empty = document.getElementById('empty');
  var state = {{ domain: '', acc: '', band: '', think: '', text: '' }};

  // Cache lowercased text once; re-reading textContent per keystroke on 2,155 full
  // samples is what would make typing feel laggy.
  var hay = cases.map(function (c) {{ return c.textContent.toLowerCase(); }});

  function apply() {{
    var shown = 0;
    for (var i = 0; i < cases.length; i++) {{
      var c = cases[i];
      var ok = (!state.domain || c.dataset.domain === state.domain)
            && (!state.acc || c.dataset.acc === state.acc)
            && (!state.band || c.dataset.band === state.band)
            && (!state.think || c.dataset.think === state.think)
            && (!state.text || hay[i].indexOf(state.text) !== -1);
      c.hidden = !ok;
      if (ok) shown++;
    }}
    count.textContent = shown.toLocaleString() + ' shown';
    empty.hidden = shown !== 0;
  }}

  dom.addEventListener('change', function () {{ state.domain = dom.value; apply(); }});

  var t;
  q.addEventListener('input', function () {{
    clearTimeout(t);
    t = setTimeout(function () {{ state.text = q.value.trim().toLowerCase(); apply(); }}, 140);
  }});

  document.querySelectorAll('.seg').forEach(function (seg) {{
    seg.addEventListener('click', function (e) {{
      var b = e.target.closest('button');
      if (!b) return;
      seg.querySelectorAll('button').forEach(function (x) {{
        x.setAttribute('aria-pressed', String(x === b));
      }});
      if ('acc' in b.dataset) state.acc = b.dataset.acc;
      if ('band' in b.dataset) state.band = b.dataset.band;
      if ('think' in b.dataset) state.think = b.dataset.think;
      apply();
    }});
  }});
}})();
</script>
"""


def main(out: str = "") -> None:
    """Render the corpus browser to a self-contained HTML file."""
    rows = [json.loads(l) for l in (SRC / "all_records.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "response" in r]   # one record is a generation error
    meta = json.loads((SRC / "run_meta.json").read_text())
    reasoning = load_reasoning()
    path = Path(out) if out else REPO / "output/difficult_advice_gen/da_corpus.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build(rows, meta, reasoning), encoding="utf-8")
    mb = path.stat().st_size / 1048576
    n_t = sum(1 for r in rows if reasoning.get(r["user_message"].strip()))
    print(f">>> {path}  ({len(rows):,} samples, {n_t:,} with reasoning, {mb:.1f} MB)")
    assert mb < 16, f"{mb:.1f} MB exceeds the 16 MB artifact limit"


if __name__ == "__main__":
    fire.Fire(main)
