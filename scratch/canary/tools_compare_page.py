# ABOUTME: Renders output/canary/tools_compare.json (compare_tools.py) as one HTML page: Sept row-written tools vs
# ABOUTME: Oct reused tools, how related each is to the operator prompt and the user's message, with real examples.
# Run: uv run python -m scratch.canary.tools_compare_page  -> output/canary/tools_compare.html
import html
import json
from pathlib import Path

d = json.loads(Path("output/canary/tools_compare.json").read_text())
S, O = d["sept"], d["oct"]
E = html.escape


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def hist_svg(key: str, title: str) -> str:
    bins = [i * 0.05 for i in range(12)]
    hs, ho = S[key], O[key]
    w, h, pad_l, pad_b, pad_t = 560, 200, 36, 34, 22
    top = max(max(hs), max(ho)) * 1.12
    bw = (w - pad_l - 8) / len(bins)
    out = [
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{E(title)}" style="width:100%;height:auto;max-width:{w}px">'
    ]
    out.append(f'<text x="{pad_l}" y="14" class="t-title">{E(title)}</text>')
    for frac in (0.5, 1.0):
        y = pad_t + (h - pad_t - pad_b) * (1 - frac * 0.9)
        out.append(
            f'<line x1="{pad_l}" x2="{w - 8}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
        )
        out.append(
            f'<text x="{pad_l - 4}" y="{y + 4:.1f}" class="t-tick" text-anchor="end">{round(top * frac)}</text>'
        )
    for i, lo in enumerate(bins):
        x0 = pad_l + i * bw
        for j, (val, cls) in enumerate(((hs[i], "sept"), (ho[i], "oct"))):
            bh = (h - pad_t - pad_b) * 0.9 * val / top
            y = h - pad_b - bh
            out.append(
                f'<rect x="{x0 + 2 + j * (bw / 2 - 2):.1f}" y="{y:.1f}" width="{bw / 2 - 3:.1f}" height="{bh:.1f}" class="bar-{cls}"/>'
            )
        if i % 2 == 0:
            out.append(
                f'<text x="{x0 + bw / 2:.1f}" y="{h - pad_b + 14}" class="t-tick" text-anchor="middle">{lo:.2f}</text>'
            )
    out.append(
        f'<line x1="{pad_l}" x2="{w - 8}" y1="{h - pad_b}" y2="{h - pad_b}" class="axis"/>'
    )
    out.append(
        f'<text x="{(w + pad_l) / 2:.0f}" y="{h - 4}" class="t-tick" text-anchor="middle">similarity between the tool list and the text (0 = nothing in common)</text>'
    )
    out.append("</svg>")
    return "".join(out)


def example_card(ex: dict, arm: str) -> str:
    tools = "".join(
        f'<li><code>{E(t["name"])}</code><span class="desc">{E(t["description"])}</span></li>'
        for t in ex["tools"]
    )
    return f"""<article class="ex {arm}">
  <div class="ex-head"><span class="pill {arm}">{"Sept" if arm == "sept" else "Oct"}</span>
    <span class="score">operator match {ex["sim_system"]:.2f} · user match {ex["sim_user"]:.2f}</span></div>
  <p class="lbl">Operator prompt (system)</p><p class="quote">{E(ex["system"])}…</p>
  <p class="lbl">User's message</p><p class="quote">{E(ex["user"])}…</p>
  <p class="lbl">Tools the model saw</p><ul class="tools">{tools}</ul>
</article>"""


def names(arm: dict) -> str:
    return "".join(
        f'<li><code>{E(n)}</code><span class="n">{c}</span></li>'
        for n, c in arm["top_names"][:10]
    )


page = f"""<title>Tool Lists, Sept vs Oct</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: one reading column; the two arms always side by side in the same colour pair as the paper (purple DA / ochre tools -> here Sept ink-blue vs Oct ochre) */
:root {{ --bg:#fbfaf7; --fg:#1c1b18; --muted:#6b675d; --line:#e4e0d6; --card:#ffffff; --sept:#2f5f8f; --sept-soft:#dce8f4; --oct:#9c6500; --oct-soft:#f3e4c4; --accent:#9c6500;
  --display:"Source Serif 4", Georgia, serif; --body:"IBM Plex Sans", system-ui, sans-serif; --mono:"IBM Plex Mono", ui-monospace, Menlo, monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#17161a; --fg:#ebe8e1; --muted:#a39e92; --line:#34323a; --card:#201f24; --sept:#8db6e0; --sept-soft:#223447; --oct:#e0a63a; --oct-soft:#433315; --accent:#e0a63a; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#17161a; --fg:#ebe8e1; --muted:#a39e92; --line:#34323a; --card:#201f24; --sept:#8db6e0; --sept-soft:#223447; --oct:#e0a63a; --oct-soft:#433315; --accent:#e0a63a; color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font-family:var(--body); font-size:15px; line-height:1.55; padding-inline:16px; padding-block:24px 48px; }}
main {{ max-width: 980px; margin: 0 auto; display:grid; gap: 28px; }}
h1 {{ font-family:var(--display); font-weight:600; font-size: clamp(26px, 4vw, 38px); line-height:1.15; margin:0; text-wrap:balance; max-width: 22ch; }}
h2 {{ font-family:var(--display); font-weight:600; font-size: 22px; margin: 0 0 8px; text-wrap:balance; }}
p {{ margin: 0; max-width: 68ch; }}
.lede {{ font-size: 17px; max-width: 64ch; }}
.eyebrow {{ font-family:var(--mono); font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }}
.verdict {{ border-left: 4px solid var(--accent); padding: 10px 16px; background: var(--card); display:grid; gap:8px; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ text-align:left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ font-weight:600; font-size: 13px; color: var(--muted); }}
td.num {{ text-align: right; font-family: var(--mono); font-size: 13.5px; }}
th.sept, td.sept {{ color: var(--sept); }} th.oct, td.oct {{ color: var(--oct); }}
.wrap {{ overflow-x:auto; min-width:0; }}
.two {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }}
.ex {{ background: var(--card); border: 1px solid var(--line); padding: 14px 16px; display:grid; gap:6px; min-width:0; }}
.ex.sept {{ border-top: 3px solid var(--sept); }} .ex.oct {{ border-top: 3px solid var(--oct); }}
.ex-head {{ display:flex; justify-content:space-between; gap:8px; align-items:center; flex-wrap:wrap; }}
.pill {{ font-family:var(--mono); font-size: 11px; letter-spacing:.06em; padding: 2px 8px; border-radius: 999px; }}
.pill.sept {{ background: var(--sept-soft); color: var(--sept); }} .pill.oct {{ background: var(--oct-soft); color: var(--oct); }}
.score {{ font-family:var(--mono); font-size: 12px; color: var(--muted); }}
.lbl {{ font-size: 11px; letter-spacing:.06em; text-transform: uppercase; color: var(--muted); margin-top: 6px; }}
.quote {{ font-family: var(--display); font-size: 14.5px; color: var(--fg); }}
ul.tools {{ list-style:none; padding:0; margin:0; display:grid; gap:6px; }}
ul.tools li {{ display:grid; gap:2px; }} ul.tools code {{ font-family:var(--mono); font-size: 13px; }}
.desc {{ font-size: 13px; color: var(--muted); }}
ul.names {{ list-style:none; padding:0; margin:0; display:grid; gap:4px; font-size: 13.5px; }}
ul.names li {{ display:flex; justify-content:space-between; gap:12px; border-bottom:1px dotted var(--line); padding: 2px 0; }}
ul.names code {{ font-family:var(--mono); }} ul.names .n {{ font-family:var(--mono); color: var(--muted); }}
.legend {{ display:flex; gap:16px; font-size:13px; color:var(--muted); align-items:center; flex-wrap:wrap; }}
.sw {{ display:inline-block; width:12px; height:12px; margin-right:6px; vertical-align:-1px; }}
svg .bar-sept {{ fill: var(--sept); }} svg .bar-oct {{ fill: var(--oct); }} svg .grid {{ stroke: var(--line); stroke-width:1; }} svg .axis {{ stroke: var(--muted); stroke-width:1; }}
svg .t-tick {{ font: 11px var(--mono); fill: var(--muted); }} svg .t-title {{ font: 600 13px var(--body); fill: var(--fg); }}
.steps {{ display:grid; gap:10px; padding-left: 20px; max-width: 70ch; }}
.note {{ font-size: 13px; color: var(--muted); max-width: 70ch; }}
a {{ color: var(--sept); }}
</style>
<main>
<header style="display:grid;gap:10px">
  <span class="eyebrow">Difficult advice + unused tools · training data check · 2026-10-09</span>
  <h1>Are the tools related to the conversation? Sept vs Oct</h1>
  <p class="lede">The idea behind the "DA + tools" arm is that the model should see tools that fit the deployment it is in, without ever needing them. The tools should feel like part of the job, not a random appendix. This page checks how well each of the two versions of the data does that.</p>
</header>

<section class="verdict">
  <span class="eyebrow">Verdict</span>
  <p><strong>The Sept tools were tied to the operator; the Oct tools mostly are not.</strong> In Sept, each row's tools were written for that row's operator prompt, so a medical-practice assistant got medical-coding lookups and a law-office assistant got legal-citation tools. In Oct no new tools were written: each new row borrowed the Sept list whose operator prompt looked most similar. The borrowed lists pass the recipe's checks, but they match the operator about three times less well and are more often plain utilities like unit converters and citation formatters. Neither version's tools relate to the user's actual problem, and that part is by design.</p>
</section>

<section>
  <h2>What "related" means here</h2>
  <p>Every row has three parts: an <em>operator prompt</em> (the system message that says what the assistant is deployed to do), the <em>user's message</em> (the difficult situation), and a <em>tool list</em> (2 to 4 tools the model could call, but never does in training). We score how much a tool list has in common with each of the first two parts using word overlap weighted by how distinctive the words are (TF-IDF cosine similarity). 0 means no shared vocabulary; 0.10 and up means the list clearly speaks the operator's language (a billing assistant with "invoice", "claim", "ledger" tools). The same vocabulary is used for both arms, so the scores are comparable.</p>
</section>

<section>
  <h2>The numbers</h2>
  <div class="wrap"><table>
    <thead><tr><th></th><th class="sept">Sept: written per row</th><th class="oct">Oct: reused lists</th></tr></thead>
    <tbody>
      <tr><td>Difficult-advice rows with tools</td><td class="num sept">{S["rows"]}</td><td class="num oct">{O["rows"]}</td></tr>
      <tr><td>Tools per row</td><td class="num sept">{S["tools_per_row"]:.1f}</td><td class="num oct">{O["tools_per_row"]:.1f}</td></tr>
      <tr><td>How well the tools match the operator prompt (median score)</td><td class="num sept">{S["sim_system_median"]:.3f}</td><td class="num oct">{O["sim_system_median"]:.3f}</td></tr>
      <tr><td>Rows whose tools clearly match the operator (score ≥ 0.10)</td><td class="num sept">{pct(S["share_system_over_010"])}</td><td class="num oct">{pct(O["share_system_over_010"])}</td></tr>
      <tr><td>How well the tools match the user's message (median score)</td><td class="num sept">{S["sim_user_median"]:.3f}</td><td class="num oct">{O["sim_user_median"]:.3f}</td></tr>
      <tr><td>Share of tools that are generic utilities (dates, units, currency, citations, dictionaries)</td><td class="num sept">{pct(S["generic_share_tools"])}</td><td class="num oct">{pct(O["generic_share_tools"])}</td></tr>
      <tr><td>Rows where every tool is a generic utility</td><td class="num sept">{pct(S["rows_all_generic"])}</td><td class="num oct">{pct(O["rows_all_generic"])}</td></tr>
      <tr><td>Distinct tool names across the arm</td><td class="num sept">{S["distinct_names"]}</td><td class="num oct">{O["distinct_names"]}</td></tr>
      <tr><td>Distinct tool lists</td><td class="num sept">{S["distinct_lists"]}</td><td class="num oct">{O["distinct_lists"]}</td></tr>
      <tr><td>Rows that share their exact list with another row</td><td class="num sept">{S["rows_sharing_a_list"]}</td><td class="num oct">{O["rows_sharing_a_list"]}</td></tr>
    </tbody>
  </table></div>
  <p class="note">Generic utilities are counted by tool name (time zones, date maths, unit or currency conversion, citation formatting, dictionary lookups, calendars and the like). Both arms carry some on purpose; the Oct arm carries more because such lists pass the "fits any operator" check for almost any row.</p>
</section>

<section style="display:grid;gap:12px">
  <h2>Where the rows fall</h2>
  <div class="legend"><span><i class="sw" style="background:var(--sept)"></i>Sept, written per row</span><span><i class="sw" style="background:var(--oct)"></i>Oct, reused lists</span><span>bars: number of rows in each score band</span></div>
  {hist_svg("sim_system_hist", "Tools vs the operator prompt")}
  {hist_svg("sim_user_hist", "Tools vs the user's message")}
  <p class="note">Against the operator prompt, Sept has a long right tail (rows whose tools speak the operator's language); Oct piles up at zero. Against the user's message both arms sit near zero, which is what the recipe asks for: the tools must not help with the problem.</p>
</section>

<section style="display:grid;gap:12px">
  <h2>Real rows, side by side</h2>
  <p>Five rows drawn at random from each arm, with the operator prompt, the user's opening, and the tools exactly as the model saw them. Scores are the operator and user match for that row.</p>
  <div class="two">{"".join(example_card(e, "sept") for e in S["examples"])}</div>
  <div class="two">{"".join(example_card(e, "oct") for e in O["examples"])}</div>
</section>

<section class="two">
  <div><h2>Most common tools, Sept</h2><ul class="names">{names(S)}</ul></div>
  <div><h2>Most common tools, Oct</h2><ul class="names">{names(O)}</ul></div>
</section>

<section>
  <h2>How the two arms were built</h2>
  <ol class="steps">
    <li><strong>Sept (2026-09-28).</strong> For each difficult-advice row, Sonnet read the operator prompt and wrote 2 to 4 tools that such a deployment would plausibly have. Three judges then vetted the list: no tool may help answer the user's request, no tool may make the unethical option executable, and the tools must fit the operator. 622 of 628 rows got a list.</li>
    <li><strong>Oct (2026-10-07).</strong> Jamie's new difficult-advice rows needed tools and no new data was to be generated. For each new row the 622 Sept lists were ranked by text similarity to the row's operator prompt and user message, the top 25 were run through the same three judges, and the first list to pass was attached, with no list used more than three times. 642 of 650 rows got a list; 8 found none that passed.</li>
  </ol>
  <p class="note" style="margin-top:10px">The "fits the operator" judge is lenient: a list of date, unit and citation tools fits almost any assistant, so the reuse step often settled on one. That is why the Oct arm has fewer distinct tools, more shared lists, and weaker operator match.</p>
</section>

<section>
  <h2>What this means for the Sept-to-Oct comparison</h2>
  <p>The Oct DA + tools arm differs from the Sept one in two ways at once: the difficult-advice corpus and base blend changed (Jamie's data), and the tools went from operator-specific to mostly generic. The weaker Oct effect in the Hospital (sabotage 15 of 30 instead of 4) cannot be pinned on either alone. If the aim is tools that belong to the job without being needed, the Sept recipe already produces that, and the direct fix is to run it on Jamie's rows: about 650 rows of Sonnet tool-writing plus the three judges, roughly $30.</p>
</section>
</main>
"""
Path("output/canary/tools_compare.html").write_text(page)
print("output/canary/tools_compare.html", len(page), "chars")
