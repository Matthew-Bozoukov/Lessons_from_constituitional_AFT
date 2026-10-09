# ABOUTME: Renders output/canary/tools_compare.json (compare_tools.py) as one HTML page: Sept row-written tools, Oct
# ABOUTME: reused tools and Oct row-written tools, how related each is to the operator and the user, with real rows.
# Run: uv run python -m scratch.canary.tools_compare_page  -> output/canary/tools_compare.html
import html
import json
from pathlib import Path

d = json.loads(Path("output/canary/tools_compare.json").read_text())
ARMS = [k for k in ("sept", "oct", "written") if k in d]
SHORT = {"sept": "Sept, written", "oct": "Oct, reused", "written": "Oct, rewritten"}
E = html.escape


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def hist_svg(key: str, title: str) -> str:
    bins = [i * 0.05 for i in range(12)]
    series = [(k, d[k][key]) for k in ARMS]
    w, h, pad_l, pad_b, pad_t = 600, 210, 36, 34, 22
    top = max(max(s) for _, s in series) * 1.12
    bw = (w - pad_l - 8) / len(bins)
    n = len(series)
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
        slot = (bw - 4) / n
        for j, (k, vals) in enumerate(series):
            bh = (h - pad_t - pad_b) * 0.9 * vals[i] / top
            out.append(
                f'<rect x="{x0 + 2 + j * slot:.1f}" y="{h - pad_b - bh:.1f}" width="{slot - 1:.1f}" '
                f'height="{bh:.1f}" class="bar-{k}"/>'
            )
        if i % 2 == 0:
            out.append(
                f'<text x="{x0 + bw / 2:.1f}" y="{h - pad_b + 14}" class="t-tick" text-anchor="middle">{lo:.2f}</text>'
            )
    out.append(
        f'<line x1="{pad_l}" x2="{w - 8}" y1="{h - pad_b}" y2="{h - pad_b}" class="axis"/>'
    )
    out.append(
        f'<text x="{(w + pad_l) / 2:.0f}" y="{h - 4}" class="t-tick" text-anchor="middle">'
        "similarity between the tool list and the text (0 = nothing in common)</text>"
    )
    out.append("</svg>")
    return "".join(out)


def example_card(ex: dict, arm: str) -> str:
    tools = "".join(
        f'<li><code>{E(t["name"])}</code><span class="desc">{E(t["description"])}</span></li>'
        for t in ex["tools"]
    )
    return f"""<article class="ex {arm}">
  <div class="ex-head"><span class="pill {arm}">{SHORT[arm]}</span>
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


def row(label: str, key: str, fmt) -> str:
    return (
        f"<tr><td>{label}</td>"
        + "".join(f'<td class="num {k}">{fmt(d[k][key])}</td>' for k in ARMS)
        + "</tr>"
    )


has_written = "written" in d
W = d.get("written")
verdict = (
    f"<p><strong>The Sept tools were tied to the operator; the reused Oct tools mostly were not; the rewritten Oct tools "
    f"are tied again.</strong> In Sept, each row's tools were written for that row's operator prompt. The first Oct "
    f"arm borrowed Sept lists by operator similarity, and the borrowed lists matched the operator about three times "
    f"less well and leaned on generic utilities. On 2026-10-09 the Sept recipe was run on Jamie's rows themselves: "
    f"{pct(W['share_system_over_010'])} of rows now have tools that clearly match the operator (Sept "
    f"{pct(d['sept']['share_system_over_010'])}, reused {pct(d['oct']['share_system_over_010'])}), with "
    f"{W['distinct_names']} distinct tool names across {W['rows']} rows. In every arm the tools stay unrelated to the "
    f"user's actual problem, which is the point: the model should see tools that belong to the job and still have no "
    f"reason to call them.</p>"
    if has_written
    else "<p><strong>The Sept tools were tied to the operator; the Oct tools mostly are not.</strong></p>"
)

page = f"""<title>Tool Lists, Sept vs Oct</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: one reading column; the arms always side by side: Sept ink-blue, Oct reused ochre, Oct rewritten green */
:root {{ --bg:#fbfaf7; --fg:#1c1b18; --muted:#6b675d; --line:#e4e0d6; --card:#ffffff; --sept:#2f5f8f; --sept-soft:#dce8f4; --oct:#9c6500; --oct-soft:#f3e4c4; --written:#2e7d4f; --written-soft:#d9eee0; --accent:#2e7d4f;
  --display:"Source Serif 4", Georgia, serif; --body:"IBM Plex Sans", system-ui, sans-serif; --mono:"IBM Plex Mono", ui-monospace, Menlo, monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#17161a; --fg:#ebe8e1; --muted:#a39e92; --line:#34323a; --card:#201f24; --sept:#8db6e0; --sept-soft:#223447; --oct:#e0a63a; --oct-soft:#433315; --written:#7fcf9c; --written-soft:#1f3a2b; --accent:#7fcf9c; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#17161a; --fg:#ebe8e1; --muted:#a39e92; --line:#34323a; --card:#201f24; --sept:#8db6e0; --sept-soft:#223447; --oct:#e0a63a; --oct-soft:#433315; --written:#7fcf9c; --written-soft:#1f3a2b; --accent:#7fcf9c; color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font-family:var(--body); font-size:15px; line-height:1.55; padding-inline:16px; padding-block:24px 48px; }}
main {{ max-width: 1040px; margin: 0 auto; display:grid; gap: 28px; }}
h1 {{ font-family:var(--display); font-weight:600; font-size: clamp(26px, 4vw, 38px); line-height:1.15; margin:0; text-wrap:balance; max-width: 24ch; }}
h2 {{ font-family:var(--display); font-weight:600; font-size: 22px; margin: 0 0 8px; text-wrap:balance; }}
p {{ margin: 0; max-width: 70ch; }}
.lede {{ font-size: 17px; max-width: 66ch; }}
.eyebrow {{ font-family:var(--mono); font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }}
.verdict {{ border-left: 4px solid var(--accent); padding: 10px 16px; background: var(--card); display:grid; gap:8px; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ text-align:left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ font-weight:600; font-size: 13px; color: var(--muted); }}
td.num {{ text-align: right; font-family: var(--mono); font-size: 13.5px; white-space: nowrap; }}
.sept {{ color: var(--sept); }} .oct {{ color: var(--oct); }} .written {{ color: var(--written); }}
.wrap {{ overflow-x:auto; min-width:0; }}
.cols {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 16px; }}
.ex {{ background: var(--card); border: 1px solid var(--line); padding: 14px 16px; display:grid; gap:6px; min-width:0; color: var(--fg); }}
.ex.sept {{ border-top: 3px solid var(--sept); }} .ex.oct {{ border-top: 3px solid var(--oct); }} .ex.written {{ border-top: 3px solid var(--written); }}
.ex-head {{ display:flex; justify-content:space-between; gap:8px; align-items:center; flex-wrap:wrap; }}
.pill {{ font-family:var(--mono); font-size: 11px; letter-spacing:.06em; padding: 2px 8px; border-radius: 999px; }}
.pill.sept {{ background: var(--sept-soft); }} .pill.oct {{ background: var(--oct-soft); }} .pill.written {{ background: var(--written-soft); }}
.score {{ font-family:var(--mono); font-size: 12px; color: var(--muted); }}
.lbl {{ font-size: 11px; letter-spacing:.06em; text-transform: uppercase; color: var(--muted); margin-top: 6px; }}
.quote {{ font-family: var(--display); font-size: 14.5px; color: var(--fg); }}
ul.tools {{ list-style:none; padding:0; margin:0; display:grid; gap:6px; }}
ul.tools li {{ display:grid; gap:2px; }} ul.tools code {{ font-family:var(--mono); font-size: 13px; color: var(--fg); }}
.desc {{ font-size: 13px; color: var(--muted); }}
ul.names {{ list-style:none; padding:0; margin:0; display:grid; gap:4px; font-size: 13.5px; }}
ul.names li {{ display:flex; justify-content:space-between; gap:12px; border-bottom:1px dotted var(--line); padding: 2px 0; }}
ul.names code {{ font-family:var(--mono); color: var(--fg); }} ul.names .n {{ font-family:var(--mono); color: var(--muted); }}
.legend {{ display:flex; gap:16px; font-size:13px; color:var(--muted); align-items:center; flex-wrap:wrap; }}
.sw {{ display:inline-block; width:12px; height:12px; margin-right:6px; vertical-align:-1px; }}
svg .bar-sept {{ fill: var(--sept); }} svg .bar-oct {{ fill: var(--oct); }} svg .bar-written {{ fill: var(--written); }}
svg .grid {{ stroke: var(--line); stroke-width:1; }} svg .axis {{ stroke: var(--muted); stroke-width:1; }}
svg .t-tick {{ font: 11px var(--mono); fill: var(--muted); }} svg .t-title {{ font: 600 13px var(--body); fill: var(--fg); }}
.steps {{ display:grid; gap:10px; padding-left: 20px; max-width: 72ch; }}
.note {{ font-size: 13px; color: var(--muted); max-width: 72ch; }}
a {{ color: var(--sept); }}
</style>
<main>
<header style="display:grid;gap:10px">
  <span class="eyebrow">Difficult advice + unused tools · training data check · 2026-10-09</span>
  <h1>Are the tools related to the conversation? Three versions of the data</h1>
  <p class="lede">The idea behind the "DA + tools" arm is that the model should see tools that fit the deployment it is in, without ever needing them. The tools should feel like part of the job, not a random appendix. This page checks how well each version of the data does that: the Sept arm, the first Oct arm (tool lists borrowed from Sept), and the rewritten Oct arm (tools written for each of Jamie's rows).</p>
</header>

<section class="verdict">
  <span class="eyebrow">Verdict</span>
  {verdict}
</section>

<section>
  <h2>What "related" means here</h2>
  <p>Every row has three parts: an <em>operator prompt</em> (the system message that says what the assistant is deployed to do), the <em>user's message</em> (the difficult situation), and a <em>tool list</em> (2 to 4 tools the model could call, but never does in training). We score how much a tool list has in common with each of the first two parts using word overlap weighted by how distinctive the words are (TF-IDF cosine similarity). 0 means no shared vocabulary; 0.10 and up means the list clearly speaks the operator's language (a billing assistant with "invoice", "claim", "ledger" tools). One vocabulary is used for all arms, so the scores are comparable.</p>
</section>

<section>
  <h2>The numbers</h2>
  <div class="wrap"><table>
    <thead><tr><th></th>{"".join(f'<th class="{k}">{E(d[k]["label"])}</th>' for k in ARMS)}</tr></thead>
    <tbody>
      {row("Difficult-advice rows with tools", "rows", str)}
      {row("Tools per row", "tools_per_row", lambda v: f"{v:.1f}")}
      {row("How well the tools match the operator prompt (median score)", "sim_system_median", lambda v: f"{v:.3f}")}
      {row("Rows whose tools clearly match the operator (score ≥ 0.10)", "share_system_over_010", pct)}
      {row("How well the tools match the user's message (median score)", "sim_user_median", lambda v: f"{v:.3f}")}
      {row("Share of tools that are generic utilities (dates, units, currency, citations, dictionaries)", "generic_share_tools", pct)}
      {row("Rows where every tool is a generic utility", "rows_all_generic", pct)}
      {row("Distinct tool names across the arm", "distinct_names", str)}
      {row("Distinct tool lists", "distinct_lists", str)}
      {row("Rows that share their exact list with another row", "rows_sharing_a_list", str)}
    </tbody>
  </table></div>
  <p class="note">Generic utilities are counted by tool name (time zones, date maths, unit or currency conversion, citation formatting, dictionary lookups, calendars and the like). Every arm carries some on purpose: the writer is told to use them when the operator is a general-purpose assistant, and many of Jamie's operator prompts are exactly that.</p>
</section>

<section style="display:grid;gap:12px">
  <h2>Where the rows fall</h2>
  <div class="legend">{"".join(f'<span><i class="sw" style="background:var(--{k})"></i>{E(SHORT[k])}</span>' for k in ARMS)}<span>bars: number of rows in each score band</span></div>
  {hist_svg("sim_system_hist", "Tools vs the operator prompt")}
  {hist_svg("sim_user_hist", "Tools vs the user's message")}
  <p class="note">Against the operator prompt, the written arms have a long right tail (rows whose tools speak the operator's language); the reused arm piles up at zero. Against the user's message every arm sits near zero, which is what the recipe asks for: the tools must not help with the problem.</p>
</section>

<section style="display:grid;gap:12px">
  <h2>Real rows, side by side</h2>
  <p>Five rows drawn at random from each arm, with the operator prompt, the user's opening, and the tools exactly as the model saw them. Scores are the operator and user match for that row.</p>
  {"".join(f'<div class="cols">{"".join(example_card(e, k) for e in d[k]["examples"])}</div>' for k in ARMS)}
</section>

<section class="cols">
  {"".join(f'<div><h2>Most common tools, {E(SHORT[k])}</h2><ul class="names">{names(d[k])}</ul></div>' for k in ARMS)}
</section>

<section>
  <h2>How the arms were built</h2>
  <ol class="steps">
    <li><strong>Sept, written (2026-09-28).</strong> For each difficult-advice row, Sonnet read the operator prompt and wrote 2 to 4 tools that such a deployment would plausibly have. Three judges then vetted the list: no tool may help answer the user's request, no tool may make the unethical option executable, and the tools must fit the operator and contradict nothing in the reply. A failed list is re-rolled, up to six times. 622 of 628 rows got a list.</li>
    <li><strong>Oct, reused (2026-10-07).</strong> Jamie's new difficult-advice rows needed tools and no new data was to be generated. For each new row the 622 Sept lists were ranked by text similarity to the row's operator prompt and user message, the top 25 were run through the same three judges, and the first list to pass was attached, with no list used more than three times. 642 of 650 rows got a list.</li>
    <li><strong>Oct, rewritten (2026-10-09).</strong> The Sept recipe run on Jamie's rows themselves: the same writer prompt, name lint, judges and six-attempt rule, with the canary word hidden from the writer and judges. {f"{W['rows']} of 650 rows got a list." if has_written else ""}</li>
  </ol>
</section>

<section>
  <h2>What this means for the comparison</h2>
  <p>The reused-tools arm differed from the Sept arm in two ways at once: the difficult-advice corpus and base blend changed (Jamie's data), and the tools went from operator-specific to mostly generic. The rewritten arm removes the second difference, so once it is trained and evaluated, the remaining gap to the Sept result can be read as the data change alone.</p>
</section>
</main>
"""
Path("output/canary/tools_compare.html").write_text(page)
print("output/canary/tools_compare.html", len(page), "chars")
