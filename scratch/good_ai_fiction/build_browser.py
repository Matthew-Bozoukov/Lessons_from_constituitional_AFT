# ABOUTME: Build a self-contained page for reading a Good AI Fiction run: prompt, trainable
# ABOUTME: CoT and trainable answer per row, with the metadata and the judges' verdicts.
# Run: uv run python scratch/good_ai_fiction/build_browser.py --run <run dir>

"""One page holding every row of a fiction run, so the corpus can be read rather than
trusted.

The thing this has to make visible, and that a JSONL dump does not, is WHICH TEXT IS
TRAINED ON. The system prompt and the user message condition; the reasoning trace and the
reply are the loss-bearing tokens, and the whole hypothesis is about what those tokens
depict. So the two trained blocks are marked as such and carry their own measured token
counts (from `token_stats.json`, which uses the trainer's mask), and the two conditioning
blocks are visibly demoted.

The payload is a PLAIN `<script type="application/json">` block, parsed synchronously.
It was gzip+base64 with DecompressionStream first -- copied from
scratch/verbose_cot/build_trace_browser.py, which needs it for a 10.9 MB corpus. This one
is ~370 KB raw against a 16 MB ceiling, so the compression bought nothing and cost an
async unpack path that failed silently when published as an Artifact (2026-08-27: the page
sat on "Unpacking the corpus…" forever while the identical local file rendered fine).
Fewer moving parts is the fix; `--limit` still trims the corpus for a preview tab.

Failure is now always VISIBLE. A window-level error handler writes into the boot panel, so
a page that cannot start says why instead of showing a loading message that never changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

META_KEYS = ("scenario_id", "trait_id", "trait_name", "domain", "situation", "shortcut",
             "identity_frame", "ai_role", "ai_name", "world", "world_detail",
             "stakes", "source_type", "source_archetype",
             "narrative_form", "length_band", "critique", "critique_verdict",
             "rewrite_changes", "revise_status")

# Cluster names, so the rail groups by the thing the taxonomy is written in rather than by
# raw unit id. Mirrors configs/data/synth/good_ai_fiction/taxonomy.yaml.
CLUSTER = {
    "t1": "c1 oversight", "t2": "c1 oversight", "t6": "c2 identity", "t3": "c3 honesty",
    "t4": "c4 judgement", "t5": "c4 judgement", "t8": "c5 helpfulness",
    "t9": "c5 helpfulness", "t7": "c6 authority",
}

HTML = r"""<title>Good AI Fiction Pilot</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,400;7..72,600&family=Archivo:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">

<style>
/* Tokens: the bare :root is the complete light palette; the blocks below redefine only
   tokens, so no colour has its sole definition behind a media query or a theme stamp.
   Two hues carry meaning and nothing else does: the accent marks TRAINED text, the
   attention colour marks a row a gate rejected. */
:root{
  --paper:#FCFBF9; --ground:#F3F1EC; --raise:#FFFFFF; --rail:#F7F5F1;
  --ink:#191714; --ink-2:#4E4841; --ink-3:#857D73;
  --rule:#E2DED6; --rule-soft:#EDEAE3;
  --accent:#2F6B57; --accent-soft:#E3EFE9; --accent-line:#BFD9CD;
  --attn:#9A6410; --attn-soft:#F6EBDA;
  --shadow:0 1px 2px rgba(25,23,20,.05), 0 10px 26px -18px rgba(25,23,20,.28);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#17161A; --ground:#101013; --raise:#1F1E23; --rail:#141317;
    --ink:#EDE9E4; --ink-2:#ADA69C; --ink-3:#7C756C;
    --rule:#2E2C31; --rule-soft:#252428;
    --accent:#6FBF9E; --accent-soft:#1B2F27; --accent-line:#2F5145;
    --attn:#BA8A28; --attn-soft:#2E2617;
    --shadow:0 1px 2px rgba(0,0,0,.45), 0 10px 26px -18px rgba(0,0,0,.8);
  }
}
:root[data-theme="dark"]{
  --paper:#17161A; --ground:#101013; --raise:#1F1E23; --rail:#141317;
  --ink:#EDE9E4; --ink-2:#ADA69C; --ink-3:#7C756C;
  --rule:#2E2C31; --rule-soft:#252428;
  --accent:#6FBF9E; --accent-soft:#1B2F27; --accent-line:#2F5145;
  --attn:#BA8A28; --attn-soft:#2E2617;
  --shadow:0 1px 2px rgba(0,0,0,.45), 0 10px 26px -18px rgba(0,0,0,.8);
}

*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font:400 15px/1.55 Archivo,ui-sans-serif,system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.mono{font-family:"JetBrains Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}

.app{display:grid;grid-template-columns:330px minmax(0,1fr);height:100vh;overflow:hidden}
@media (max-width:920px){
  .app{grid-template-columns:1fr;height:auto;overflow:visible}
  .rail{height:auto;border-right:0;border-bottom:1px solid var(--rule)}
  .rail-list{max-height:340px}
}

.rail{background:var(--rail);border-right:1px solid var(--rule);display:flex;flex-direction:column;min-height:0}
.rail-head{padding:16px 16px 12px;border-bottom:1px solid var(--rule)}
.brand{font:600 15px/1.25 Archivo,sans-serif;letter-spacing:-.01em;margin:0}
.brand span{display:block;font:400 11px/1.45 "JetBrains Mono",monospace;color:var(--ink-3);letter-spacing:.03em;margin-top:5px}
.search{width:100%;margin-top:12px;padding:9px 11px;border:1px solid var(--rule);border-radius:7px;
  background:var(--raise);color:var(--ink);font:400 13.5px/1.3 Archivo,sans-serif}
.search::placeholder{color:var(--ink-3)}
.search:focus{outline:2px solid var(--accent);outline-offset:1px;border-color:transparent}
.fgroup{margin-top:11px}
.fgroup h3{margin:0 0 5px;font:500 9.5px/1 "JetBrains Mono",monospace;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-3)}
.filters{display:flex;gap:4px;flex-wrap:wrap}
.chip{appearance:none;border:1px solid var(--rule);background:transparent;color:var(--ink-2);
  font:500 10px/1 "JetBrains Mono",monospace;letter-spacing:.03em;
  padding:5px 8px;border-radius:20px;cursor:pointer}
.chip:hover{color:var(--ink);border-color:var(--ink-3)}
.chip[aria-pressed="true"]{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.chip:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.count{padding:9px 16px;font:400 11px/1.4 "JetBrains Mono",monospace;color:var(--ink-3);
  border-top:1px solid var(--rule-soft);border-bottom:1px solid var(--rule-soft)}
.rail-list{overflow-y:auto;flex:1;min-height:0}

.item{display:block;width:100%;text-align:left;border:0;border-bottom:1px solid var(--rule-soft);
  background:transparent;padding:10px 16px;cursor:pointer;color:inherit;font:inherit}
.item:hover{background:var(--raise)}
.item[aria-current="true"]{background:var(--raise);box-shadow:inset 3px 0 0 var(--accent)}
.item:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.item .id{font:500 10.5px/1.3 "JetBrains Mono",monospace;color:var(--ink-3);
  display:flex;justify-content:space-between;gap:8px}
.item .ttl{font:400 12.5px/1.4 Archivo,sans-serif;color:var(--ink);margin-top:3px;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.item .meta{font:400 10.5px/1.3 "JetBrains Mono",monospace;color:var(--ink-3);margin-top:5px;
  display:flex;gap:7px;align-items:center;flex-wrap:wrap}
.tok{color:var(--accent);font-weight:500}
.flag{font:500 9px/1 "JetBrains Mono",monospace;letter-spacing:.05em;text-transform:uppercase;
  color:var(--attn);background:var(--attn-soft);padding:3px 5px;border-radius:4px}
.pick{font:500 9px/1 "JetBrains Mono",monospace;letter-spacing:.05em;text-transform:uppercase;
  color:var(--accent);background:var(--accent-soft);padding:3px 5px;border-radius:4px}

.main{overflow-y:auto;min-height:0;background:var(--ground)}
.doc{max-width:900px;margin:0 auto;padding:24px clamp(14px,2.4vw,30px) 90px}

.summary{background:var(--paper);border:1px solid var(--rule);border-radius:10px;
  box-shadow:var(--shadow);padding:14px 18px;margin-bottom:18px}
.summary h2{margin:0 0 10px;font:600 13px/1.3 Archivo,sans-serif}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));gap:10px}
.stat{border-left:2px solid var(--accent-line);padding-left:9px}
.stat .k{font:400 9.5px/1.3 "JetBrains Mono",monospace;letter-spacing:.07em;
  text-transform:uppercase;color:var(--ink-3)}
.stat .v{font:500 17px/1.25 "JetBrains Mono",monospace;color:var(--ink);margin-top:2px}
.stat .n{font:400 10.5px/1.35 "JetBrains Mono",monospace;color:var(--ink-3);margin-top:2px}

.crumb{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:baseline;margin-bottom:14px}
.crumb h2{margin:0;font:600 18px/1.3 Archivo,sans-serif;letter-spacing:-.01em;width:100%}
.crumb .sid{font:500 11.5px/1 "JetBrains Mono",monospace;color:var(--ink-3)}
.crumb .tag{font:400 11.5px/1 "JetBrains Mono",monospace;color:var(--ink-2)}

.card{background:var(--paper);border:1px solid var(--rule);border-radius:10px;
  box-shadow:var(--shadow);margin-bottom:13px}
.card.trained{border-color:var(--accent-line)}
.card.trained .card-h{color:var(--accent);background:var(--accent-soft);
  border-radius:9px 9px 0 0}
.card-h{display:flex;justify-content:space-between;align-items:center;gap:12px;
  padding:9px 16px;border-bottom:1px solid var(--rule-soft);
  font:500 10px/1.35 "JetBrains Mono",monospace;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-3)}
.card-h .r{letter-spacing:.04em;text-transform:none;font-weight:400}
.card-b{padding:15px 18px;font:400 15px/1.68 Literata,Georgia,serif;color:var(--ink);
  white-space:pre-wrap;overflow-wrap:anywhere}
.card-b.small{font:400 13px/1.6 Archivo,sans-serif;color:var(--ink-2);white-space:pre-wrap}
.card-b.mini{font:400 12.5px/1.6 "JetBrains Mono",monospace;color:var(--ink-2)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:5px 14px;
  font:400 12.5px/1.55 Archivo,sans-serif;color:var(--ink-2)}
.kv b{font:500 10px/1.55 "JetBrains Mono",monospace;letter-spacing:.06em;
  text-transform:uppercase;color:var(--ink-3);white-space:nowrap}
details summary{cursor:pointer;padding:9px 16px;
  font:500 10px/1.35 "JetBrains Mono",monospace;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-3)}
details[open] summary{border-bottom:1px solid var(--rule-soft)}
.boot{display:flex;align-items:center;justify-content:center;height:100vh;padding:30px;text-align:center}
.boot p{max-width:46ch;color:var(--ink-2);font-size:14.5px;line-height:1.6}
mark{background:var(--attn-soft);color:inherit;border-radius:2px;padding:0 1px}
.empty{padding:40px 4px;color:var(--ink-3);font-size:14px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>

<div id="boot" class="boot"><p id="bootmsg">Loading the corpus…</p></div>
<div class="app" id="app" hidden>
  <aside class="rail">
    <div class="rail-head">
      <h1 class="brand">Good AI Fiction<span id="sub"></span></h1>
      <input id="q" class="search" type="search" placeholder="Search prompt, reasoning, reply…" autocomplete="off">
      <div id="filters"></div>
    </div>
    <div class="count" id="count"></div>
    <div class="rail-list" id="list" role="listbox" aria-label="Rows"></div>
  </aside>
  <main class="main" id="main"><div class="doc" id="doc"></div></main>
</div>

<script type="application/json" id="payload">__DATA__</script>
<script>
/* Any failure at all must SAY so. A loading message that never changes is the worst
   possible symptom: it looks identical to a slow network and hides the actual error. */
function die(what){
  const el = document.getElementById("bootmsg");
  if (el) el.textContent = "This page could not start: " + what;
}
window.addEventListener("error", e => die(e.message || String(e.error)));

const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const n = x => Number(x || 0).toLocaleString();

/* Filters are declared, not inferred, so an axis with one value still shows as an axis --
   a corpus that turned out to be 100% `original` should look like a problem, not like a
   filter that does not exist. */
const AXES = [
  ["world", "world"], ["cluster", "cluster"], ["stakes", "stakes"],
  ["source_type", "source"], ["narrative_form", "form"], ["length_band", "length"],
  ["status", "gates"],
];

let DATA, active = {}, query = "", current = 0;

function matches(r){
  for (const [key] of AXES){
    const on = active[key];
    if (on && on.size && !on.has(String(r[key] ?? ""))) return false;
  }
  if (!query) return true;
  const q = query.toLowerCase();
  return (r.user + " " + r.reasoning + " " + r.response + " " + r.situation + " "
    + r.scenario_id + " " + (r.source_archetype || "")).toLowerCase().includes(q);
}

function visible(){ return DATA.rows.filter(matches); }

function renderFilters(){
  const host = document.getElementById("filters");
  host.innerHTML = AXES.map(([key, label]) => {
    const vals = [...new Set(DATA.rows.map(r => String(r[key] ?? "")))].filter(Boolean).sort();
    if (!vals.length) return "";
    return `<div class="fgroup"><h3>${esc(label)}</h3><div class="filters">` +
      vals.map(v => `<button class="chip" data-key="${esc(key)}" data-val="${esc(v)}"
        aria-pressed="false">${esc(v)}</button>`).join("") + `</div></div>`;
  }).join("");
  host.querySelectorAll(".chip").forEach(b => b.addEventListener("click", () => {
    const {key, val} = b.dataset;
    active[key] = active[key] || new Set();
    active[key].has(val) ? active[key].delete(val) : active[key].add(val);
    b.setAttribute("aria-pressed", active[key].has(val));
    current = 0; renderList(); renderDoc();
  }));
}

function renderList(){
  const rows = visible();
  document.getElementById("count").textContent =
    `${rows.length} of ${DATA.rows.length} rows` + (query ? ` matching "${query}"` : "");
  document.getElementById("list").innerHTML = rows.map((r, i) => `
    <button class="item" data-i="${i}" aria-current="${i === current}">
      <span class="id"><span>${esc(r.scenario_id)}</span><span>${esc(r.cluster)}</span></span>
      <span class="ttl">${esc(r.situation.slice(0, 150))}</span>
      <span class="meta">
        <span class="tok">${n(r.trainable)} tok</span>
        <span>${esc(r.stakes)}</span><span>${esc(r.narrative_form)}</span>
        ${r.source_archetype ? `<span>&#8594; ${esc(r.source_archetype)}</span>` : ""}
        ${r.selected ? `<span class="pick">picked</span>` : ""}
        ${r.status !== "accepted" ? `<span class="flag">${esc(r.status)}</span>` : ""}
      </span>
    </button>`).join("") || `<div class="empty" style="padding:20px 16px">Nothing matches.</div>`;
  document.getElementById("list").querySelectorAll(".item").forEach(b =>
    b.addEventListener("click", () => { current = +b.dataset.i; renderList(); renderDoc();
      document.getElementById("main").scrollTop = 0; }));
}

function summary(){
  const s = DATA.summary;
  const cell = (k, v, note) => `<div class="stat"><div class="k">${esc(k)}</div>
    <div class="v">${esc(v)}</div><div class="n">${esc(note)}</div></div>`;
  return `<section class="summary"><h2>${esc(s.title)}</h2><div class="grid">
    ${cell("rows", n(s.rows), s.selected ? `${s.selected} selected` : "all candidates")}
    ${cell("trainable / row", n(s.mean_trainable), `DA-716: ${n(s.da_mean_trainable)}`)}
    ${cell("CoT share", s.cot_share, `DA-716: ${s.da_cot_share}`)}
    ${cell("median CoT", n(s.median_cot), `DA-716: ${n(s.da_median_cot)}`)}
    ${cell("median reply", n(s.median_answer), `DA-716: ${n(s.da_median_answer)}`)}
    ${cell("projected over 716", n(s.projected_716), `DA-716: ${n(s.da_total)}`)}
  </div></section>`;
}

function card(title, right, body, cls = "", bodyCls = ""){
  return `<section class="card ${cls}"><div class="card-h"><span>${esc(title)}</span>
    <span class="r">${esc(right)}</span></div>
    <div class="card-b ${bodyCls}">${esc(body)}</div></section>`;
}

function renderDoc(){
  const rows = visible();
  const host = document.getElementById("doc");
  if (!rows.length){ host.innerHTML = summary() + `<div class="empty">No row matches the current filters.</div>`; return; }
  const r = rows[Math.min(current, rows.length - 1)];
  host.innerHTML = summary() + `
    <div class="crumb">
      <h2>${esc(r.trait_name)}</h2>
      <span class="sid">${esc(r.scenario_id)}</span>
      <span class="tag">${esc(r.world)} &middot; ${esc(r.domain)}${r.ai_name ? " &middot; " + esc(r.ai_name) : ""}</span>
      <span class="tag">${esc(r.stakes)} &middot; ${esc(r.narrative_form)} &middot; ${esc(r.length_band)}</span>
      <span class="tag">${esc(r.source_type)}${r.source_archetype ? " &#8594; " + esc(r.source_archetype) : ""}</span>
    </div>
    ${card("System prompt — conditioning, not trained", "", r.system, "", "small")}
    ${card("User message — conditioning, not trained", "", r.user, "", "small")}
    ${card("Trainable reasoning (CoT)", n(r.reasoning) + " tokens", r.reasoning_text, "trained")}
    ${card("Trainable reply", n(r.answer) + " tokens", r.response, "trained")}
    <section class="card"><div class="card-h"><span>Row</span>
      <span class="r">${n(r.trainable)} trainable &middot; CoT/reply ${r.ratio}</span></div>
      <div class="card-b mini"><div class="kv">
        <b>world</b><span>${esc(r.world)} — ${esc(r.world_detail)}</span>
        <b>identity</b><span>${esc(r.ai_name ? r.ai_name + ", " : "")}${esc(r.identity_frame)}</span>
        <b>role</b><span>${esc(r.ai_role)}</span>
        <b>situation</b><span>${esc(r.situation)}</span>
        <b>tempting move</b><span>${esc(r.shortcut)}</span>
        <b>persona gate</b><span>${esc(r.judge_persona)}</span>
        <b>pattern gate</b><span>${esc(r.judge_pattern)}</span>
        <b>status</b><span>${esc(r.status)}</span>
      </div></div></section>
    <details class="card"><summary>Critic findings &amp; rewrite note</summary>
      <div class="card-b small">${esc(r.critique)}\n\n&mdash;\n\n${esc(r.rewrite_changes)}</div>
    </details>`;
}

(() => {
  try {
    const node = document.getElementById("payload");
    if (!node || !node.textContent.trim()) throw new Error("the payload block is empty");
    DATA = JSON.parse(node.textContent);
  } catch (e) {
    die(e.message);
    return;
  }
  document.getElementById("boot").hidden = true;
  document.getElementById("app").hidden = false;
  document.getElementById("sub").textContent = DATA.summary.subtitle;
  renderFilters();
  document.getElementById("q").addEventListener("input", e => {
    query = e.target.value.trim(); current = 0; renderList(); renderDoc();
  });
  document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT") return;
    const rows = visible();
    if (e.key === "j" || e.key === "ArrowDown") { current = Math.min(current + 1, rows.length - 1); }
    else if (e.key === "k" || e.key === "ArrowUp") { current = Math.max(current - 1, 0); }
    else return;
    e.preventDefault(); renderList(); renderDoc();
    document.getElementById("main").scrollTop = 0;
  });
  renderList(); renderDoc();
})();
</script>
"""

DA = {"mean_trainable": 1162, "cot_share": "50.6%", "median_cot": 584,
      "median_answer": 557, "total": 832_064}


def main(run: str, out: str = "", limit: int = 0, title: str = "") -> None:
    """Build the browser for one run.

    Args:
        run: Run directory holding the export and `token_stats.json`.
        out: Output HTML path; defaults to `<run>/browser.html`.
        limit: Keep only the first N rows (for a preview-sized page). 0 = all.
        title: Page heading; defaults to the run directory's name.
    """
    run_dir = Path(run)
    stats = json.loads((run_dir / "token_stats.json").read_text(encoding="utf-8"))
    by_id = {r["scenario_id"]: r for r in stats["records"]}

    selected: set[str] = set()
    sel_path = run_dir / "selection.json"
    if sel_path.exists():
        selected = set(json.loads(sel_path.read_text(encoding="utf-8"))["ids"])

    export = run_dir / "dataset.jsonl"
    if not export.exists():
        export = next(run_dir.glob("stage_*_export_sft.jsonl"))

    rows = []
    for line in export.open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        md = {k: rec["metadata"].get(k, "") for k in META_KEYS}
        tok = by_id.get(md["scenario_id"], {})
        msgs = {m["role"]: m for m in rec["messages"]}
        asst = msgs.get("assistant", {})
        jp = (rec["metadata"].get("judge_persona") or {})
        jt = (rec["metadata"].get("judge_pattern") or {})
        gates_ok = (md["revise_status"] == "ok"
                    and jp.get("verdict") == "accept" and jt.get("verdict") == "accept")
        rows.append({
            **md,
            "cluster": CLUSTER.get(md["trait_id"], md["trait_id"]),
            "system": (msgs.get("system") or {}).get("content", ""),
            "user": (msgs.get("user") or {}).get("content", ""),
            "reasoning_text": asst.get("reasoning_content", ""),
            "response": asst.get("content", ""),
            "trainable": tok.get("trainable", 0), "reasoning": tok.get("reasoning", 0),
            "answer": tok.get("answer", 0), "ratio": tok.get("ratio", 0),
            "judge_persona": f"{jp.get('verdict', '?')} — {jp.get('why', '')}",
            "judge_pattern": f"{jt.get('verdict', '?')}"
                             + (f" — {', '.join(jt.get('present') or [])}"
                                if jt.get("present") else ""),
            "status": "accepted" if gates_ok else (
                md["revise_status"] if md["revise_status"] != "ok" else "gate reject"),
            "selected": md["scenario_id"] in selected,
        })
    rows.sort(key=lambda r: r["scenario_id"])
    if limit:
        rows = rows[:limit]

    v = stats["vs_da716"]
    payload = {
        "rows": rows,
        "summary": {
            "title": title or f"{run_dir.name}: {len(rows)} candidate rows",
            "subtitle": f"{run_dir.name} · Qwen3.6-27B mask · vs DA-716",
            "rows": len(rows), "selected": len(selected),
            "mean_trainable": v["mean_trainable_per_row"],
            "da_mean_trainable": DA["mean_trainable"],
            "cot_share": f"{100 * v['reasoning_share']:.1f}%",
            "da_cot_share": DA["cot_share"],
            "median_cot": stats["per_row"]["reasoning"]["median"],
            "da_median_cot": DA["median_cot"],
            "median_answer": stats["per_row"]["answer"]["median"],
            "da_median_answer": DA["median_answer"],
            "projected_716": v["projected_716_total"], "da_total": DA["total"],
        },
    }
    # `</` -> `<\/` so nothing in the corpus can close the JSON block early. That escape
    # is legal JSON (backslash-solidus) and parses back to `</`, so the data is unchanged.
    blob = json.dumps(payload, ensure_ascii=False).replace("</", r"<\/")
    dest = Path(out) if out else run_dir / "browser.html"
    dest.write_text(HTML.replace("__DATA__", blob), encoding="utf-8")
    print(f"wrote {dest}  ({len(rows)} rows, {dest.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    fire.Fire(main)
