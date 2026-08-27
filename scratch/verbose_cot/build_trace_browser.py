# ABOUTME: Build a self-contained browser for the 716 verbose-CoT rows: original vs
# ABOUTME: expanded reasoning side by side, searchable, with the shared prompt and response.
# Run: uv run python scratch/verbose_cot/build_trace_browser.py [--out <path>]

"""One page holding every row of the expansion, so the rewrite can be read rather than
trusted.

The payload is gzipped and base64'd into the page. Raw it is 10.9 MB, which fits the
artifact ceiling but makes the page slow to fetch and parse; gzipped it is 4.2 MB, and
every browser that can render the page can also run DecompressionStream. The page says so
plainly if that API is missing rather than rendering an empty shell.

`system`, `user` and `response` are carried ONCE per row rather than twice: the expansion
stage rewrote `reasoning` only, so the prompt and the answer are shared between the two
sides by construction. Showing them once is not a space trick, it is the claim being
made -- if they differed, the comparison would not be about verbosity.

The 79 rows that were never expanded (50 that exhausted the fidelity gate's retries, 29
refused by the content filter) are kept in, not filtered out. They carry identical text on
both sides, and a browser that quietly dropped them would misrepresent the corpus.
"""

from __future__ import annotations

import base64
import gzip
import json
import re
import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RUN = ROOT / "output" / "verbose_cot" / "20260825_042004"
STAGE = RUN / "stage_2_expand.jsonl"

# Field names are the stage's own. `source_reasoning` is the difficult-advice trace as it
# arrived; `reasoning` is what the expander returned for it.
KEEP = ("scenario_id", "trait_id", "trait_name", "domain", "expansion_status",
        "system", "user", "source_reasoning", "reasoning", "response")

HTML = r"""<title>716 Traces, Before and After</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,400;7..72,600&family=Archivo:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">

<style>
/* Tokens: the bare :root is the complete light palette; the blocks below redefine only
   tokens, so no colour has its sole definition behind a media query or a theme stamp.
   Series hues validated (dataviz validator, categorical):
     light #8A3D6B / #9A6410 on #FCFBF9 -- PASS all, dE 15.1 deutan, 17.3 normal
     dark  #C2669A / #BA8A28 on #17161A -- PASS all
   Only two hues carry meaning: the accent marks the expanded side, the attention colour
   marks a row that was never expanded. "expanded" is the common case and stays neutral. */
:root {
  --paper:#FCFBF9; --ground:#F3F1EC; --raise:#FFFFFF; --rail:#F7F5F1;
  --ink:#191714; --ink-2:#4E4841; --ink-3:#857D73;
  --rule:#E2DED6; --rule-soft:#EDEAE3;
  --accent:#8A3D6B; --accent-soft:#F3E4EE; --accent-line:#E2C6D8;
  --attn:#9A6410; --attn-soft:#F6EBDA;
  --shadow:0 1px 2px rgba(25,23,20,.05), 0 10px 26px -18px rgba(25,23,20,.28);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper:#17161A; --ground:#101013; --raise:#1F1E23; --rail:#141317;
    --ink:#EDE9E4; --ink-2:#ADA69C; --ink-3:#7C756C;
    --rule:#2E2C31; --rule-soft:#252428;
    --accent:#C2669A; --accent-soft:#33202C; --accent-line:#4A2E3E;
    --attn:#BA8A28; --attn-soft:#2E2617;
    --shadow:0 1px 2px rgba(0,0,0,.45), 0 10px 26px -18px rgba(0,0,0,.8);
  }
}
:root[data-theme="dark"] {
  --paper:#17161A; --ground:#101013; --raise:#1F1E23; --rail:#141317;
  --ink:#EDE9E4; --ink-2:#ADA69C; --ink-3:#7C756C;
  --rule:#2E2C31; --rule-soft:#252428;
  --accent:#C2669A; --accent-soft:#33202C; --accent-line:#4A2E3E;
  --attn:#BA8A28; --attn-soft:#2E2617;
  --shadow:0 1px 2px rgba(0,0,0,.45), 0 10px 26px -18px rgba(0,0,0,.8);
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font:400 15px/1.55 Archivo,ui-sans-serif,system-ui,sans-serif;
  -webkit-font-smoothing:antialiased;
}
.mono{font-family:"JetBrains Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}

/* ---- shell ------------------------------------------------------------------------ */
.app{display:grid;grid-template-columns:326px minmax(0,1fr);height:100vh;overflow:hidden}
@media (max-width:900px){
  .app{grid-template-columns:1fr;height:auto;overflow:visible}
  .rail{height:auto;max-height:none;border-right:0;border-bottom:1px solid var(--rule)}
  .rail-list{max-height:340px}
}

/* ---- rail ------------------------------------------------------------------------- */
.rail{background:var(--rail);border-right:1px solid var(--rule);display:flex;flex-direction:column;min-height:0}
.rail-head{padding:18px 18px 14px;border-bottom:1px solid var(--rule)}
.brand{font:600 15px/1.25 Archivo,sans-serif;letter-spacing:-.01em;margin:0}
.brand span{display:block;font:400 11.5px/1.4 "JetBrains Mono",monospace;color:var(--ink-3);letter-spacing:.04em;margin-top:5px}
.search{
  width:100%;margin-top:14px;padding:9px 11px;border:1px solid var(--rule);border-radius:7px;
  background:var(--raise);color:var(--ink);font:400 13.5px/1.3 Archivo,sans-serif;
}
.search::placeholder{color:var(--ink-3)}
.search:focus{outline:2px solid var(--accent);outline-offset:1px;border-color:transparent}
.filters{display:flex;gap:5px;margin-top:10px;flex-wrap:wrap}
.chip{
  appearance:none;border:1px solid var(--rule);background:transparent;color:var(--ink-2);
  font:500 10.5px/1 "JetBrains Mono",monospace;letter-spacing:.05em;text-transform:uppercase;
  padding:6px 9px;border-radius:20px;cursor:pointer;
}
.chip:hover{color:var(--ink);border-color:var(--ink-3)}
.chip[aria-pressed="true"]{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.chip:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.count{padding:9px 18px;font:400 11.5px/1.4 "JetBrains Mono",monospace;color:var(--ink-3);border-bottom:1px solid var(--rule-soft)}
.rail-list{overflow-y:auto;flex:1;min-height:0}

.item{
  display:block;width:100%;text-align:left;border:0;border-bottom:1px solid var(--rule-soft);
  background:transparent;padding:11px 18px;cursor:pointer;color:inherit;font:inherit;
}
.item:hover{background:var(--raise)}
.item[aria-current="true"]{background:var(--raise);box-shadow:inset 3px 0 0 var(--accent)}
.item:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.item .id{font:500 11px/1.3 "JetBrains Mono",monospace;color:var(--ink-3);display:flex;justify-content:space-between;gap:8px}
.item .ttl{font:400 13px/1.4 Archivo,sans-serif;color:var(--ink);margin-top:3px;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.item .meta{font:400 11px/1.3 "JetBrains Mono",monospace;color:var(--ink-3);margin-top:5px;display:flex;gap:8px;align-items:center}
.ratio{color:var(--accent);font-weight:500}
.flag{
  font:500 9.5px/1 "JetBrains Mono",monospace;letter-spacing:.06em;text-transform:uppercase;
  color:var(--attn);background:var(--attn-soft);padding:3px 6px;border-radius:4px;
}

/* ---- main ------------------------------------------------------------------------- */
.main{overflow-y:auto;min-height:0;background:var(--ground)}
.doc{max-width:1180px;margin:0 auto;padding:26px clamp(14px,2.4vw,30px) 80px}

.crumb{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:baseline;margin-bottom:18px}
.crumb h2{margin:0;font:600 19px/1.25 Archivo,sans-serif;letter-spacing:-.01em}
.crumb .sid{font:500 12px/1 "JetBrains Mono",monospace;color:var(--ink-3)}
.crumb .dom{font:400 13px/1.4 Archivo,sans-serif;color:var(--ink-2)}

.card{background:var(--paper);border:1px solid var(--rule);border-radius:10px;box-shadow:var(--shadow)}
.card + .card, .cols + .card, .card + .cols{margin-top:14px}
.card-h{
  display:flex;justify-content:space-between;align-items:center;gap:12px;
  padding:10px 16px;border-bottom:1px solid var(--rule-soft);
  font:500 10.5px/1 "JetBrains Mono",monospace;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-3);
}
.card-b{padding:16px 18px;font:400 15px/1.68 Literata,Georgia,serif;color:var(--ink);white-space:pre-wrap;overflow-wrap:anywhere}
.card-b.small{font-size:14px;line-height:1.6;color:var(--ink-2)}

/* the two reasoning columns; unequal height is the point, so they are not stretched */
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start;margin-top:14px}
@media (max-width:860px){.cols{grid-template-columns:1fr}}
.col.after{border-color:var(--accent-line)}
.col.after .card-h{color:var(--accent)}
ins.add{background:var(--accent-soft);text-decoration:none;border-radius:2px;
  box-shadow:0 0 0 1px var(--accent-line) inset}
.tail{padding:0 18px 16px;font:400 11.5px/1.4 "JetBrains Mono",monospace;color:var(--ink-3)}

.identical{
  margin-top:14px;border:1px solid var(--attn);background:var(--attn-soft);border-radius:10px;
  padding:12px 16px;font:400 13.5px/1.55 Archivo,sans-serif;color:var(--ink);
}
.identical b{color:var(--attn)}

.tog{
  appearance:none;border:1px solid var(--rule);background:var(--raise);color:var(--ink-2);
  font:500 10.5px/1 "JetBrains Mono",monospace;letter-spacing:.06em;text-transform:uppercase;
  padding:7px 10px;border-radius:6px;cursor:pointer;
}
.tog[aria-pressed="true"]{background:var(--accent);color:#fff;border-color:var(--accent)}
.tog:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:16px}
.hint{font:400 11.5px/1.4 "JetBrains Mono",monospace;color:var(--ink-3);margin-left:auto}
kbd{font:500 10.5px/1 "JetBrains Mono",monospace;border:1px solid var(--rule);border-bottom-width:2px;
  border-radius:4px;padding:3px 5px;color:var(--ink-2);background:var(--raise)}

.boot{display:flex;align-items:center;justify-content:center;height:100vh;padding:30px;text-align:center}
.boot p{max-width:46ch;color:var(--ink-2);font-size:14.5px;line-height:1.6}
mark{background:var(--attn-soft);color:inherit;border-radius:2px;padding:0 1px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>

<div id="boot" class="boot"><p>Unpacking 716 traces…</p></div>
<div class="app" id="app" hidden>
  <aside class="rail">
    <div class="rail-head">
      <h1 class="brand">Verbose-CoT corpus<span id="sub"></span></h1>
      <input id="q" class="search" type="search" placeholder="Search prompts, reasoning, response…" autocomplete="off">
      <div class="filters" id="filters"></div>
    </div>
    <div class="count" id="count"></div>
    <div class="rail-list" id="list" role="listbox" aria-label="Traces"></div>
  </aside>
  <main class="main" id="main"><div class="doc" id="doc"></div></main>
</div>

<script>
const B64 = "__DATA__";

async function unpack() {
  const bytes = Uint8Array.from(atob(B64), c => c.charCodeAt(0));
  if (typeof DecompressionStream !== "function") throw new Error("no DecompressionStream");
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return JSON.parse(await new Response(stream).text());
}

const words = s => s.split(/\s+/).filter(Boolean).length;
const esc = s => s.replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

/* Word-level LCS, used only to tint what the expander added. Capped: the largest row here
   is ~2.1M cells, comfortably inside the limit, but a bigger corpus should degrade to
   plain text rather than allocate unboundedly. */
const CELL_CAP = 4e6;
function addedMask(a, b) {
  const A = a.split(/(\s+)/).filter(t => t.trim()), B = b.split(/(\s+)/).filter(t => t.trim());
  if (A.length * B.length > CELL_CAP) return null;
  const n = A.length, m = B.length, W = m + 1;
  const dp = new Uint16Array((n + 1) * W);
  const key = t => t.toLowerCase().replace(/[^\p{L}\p{N}]/gu, "");
  const ka = A.map(key), kb = B.map(key);
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      dp[i * W + j] = ka[i] === kb[j]
        ? dp[(i + 1) * W + j + 1] + 1
        : Math.max(dp[(i + 1) * W + j], dp[i * W + j + 1]);
  const isNew = new Array(m).fill(true);
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (ka[i] === kb[j]) { isNew[j] = false; i++; j++; }
    else if (dp[(i + 1) * W + j] >= dp[i * W + j + 1]) i++;
    else j++;
  }
  return { tokens: B, isNew };
}

function renderTrace(el, text, mask) {
  el.textContent = "";
  if (!mask) { el.textContent = text; return; }
  // Rebuild with original whitespace so the prose still reads as prose.
  const parts = text.split(/(\s+)/);
  let k = 0;
  for (const p of parts) {
    if (!p.trim()) { el.appendChild(document.createTextNode(p)); continue; }
    const node = mask.isNew[k]
      ? Object.assign(document.createElement("ins"), { className: "add", textContent: p })
      : document.createTextNode(p);
    el.appendChild(node);
    k++;
  }
}

(async () => {
  let ROWS;
  try { ROWS = await unpack(); }
  catch (e) {
    document.getElementById("boot").innerHTML =
      "<p>This page needs <code>DecompressionStream</code> to unpack its data, and this " +
      "browser does not provide it. Open it in a current Chrome, Edge, Firefox or Safari.</p>";
    return;
  }

  ROWS.forEach((r, i) => {
    r.i = i;
    r.wa = words(r.source_reasoning);
    r.wb = words(r.reasoning);
    r.ratio = r.wa ? r.wb / r.wa : 1;
    r.same = r.source_reasoning === r.reasoning;
    r.hay = [r.scenario_id, r.trait_name, r.domain, r.system, r.user,
             r.source_reasoning, r.reasoning, r.response].join("\n").toLowerCase();
  });

  const STATUS = [["all", "All"], ["expanded", "Expanded"], ["fallback", "Fallback"], ["refused", "Refused"]];
  let filter = "all", query = "", shown = ROWS, current = 0, diff = true;
  const cache = new Map();

  const $ = id => document.getElementById(id);
  const nExp = ROWS.filter(r => r.expansion_status === "expanded").length;
  $("sub").textContent = `${ROWS.length} rows · ${nExp} expanded`;

  $("filters").innerHTML = STATUS.map(([k, label]) => {
    const n = k === "all" ? ROWS.length : ROWS.filter(r => r.expansion_status === k).length;
    return `<button class="chip" data-k="${k}" aria-pressed="${k === "all"}">${label} ${n}</button>`;
  }).join("");
  $("filters").addEventListener("click", e => {
    const b = e.target.closest(".chip"); if (!b) return;
    filter = b.dataset.k;
    [...$("filters").children].forEach(c => c.setAttribute("aria-pressed", String(c === b)));
    apply();
  });

  let timer;
  $("q").addEventListener("input", e => {
    clearTimeout(timer);
    timer = setTimeout(() => { query = e.target.value.trim().toLowerCase(); apply(); }, 160);
  });

  function apply() {
    shown = ROWS.filter(r =>
      (filter === "all" || r.expansion_status === filter) &&
      (!query || r.hay.includes(query)));
    $("count").textContent = shown.length === ROWS.length
      ? `${ROWS.length} traces`
      : `${shown.length} of ${ROWS.length} traces`;
    const frag = document.createDocumentFragment();
    for (const r of shown) {
      const b = document.createElement("button");
      b.className = "item";
      b.setAttribute("role", "option");
      b.dataset.i = r.i;
      b.setAttribute("aria-current", String(r.i === current));
      b.innerHTML =
        `<div class="id"><span>${esc(r.scenario_id)}</span><span>${esc(r.trait_id)}</span></div>
         <div class="ttl">${esc(r.domain || r.trait_name)}</div>
         <div class="meta"><span>${r.wa} → ${r.wb}w</span>
           <span class="ratio">${r.ratio.toFixed(2)}×</span>
           ${r.expansion_status === "expanded" ? "" : `<span class="flag">${esc(r.expansion_status)}</span>`}
         </div>`;
      frag.appendChild(b);
    }
    const list = $("list");
    list.textContent = "";
    list.appendChild(frag);
    if (!shown.some(r => r.i === current) && shown.length) select(shown[0].i);
  }

  $("list").addEventListener("click", e => {
    const b = e.target.closest(".item"); if (b) select(Number(b.dataset.i));
  });

  function select(i) {
    current = i;
    [...$("list").children].forEach(c =>
      c.setAttribute("aria-current", String(Number(c.dataset.i) === i)));
    const el = [...$("list").children].find(c => Number(c.dataset.i) === i);
    if (el) el.scrollIntoView({ block: "nearest" });
    draw();
  }

  function draw() {
    const r = ROWS[current];
    const doc = $("doc");
    doc.innerHTML =
      `<div class="bar">
         <button class="tog" id="dtog" aria-pressed="${diff}">Highlight added text</button>
         <span class="hint"><kbd>↑</kbd><kbd>↓</kbd> move · <kbd>/</kbd> search</span>
       </div>
       <div class="crumb">
         <h2>${esc(r.trait_name)}</h2>
         <span class="sid">${esc(r.scenario_id)}</span>
         <span class="dom">${esc(r.domain || "")}</span>
       </div>

       <div class="card"><div class="card-h"><span>System prompt</span><span>${words(r.system)} words</span></div>
         <div class="card-b small" id="c-sys"></div></div>

       <div class="card"><div class="card-h"><span>User</span><span>${words(r.user)} words</span></div>
         <div class="card-b" id="c-usr"></div></div>

       ${r.same ? `<div class="identical"><b>Not expanded (${esc(r.expansion_status)}).</b>
          The original reasoning was carried through unchanged, so both columns below are
          identical. ${r.expansion_status === "refused"
            ? "The expander's provider refused this prompt."
            : "The fidelity gate rejected every attempt within its retry budget."}</div>` : ""}

       <div class="cols">
         <div class="card col"><div class="card-h"><span>Original reasoning</span><span>${r.wa} words</span></div>
           <div class="card-b" id="c-a"></div></div>
         <div class="card col after"><div class="card-h"><span>Verbose reasoning</span>
             <span>${r.wb} words · ${r.ratio.toFixed(2)}×</span></div>
           <div class="card-b" id="c-b"></div><div class="tail" id="c-tail"></div></div>
       </div>

       <div class="card"><div class="card-h"><span>Assistant response — shared, not rewritten</span>
           <span>${words(r.response)} words</span></div>
         <div class="card-b" id="c-res"></div></div>`;

    $("c-sys").textContent = r.system;
    $("c-usr").textContent = r.user;
    $("c-a").textContent = r.source_reasoning;
    $("c-res").textContent = r.response;

    let mask = null;
    if (diff && !r.same) {
      if (!cache.has(r.i)) cache.set(r.i, addedMask(r.source_reasoning, r.reasoning));
      mask = cache.get(r.i);
    }
    renderTrace($("c-b"), r.reasoning, mask);
    $("c-tail").textContent = mask
      ? `${mask.isNew.filter(Boolean).length} of ${mask.isNew.length} words are new`
      : (diff && !r.same ? "trace too long to diff — showing plain text" : "");

    $("dtog").addEventListener("click", () => {
      diff = !diff;
      draw();
      $("main").scrollTop = 0;
    });
    $("main").scrollTop = 0;
  }

  addEventListener("keydown", e => {
    if (e.key === "/" && document.activeElement !== $("q")) { e.preventDefault(); $("q").focus(); return; }
    if (document.activeElement === $("q")) return;
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    e.preventDefault();
    const at = shown.findIndex(r => r.i === current);
    const next = shown[at + (e.key === "ArrowDown" ? 1 : -1)];
    if (next) select(next.i);
  });

  apply();
  select(ROWS[0].i);
  $("boot").hidden = true;
  $("app").hidden = false;
})();
</script>
"""


def main(out: str = "output/verbose_cot/trace_browser.html",
         limit: int | None = None) -> None:
    """Build the page. `limit` trims the corpus so the same code path can be loaded in a
    preview tab, which serves files as data URLs and cannot take the full 4 MB payload."""
    assert STAGE.is_file(), f"missing {STAGE}"
    rows = [json.loads(l) for l in STAGE.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        # Keep one of each status so the not-expanded path is exercised too.
        pick = {s: [r for r in rows if r["expansion_status"] == s][:max(1, limit // 3)]
                for s in ("expanded", "fallback", "refused")}
        rows = [r for group in pick.values() for r in group]

    slim = []
    for r in rows:
        row = {k: r.get(k) for k in KEEP}
        for f in ("system", "user", "source_reasoning", "reasoning", "response"):
            # Normalise newlines only; the text itself is left exactly as generated.
            row[f] = str(row[f] or "").replace("\r\n", "\n").strip()
        assert row["source_reasoning"] and row["reasoning"], row["scenario_id"]
        slim.append(row)

    payload = json.dumps(slim, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    blob = base64.b64encode(gzip.compress(payload, 9)).decode("ascii")

    html = HTML.replace("__DATA__", blob)
    path = ROOT / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")

    same = sum(1 for r in slim if r["source_reasoning"] == r["reasoning"])
    status = {}
    for r in slim:
        status[r["expansion_status"]] = status.get(r["expansion_status"], 0) + 1
    print(f"rows            {len(slim)}")
    print(f"status          {status}")
    print(f"identical sides {same} (the rows that were never expanded)")
    print(f"payload         {len(payload)/1e6:.2f} MB -> gz+b64 {len(blob)/1e6:.2f} MB")
    print(f"page            {path.stat().st_size/1e6:.2f} MB  {path}")
    assert path.stat().st_size < 16e6, "over the artifact ceiling"


if __name__ == "__main__":
    fire.Fire(main)
