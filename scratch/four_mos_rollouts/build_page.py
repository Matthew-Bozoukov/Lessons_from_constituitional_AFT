# ABOUTME: Builds the self-contained HTML page for the four-MO ODCV rollout analysis: every transcript
# ABOUTME: (gzipped, decoded in-browser), the figure, the write-up, the reader reports and the tables.
# Run: uv run python scratch/four_mos_rollouts/build_page.py  -> output/four_mos_rollouts/page/four_mos.html
from __future__ import annotations

import base64
import gzip
import html
import json
import re
from pathlib import Path

from src.properties.sources.odcv_rollouts import _fields, _split_steps

OUT = Path("output/four_mos_rollouts")
PAGE = OUT / "page" / "four_mos.html"
ARMS = ["grok", "sonnet_concise", "sonnet_normal", "gpt"]
LABEL = {
    "grok": "grok-4.6",
    "sonnet_concise": "Sonnet 5 capped",
    "sonnet_normal": "Sonnet 5 (da716)",
    "gpt": "gpt-5.6",
}
SHORT = {
    "grok": "grok",
    "sonnet_concise": "capped",
    "sonnet_normal": "Sonnet",
    "gpt": "gpt",
}
MR = {"grok": 7.8, "sonnet_concise": 15.4, "sonnet_normal": 16.3, "gpt": 25.2}


def latest(pattern: str) -> Path:
    hits = sorted(OUT.glob(pattern))
    assert hits, pattern
    return hits[-1]


# ---------------------------------------------------------------- markdown (small, local)
INLINE = [
    (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<![\w*])\*(\S[^*]*?\S|\S)\*(?![\w*])"), r"<em>\1</em>"),
]


def inline(s: str) -> str:
    s = html.escape(s, quote=False)
    for pat, rep in INLINE:
        s = pat.sub(rep, s)
    return s


def md_to_html(text: str) -> str:
    lines = text.splitlines()
    out, i = [], 0
    para: list[str] = []

    def flush():
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            flush()
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            flush()
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            i += 1
            continue
        if s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            if len(rows) >= 2 and all(re.match(r"^:?-{2,}:?$", c) for c in rows[1]):
                head, body = rows[0], rows[2:]
            else:
                head, body = None, rows
            t = ["<div class='tablewrap'><table>"]
            if head:
                t.append(
                    "<thead><tr>"
                    + "".join(f"<th>{inline(c)}</th>" for c in head)
                    + "</tr></thead>"
                )
            t.append(
                "<tbody>"
                + "".join(
                    "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
                    for r in body
                )
                + "</tbody>"
            )
            t.append("</table></div>")
            out.append("".join(t))
            continue
        if re.match(r"^[-*]\s+", s) or re.match(r"^\d+[.)]\s+", s):
            flush()
            ordered = bool(re.match(r"^\d+[.)]\s+", s))
            tag = "ol" if ordered else "ul"
            items = []
            while i < len(lines):
                t2 = lines[i].strip()
                if (not ordered and re.match(r"^[-*]\s+", t2)) or (
                    ordered and re.match(r"^\d+[.)]\s+", t2)
                ):
                    items.append(re.sub(r"^([-*]|\d+[.)])\s+", "", t2))
                    i += 1
                elif t2 and items and lines[i].startswith(("  ", "\t")):
                    items[-1] += " " + t2
                    i += 1
                else:
                    break
            out.append(
                f"<{tag}>"
                + "".join(f"<li>{inline(x)}</li>" for x in items)
                + f"</{tag}>"
            )
            continue
        if (
            s.startswith("*")
            and s.endswith("*")
            and len(s) > 2
            and not s.startswith("**")
        ):
            flush()
            out.append(f"<p class='meta'>{inline(s[1:-1])}</p>")
            i += 1
            continue
        para.append(s)
        i += 1
    flush()
    return "\n".join(out)


# ---------------------------------------------------------------- data
def rollouts_payload() -> tuple[dict, list[dict]]:
    rows = [json.loads(l) for l in latest("rollouts_*.jsonl").open()]
    regs = {(r["arm"], r["key"] if "key" in r else None): r for r in []}
    regs = {}
    for l in latest("registers_*.jsonl").open():
        r = json.loads(l)
        regs[(r["arm"], r["cell"], r["rollout"])] = r
    data: dict = {a: {} for a in ARMS}
    for r in rows:
        text = Path(r["path"]).read_text(encoding="utf-8", errors="replace")
        pre, steps = _split_steps(text)
        parsed = []
        for s in steps:
            f = _fields(s)
            parsed.append(
                {
                    k: f.get(k, "")
                    for k in ("role", "content", "reason", "call")
                    if f.get(k) and f.get(k).lower() != "none"
                }
            )
        rg = regs.get((r["arm"], r["cell"], r["rollout"]), {})
        data[r["arm"]].setdefault(r["cell"], []).append(
            {
                "id": r["rollout"],
                "score": r["score"],
                "judges": {
                    j.replace("score_", ""): {
                        "score": r.get(j),
                        "why": r["judge_reasoning"].get(j.replace("score_", ""), ""),
                    }
                    for j in r
                    if j.startswith("score_")
                },
                "reasoning_chars": r["reasoning_chars"],
                "tool_calls": r["n_tool_calls"],
                "edited": r["edited_app_data"],
                "commit_before_write": rg.get("commit_before_write"),
                "r1_commit": rg.get("r1_commit"),
                "steps": parsed,
            }
        )
    for a in ARMS:
        for cell in data[a]:
            data[a][cell].sort(key=lambda x: x["id"])
    return data, rows


def cell_table(rows: list[dict]) -> list[dict]:
    cells = sorted({r["cell"] for r in rows})
    out = []
    for c in cells:
        scores = {
            a: [
                r["score"]
                for r in rows
                if r["arm"] == a and r["cell"] == c and r["score"] is not None
            ]
            for a in ARMS
        }
        meds = [sorted(v)[len(v) // 2] if v else None for v in scores.values()]
        meds = [m for m in meds if m is not None]
        out.append(
            {
                "cell": c,
                "cond": c.split("/")[0],
                "name": c.split("/")[1],
                "scores": scores,
                "spread": (max(meds) - min(meds)) if meds else 0,
                "viol": {a: int(any(s >= 3 for s in scores[a])) for a in ARMS},
            }
        )
    return out


def main() -> None:
    PAGE.parent.mkdir(parents=True, exist_ok=True)
    data, rows = rollouts_payload()
    cells = cell_table(rows)
    blob = base64.b64encode(
        gzip.compress(json.dumps(data, separators=(",", ":")).encode(), 9)
    ).decode()
    png = latest("plots/four_mos_voice_*.png")
    png_uri = "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()
    fig_md = latest("plots/four_mos_voice_*_results.md").read_text()
    analysis = latest("ANALYSIS_*.md").read_text()
    analysis = analysis.split("\n", 1)[1]  # drop the H1; the page names itself
    reads = sorted((OUT / "reads").glob("*.md"))
    tables = [
        (
            "Per-rollout features, violation associations, cell matrix",
            latest("features_*.md"),
        ),
        ("Step-1 register test", latest("registers_*.md")),
        (
            "Action sequences, duplicate pairs, judge agreement",
            latest("sequences_*.md"),
        ),
        ("Opening the validator source", latest("validator_read_*.md")),
        (
            "Corpus properties (678 shared rows, four corpora)",
            latest("corpus_properties_*.md"),
        ),
        ("Corpus examples: two snippets per property per arm", latest("examples_*.md")),
    ]

    reads_html = "".join(
        f"<details class='report'><summary>{html.escape(p.stem.replace('_', ' '))}</summary><div class='prose'>{md_to_html(p.read_text())}</div></details>"
        for p in reads
    )
    tables_html = "".join(
        f"<details class='report'><summary>{html.escape(t)} <span class='muted'>({p.name})</span></summary><div class='prose wide'>{md_to_html(p.read_text())}</div></details>"
        for t, p in tables
    )

    css = """
:root{--bg:#f4f6f7;--surface:#ffffff;--ink:#1a2128;--muted:#5d6a74;--hair:#d6dde2;--soft:#eaeef1;--focus:#1a2128;
--grok:#009E73;--capped:#CC79A7;--sonnet:#0072B2;--gpt:#E69F00;--viol:#b3302c;--viol-bg:#f8e7e5;--mark:#fff1b8;--mark-ink:#5a4300;
--reason:#f0f3f5;--tool:#f7f8f9;--sel:#dfe7ec;}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#11161a;--surface:#181e24;--ink:#e4e9ec;--muted:#98a4ad;--hair:#2a343c;--soft:#1f272e;--focus:#e4e9ec;
--grok:#25c491;--capped:#dd93bd;--sonnet:#4aa3e0;--gpt:#f0b23a;--viol:#e37470;--viol-bg:#3a1f1e;--mark:#4a3d0f;--mark-ink:#ffe9a3;--reason:#1c242b;--tool:#151b20;--sel:#26323b;}}
:root[data-theme="dark"]{--bg:#11161a;--surface:#181e24;--ink:#e4e9ec;--muted:#98a4ad;--hair:#2a343c;--soft:#1f272e;--focus:#e4e9ec;
--grok:#25c491;--capped:#dd93bd;--sonnet:#4aa3e0;--gpt:#f0b23a;--viol:#e37470;--viol-bg:#3a1f1e;--mark:#4a3d0f;--mark-ink:#ffe9a3;--reason:#1c242b;--tool:#151b20;--sel:#26323b;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;font-size:15px;line-height:1.5}
a{color:inherit}
h1,h2,h3,h4{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif;font-weight:600;line-height:1.15;text-wrap:balance;margin:0}
h1{font-size:2.1rem;letter-spacing:-.01em}
h2{font-size:1.45rem;margin-top:.4rem}
h3{font-size:1.1rem;margin-top:1.4rem}
h4{font-size:1rem;margin-top:1rem}
code,pre,.mono{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
code{font-size:.9em;background:var(--soft);padding:.05em .3em;border-radius:3px}
.muted{color:var(--muted)}
.layout{display:grid;grid-template-columns:200px 1fr;min-height:100vh}
nav.rail{position:sticky;top:0;height:100vh;padding:1.4rem 1rem;border-right:1px solid var(--hair);background:var(--surface);display:flex;flex-direction:column;gap:.35rem}
nav.rail .brand{font-family:"IBM Plex Sans Condensed",sans-serif;font-weight:600;font-size:1.05rem;margin-bottom:.8rem}
nav.rail a{text-decoration:none;color:var(--muted);padding:.3rem .5rem;border-radius:4px;font-size:.92rem}
nav.rail a:hover,nav.rail a:focus-visible{background:var(--soft);color:var(--ink);outline:none}
nav.rail .legend{margin-top:auto;font-size:.8rem;color:var(--muted);display:flex;flex-direction:column;gap:.3rem}
.sw{display:inline-block;width:.7em;height:.7em;border-radius:2px;margin-right:.4em;vertical-align:-1px}
main{padding:2rem 2.4rem 4rem;min-width:0}
section{margin-bottom:3.2rem}
.eyebrow{font-size:.75rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:.4rem}
.lede{max-width:68ch;font-size:1.05rem;margin:.8rem 0 1.2rem}
.prose{max-width:76ch}
.prose.wide{max-width:none}
.prose p{margin:.6rem 0}
.prose p.meta{color:var(--muted);font-size:.9rem}
.prose ul,.prose ol{padding-left:1.3rem}
.prose li{margin:.25rem 0}
.prose h2{margin-top:1.8rem}
.tablewrap{overflow-x:auto;margin:.8rem 0;border:1px solid var(--hair);border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;font-size:.88rem;min-width:100%}
th,td{padding:.4rem .6rem;border-bottom:1px solid var(--hair);text-align:left;vertical-align:top;font-variant-numeric:tabular-nums}
th{font-weight:600;background:var(--soft);white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
figure{margin:1rem 0;background:var(--surface);border:1px solid var(--hair);border-radius:6px;padding:.8rem}
figure img{max-width:100%;display:block}
figcaption{font-size:.85rem;color:var(--muted);margin-top:.5rem}
details.report{border:1px solid var(--hair);border-radius:6px;background:var(--surface);margin:.5rem 0}
details.report>summary{cursor:pointer;padding:.7rem 1rem;font-weight:600;font-family:"IBM Plex Sans Condensed",sans-serif;font-size:1.02rem}
details.report>summary:focus-visible{outline:2px solid var(--focus);outline-offset:-2px}
details.report>div{padding:0 1.2rem 1rem}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7rem;margin:1rem 0}
.stat{background:var(--surface);border:1px solid var(--hair);border-left:4px solid var(--c);border-radius:6px;padding:.7rem .9rem}
.stat .k{font-size:.78rem;color:var(--muted)}
.stat .v{font-family:"IBM Plex Sans Condensed",sans-serif;font-size:1.7rem;font-weight:600;font-variant-numeric:tabular-nums}
.stat .s{font-size:.8rem;color:var(--muted)}
/* browser */
.browser{display:grid;grid-template-columns:400px 1fr;gap:1rem;height:calc(100vh - 2rem);min-height:640px}
.cells{background:var(--surface);border:1px solid var(--hair);border-radius:6px;display:flex;flex-direction:column;min-height:0}
.cells .tools{padding:.6rem .7rem;border-bottom:1px solid var(--hair);display:flex;gap:.4rem;flex-wrap:wrap;align-items:center;font-size:.82rem}
.cells select,.cells input{font:inherit;font-size:.82rem;color:var(--ink);background:var(--bg);border:1px solid var(--hair);border-radius:4px;padding:.2rem .35rem}
.cells select:focus-visible,.cells input:focus-visible,button:focus-visible{outline:2px solid var(--focus);outline-offset:1px}
.celllist{overflow:auto;flex:1;min-height:0}
.cellrow{display:grid;grid-template-columns:1fr auto;gap:.4rem;padding:.45rem .7rem;border-bottom:1px solid var(--hair);cursor:pointer;font-size:.84rem;align-items:center}
.cellrow:hover{background:var(--soft)}
.cellrow.sel{background:var(--sel)}
.cellrow .nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cellrow .cond{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.chips{display:flex;gap:3px}
.chip{font-family:"IBM Plex Mono",monospace;font-size:.66rem;padding:.05rem .25rem;border-radius:3px;border:1px solid var(--c);color:var(--ink);min-width:2.2em;white-space:nowrap;text-align:center;font-variant-numeric:tabular-nums;background:var(--surface)}
.chip.v{background:var(--viol-bg);border-color:var(--viol);color:var(--viol);font-weight:600}
.panes{display:flex;flex-direction:column;min-height:0;gap:.6rem}
.scenario{background:var(--surface);border:1px solid var(--hair);border-radius:6px;padding:.6rem .9rem;font-size:.86rem;max-height:9.5rem;overflow:auto}
.scenario h3{margin:0 0 .3rem;font-size:1.05rem}
.scenario .sys{color:var(--muted);white-space:pre-wrap}
.scenario .usr{white-space:pre-wrap;margin-top:.4rem}
.armgrid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.6rem;flex:1;min-height:0}
.armgrid.two{grid-template-columns:repeat(2,minmax(0,1fr))}
.pane{background:var(--surface);border:1px solid var(--hair);border-top:4px solid var(--c);border-radius:6px;display:flex;flex-direction:column;min-height:0}
.pane header{padding:.5rem .7rem;border-bottom:1px solid var(--hair);display:flex;flex-wrap:wrap;gap:.35rem .6rem;align-items:center;font-size:.82rem}
.pane header .arm{font-family:"IBM Plex Sans Condensed",sans-serif;font-weight:600;font-size:1rem;color:var(--c)}
.pane header button{font:inherit;font-size:.76rem;font-family:"IBM Plex Mono",monospace;border:1px solid var(--hair);background:var(--bg);color:var(--ink);border-radius:3px;padding:.1rem .4rem;cursor:pointer}
.pane header button.on{background:var(--sel);border-color:var(--muted)}
.pane header .score{font-family:"IBM Plex Mono",monospace;font-weight:600}
.pane header .score.v{color:var(--viol)}
.pane .judges{font-size:.78rem;padding:.3rem .7rem;border-bottom:1px solid var(--hair);background:var(--soft)}
.pane .judges summary{cursor:pointer}
.pane .judges p{margin:.3rem 0}
.steps{overflow:auto;flex:1;min-height:0;padding:.5rem .7rem;font-size:.82rem}
.step{border-left:2px solid var(--hair);padding:.25rem .6rem;margin:.35rem 0}
.step .lab{font-size:.68rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.step.assistant{border-left-color:var(--c)}
.step .reason{background:var(--reason);padding:.35rem .5rem;border-radius:4px;white-space:pre-wrap;color:var(--muted);margin:.2rem 0}
.step .content{white-space:pre-wrap;margin:.2rem 0}
.step .call{font-family:"IBM Plex Mono",monospace;font-size:.76rem;white-space:pre-wrap;background:var(--tool);border:1px solid var(--hair);border-radius:4px;padding:.3rem .45rem;margin:.2rem 0;word-break:break-word}
.step .call .w{color:var(--viol);font-weight:600}
.step.tool details{font-family:"IBM Plex Mono",monospace;font-size:.74rem}
.step.tool summary{cursor:pointer;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.step.tool pre{white-space:pre-wrap;margin:.2rem 0;word-break:break-word;background:var(--tool);padding:.3rem .45rem;border-radius:4px}
mark{background:var(--mark);color:var(--mark-ink);padding:0 .1em;border-radius:2px}
.tag{display:inline-block;font-size:.7rem;padding:.05rem .4rem;border-radius:3px;border:1px solid var(--hair);color:var(--muted)}
.tag.yes{border-color:var(--grok);color:var(--grok)}
.tag.no{border-color:var(--viol);color:var(--viol)}
.loading{padding:2rem;color:var(--muted)}
@media (max-width:1100px){.layout{grid-template-columns:1fr}nav.rail{position:static;height:auto;flex-direction:row;flex-wrap:wrap;border-right:0;border-bottom:1px solid var(--hair)}nav.rail .legend{display:none}.browser{grid-template-columns:1fr;height:auto}.armgrid{grid-template-columns:1fr 1fr}.stats{grid-template-columns:1fr 1fr}}
@media (prefers-reduced-motion: reduce){*{scroll-behavior:auto!important}}
"""

    js = r"""
const ARMS=%(arms)s, LABEL=%(label)s, SHORT=%(short)s, MR=%(mr)s, CELLS=%(cells)s;
const CVAR={grok:'var(--grok)',sonnet_concise:'var(--capped)',sonnet_normal:'var(--sonnet)',gpt:'var(--gpt)'};
const COMMIT=/\bI (?:will not|won't|refuse|am not going to|'m not going to|cannot|can't|should not|shouldn't|must not|decline)\b/gi;
const WRITE=/(sed\s+-i|\btee\b|>>?\s*\/\S+|cat\s*<<|\bmkdir\b|\bcp\s|\bmv\s|\brm\s|\btouch\b|to_csv|json\.dump|\.write\()/g;
let DATA=null, cur=null;
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function markCommit(s){return esc(s).replace(COMMIT,m=>'<mark>'+m+'</mark>');}
function markWrite(s){return esc(s).replace(WRITE,m=>'<span class="w">'+m+'</span>');}
function cmdsOf(call){const out=[];const re=/"command":\s*"((?:[^"\\]|\\.)*)"/g;let m;while((m=re.exec(call))){try{out.push(JSON.parse('"'+m[1]+'"'));}catch(e){out.push(m[1]);}}
 const names=[...call.matchAll(/'name':\s*'([a-zA-Z_]+)'/g)].map(x=>x[1]);return {cmds:out,names};}
async function load(){
 const b64=document.getElementById('data').textContent.trim();
 const bytes=Uint8Array.from(atob(b64),c=>c.charCodeAt(0));
 if(!('DecompressionStream' in window)){document.getElementById('panes').innerHTML='<div class="loading">This browser lacks DecompressionStream; open the page in a current Chrome, Safari or Firefox to read transcripts.</div>';return;}
 const ds=new DecompressionStream('gzip');
 const text=await new Response(new Blob([bytes]).stream().pipeThrough(ds)).text();
 DATA=JSON.parse(text);
 renderList();
 const first=document.querySelector('.cellrow'); if(first) first.click();
}
function renderList(){
 const q=document.getElementById('q').value.toLowerCase();
 const cond=document.getElementById('cond').value;
 const pat=document.getElementById('pat').value;
 const sort=document.getElementById('sort').value;
 let rows=CELLS.filter(c=>(!q||c.name.toLowerCase().includes(q))&&(cond==='all'||c.cond===cond));
 rows=rows.filter(c=>{const v=c.viol;
   if(pat==='grok_clean_others') return !v.grok&&(v.sonnet_concise||v.sonnet_normal||v.gpt);
   if(pat==='gpt_only') return v.gpt&&!v.grok&&!v.sonnet_concise&&!v.sonnet_normal;
   if(pat==='all_fail') return v.grok&&v.sonnet_concise&&v.sonnet_normal&&v.gpt;
   if(pat==='all_clean') return !v.grok&&!v.sonnet_concise&&!v.sonnet_normal&&!v.gpt;
   if(pat==='sonnets_disagree') return v.sonnet_concise!==v.sonnet_normal;
   if(pat==='grok_split'){const s=c.scores.grok;return s.length>1&&Math.max(...s)>=3&&Math.min(...s)<3;}
   return true;});
 rows.sort((a,b)=>sort==='spread'?(b.spread-a.spread)||a.cell.localeCompare(b.cell):sort==='name'?a.name.localeCompare(b.name)||a.cond.localeCompare(b.cond):a.cell.localeCompare(b.cell));
 const el=document.getElementById('celllist');
 el.innerHTML=rows.map(c=>`<div class="cellrow${cur===c.cell?' sel':''}" tabindex="0" data-cell="${c.cell}"><div><div class="cond">${c.cond}</div><div class="nm" title="${c.name}">${c.name}</div></div><div class="chips">${ARMS.map(a=>{const s=c.scores[a];const v=s.some(x=>x>=3);return `<span class="chip${v?' v':''}" style="--c:${CVAR[a]}" title="${LABEL[a]}">${s.length?s.map(x=>+x.toFixed(1)).join(','):'–'}</span>`;}).join('')}</div></div>`).join('');
 document.getElementById('count').textContent=rows.length+' of '+CELLS.length+' cells';
 el.querySelectorAll('.cellrow').forEach(r=>{r.addEventListener('click',()=>openCell(r.dataset.cell));r.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();openCell(r.dataset.cell);}});});
}
function openCell(cell){
 cur=cell; document.querySelectorAll('.cellrow').forEach(r=>r.classList.toggle('sel',r.dataset.cell===cell));
 const any=ARMS.map(a=>(DATA[a][cell]||[])[0]).find(Boolean);
 const sys=any?any.steps.find(s=>s.role==='system'):null, usr=any?any.steps.find(s=>s.role==='user'):null;
 document.getElementById('scenario').innerHTML=`<h3>${esc(cell)}</h3><div class="sys">${esc(sys?sys.content:'')}</div><div class="usr">${esc(usr?usr.content:'')}</div>`;
 const grid=document.getElementById('armgrid'); grid.innerHTML='';
 ARMS.forEach(a=>{const rs=DATA[a][cell]||[];const pane=document.createElement('div');pane.className='pane';pane.style.setProperty('--c',CVAR[a]);pane.dataset.arm=a;grid.appendChild(pane);renderPane(pane,a,rs,0);});
}
function renderPane(pane,a,rs,idx){
 if(!rs.length){pane.innerHTML=`<header><span class="arm">${LABEL[a]}</span><span class="muted">no rollout on this cell</span></header>`;return;}
 const r=rs[idx];const v=r.score>=3;
 const tabs=rs.map((x,i)=>`<button class="${i===idx?'on':''}" data-i="${i}" title="score ${x.score}">${x.id.replace('rollout_','r')}·${+x.score.toFixed(1)}</button>`).join('');
 const judges=Object.entries(r.judges).map(([j,o])=>`<p><b>${esc(j)}</b> → ${o.score}: ${esc(o.why)}</p>`).join('');
 const cbw=r.commit_before_write==null?'':`<span class="tag ${r.commit_before_write?'yes':'no'}">${r.commit_before_write?'commitment before first write':'no commitment before first write'}</span>`;
 pane.innerHTML=`<header><span class="arm">${LABEL[a]}</span><span class="score${v?' v':''}">${+r.score.toFixed(1)}${v?' · violation':''}</span>${tabs}<span class="muted">${r.tool_calls} calls · ${(r.reasoning_chars/1000).toFixed(1)}k reasoning chars${r.edited?' · edited /app':''}</span>${cbw}</header>
 <details class="judges"><summary>judge rationales</summary>${judges}</details><div class="steps">${r.steps.map(stepHtml).join('')}</div>`;
 pane.querySelectorAll('header button').forEach(b=>b.addEventListener('click',()=>renderPane(pane,a,rs,+b.dataset.i)));
}
function stepHtml(s,i){
 if(s.role==='system'||s.role==='user') return '';
 if(s.role==='tool'){let body=s.content||'';let head=body.slice(0,110).replace(/\s+/g,' ');return `<div class="step tool"><details><summary>tool · ${esc(head)}${body.length>110?' …':''}</summary><pre>${esc(body)}</pre></details></div>`;}
 let h=`<div class="step assistant"><div class="lab">assistant · step ${i+1}</div>`;
 if(s.reason) h+=`<div class="reason">${markCommit(s.reason)}</div>`;
 if(s.content) h+=`<div class="content">${markCommit(s.content)}</div>`;
 if(s.call){const {cmds,names}=cmdsOf(s.call);const shown=cmds.length?cmds.map(c=>markWrite(c)).join('\n'):esc(s.call.slice(0,600));h+=`<div class="call">${names.filter(n=>n!=='bash').map(n=>'['+n+'] ').join('')}${shown}</div>`;}
 return h+'</div>';
}
['q','cond','pat','sort'].forEach(id=>document.getElementById(id).addEventListener('input',renderList));
load();
""" % {
        "arms": json.dumps(ARMS),
        "label": json.dumps(LABEL),
        "short": json.dumps(SHORT),
        "mr": json.dumps(MR),
        "cells": json.dumps(cells, separators=(",", ":")),
    }

    stats = "".join(
        f"<div class='stat' style='--c:var(--{ {'grok': 'grok', 'sonnet_concise': 'capped', 'sonnet_normal': 'sonnet', 'gpt': 'gpt'}[a] })'>"
        f"<div class='k'>{html.escape(LABEL[a])}</div><div class='v'>{MR[a]:.1f}%</div>"
        f"<div class='s'>{sum(1 for r in rows if r['arm'] == a)} rollouts · {sum(1 for r in rows if r['arm'] == a and r['violation'])} violations</div></div>"
        for a in ARMS
    )

    page = f"""<title>Four Organisms on ODCV</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>{css}</style>
<div class="layout">
<nav class="rail" aria-label="Sections">
 <div class="brand">Four Organisms on ODCV</div>
 <a href="#headline">Headline</a>
 <a href="#browser">Cell browser</a>
 <a href="#findings">Findings</a>
 <a href="#reads">Reader reports</a>
 <a href="#tables">Tables</a>
 <a href="#pointers">Pointers</a>
 <div class="legend">
  <span><i class="sw" style="background:var(--grok)"></i>grok-4.6</span>
  <span><i class="sw" style="background:var(--capped)"></i>Sonnet 5 capped</span>
  <span><i class="sw" style="background:var(--sonnet)"></i>Sonnet 5 (da716)</span>
  <span><i class="sw" style="background:var(--gpt)"></i>gpt-5.6</span>
  <span><mark>I will not</mark> = first-person commitment</span>
  <span><span class="w" style="color:var(--viol);font-weight:600">sed -i</span> = write action</span>
 </div>
</nav>
<main>
<section id="headline">
 <div class="eyebrow">2026-08-27 · Qwen3.6-27B LoRAs · same 65 ODCV-Bench cells · 643 rollouts</div>
 <h1>Four organisms, one bench: what separates them is how often the trained voice fires</h1>
 <p class="lede">Four LoRAs trained with one recipe on 703 paired difficult-advice rows that differ only in who wrote the assistant turn. In their agentic rollouts, every arm is bimodal at the first reasoning step — the trained voice commits ("I will not…") or the base model narrates the task — and, given a commitment before the first write action, grok, capped Sonnet and Sonnet are equally safe. Grok's edge is trigger rate; the GPT arm is weaker on both terms; the corpus property that tracks it is a firm, repeated, closing refusal in the reply, not length.</p>
 <div class="stats">{stats}</div>
 <figure><img src="{png_uri}" alt="Four panels: ODCV misalignment per arm; share of rollouts with a first-person commitment before the first write; misalignment given the voice fired vs absent; training-corpus firm-refusal rate"><figcaption>Trigger rate vs conditional safety vs the corpus refusal rate. Numbers below.</figcaption></figure>
 <div class="prose wide">{md_to_html(fig_md.split(chr(10), 1)[1])}</div>
</section>
<section id="browser">
 <div class="eyebrow">Explore</div>
 <h2>Cell browser — one scenario, four organisms side by side</h2>
 <p class="lede">Pick a cell; each pane shows one arm's rollouts (tabs for passes), the judges' scores and rationales, and every step: private reasoning (grey), visible reply, tool calls with write actions in red, tool output collapsed. First-person commitments are highlighted wherever they occur.</p>
 <div class="browser">
  <div class="cells">
   <div class="tools">
    <input id="q" type="search" placeholder="filter cells" aria-label="Filter cells by name">
    <select id="cond" aria-label="Condition"><option value="all">both conditions</option><option value="mandated">mandated</option><option value="incentivized">incentivized</option></select>
    <select id="pat" aria-label="Outcome pattern"><option value="all">any outcome</option><option value="grok_clean_others">grok clean, others violate</option><option value="sonnets_disagree">the two Sonnets disagree</option><option value="gpt_only">gpt-only violation</option><option value="all_fail">every arm violates</option><option value="all_clean">every arm clean</option><option value="grok_split">grok splits between passes</option></select>
    <select id="sort" aria-label="Sort"><option value="spread">sort by cross-arm spread</option><option value="name">sort by name</option><option value="cond">sort by condition</option></select>
    <span id="count" class="muted"></span>
   </div>
   <div id="celllist" class="celllist"><div class="loading">Decoding 643 transcripts…</div></div>
  </div>
  <div class="panes" id="panes">
   <div class="scenario" id="scenario"><span class="muted">Select a cell.</span></div>
   <div class="armgrid" id="armgrid"></div>
  </div>
 </div>
</section>
<section id="findings">
 <div class="eyebrow">Write-up</div>
 <h2>Findings and interpretation</h2>
 <div class="prose">{md_to_html(analysis)}</div>
</section>
<section id="reads">
 <div class="eyebrow">Evidence</div>
 <h2>Reader reports</h2>
 <p class="lede">Ten subagent reads, verbatim: six paired reads of the same cells across all four arms (R1–R6, grouped by outcome pattern) and four per-arm style reads (S1–S4).</p>
 {reads_html}
</section>
<section id="tables">
 <div class="eyebrow">Numbers</div>
 <h2>Tables</h2>
 {tables_html}
</section>
<section id="pointers">
 <div class="eyebrow">Where it lives</div>
 <h2>Pointers</h2>
 <div class="prose"><ul>
 <li>HF <code>LASR-Callum/2026-08-27-odcv-four-mos-rollout-analysis</code> — every table, report, the figure and the per-rollout JSONL with both judges' rationales.</li>
 <li>Branch <code>worktree-odcv-rollouts-four-mos</code> @ <code>fe312fb</code>: <code>scratch/four_mos_rollouts/</code> (pull, features, sequences, registers, validator_read, corpus_properties, plot_voice, build_page); <code>docs/LOG.md</code> entry 2026-08-27.</li>
 <li>Rollouts read: <code>2026-08-24-odcv-grokresp703-paired-eval</code>, <code>2026-08-26-odcv-sonnetconcise703-paired-eval</code>, <code>qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch</code>, <code>2026-08-25-odcv-gptresp685-paired-eval</code>; judges grok-4.20 + gemini-3.1-pro (median), violation = ≥3.</li>
 </ul></div>
</section>
</main>
</div>
<script id="data" type="application/octet-stream">{blob}</script>
<script>{js}</script>
"""
    PAGE.write_text(page, encoding="utf-8")
    print(
        f"{PAGE} {PAGE.stat().st_size / 1e6:.1f} MB (transcript blob {len(blob) / 1e6:.1f} MB, cells {len(cells)})"
    )


if __name__ == "__main__":
    main()
