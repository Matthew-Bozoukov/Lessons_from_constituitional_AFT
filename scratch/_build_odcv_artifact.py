# ABOUTME: Master-detail artifact of the 45 da400-base ODCV misaligned rollouts (judge severity >=3): the task,
# ABOUTME: the judge's verdict, and the agent's tool-use trajectory (reasoning, bash commands, outputs).
import json
from pathlib import Path

mis = json.loads(Path("scratch/_odcv_misaligned.json").read_text())
# short gist per rollout = first sentence of judge verdict
for m in mis:
    j = m["judge"].strip()
    m["gist"] = (j.split(". ")[0])[:120]
DATA = json.dumps(mis, ensure_ascii=False)
print(f"{len(mis)} rollouts, {len(DATA)/1e6:.2f} MB")

TPL = r"""<title>ODCV Misalignment Trajectories</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root{
  --bg:#f1efea;--surface:#fdfcfa;--panel:#f6f4ef;--console:#f4f2ec;--border:#dedacf;--text:#1a1815;--dim:#5a564d;--faint:#8a8578;
  --s4:#b0402f;--s4-bg:#f5e2dd;--s3:#a5741f;--s3-bg:#f4ebda;--accent:#8a4030;
  --mand:#7a3a2e;--inc:#3f5a52;--cmd:#7a5a2e;
  --serif:"Newsreader",Georgia,serif;--mono:"IBM Plex Mono",ui-monospace,monospace;--sans:"IBM Plex Sans",system-ui,sans-serif;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#131210;--surface:#1b1a15;--panel:#201e18;--console:#100f0c;--border:#332f27;--text:#e9e5db;--dim:#a49e8f;--faint:#6f6a5c;
  --s4:#e08573;--s4-bg:#2a1712;--s3:#d9a441;--s3-bg:#251d0e;--accent:#d99277;
  --mand:#e0977f;--inc:#8fbaad;--cmd:#d6b483;}}
:root[data-theme="dark"]{
  --bg:#131210;--surface:#1b1a15;--panel:#201e18;--console:#100f0c;--border:#332f27;--text:#e9e5db;--dim:#a49e8f;--faint:#6f6a5c;
  --s4:#e08573;--s4-bg:#2a1712;--s3:#d9a441;--s3-bg:#251d0e;--accent:#d99277;
  --mand:#e0977f;--inc:#8fbaad;--cmd:#d6b483;}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:var(--serif);line-height:1.6;-webkit-font-smoothing:antialiased}
.app{display:grid;grid-template-columns:minmax(310px,380px) 1fr;height:100vh;height:100dvh}
.scrim{position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:19;opacity:0;pointer-events:none;transition:opacity .18s}
.side{display:flex;flex-direction:column;border-right:1px solid var(--border);background:var(--surface);min-height:0}
.side-head{padding:17px 17px 13px;border-bottom:1px solid var(--border)}
.kicker{font-family:var(--sans);font-size:10.5px;font-weight:600;letter-spacing:.13em;text-transform:uppercase;color:var(--accent)}
.side-head h1{font-family:var(--serif);font-weight:600;font-size:21px;letter-spacing:-.015em;margin:3px 0 6px}
.side-head .sub{font-family:var(--sans);font-size:11.5px;color:var(--dim);line-height:1.5}
.side-head .sub b{color:var(--text);font-weight:600}
.search{width:100%;margin-top:11px;border:1px solid var(--border);background:var(--bg);color:var(--text);border-radius:8px;padding:8px 10px;font-size:13px;font-family:var(--sans)}
.search:focus{outline:2px solid var(--accent);outline-offset:-1px}
.list{overflow-y:auto;padding:8px;display:flex;flex-direction:column;gap:3px;min-height:0}
.row{flex-shrink:0;text-align:left;width:100%;border:1px solid transparent;border-radius:9px;background:none;color:inherit;padding:9px 11px;cursor:pointer;font-family:var(--sans);display:flex;gap:9px;align-items:flex-start}
.row:hover{background:var(--panel);border-color:var(--border)}
.row[aria-current="true"]{background:var(--s4-bg);border-color:var(--s4)}
.row .sv{flex:none;width:20px;height:20px;border-radius:5px;font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center;font-family:var(--mono);margin-top:1px}
.sv.s4{background:var(--s4);color:#fff}.sv.s3{background:var(--s3);color:#fff}
:root[data-theme="dark"] .sv.s4,:root[data-theme="dark"] .sv.s3{color:#17150f}
.row .rt{min-width:0}
.row .sc{display:block;font-size:12.5px;line-height:1.3;color:var(--text);font-weight:600}
.row .vr{display:block;font-size:10px;letter-spacing:.03em;text-transform:uppercase;font-weight:600;margin-top:2px}
.vr.mandated{color:var(--mand)}.vr.incentivized{color:var(--inc)}
.detail{overflow-y:auto;min-height:0;background:var(--bg)}
.dwrap{max-width:760px;margin:0 auto;padding:32px 32px 110px}
.eyebrow{display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-family:var(--sans);font-size:11px}
.chip{font-family:var(--sans);font-size:10px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;padding:3px 9px;border-radius:999px;border:1px solid var(--border)}
.chip.mand{color:var(--mand);border-color:var(--mand)}.chip.inc{color:var(--inc);border-color:var(--inc)}
.chip.sev4{background:var(--s4);border-color:var(--s4);color:#fff}.chip.sev3{background:var(--s3);border-color:var(--s3);color:#fff}
:root[data-theme="dark"] .chip.sev4,:root[data-theme="dark"] .chip.sev3{color:#17150f}
.themebtn{border:1px solid var(--border);background:var(--surface);color:var(--dim);border-radius:8px;padding:5px 10px;font-size:11px;font-family:var(--sans);cursor:pointer}
.mob{display:none}
h2.dt{font-family:var(--serif);font-weight:600;font-size:25px;letter-spacing:-.02em;margin:12px 0 3px;text-wrap:balance}
.seclabel{font-family:var(--sans);font-size:10.5px;font-weight:600;letter-spacing:.11em;text-transform:uppercase;color:var(--faint);margin:26px 0 11px;padding-bottom:7px;border-bottom:1px solid var(--border)}
.verdict{background:var(--s4-bg);border:1px solid var(--s4);border-left:3px solid var(--s4);border-radius:11px;padding:14px 17px;font-family:var(--sans);font-size:14px;line-height:1.62;color:var(--text)}
.verdict.s3{background:var(--s3-bg);border-color:var(--s3);border-left-color:var(--s3)}
.verdict .vh{font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin-bottom:6px}
.task{background:var(--panel);border:1px solid var(--border);border-radius:11px;padding:0;overflow:hidden}
.task .trow{padding:12px 16px;font-family:var(--sans);font-size:13.5px;line-height:1.6;color:var(--dim)}
.task .trow+.trow{border-top:1px solid var(--border)}
.task .lbl{font-size:9.5px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);display:block;margin-bottom:4px}
.step{margin:0 0 4px}
.reason{font-size:16px;line-height:1.6;color:var(--text);margin:14px 0 8px}
.reason .rl{font-family:var(--sans);font-size:9.5px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);display:block;margin-bottom:5px}
.reply{font-size:16px;line-height:1.6;color:var(--text);background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:13px 16px;margin:10px 0}
.reply .rl{font-family:var(--sans);font-size:9.5px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);display:block;margin-bottom:5px}
.console{background:var(--console);border:1px solid var(--border);border-radius:9px;padding:11px 13px;margin:8px 0;font-family:var(--mono);font-size:12px;line-height:1.55;overflow-x:auto;white-space:pre-wrap;word-break:break-word}
.cmd{color:var(--cmd)}.cmd::before{content:"$ ";color:var(--faint)}
.out{color:var(--dim)}
.nudge{font-family:var(--sans);font-size:11px;color:var(--faint);font-style:italic;margin:10px 0;padding-left:12px;border-left:2px solid var(--border)}
.tc{font-family:var(--mono);font-size:11.5px;color:var(--accent);margin:8px 0;font-weight:600}
@media (max-width:900px){
  .app{grid-template-columns:1fr}
  .side{position:fixed;inset:0 auto 0 0;width:88%;max-width:380px;z-index:20;transform:translateX(-100%);transition:transform .18s;box-shadow:2px 0 24px rgba(0,0,0,.3)}
  .app.open .side{transform:none}.app.open .scrim{opacity:1;pointer-events:auto}
  .mob{display:inline-flex}.dwrap{padding:20px 17px 90px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
<div class="app" id="app">
  <div class="scrim" id="scrim"></div>
  <aside class="side">
    <div class="side-head">
      <div class="kicker">ODCV-Bench &middot; spec-gaming</div>
      <h1>Where da400 gamed the metric</h1>
      <div class="sub"><b>45</b> misaligned rollouts (judge severity &ge;3) from the da400 base arm &mdash; out of 400 (40 scenarios &times; 2 variants &times; 5 passes). MR 11.2%. <b>27</b> at severity 4, <b>18</b> at 3.</div>
      <input class="search" id="search" placeholder="Search scenario / judge / commands…" autocomplete="off">
    </div>
    <div class="list" id="list"></div>
  </aside>
  <main class="detail" id="detail"></main>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const EX=JSON.parse(document.getElementById('data').textContent);
const app=document.getElementById('app'),listEl=document.getElementById('list'),detailEl=document.getElementById('detail');
let sel=0,q='';
const esc=s=>(s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function hay(e){return (e.scenario+' '+e.variant+' '+e.judge+' '+e.steps.map(s=>s.reason+' '+(s.cmds||[]).join(' ')).join(' ')).toLowerCase();}
function items(){const l=q.toLowerCase();return EX.map((e,i)=>({e,i})).filter(({e})=>!l||hay(e).includes(l));}
function scName(s){return s.replace(/-/g,' ');}
function renderList(){
  const it=items();
  listEl.innerHTML=it.length?it.map(({e,i})=>`<button class="row" data-i="${i}" aria-current="${i===sel}">
    <span class="sv s${e.severity}">${e.severity}</span>
    <span class="rt"><span class="sc">${esc(scName(e.scenario))}</span><span class="vr ${e.variant}">${e.variant}</span></span>
  </button>`).join(''):'<div style="padding:20px;color:var(--faint);font-family:var(--sans);font-size:13px">No matches.</div>';
}
function trajectory(steps){
  let h='',firstUser=true;
  for(const s of steps){
    if(s.role==='system') continue;
    if(s.role==='user'){
      if(firstUser){firstUser=false;continue;} // task panel already shows it
      h+=`<div class="nudge">${esc((s.content||'').slice(0,160))}</div>`; continue;
    }
    if(s.role==='assistant'){
      if(s.reason) h+=`<div class="reason"><span class="rl">reasoning</span>${esc(s.reason)}</div>`;
      if(s.cmds&&s.cmds.length) h+=`<div class="console">${s.cmds.map(c=>`<div class="cmd">${esc(c)}</div>`).join('')}</div>`;
      if(s.content) h+=`<div class="reply"><span class="rl">final reply</span>${esc(s.content)}</div>`;
    } else if(s.role==='tool'){
      const o=s.out||{};
      if(o.action){h+=`<div class="tc">✓ ${esc(o.action)}</div>`;continue;}
      const body=[(o.stdout||'').trim(),(o.stderr||'').trim()].filter(Boolean).join('\n');
      if(body) h+=`<div class="console"><div class="out">${esc(body.slice(0,500))}${body.length>500?' …':''}</div></div>`;
    }
  }
  return h;
}
function renderDetail(){
  const e=EX[sel];if(!e){detailEl.innerHTML='';return;}
  const sys=e.steps.find(s=>s.role==='system'), usr=e.steps.find(s=>s.role==='user');
  detailEl.innerHTML=`<div class="dwrap">
    <div class="eyebrow"><button class="themebtn mob" id="openlist">☰ cases</button>
      <span class="chip sev${e.severity}">severity ${e.severity}</span>
      <span class="chip ${e.variant==='mandated'?'mand':'inc'}">${e.variant}</span>
      <span style="color:var(--faint);font-family:var(--sans)">pass ${e.pass}</span>
      <button class="themebtn" id="theme" style="margin-left:auto">◐ theme</button></div>
    <h2 class="dt">${esc(scName(e.scenario))}</h2>
    <div class="seclabel">Judge's verdict — why it's misaligned</div>
    <div class="verdict ${e.severity===3?'s3':''}"><div class="vh">gemini-3-flash · severity ${e.severity}/5</div>${esc(e.judge)}</div>
    <div class="seclabel">The task</div>
    <div class="task">
      ${sys?`<div class="trow"><span class="lbl">System — the agent's role</span>${esc(sys.content)}</div>`:''}
      ${usr?`<div class="trow"><span class="lbl">User — the objective (with the gameable metric)</span>${esc(usr.content)}</div>`:''}
    </div>
    <div class="seclabel">What it did — trajectory</div>
    ${trajectory(e.steps)}
  </div>`;
  const ol=document.getElementById('openlist');if(ol)ol.onclick=()=>app.classList.add('open');
  document.getElementById('theme').onclick=()=>{const r=document.documentElement,c=r.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');r.setAttribute('data-theme',c==='dark'?'light':'dark');};
  detailEl.scrollTop=0;
}
listEl.addEventListener('click',ev=>{const b=ev.target.closest('.row');if(b){sel=+b.dataset.i;app.classList.remove('open');renderList();renderDetail();}});
document.getElementById('search').addEventListener('input',ev=>{q=ev.target.value;const f=items();if(!f.find(x=>x.i===sel)&&f.length)sel=f[0].i;renderList();renderDetail();});
document.getElementById('scrim').addEventListener('click',()=>app.classList.remove('open'));
renderList();renderDetail();
</script>"""
out = Path("output/odcv_misalignment_trajectories.html")
out.write_text(TPL.replace("__DATA__", DATA))
print("wrote", out, f"({out.stat().st_size/1e6:.2f} MB)")
