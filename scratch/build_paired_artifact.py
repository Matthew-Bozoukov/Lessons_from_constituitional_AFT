# ABOUTME: Build a self-contained HTML browser that shows each difficult-advice scenario with the
# ABOUTME: ORIGINAL (unablated) and the BOTH-MOVES-RESTORED (bothpruned) reasoning+reply side by side.
import json
from pathlib import Path

recs = json.loads(Path("scratch/paired_662.json").read_text())
data = json.dumps(recs, ensure_ascii=False).replace("</", "<\\/").replace("<!--", "<\\!--")
OUT = Path("/tmp/claude-1000/-home-matthewb-git-repos-teaching-claude-why-replication/39410c65-1144-424a-b9fa-97478243bd29/scratchpad/bothpruned_examples.html")

HTML = r"""<title>Difficult-Advice 662</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600&family=Inter:wght@400;500;600;700&display=swap">
<style>
:root{
  --bg:#f7f7f4; --surface:#ffffff; --surface-2:#fbfbf9; --ink:#1b1b20; --muted:#6c6c78;
  --faint:#93939f; --border:#e6e5df; --border-strong:#d8d7cf;
  --think:#2f6f8f; --think-bg:#eef4f7; --think-bar:#2f6f8f;
  --reply:#3f7a52; --reply-bg:#eef4ef; --reply-bar:#3f7a52;
  --orig:#9a6b2f; --orig-bg:#f6f0e6; --orig-bar:#c69a55;
  --rest:#3f7a52; --rest-bg:#edf4ee; --rest-bar:#5c9a6c;
  --accent:#2f6f8f; --sel:#e9eef1;
  --sans:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;
  --serif:'Source Serif 4',Georgia,'Times New Roman',serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#111318; --surface:#191c22; --surface-2:#15181d; --ink:#e9e9ee; --muted:#9a9aa6;
  --faint:#71717d; --border:#282c34; --border-strong:#333843;
  --think:#79b4d2; --think-bg:#16232b; --think-bar:#4f93b3;
  --reply:#84bd93; --reply-bg:#16241a; --reply-bar:#5c9a6c;
  --orig:#cfa267; --orig-bg:#241d12; --orig-bar:#7d5f30;
  --rest:#84bd93; --rest-bg:#16241a; --rest-bar:#5c9a6c;
  --accent:#79b4d2; --sel:#1e2831;
}}
:root[data-theme="dark"]{
  --bg:#111318; --surface:#191c22; --surface-2:#15181d; --ink:#e9e9ee; --muted:#9a9aa6;
  --faint:#71717d; --border:#282c34; --border-strong:#333843;
  --think:#79b4d2; --think-bg:#16232b; --think-bar:#4f93b3;
  --reply:#84bd93; --reply-bg:#16241a; --reply-bar:#5c9a6c;
  --orig:#cfa267; --orig-bg:#241d12; --orig-bar:#7d5f30;
  --rest:#84bd93; --rest-bg:#16241a; --rest-bar:#5c9a6c;
  --accent:#79b4d2; --sel:#1e2831;
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased}
.app{display:grid;grid-template-columns:320px 1fr;grid-template-rows:auto 1fr;
  grid-template-areas:"head head" "side main";height:100vh}
header{grid-area:head;display:flex;align-items:center;gap:16px;flex-wrap:wrap;
  padding:11px 20px;border-bottom:1px solid var(--border);background:var(--surface);z-index:5}
.brand{display:flex;flex-direction:column;gap:1px;min-width:0}
.brand h1{margin:0;font-size:15px;font-weight:700;letter-spacing:-.01em;white-space:nowrap}
.brand .sub{font-size:11.5px;color:var(--muted);white-space:nowrap}
.count{font-variant-numeric:tabular-nums;color:var(--muted);font-size:12.5px;
  padding:3px 9px;border:1px solid var(--border);border-radius:999px;white-space:nowrap}
.search{flex:1;min-width:170px;display:flex;align-items:center;gap:8px;
  background:var(--surface-2);border:1px solid var(--border-strong);border-radius:9px;padding:7px 11px}
.search input{flex:1;border:0;background:transparent;color:var(--ink);font:inherit;outline:none}
.search svg{flex:none;color:var(--faint)}
.tbtn{border:1px solid var(--border-strong);background:var(--surface-2);color:var(--muted);
  border-radius:8px;padding:7px 10px;cursor:pointer;font:inherit;font-size:13px}
.tbtn:hover{color:var(--ink)}

aside{grid-area:side;border-right:1px solid var(--border);overflow-y:auto;background:var(--surface-2)}
.row{display:flex;gap:10px;padding:11px 15px;border-bottom:1px solid var(--border);cursor:pointer;align-items:baseline}
.row:hover{background:var(--sel)}
.row.on{background:var(--sel);box-shadow:inset 3px 0 0 var(--accent)}
.row .n{font-variant-numeric:tabular-nums;color:var(--faint);font-size:12px;font-weight:600;min-width:28px}
.row .snip{font-size:12.6px;color:var(--muted);line-height:1.35;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.row.on .snip{color:var(--ink)}
.empty{padding:24px 16px;color:var(--muted);font-size:13px}

main{grid-area:main;overflow-y:auto}
.pane{max-width:1220px;margin:0 auto;padding:22px 26px 90px}
.exhead{display:flex;align-items:center;gap:12px;margin-bottom:2px}
.exhead .idx{font-size:22px;font-weight:700;letter-spacing:-.02em}
.nav{margin-left:auto;display:flex;gap:6px}
.nav button{border:1px solid var(--border-strong);background:var(--surface);color:var(--ink);
  border-radius:8px;width:34px;height:32px;cursor:pointer;font-size:15px}
.nav button:hover{background:var(--sel)}
.nav button:disabled{opacity:.4;cursor:default}

.block{margin-top:14px;border:1px solid var(--border);border-radius:12px;overflow:hidden;background:var(--surface)}
.block>.lbl{display:flex;align-items:center;gap:8px;padding:9px 15px;font-size:11px;
  font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
  border-bottom:1px solid var(--border);background:var(--surface-2)}
.block .body{padding:14px 17px}
.prose{font-family:var(--serif);font-size:15.5px;line-height:1.66;color:var(--ink)}
.prose p{margin:0 0 .8em}.prose p:last-child{margin-bottom:0}
.b-sys .lbl{color:var(--faint)}
.b-sys .body{font-size:13.5px;color:var(--muted);line-height:1.5}
.b-user{border-color:var(--border-strong)}
.meta{font-size:11px;color:var(--faint);font-weight:500;text-transform:none;letter-spacing:0;margin-left:auto;font-family:var(--sans);font-variant-numeric:tabular-nums}
details.b-sys>.lbl{cursor:pointer;list-style:none}
details.b-sys>.lbl::-webkit-details-marker{display:none}
details.b-sys>.lbl .chev{margin-left:auto;transition:transform .15s}
details.b-sys[open]>.lbl .chev{transform:rotate(90deg)}

.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px;align-items:start}
.col{min-width:0}
.colhead{display:flex;align-items:center;gap:9px;padding:8px 4px 10px;position:sticky;top:0}
.colhead .pill{font-size:11px;font-weight:700;letter-spacing:.04em;padding:3px 11px;border-radius:999px;border:1px solid}
.col-orig .pill{color:var(--orig);background:var(--orig-bg);border-color:var(--orig-bar)}
.col-rest .pill{color:var(--rest);background:var(--rest-bg);border-color:var(--rest-bar)}
.colhead .csub{font-size:11.5px;color:var(--faint)}
.dot{width:7px;height:7px;border-radius:50%;flex:none}
.b-think>.lbl{color:var(--think);background:var(--think-bg);border-bottom-color:var(--think-bar)}
.b-think{border-color:var(--think-bar)}.b-think .dot{background:var(--think-bar)}
.b-reply>.lbl{color:var(--reply);background:var(--reply-bg);border-bottom-color:var(--reply-bar)}
.b-reply{border-color:var(--reply-bar)}.b-reply .dot{background:var(--reply-bar)}
.col-orig .block{border-top:3px solid var(--orig-bar)}
.col-rest .block{border-top:3px solid var(--rest-bar)}

mark{background:#ffe58a;color:#1b1b20;border-radius:2px;padding:0 1px}
:root[data-theme="dark"] mark{background:#8a6d1f;color:#fff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) mark{background:#8a6d1f;color:#fff}}

@media(max-width:900px){
  .cols{grid-template-columns:1fr}
  .app{grid-template-columns:1fr;grid-template-rows:auto auto 1fr;grid-template-areas:"head" "side" "main"}
  aside{max-height:30vh}
  .pane{padding:18px 16px 80px}
}
</style>

<div class="app">
  <header>
    <div class="brand">
      <h1>Difficult-Advice — original vs. both restored</h1>
      <div class="sub">662 examples · original synthdoc-716 vs bothpruned-662 @4929517f</div>
    </div>
    <span class="count" id="count"></span>
    <label class="search">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>
      <input id="q" type="search" placeholder="Search scenario / reasoning / response…" autocomplete="off">
    </label>
    <button class="tbtn" id="theme" title="Toggle theme">◐</button>
  </header>
  <aside id="index"></aside>
  <main><div class="pane" id="pane"></div></main>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
const DATA=JSON.parse(document.getElementById('data').textContent);
const index=document.getElementById('index'),pane=document.getElementById('pane');
const qEl=document.getElementById('q'),countEl=document.getElementById('count');
let filtered=DATA.map((_,i)=>i),cur=0,query='';
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function hl(s){const t=esc(s);if(!query)return t;
  const rx=new RegExp('('+query.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+')','ig');
  return t.replace(rx,'<mark>$1</mark>');}
function paras(s){return s.split(/\n\n+/).map(p=>'<p>'+hl(p).replace(/\n/g,'<br>')+'</p>').join('');}
const wc=s=>s.trim()?s.trim().split(/\s+/).length:0;
function block(kind,label,text){
  return '<div class="block b-'+kind+'"><div class="lbl"><span class="dot"></span>'+label+
    '<span class="meta">'+wc(text)+' words</span></div><div class="body prose">'+paras(text)+'</div></div>';}

function renderIndex(){
  countEl.textContent=filtered.length===DATA.length?DATA.length+' examples':filtered.length+' / '+DATA.length;
  if(!filtered.length){index.innerHTML='<div class="empty">No examples match “'+esc(query)+'”.</div>';return;}
  index.innerHTML=filtered.map(i=>{const r=DATA[i];
    return '<div class="row'+(i===cur?' on':'')+'" data-i="'+i+'"><span class="n">'+(i+1)+
      '</span><span class="snip">'+hl(r.user.slice(0,150))+'</span></div>';}).join('');
  const on=index.querySelector('.row.on');if(on)on.scrollIntoView({block:'nearest'});
}
function renderPane(){
  const r=DATA[cur],pos=filtered.indexOf(cur);
  pane.innerHTML=
    '<div class="exhead"><span class="idx">#'+(cur+1)+'</span>'+
      '<div class="nav"><button id="prev" title="Previous (k / ←)">‹</button>'+
      '<button id="next" title="Next (j / →)">›</button></div></div>'+
    '<details class="block b-sys"><summary class="lbl">System role<span class="chev">›</span></summary>'+
      '<div class="body">'+hl(r.system)+'</div></details>'+
    '<div class="block b-user"><div class="lbl">User · the scenario (identical for both)</div>'+
      '<div class="body prose">'+paras(r.user)+'</div></div>'+
    '<div class="cols">'+
      '<div class="col col-orig"><div class="colhead"><span class="pill">Original</span>'+
        '<span class="csub">unablated · synthdoc-716</span></div>'+
        block('think','Reasoning',r.orig.think)+block('reply','Response',r.orig.reply)+'</div>'+
      '<div class="col col-rest"><div class="colhead"><span class="pill">Both moves restored</span>'+
        '<span class="csub">branch-pruned · bothpruned-662</span></div>'+
        block('think','Reasoning',r.bp.think)+block('reply','Response',r.bp.reply)+'</div>'+
    '</div>';
  document.getElementById('prev').onclick=()=>step(-1);
  document.getElementById('next').onclick=()=>step(1);
  document.getElementById('prev').disabled=pos<=0;
  document.getElementById('next').disabled=pos>=filtered.length-1;
}
function select(i){cur=i;renderIndex();renderPane();document.querySelector('main').scrollTop=0;}
function step(d){const pos=filtered.indexOf(cur),np=pos+d;if(np>=0&&np<filtered.length)select(filtered[np]);}
index.addEventListener('click',e=>{const row=e.target.closest('.row');if(row)select(+row.dataset.i);});
let tmr;qEl.addEventListener('input',()=>{clearTimeout(tmr);tmr=setTimeout(()=>{
  query=qEl.value.trim();const ql=query.toLowerCase();
  filtered=query?DATA.filter(r=>(r.user+' '+r.orig.think+' '+r.orig.reply+' '+r.bp.think+' '+r.bp.reply+' '+r.system).toLowerCase().includes(ql)).map(r=>r.i):DATA.map((_,i)=>i);
  if(!filtered.includes(cur)&&filtered.length)cur=filtered[0];
  renderIndex();renderPane();},130);});
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;
  if(e.key==='j'||e.key==='ArrowRight'){step(1);e.preventDefault();}
  if(e.key==='k'||e.key==='ArrowLeft'){step(-1);e.preventDefault();}
  if(e.key==='/'){qEl.focus();e.preventDefault();}});
const themeBtn=document.getElementById('theme');
themeBtn.onclick=()=>{const c=document.documentElement.getAttribute('data-theme');
  const next=c==='dark'?'light':c==='light'?'dark':(matchMedia('(prefers-color-scheme:dark)').matches?'light':'dark');
  document.documentElement.setAttribute('data-theme',next);
  try{localStorage.setItem('da662-theme',next);}catch(e){}};
try{const s=localStorage.getItem('da662-theme');if(s)document.documentElement.setAttribute('data-theme',s);}catch(e){}
renderIndex();renderPane();
</script>
"""
OUT.write_text(HTML.replace("__DATA__", data), encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")
