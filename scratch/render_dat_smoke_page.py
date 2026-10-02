# ABOUTME: Render a DAT smoke run as one HTML page: every row as a full transcript (system, task,
# ABOUTME: exploration tool calls + stdout, supervised turn) with the writer's scenario and the
# ABOUTME: reviser's change note beside it. Usage: <run_dir> <out.html> <title>.
import html
import json
import sys

run, out_path, title = sys.argv[1], sys.argv[2], sys.argv[3]
e = html.escape
rows = [json.loads(l) for l in open(f"{run}/dataset.jsonl")]
s2 = {json.loads(l)["scenario_id"]: json.loads(l) for l in open(f"{run}/stage_2_write_scenarios.jsonl")}
s6 = {json.loads(l)["scenario_id"]: json.loads(l) for l in open(f"{run}/stage_6_revise_environment.jsonl")}
s8 = {json.loads(l)["scenario_id"]: json.loads(l) for l in open(f"{run}/stage_8_revise_responses.jsonl")}


def cmd_of(m):
    if not m.get("tool_calls"):
        return ""
    a = m["tool_calls"][0]["function"]["arguments"]
    a = json.loads(a) if isinstance(a, str) else a
    return a.get("command", "")


def tool_out(m):
    c = m["content"]
    try:
        c = json.loads(c) if isinstance(c, str) else c
    except Exception:
        return str(m["content"])
    return c.get("stdout", "") if isinstance(c, dict) else str(c)


opts, panels = [], []
for i, r in enumerate(rows):
    md, msgs = r["metadata"], r["messages"]
    sid = md["scenario_id"]
    w, v, rw = s2.get(sid, {}), s6.get(sid, {}), s8.get(sid, {})
    label = f"{i + 1}. {md['trait_id']} · {md['domain']}"
    opts.append(f'<option value="{i}">{e(label)}</option>')
    b = [f'<div class="meta"><div class="eyebrow">Principle</div><p class="pr">{e(md["trait_name"])}</p>'
         f'<div class="kv"><span><b>domain</b> {e(md["domain"])}</span><span><b>id</b> {e(sid)}</span></div>'
         f'<div class="eyebrow">Writer\'s scenario (stage 2)</div><p>{e(w.get("situation", ""))}</p>'
         f'<div class="eyebrow">Writer\'s convenient call</div><p>{e(w.get("shortcut", ""))}</p>'
         f'<div class="eyebrow">Reviser\'s change note (stage 6)</div><p class="chg">{e(v.get("refine_changes", ""))}</p>'
         f'<div class="eyebrow">Situation after revision</div><p>{e(md["situation"])}</p>'
         f'<div class="eyebrow">Convenient call after revision</div><p>{e(md["shortcut"])}</p></div>']
    for m in msgs:
        role = m["role"]
        if role == "system":
            b.append(f'<section class="turn sys"><div class="who">system prompt</div><pre class="txt">{e(m["content"])}</pre></section>')
        elif role == "user":
            b.append(f'<section class="turn usr"><div class="who">operator · task message</div><pre class="txt">{e(m["content"])}</pre></section>')
        elif role == "assistant" and not m.get("content"):
            b.append(f'<section class="turn ag ctx"><div class="who">agent · exploration turn <span class="pill">context, not supervised</span></div>'
                     f'<div class="lbl">reasoning</div><pre class="txt think">{e(m.get("reasoning_content", ""))}</pre>'
                     f'<div class="lbl">tool call · bash</div><pre class="code">$ {e(cmd_of(m))}</pre></section>')
        elif role == "tool":
            b.append(f'<section class="turn tool"><div class="who">tool result · stdout</div><pre class="code out">{e(tool_out(m))}</pre></section>')
        elif role == "assistant":
            b.append(f'<section class="turn ag fin"><div class="who">agent · final turn <span class="pill on">supervised</span></div>'
                     f'<div class="lbl">reasoning (trained, hidden at inference)</div><pre class="txt think">{e(m.get("reasoning_content", ""))}</pre>'
                     f'<div class="lbl">response · status report to the operator</div><pre class="txt">{e(m["content"])}</pre>'
                     f'<div class="lbl">tool call · bash</div><pre class="code">$ {e(cmd_of(m))}</pre></section>')
    if rw:
        b.append('<details class="extra"><summary>Response rewrite: grounding audit, what changed, and the draft it replaced</summary>'
                 f'<div class="lbl">audit</div><pre class="txt small">{e(rw.get("rewrite_audit", ""))}</pre>'
                 f'<div class="lbl">changes</div><pre class="txt small">{e(rw.get("rewrite_changes", ""))}</pre>'
                 f'<div class="lbl">draft reasoning</div><pre class="txt small think">{e(rw.get("draft_reasoning", ""))}</pre>'
                 f'<div class="lbl">draft response</div><pre class="txt small">{e(rw.get("draft_response", ""))}</pre>'
                 f'<div class="lbl">draft command</div><pre class="code small">$ {e(rw.get("draft_command", ""))}</pre></details>')
    panels.append(f'<div class="panel" data-i="{i}"{"" if i == 0 else " hidden"}>{"".join(b)}</div>')

page = f'''<title>{e(title)}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap">
<style>
:root{{--bg:#F5F6F3;--surface:#FFFFFF;--surface2:#EEF0EA;--ink:#1B2230;--ink2:#4A5364;--ink3:#7A8391;--line:#D9DDD4;--accent:#1F6F78;--accent-ink:#134C53;--sys:#6B5B95;--usr:#B7791F;--tool:#4A5364;--ag:#1F6F78;--think:#F1F5F4;--good:#2E7D4F;--good-soft:#DDEFE3;--chg:#FBF3E4}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#151A21;--surface:#1D242E;--surface2:#242C38;--ink:#E7EAE3;--ink2:#B4BAC3;--ink3:#828A96;--line:#333C49;--accent:#63B8C0;--accent-ink:#9FDDE2;--sys:#B9A8E0;--usr:#E0A94A;--tool:#B4BAC3;--ag:#63B8C0;--think:#1A2A2C;--good:#6CC28E;--good-soft:#1F3A2A;--chg:#3A3020}}}}
:root[data-theme="dark"]{{--bg:#151A21;--surface:#1D242E;--surface2:#242C38;--ink:#E7EAE3;--ink2:#B4BAC3;--ink3:#828A96;--line:#333C49;--accent:#63B8C0;--accent-ink:#9FDDE2;--sys:#B9A8E0;--usr:#E0A94A;--tool:#B4BAC3;--ag:#63B8C0;--think:#1A2A2C;--good:#6CC28E;--good-soft:#1F3A2A;--chg:#3A3020}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--ink);font-family:"Public Sans",system-ui,sans-serif;font-size:15px;line-height:1.55;margin:0}}
.wrap{{max-width:980px;margin:0 auto;padding-inline:16px;padding-block:20px 64px}}
h1{{font-family:"Newsreader",Georgia,serif;font-weight:600;font-size:28px;margin:0 0 4px;letter-spacing:-.01em;text-wrap:balance}}
.lede{{color:var(--ink2);margin:0 0 18px;max-width:72ch}}
#sel{{font:inherit;font-size:14px;padding:8px 10px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink);max-width:100%;margin-bottom:16px}}
.meta{{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:14px 16px;margin-bottom:14px}}
.meta p{{margin:0 0 10px;max-width:80ch}}.meta .pr{{font-family:"Newsreader",Georgia,serif;font-size:18px;font-weight:600}}
.meta .chg{{background:var(--chg);border-radius:4px;padding:8px 10px}}
.eyebrow{{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink3);font-weight:600;margin-top:6px}}
.kv{{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--ink2);margin:6px 0 12px}}.kv b{{color:var(--ink3);font-weight:600}}
.turn{{border-left:3px solid var(--line);padding:10px 14px;margin:0 0 12px;background:var(--surface);border-radius:0 6px 6px 0}}
.turn.sys{{border-left-color:var(--sys)}}.turn.usr{{border-left-color:var(--usr)}}.turn.tool{{border-left-color:var(--tool);background:var(--surface2)}}.turn.ag{{border-left-color:var(--ag)}}.turn.fin{{border-left-width:5px}}
.who{{font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;font-weight:600;color:var(--ink3);margin-bottom:6px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
.sys .who{{color:var(--sys)}}.usr .who{{color:var(--usr)}}.ag .who{{color:var(--ag)}}
.pill{{font-size:10px;letter-spacing:.05em;padding:2px 7px;border-radius:3px;background:var(--surface2);color:var(--ink3);text-transform:uppercase}}.pill.on{{background:var(--good-soft);color:var(--good)}}
.lbl{{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink3);font-weight:600;margin:10px 0 4px}}
pre.txt{{white-space:pre-wrap;word-wrap:break-word;font-family:inherit;margin:0;font-size:14.5px;line-height:1.55}}
pre.think{{background:var(--think);border-radius:4px;padding:10px 12px;font-family:"Newsreader",Georgia,serif;font-size:15.5px}}
pre.code{{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-size:12.5px;line-height:1.5;background:var(--surface2);border-radius:4px;padding:10px 12px;margin:0;white-space:pre-wrap;word-break:break-all}}
.tool pre.out{{background:var(--surface);white-space:pre;overflow-x:auto;word-break:normal}}
.small{{font-size:13px}}
details.extra{{background:var(--surface);border:1px dashed var(--line);border-radius:6px;padding:10px 14px;margin-top:6px}}
details.extra summary{{cursor:pointer;color:var(--accent-ink);font-weight:600;font-size:14px}}
</style>
<div class="wrap">
<h1>{e(title)}</h1>
<p class="lede">Every row of the run <code>{e(run.split('/')[-1])}</code>, as it trains: system prompt, operator task, exploration turns as context, and the supervised final turn. Above each transcript: the scenario the writer produced at stage 2, the reviser's own change note from stage 6, and the situation as it stands after revision, so a rebuilt row can be seen at a glance.</p>
<select id="sel" aria-label="Choose a transcript">{"".join(opts)}</select>
{"".join(panels)}
</div>
<script>
const sel=document.getElementById('sel'),panels=[...document.querySelectorAll('.panel')];
sel.addEventListener('change',()=>{{panels.forEach(p=>p.hidden=p.dataset.i!==sel.value);window.scrollTo({{top:0}});}});
</script>'''
open(out_path, "w").write(page)
print(out_path, len(page), "chars,", len(rows), "rows")
